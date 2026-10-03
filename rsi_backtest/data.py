"""Pluggable data layer.

Providers all return the SAME normalized frame per symbol:
    index : tz-naive DatetimeIndex (daily, ascending, unique)
    cols  : open, high, low, close, volume  (float)

Providers
---------
YFinanceProvider : spec default. Pulls split/bonus-ADJUSTED OHLCV from Yahoo
    (auto_adjust=True) with NSE's ".NS" suffix, in polite throttled batches,
    and caches each symbol to parquet so re-runs never re-hit Yahoo.
CacheProvider : reads whatever is already cached in data_dir (parquet/csv).
    Used for offline runs and for data ingested from other sources (e.g. the
    Kite MCP). Never hits the network.

Data hygiene (both providers): no fabrication, no forward-fill of OHLCV.
Symbols with fewer than cfg.min_history_days rows are skipped and reported.
"""
from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _cache_path(cfg: Config, symbol: str) -> str:
    return os.path.join(cfg.data_dir, f"{symbol}.parquet")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame to the normalized schema."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = df.copy()
    # flatten possible MultiIndex columns (yfinance single-ticker edge cases)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    # yfinance uses "adj close"/"close"; with auto_adjust=True "close" is adjusted
    keep = {}
    for col in OHLCV_COLS:
        if col in df.columns:
            keep[col] = df[col]
    out = pd.DataFrame(keep)
    out = out[[c for c in OHLCV_COLS if c in out.columns]]
    # index -> tz-naive daily timestamps
    out.index = pd.to_datetime(out.index)
    try:
        out.index = out.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # drop rows with no price (do NOT forward-fill)
    out = out.dropna(subset=[c for c in ("open", "high", "low", "close") if c in out.columns])
    return out


def clean_ohlcv(df: pd.DataFrame, max_gap_days: int = 45, head_frac: float = 0.25
                ) -> pd.DataFrame:
    """Drop isolated LEADING fragments separated from the main series by a large
    calendar gap.

    Some Kite/yfinance series carry a handful of spurious placeholder candles
    dated well before the stock's real listing (e.g. sub-rupee rows years early),
    then a long gap, then the genuine history. Those stray early candles can
    manufacture a FAKE 52-week low. If a gap larger than `max_gap_days` occurs
    within the first `head_frac` of the rows, everything up to and including that
    gap is dropped. Normal continuous series are untouched.
    """
    if df is None or len(df) < 10:
        return df
    gaps = df.index.to_series().diff().dt.days.to_numpy()
    cutoff = max(1, int(len(df) * head_frac))
    big = [i for i in range(1, cutoff) if gaps[i] > max_gap_days]
    if big:
        return df.iloc[big[-1]:]
    return df


def load_from_cache(cfg: Config, symbol: str) -> pd.DataFrame | None:
    path = _cache_path(cfg, symbol)
    if os.path.exists(path):
        try:
            return normalize_ohlcv(pd.read_parquet(path))
        except Exception:
            return None
    # also allow a csv fallback
    csv = os.path.join(cfg.data_dir, f"{symbol}.csv")
    if os.path.exists(csv):
        try:
            return normalize_ohlcv(pd.read_csv(csv, index_col=0, parse_dates=True))
        except Exception:
            return None
    return None


def save_to_cache(cfg: Config, symbol: str, df: pd.DataFrame) -> None:
    os.makedirs(cfg.data_dir, exist_ok=True)
    try:
        df.to_parquet(_cache_path(cfg, symbol))
    except Exception:
        df.to_csv(os.path.join(cfg.data_dir, f"{symbol}.csv"))


# --------------------------------------------------------------------------
# Kite ingestion (used for the real-data run in this restricted environment)
# --------------------------------------------------------------------------
def ingest_kite_candles(candles: list, symbol: str, cfg: Config) -> pd.DataFrame:
    """Convert Kite historical `day` candles -> normalized frame and cache it.

    Each candle is {"date","open","high","low","close","volume","oi"} or the
    list form [date, o, h, l, c, v, oi]. Kite daily data is already adjusted
    for splits/bonuses.
    """
    rows = []
    for row in candles:
        if isinstance(row, dict):
            rows.append((row["date"], row["open"], row["high"], row["low"],
                         row["close"], row.get("volume", 0)))
        else:
            rows.append((row[0], row[1], row[2], row[3], row[4],
                         row[5] if len(row) > 5 else 0))
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    # Kite stamps each daily candle at 00:00:00 IST; the first 10 chars ARE the
    # trading date. Take them directly -- converting via UTC would shift IST
    # midnight back to the previous calendar day.
    df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10), format="%Y-%m-%d")
    df = df.set_index("date")
    df = normalize_ohlcv(df)
    save_to_cache(cfg, symbol, df)
    return df


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
class CacheProvider:
    """Reads only from the local cache; never touches the network."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def get(self, symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str]]]:
        frames, skipped = {}, []
        for sym in symbols:
            df = load_from_cache(self.cfg, sym)
            if df is None or len(df) == 0:
                skipped.append((sym, "no cached data"))
                continue
            if len(df) < self.cfg.min_history_days:
                skipped.append((sym, f"only {len(df)} rows (< {self.cfg.min_history_days})"))
                continue
            frames[sym] = df
        return frames, skipped


class YFinanceProvider:
    """Spec default: adjusted OHLCV from Yahoo via yfinance, batched + cached.

    NOTE: requires outbound access to Yahoo Finance. In sandboxes where Yahoo
    is firewalled this will fail per symbol; use the CacheProvider instead.
    """

    def __init__(self, cfg: Config, force_refresh: bool = False):
        self.cfg = cfg
        self.force_refresh = force_refresh

    def _to_yahoo(self, symbol: str) -> str:
        return symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    def get(self, symbols: list[str]) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str]]]:
        import yfinance as yf
        cfg = self.cfg
        frames, skipped = {}, []
        period = f"{cfg.years + 1}y"   # a little extra for the 252-day warm-up

        # 1) satisfy from cache first
        to_fetch = []
        for sym in symbols:
            if not self.force_refresh:
                cached = load_from_cache(cfg, sym)
                if cached is not None and len(cached) >= cfg.min_history_days:
                    frames[sym] = cached
                    continue
            to_fetch.append(sym)

        # 2) batched throttled download for the rest
        for i in range(0, len(to_fetch), cfg.yf_batch_size):
            batch = to_fetch[i:i + cfg.yf_batch_size]
            ymap = {self._to_yahoo(s): s for s in batch}
            try:
                raw = yf.download(list(ymap.keys()), period=period, interval="1d",
                                  auto_adjust=True, group_by="ticker",
                                  threads=False, progress=False)
            except Exception as e:
                for s in batch:
                    skipped.append((s, f"download error: {e!r}"))
                continue

            for ytk, sym in ymap.items():
                try:
                    sub = raw[ytk] if isinstance(raw.columns, pd.MultiIndex) else raw
                except Exception:
                    sub = None
                df = normalize_ohlcv(sub) if sub is not None else None
                if df is None or len(df) < cfg.min_history_days:
                    n = 0 if df is None else len(df)
                    skipped.append((sym, f"sparse/missing ({n} rows)"))
                    continue
                save_to_cache(cfg, sym, df)
                frames[sym] = df
            time.sleep(cfg.yf_pause_sec)   # be polite; avoid rate limits

        return frames, skipped


def get_provider(cfg: Config, force_refresh: bool = False):
    if cfg.provider == "yfinance":
        return YFinanceProvider(cfg, force_refresh=force_refresh)
    if cfg.provider == "cache":
        return CacheProvider(cfg)
    raise ValueError(f"unknown provider: {cfg.provider}")


def load_universe_data(symbols: list[str], cfg: Config, force_refresh: bool = False
                       ) -> tuple[dict[str, pd.DataFrame], list[tuple[str, str]]]:
    """Load normalized OHLCV for a universe, trimmed to the last `cfg.years`."""
    provider = get_provider(cfg, force_refresh=force_refresh)
    frames, skipped = provider.get(symbols)
    # clean isolated pre-listing fragments, then trim to the requested window
    if frames:
        end = max(df.index.max() for df in frames.values())
        start = end - pd.Timedelta(days=int((cfg.years + 1) * 365.25))
        for sym in list(frames.keys()):
            df = clean_ohlcv(frames[sym])
            df = df.loc[df.index >= start]
            if len(df) < cfg.min_history_days:
                skipped.append((sym, f"too few rows after clean/trim ({len(df)})"))
                del frames[sym]
            else:
                frames[sym] = df
    return frames, skipped
