"""Mechanical 52-week-low reversal signal detection.

A *signal candle* occurs on day T when BOTH hold:

  (1) The stock was at fresh 52-week lows coming in, and oversold there:
        - a NEW 252-day low occurred on T-1 (strict, low_lookback_days == 1),
          or within the last `low_lookback_days` days before T; AND
        - RSI(rsi_period) is in [rsi_min, rsi_max) on the capitulation day
          (T-1 by default; switchable to T).
      RSI is tested on T-1 by convention: by the time T prints green, RSI has
      already ticked up off the bottom, so testing it on T would wrongly
      reject good signals.

  (2) Day T breaks the streak and turns green:
        - day T's low does NOT undercut the running 252-day low (it does not
          make a fresh new low); AND
        - day T closes green: close > open.

Day T's high and low define the trade (entry trigger / initial stop).

No lookahead: every column on row T uses only data through row T.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from . import indicators as ind

# Tiny epsilon for float-equality on adjusted prices. Much smaller than a tick.
_EPS = 1e-6


@dataclass
class Signal:
    """A detected signal candle and the levels it defines."""
    symbol: str
    t_index: int            # positional index of day T in the stock frame
    t_date: pd.Timestamp    # signal candle date (T)
    tminus1_date: pd.Timestamp
    signal_high: float      # T's high  -> buy-stop level (+1 tick)
    signal_low: float       # T's low   -> initial stop  (-1 tick)
    rsi_value: float        # RSI on the day it was evaluated (T-1 or T)
    capitulation_low: float # the 252-day-low value that qualified the setup


def compute_signal_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a copy of `df` with indicator + signal columns added.

    Expected input columns: open, high, low, close, volume (float), a
    DatetimeIndex sorted ascending.
    """
    out = df.copy()
    low = out["low"].astype(float)
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)

    # --- indicators ----------------------------------------------------
    out["rsi"] = ind.rsi_wilder(close, cfg.rsi_period)
    rollmin = ind.rolling_min(low, cfg.week52_lookback)  # min incl. current bar
    out["rollmin252"] = rollmin

    # A fresh 252-day low on day t: t's low is (at/below) the trailing-252 min.
    # Since rollmin includes t, low[t] >= rollmin[t] always; equality => new low.
    new_low = (low <= rollmin + _EPS) & rollmin.notna()
    out["new_low"] = new_low

    # --- condition 1: recent new low + oversold RSI on capitulation day ---
    # "A new 252-day low occurred within the last `low_lookback_days` days
    #  before T" == any new_low over days [T-low_lookback_days .. T-1].
    lb = max(1, int(cfg.low_lookback_days))
    recent_new_low = (
        new_low.astype(float).shift(1).rolling(window=lb, min_periods=1).max() >= 1.0
    )
    recent_new_low = recent_new_low.fillna(False)

    # RSI evaluated on T-1 (default) or T.
    if cfg.rsi_on_day == "T-1":
        rsi_eval = out["rsi"].shift(1)
    else:  # "T"
        rsi_eval = out["rsi"]
    rsi_ok = (rsi_eval >= cfg.rsi_min) & (rsi_eval < cfg.rsi_max) & rsi_eval.notna()
    out["rsi_eval"] = rsi_eval
    cond1 = recent_new_low & rsi_ok

    # --- condition 2: T green and T does NOT undercut the running low ------
    # Running 252-day floor coming INTO T = trailing-252 min ending at T-1.
    floor_prev = rollmin.shift(1)
    not_undercut = (low >= (floor_prev - _EPS)) & floor_prev.notna()
    green = close > open_
    cond2 = not_undercut & green

    out["signal"] = (cond1 & cond2).fillna(False)
    out["floor_prev"] = floor_prev
    return out


def extract_signals(df_with_cols: pd.DataFrame, symbol: str) -> list[Signal]:
    """Turn the boolean `signal` column into a list of Signal objects."""
    sigs: list[Signal] = []
    idx = df_with_cols.index
    sig_rows = np.flatnonzero(df_with_cols["signal"].to_numpy())
    for pos in sig_rows:
        if pos == 0:
            continue  # need a T-1
        row = df_with_cols.iloc[pos]
        sigs.append(Signal(
            symbol=symbol,
            t_index=int(pos),
            t_date=idx[pos],
            tminus1_date=idx[pos - 1],
            signal_high=float(row["high"]),
            signal_low=float(row["low"]),
            rsi_value=float(row["rsi_eval"]) if not np.isnan(row["rsi_eval"]) else float("nan"),
            capitulation_low=float(row["floor_prev"]) if not np.isnan(row["floor_prev"]) else float("nan"),
        ))
    return sigs


def flag_corporate_action(df: pd.DataFrame, t_index: int, window: int = 5,
                          jump_threshold: float = 0.35) -> bool:
    """Heuristic flag: did a suspiciously large overnight move occur near the
    signal, of the sort an un/mis-adjusted split or bonus manufactures?

    This does NOT prove a bad adjustment; it marks signals worth spot-checking,
    per the known yfinance NSE-adjustment caveat. A single-day close-to-close
    move beyond +/- `jump_threshold` within +/- `window` days of T is flagged.
    """
    close = df["close"].astype(float)
    lo = max(1, t_index - window)
    hi = min(len(close) - 1, t_index + window)
    seg = close.iloc[lo - 1: hi + 1]
    if len(seg) < 2:
        return False
    rets = seg.pct_change().abs().dropna()
    return bool((rets > jump_threshold).any())
