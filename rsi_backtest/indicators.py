"""Technical indicators, implemented from first principles.

All functions are pure: they take a price Series/DataFrame and return a Series
aligned to the same index, using ONLY past-and-current data (no lookahead).
NaNs are produced during the warm-up period rather than back-filled.

Indicators provided:
  * rsi_wilder   -- Wilder's RSI (the standard RSI, NOT an SMA variant)
  * ema          -- exponential moving average (close below => trail exit)
  * atr_wilder   -- Average True Range with Wilder smoothing
  * rolling_min  -- trailing rolling minimum (the "52-week low" machinery)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (a.k.a. RMA / Modified Moving Average).

    Seed = simple average of the first `period` values; thereafter
        s[i] = (s[i-1] * (period - 1) + values[i]) / period
    Output positions before the seed are NaN; the seed sits at index
    `period - 1` of the valid (non-NaN-input) region.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return out
    # Seed: simple mean of the first `period` observations.
    seed = np.mean(values[:period])
    out[period - 1] = seed
    prev = seed
    for i in range(period, n):
        prev = (prev * (period - 1) + values[i]) / period
        out[i] = prev
    return out


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI.

    delta  = close.diff()
    gain   = max(delta, 0);  loss = max(-delta, 0)
    avg_gain/avg_loss via Wilder smoothing of gains/losses
    RS     = avg_gain / avg_loss
    RSI    = 100 - 100 / (1 + RS)

    Conventions:
      * The first delta is NaN (no prior close); it is dropped from the
        smoothing input and the result is realigned, so the first RSI value
        appears on the (period+1)-th bar -- i.e. it needs `period` deltas.
      * avg_loss == 0  -> RSI = 100 (no down moves).
      * avg_gain == 0 and avg_loss == 0 -> RSI = 100 (flat; conventional).
    """
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Drop the leading NaN delta so the Wilder seed uses `period` real moves.
    gain_v = gain.to_numpy()[1:]
    loss_v = loss.to_numpy()[1:]

    avg_gain = _wilder_smooth(gain_v, period)
    avg_loss = _wilder_smooth(loss_v, period)

    rsi_v = np.full(len(gain_v), np.nan, dtype=float)
    for i in range(len(gain_v)):
        ag, al = avg_gain[i], avg_loss[i]
        if np.isnan(ag) or np.isnan(al):
            continue
        if al == 0.0:
            rsi_v[i] = 100.0            # no losses -> fully overbought
        elif ag == 0.0:
            rsi_v[i] = 0.0             # no gains  -> fully oversold
        else:
            rs = ag / al
            rsi_v[i] = 100.0 - 100.0 / (1.0 + rs)

    # Realign: rsi_v[i] corresponds to close index i+1 (we dropped index 0).
    out = pd.Series(np.nan, index=close.index, dtype=float)
    out.iloc[1:] = rsi_v
    return out


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, alpha = 2/(period+1), recursive (adjust=False).

    This is the standard "N-day EMA" charting definition. The first `period-1`
    values are NaN (warm-up); the seed at index period-1 is the SMA of the
    first `period` values, matching common charting packages.
    """
    s = series.astype(float)
    # Seed with SMA of first `period` values for a stable, charting-style EMA.
    ema_vals = s.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema_vals


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(H-L, |H - prevC|, |L - prevC|)."""
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # First bar has no prev close: TR = H - L.
    tr.iloc[0] = (high.iloc[0] - low.iloc[0])
    return tr


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing of the True Range.

    Seed (at index `period`, i.e. the (period+1)-th bar relative to the first
    TR) = simple mean of the first `period` True Ranges. The first TR itself
    uses H-L (no prior close), matching the standard definition.
    """
    tr = true_range(high, low, close)
    atr_v = _wilder_smooth(tr.to_numpy(), period)
    return pd.Series(atr_v, index=close.index, dtype=float)


def rolling_min(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing rolling minimum INCLUDING the current bar.

    Used for the 52-week-low logic. `min_periods` defaults to `window`, so a
    value only appears once a full window of history exists -- this prevents
    spurious "new lows" early in a series when the window is not yet full.
    """
    if min_periods is None:
        min_periods = window
    return series.astype(float).rolling(window=window, min_periods=min_periods).min()
