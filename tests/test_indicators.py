"""Hand-checked tests for the indicator layer.

These use tiny series with values computed BY HAND in the comments so the
indicator maths can be trusted independently of any data source.
Run: python -m pytest tests/test_indicators.py -q   (or:  python tests/test_indicators.py)
"""
import math
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rsi_backtest import indicators as ind


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


def test_rsi_wilder_hand_checked():
    # closes:        10   12   11   13   14   13   15
    # deltas:            +2   -1   +2   +1   -1   +2
    # gains:              2    0    2    1    0    2
    # losses:             0    1    0    0    1    0
    # period = 3. Wilder seed = SMA of first 3 gains/losses.
    #   seed avg_gain = (2+0+2)/3 = 1.33333 ; seed avg_loss = (0+1+0)/3 = 0.33333
    #   -> at close idx 3 (val 13): RS = 4.0  -> RSI = 100 - 100/5 = 80.0
    #   step: avg_gain=(1.33333*2+1)/3=1.22222, avg_loss=(0.33333*2+0)/3=0.22222
    #   -> at close idx 4 (val 14): RS=5.5 -> RSI = 100 - 100/6.5 = 84.61538
    #   step: avg_gain=(1.22222*2+0)/3=0.81481, avg_loss=(0.22222*2+1)/3=0.48148
    #   -> at close idx 5 (val 13): RS=1.69231 -> RSI = 62.85714
    #   step: avg_gain=(0.81481*2+2)/3=1.20988, avg_loss=(0.48148*2+0)/3=0.32099
    #   -> at close idx 6 (val 15): RS=3.76923 -> RSI = 79.03226
    close = pd.Series([10, 12, 11, 13, 14, 13, 15], dtype=float)
    rsi = ind.rsi_wilder(close, period=3)

    # Warm-up: first `period` positions (0,1,2) have no RSI.
    assert math.isnan(rsi.iloc[0]) and math.isnan(rsi.iloc[1]) and math.isnan(rsi.iloc[2])
    assert approx(rsi.iloc[3], 80.0), rsi.iloc[3]
    assert approx(rsi.iloc[4], 84.61538), rsi.iloc[4]
    assert approx(rsi.iloc[5], 62.85714), rsi.iloc[5]
    assert approx(rsi.iloc[6], 79.03226), rsi.iloc[6]
    print("RSI Wilder hand-check OK:",
          [round(x, 4) for x in rsi.iloc[3:].tolist()])


def test_rsi_all_gains_is_100():
    close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    rsi = ind.rsi_wilder(close, period=3)
    assert approx(rsi.iloc[-1], 100.0), rsi.iloc[-1]


def test_ema_hand_checked():
    # period=3, seed = SMA(first 3) = (10+12+11)/3 = 11 at idx 2; alpha = 0.5
    #   idx3 = 0.5*13 + 0.5*11 = 12
    #   idx4 = 0.5*14 + 0.5*12 = 13
    s = pd.Series([10, 12, 11, 13, 14], dtype=float)
    e = ind.ema(s, period=3)
    assert math.isnan(e.iloc[0]) and math.isnan(e.iloc[1])
    assert approx(e.iloc[2], 11.0), e.iloc[2]
    assert approx(e.iloc[3], 12.0), e.iloc[3]
    assert approx(e.iloc[4], 13.0), e.iloc[4]
    print("EMA hand-check OK:", e.dropna().tolist())


def test_atr_wilder_hand_checked():
    # OHLC with prev-close dependence:
    #   H:  11 12 13 14   L:  9 10 11 12   C: 10 11 12 13
    # TR[0] = H-L = 2
    # TR[1] = max(12-10, |12-10|, |10-10|) = max(2,2,0)=2
    # TR[2] = max(13-11, |13-11|, |11-11|) = 2
    # TR[3] = max(14-12, |14-12|, |12-12|) = 2
    # period=3 -> seed ATR at idx2 = mean(TR[0:3]) = 2 ; idx3 = (2*2+2)/3 = 2
    high = pd.Series([11, 12, 13, 14], dtype=float)
    low = pd.Series([9, 10, 11, 12], dtype=float)
    close = pd.Series([10, 11, 12, 13], dtype=float)
    atr = ind.atr_wilder(high, low, close, period=3)
    assert math.isnan(atr.iloc[0]) and math.isnan(atr.iloc[1])
    assert approx(atr.iloc[2], 2.0), atr.iloc[2]
    assert approx(atr.iloc[3], 2.0), atr.iloc[3]

    # A gappy bar exercises the prev-close terms.
    #   add a bar: H=20 L=18 C=19, prevC=13 -> TR = max(2, |20-13|, |18-13|)=7
    #   atr idx4 = (2*2 + 7)/3 = 11/3 = 3.66667
    high2 = pd.Series([11, 12, 13, 14, 20], dtype=float)
    low2 = pd.Series([9, 10, 11, 12, 18], dtype=float)
    close2 = pd.Series([10, 11, 12, 13, 19], dtype=float)
    atr2 = ind.atr_wilder(high2, low2, close2, period=3)
    assert approx(atr2.iloc[4], 11.0 / 3.0), atr2.iloc[4]
    print("ATR Wilder hand-check OK")


def test_rolling_min_includes_current_and_warms_up():
    low = pd.Series([5, 4, 6, 3, 7], dtype=float)
    rm = ind.rolling_min(low, window=3)
    # min_periods defaults to window=3 -> first two are NaN
    assert math.isnan(rm.iloc[0]) and math.isnan(rm.iloc[1])
    assert rm.iloc[2] == 4.0   # min(5,4,6)
    assert rm.iloc[3] == 3.0   # min(4,6,3)  -- includes current bar
    assert rm.iloc[4] == 3.0   # min(6,3,7)
    print("rolling_min hand-check OK")


if __name__ == "__main__":
    test_rsi_wilder_hand_checked()
    test_rsi_all_gains_is_100()
    test_ema_hand_checked()
    test_atr_wilder_hand_checked()
    test_rolling_min_includes_current_and_warms_up()
    print("\nAll indicator hand-checks passed.")
