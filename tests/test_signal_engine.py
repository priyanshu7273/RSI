"""Signal + engine validation on hand-built scenarios.

A long monotonic decline guarantees: fresh 252-day lows every day and RSI -> ~0
(well under 25) on the capitulation day. We then append a green "signal candle"
that does not undercut the floor, and custom bars after it to exercise entry and
each exit type. Because the setup is fully controlled, the expected signal index,
entry fill and exit reason are known exactly.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rsi_backtest.config import Config
from rsi_backtest import signal as sigmod
from rsi_backtest import engine

BASE_N = 260                      # declining days, indices 0..259 (T-1 = 259)
TICK = 0.05


def build_frame(post_bars):
    """Decline for BASE_N days, a green signal candle at index 260, then post_bars."""
    o, h, l, c = [], [], [], []
    for i in range(BASE_N):
        close = 100.0 - i * 0.25
        c.append(close)
        l.append(close - 0.25)    # new low every day
        h.append(close + 0.15)
        o.append(close + 0.20)    # red candle (open > close)
    # signal candle at index 260: green, low does NOT undercut floor (low[259]=35.0)
    o.append(35.30); h.append(37.00); l.append(35.10); c.append(36.80)
    for (po, ph, pl, pc) in post_bars:
        o.append(po); h.append(ph); l.append(pl); c.append(pc)
    idx = pd.bdate_range("2015-01-01", periods=len(c))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": [1_000_000] * len(c)}, index=idx)


def base_cfg(**kw):
    return Config(min_history_days=100, **kw)


def test_signal_fires_at_T():
    df = build_frame([(36.9, 38.0, 36.8, 37.5)])
    cfg = base_cfg()
    df2 = sigmod.compute_signal_columns(df, cfg)
    sigs = sigmod.extract_signals(df2, "TEST")
    assert len(sigs) == 1, f"expected exactly 1 signal, got {len(sigs)}"
    s = sigs[0]
    assert s.t_index == 260, s.t_index
    assert abs(s.signal_high - 37.00) < 1e-9
    assert abs(s.signal_low - 35.10) < 1e-9
    assert s.rsi_value < 25.0, s.rsi_value      # oversold on T-1
    print("signal fires at T=260, RSI(T-1)=", round(s.rsi_value, 3))


def test_no_signal_if_T_undercuts():
    # signal candle that makes a NEW low (undercuts) must NOT fire
    df = build_frame([(36.9, 38.0, 36.8, 37.5)])
    # overwrite index 260 low to undercut the floor (low[259]=35.0 -> set 34.0)
    df.iloc[260, df.columns.get_loc("low")] = 34.0
    cfg = base_cfg()
    df2 = sigmod.compute_signal_columns(df, cfg)
    sigs = sigmod.extract_signals(df2, "TEST")
    assert len(sigs) == 0, "signal should not fire when T undercuts the floor"
    print("no signal when T undercuts floor: OK")


def test_no_signal_if_T_not_green():
    df = build_frame([(36.9, 38.0, 36.8, 37.5)])
    # make T red: close < open
    df.iloc[260, df.columns.get_loc("open")] = 37.0
    df.iloc[260, df.columns.get_loc("close")] = 35.5
    cfg = base_cfg()
    df2 = sigmod.compute_signal_columns(df, cfg)
    sigs = sigmod.extract_signals(df2, "TEST")
    assert len(sigs) == 0
    print("no signal when T not green: OK")


def test_entry_and_target_exit():
    # entry on index 261 at trigger 37.05; then run to +20% (44.46)
    post = [
        (36.90, 38.00, 36.80, 37.50),   # 261: high 38 >= 37.05 -> entry at 37.05
        (37.50, 40.00, 37.20, 39.80),   # 262: no +10% yet (40 < 40.755)
        (40.00, 45.00, 39.90, 44.90),   # 263: high 45 >= target 44.46 -> target exit
    ]
    df = build_frame(post)
    cfg = base_cfg()
    res = engine.run_stock(df, "TEST", cfg)
    assert res.n_entered == 1, res.n_entered
    t = res.trades[0]
    assert abs(t.entry_price - 37.05) < 1e-6, t.entry_price
    assert abs(t.initial_stop - 35.05) < 1e-6, t.initial_stop
    assert t.exit_reason == engine.R_TARGET, t.exit_reason
    assert abs(t.return_pct - 20.0) < 1e-6, t.return_pct
    assert abs(t.r_multiple - (44.46 - 37.05) / (37.05 - 35.05)) < 1e-6
    print(f"target exit OK: entry {t.entry_price}, exit {t.exit_price}, R={t.r_multiple:.3f}")


def test_initial_stop_exit_with_gap():
    # entry at 37.05, then a gap-down THROUGH the initial stop (35.05): fill at open
    post = [
        (36.90, 38.00, 36.80, 37.50),   # 261: entry at 37.05
        (34.00, 34.50, 33.50, 33.80),   # 262: opens 34.00 < stop 35.05 -> fill at open
    ]
    df = build_frame(post)
    cfg = base_cfg()
    res = engine.run_stock(df, "TEST", cfg)
    t = res.trades[0]
    assert t.exit_reason == engine.R_INITIAL, t.exit_reason
    assert abs(t.exit_price - 34.00) < 1e-6, t.exit_price   # gap -> open, worse than stop
    assert t.return_pct < 0
    print(f"initial-stop gap exit OK: exit at open {t.exit_price} (< stop 35.05)")


def test_activation_then_breakeven_exit():
    # reach +10% (activate, floor=breakeven=entry), then a down-WICK to breakeven
    # while the close stays ABOVE the EMA (so the ema10 trail rule does NOT fire,
    # isolating the breakeven floor -- priority 3).
    post = [
        (36.90, 38.00, 36.80, 37.50),   # 261: entry 37.05
        (38.00, 41.00, 37.80, 40.50),   # 262: high 41 >= 40.755 (+10%) -> activate
        (38.00, 38.20, 36.50, 38.00),   # 263: close 38 > EMA (no trail), low 36.5 <= 37.05 -> BE
    ]
    df = build_frame(post)
    cfg = base_cfg(trail=Config().trail)   # ema10 default
    res = engine.run_stock(df, "TEST", cfg)
    t = res.trades[0]
    assert t.activated is True
    assert t.exit_reason == engine.R_BREAKEVEN, t.exit_reason
    assert abs(t.exit_price - 37.05) < 1e-6, t.exit_price
    print(f"activation->breakeven OK: exit {t.exit_price}")


def test_lastlow_trail_exit():
    from rsi_backtest.config import TrailConfig
    post = [
        (36.90, 38.00, 36.80, 37.50),   # 261 entry 37.05
        (38.00, 41.00, 37.80, 40.60),   # 262 activate (+10%)
        (40.60, 42.00, 40.00, 41.50),   # 263 low 40.00
        (41.50, 42.00, 39.00, 39.20),   # 264 low 39.00 (ratchets trail up)
        (39.20, 39.50, 38.00, 38.20),   # 265 low 38.00 <= trail(min(40.00,39.00)-tick=38.95)
    ]
    df = build_frame(post)
    cfg = base_cfg(trail=TrailConfig(method="lastlow", lastlow_n=2))
    res = engine.run_stock(df, "TEST", cfg)
    t = res.trades[0]
    assert t.activated is True
    assert t.exit_reason == "trail_lastlow", t.exit_reason
    assert abs(t.exit_price - 38.95) < 1e-6, t.exit_price
    print(f"lastlow trail OK: exit {t.exit_price} reason {t.exit_reason}")


def test_signal_ignored_while_position_open():
    # a second signal while the first position is open must be skipped
    post = [
        (36.90, 38.00, 36.80, 37.50),   # 261 entry
        (37.50, 38.50, 37.20, 38.00),   # 262 hold
        (38.00, 38.50, 37.00, 37.20),   # 263 hold
    ] + [(37.0, 37.5, 36.9, 37.1)] * 40  # long hold, no exit -> ends at eod
    df = build_frame(post)
    cfg = base_cfg()
    res = engine.run_stock(df, "TEST", cfg)
    assert res.n_entered == 1
    print(f"one position at a time OK: entered={res.n_entered}, skipped={res.n_skipped_overlap}")


if __name__ == "__main__":
    test_signal_fires_at_T()
    test_no_signal_if_T_undercuts()
    test_no_signal_if_T_not_green()
    test_entry_and_target_exit()
    test_initial_stop_exit_with_gap()
    test_activation_then_breakeven_exit()
    test_lastlow_trail_exit()
    test_signal_ignored_while_position_open()
    print("\nAll signal/engine scenario tests passed.")
