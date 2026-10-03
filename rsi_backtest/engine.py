"""Event-driven, per-stock backtest engine.

No lookahead:
  * signals use data through day T's close;
  * a buy-stop can only trigger on T+1 .. T+entry_window;
  * all indicators use only past-and-current data.

Realistic fills:
  * entry: buy-stop at signal_high + 1 tick; fills at that level, or at the
    day's OPEN if the day gaps above it;
  * stop exits: fill at the stop level, or at the OPEN if the day gaps THROUGH
    it (so a gap-down loss can exceed the nominal stop);
  * +20% target: fills at the target, or at the OPEN if the day gaps above it.

Exit priority each day (per spec):
  1. hard +20% target  2. trailing stop  3. initial / breakeven stop

For every entered trade we also simulate the hypothetical "uncapped" exit
(same rules with the +20% target disabled) so the cost of the cap can be
measured directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from . import indicators as ind
from . import signal as sig_mod


# ---- exit reason labels ---------------------------------------------------
R_TARGET = "target_20pct"
R_TRAIL = "trail"          # suffixed with method, e.g. trail_ema10
R_INITIAL = "initial_stop"
R_BREAKEVEN = "breakeven"
R_EOD = "end_of_data"


@dataclass
class Trade:
    symbol: str
    tminus1_date: pd.Timestamp
    t_date: pd.Timestamp            # signal candle (T)
    entry_date: pd.Timestamp
    entry_index: int
    entry_price: float
    initial_stop: float
    exit_date: pd.Timestamp
    exit_index: int
    exit_price: float
    exit_reason: str
    return_pct: float               # GROSS, before costs
    r_multiple: float               # GROSS
    days_held: int
    activated: bool                 # did it ever reach +10%?
    rsi_value: float
    ca_flag: bool                   # possible corporate action near signal
    # hypothetical uncapped outcome (+20% target disabled)
    uncapped_return_pct: float
    uncapped_exit_reason: str


@dataclass
class StockResult:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    n_signals: int = 0
    n_entered: int = 0
    n_expired: int = 0              # buy-stop expired unused in the window
    n_skipped_overlap: int = 0      # signal ignored (position/order already live)


@dataclass
class ExitOutcome:
    index: int
    price: float
    reason: str
    activated: bool


def _simulate_exit(o, h, l, c, ema_tr, atr, low_prevN, running_high_seed,
                   entry_index: int, entry_price: float, initial_stop: float,
                   cfg: Config, use_hard_target: bool) -> ExitOutcome:
    """Walk forward from the entry bar and return the exit outcome.

    All inputs are numpy arrays aligned to the stock frame. `running_high_seed`
    seeds the ATR trail's running high (the entry bar's high).
    """
    n = len(c)
    method = cfg.trail.method
    tick = cfg.tick
    target_level = entry_price * (1.0 + cfg.hard_target_pct)
    activate_level = entry_price * (1.0 + cfg.activate_pct)
    breakeven = entry_price

    activated = False
    trailing_level = -np.inf
    running_high = running_high_seed

    for d in range(entry_index, n):
        H, L, O, C = h[d], l[d], o[d], c[d]

        # (1) hard +20% target -----------------------------------------
        if use_hard_target and H >= target_level:
            px = O if O >= target_level else target_level  # gap up -> open
            return ExitOutcome(d, px, R_TARGET, activated)

        # running high (for ATR trail) includes today
        if H > running_high:
            running_high = H

        # (2) activation at +10% (intraday touch) ----------------------
        if not activated and H >= activate_level:
            activated = True

        # effective numeric stop for this bar
        if activated:
            if method == "ema10":
                eff_stop = breakeven                       # floor; trail is close-rule
            elif method == "lastlow":
                cand = (low_prevN[d] - tick) if not np.isnan(low_prevN[d]) else -np.inf
                if cand > trailing_level:
                    trailing_level = cand
                eff_stop = max(trailing_level, breakeven)
            elif method == "atr":
                cand = running_high - cfg.trail.atr_mult * atr[d] if not np.isnan(atr[d]) else -np.inf
                if cand > trailing_level:
                    trailing_level = cand
                eff_stop = max(trailing_level, breakeven)
            else:
                eff_stop = breakeven
        else:
            eff_stop = initial_stop

        # (3) trailing / initial / breakeven exits ---------------------
        if activated and method == "ema10":
            # trailing stop: daily CLOSE below the 10-day EMA
            if not np.isnan(ema_tr[d]) and C < ema_tr[d]:
                return ExitOutcome(d, C, f"{R_TRAIL}_ema10", activated)
            # breakeven floor (intraday); gap-down fills at open
            if L <= breakeven:
                px = O if O < breakeven else breakeven
                return ExitOutcome(d, px, R_BREAKEVEN, activated)
        else:
            if L <= eff_stop:
                px = O if O < eff_stop else eff_stop       # gap-through -> open
                if not activated:
                    reason = R_INITIAL
                elif trailing_level > breakeven:
                    reason = f"{R_TRAIL}_{method}"
                else:
                    reason = R_BREAKEVEN
                return ExitOutcome(d, px, reason, activated)

    # ran out of data -> exit at last close
    return ExitOutcome(n - 1, c[n - 1], R_EOD, activated)


def run_stock(df: pd.DataFrame, symbol: str, cfg: Config) -> StockResult:
    """Generate all trades for one stock (sizing-agnostic)."""
    res = StockResult(symbol=symbol)
    if len(df) < max(cfg.week52_lookback + 2, cfg.min_history_days):
        return res

    df2 = sig_mod.compute_signal_columns(df, cfg)
    signals = sig_mod.extract_signals(df2, symbol)
    res.n_signals = len(signals)
    if not signals:
        return res

    o = df2["open"].to_numpy(float)
    h = df2["high"].to_numpy(float)
    l = df2["low"].to_numpy(float)
    c = df2["close"].to_numpy(float)
    idx = df2.index

    ema_tr = ind.ema(df2["close"], cfg.trail.ema_period).to_numpy(float)
    atr = ind.atr_wilder(df2["high"], df2["low"], df2["close"], cfg.trail.atr_period).to_numpy(float)
    # "low of the last N candles" = last N COMPLETED candles (excludes the
    # current bar), so today's action can breach it and it ratchets up as price
    # rises. Including the current bar would make it un-triggerable intraday.
    low_prevN = df2["low"].rolling(cfg.trail.lastlow_n, min_periods=1).min().shift(1).to_numpy(float)

    tick = cfg.tick
    n = len(c)
    block_until = -1  # positional index up to which new signals are ignored

    for s in signals:
        p = s.t_index
        if p <= block_until:
            res.n_skipped_overlap += 1
            continue

        trigger = s.signal_high + tick
        entry_idx = None
        window_end = min(p + cfg.entry_window, n - 1)
        for d in range(p + 1, window_end + 1):
            if h[d] >= trigger:
                entry_idx = d
                break

        if entry_idx is None:
            res.n_expired += 1
            block_until = window_end     # don't stack orders during the dead window
            continue

        entry_price = max(trigger, o[entry_idx])   # gap above -> fill at open
        initial_stop = s.signal_low - tick
        per_share_risk = entry_price - initial_stop
        if per_share_risk <= 0:
            res.n_skipped_overlap += 1
            continue

        capped = _simulate_exit(o, h, l, c, ema_tr, atr, low_prevN, h[entry_idx],
                                entry_idx, entry_price, initial_stop, cfg, True)
        uncapped = _simulate_exit(o, h, l, c, ema_tr, atr, low_prevN, h[entry_idx],
                                  entry_idx, entry_price, initial_stop, cfg, False)

        ret = (capped.price / entry_price - 1.0) * 100.0
        r_mult = (capped.price - entry_price) / per_share_risk
        unc_ret = (uncapped.price / entry_price - 1.0) * 100.0

        res.trades.append(Trade(
            symbol=symbol,
            tminus1_date=s.tminus1_date,
            t_date=s.t_date,
            entry_date=idx[entry_idx],
            entry_index=entry_idx,
            entry_price=entry_price,
            initial_stop=initial_stop,
            exit_date=idx[capped.index],
            exit_index=capped.index,
            exit_price=capped.price,
            exit_reason=capped.reason,
            return_pct=ret,
            r_multiple=r_mult,
            days_held=capped.index - entry_idx,
            activated=capped.activated,
            rsi_value=s.rsi_value,
            ca_flag=sig_mod.flag_corporate_action(df2, p),
            uncapped_return_pct=unc_ret,
            uncapped_exit_reason=uncapped.reason,
        ))
        res.n_entered += 1
        block_until = capped.index       # ignore signals until this position closes

    return res


def run_all(frames: dict[str, pd.DataFrame], cfg: Config) -> list[StockResult]:
    """Run every stock. `frames` maps symbol -> OHLCV DataFrame."""
    results = []
    for sym, df in frames.items():
        try:
            results.append(run_stock(df, sym, cfg))
        except Exception as e:  # keep going; report at the end
            print(f"  [engine] {sym}: ERROR {e!r}")
    return results
