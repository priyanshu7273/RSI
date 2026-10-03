"""Portfolio-level simulation: turn per-stock trades into an equity curve.

The per-trade statistics (win rate, R, expectancy) are sizing-independent and
come straight from engine.Trade. But total return, CAGR, max drawdown and
Sharpe need a capital-aware forward simulation, because:

  * risk-based sizing depends on the equity available at entry;
  * 52-week-low signals CLUSTER (many stocks bottom together in a crash), so a
    real portfolio cannot take every signal -- capital and a concurrent-position
    cap bind exactly when signals are densest. Ignoring that flatters results.

Two sizing modes are simulated:
  * "risk"  -- risk `risk_per_trade` of equity on the entry-to-initial-stop gap
  * "equal" -- deploy `equal_weight_frac` of equity per position

Costs (round-trip) are split half on entry notional, half on exit notional, and
can be switched off to report gross vs net.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from .config import Config
from .engine import Trade


@dataclass
class PortfolioResult:
    sizing: str
    with_costs: bool
    equity_curve: pd.Series
    n_taken: int
    n_skipped_capital: int
    n_skipped_concurrency: int
    realized_pnl: float
    trade_pnls: list[float] = field(default_factory=list)

    # headline metrics (filled by compute_metrics)
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    return_over_maxdd: float = 0.0
    sharpe: float = 0.0


def _build_close_matrix(frames: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex
                        ) -> dict[str, np.ndarray]:
    """Per-symbol close reindexed to the global calendar and forward-filled.

    Forward-fill is used ONLY for marking open positions to market on exchange
    holidays / missing bars -- never to create signals or trades.
    """
    out = {}
    pos = {d: i for i, d in enumerate(calendar)}
    for sym, df in frames.items():
        s = df["close"].astype(float).reindex(calendar).ffill()
        out[sym] = s.to_numpy(float)
    return out, pos


def simulate(trades: list[Trade], frames: dict[str, pd.DataFrame], cfg: Config,
             sizing: str, with_costs: bool) -> PortfolioResult:
    calendar = pd.DatetimeIndex(sorted(set().union(*[df.index for df in frames.values()])))
    close_mat, dpos = _build_close_matrix(frames, calendar)
    cost = cfg.round_trip_cost if with_costs else 0.0
    half = cost / 2.0

    # index trades by entry/exit date
    by_entry: dict[pd.Timestamp, list[int]] = {}
    by_exit: dict[pd.Timestamp, list[int]] = {}
    for i, t in enumerate(trades):
        by_entry.setdefault(t.entry_date, []).append(i)
        by_exit.setdefault(t.exit_date, []).append(i)

    cash = cfg.starting_equity
    open_pos: dict[int, dict] = {}       # trade idx -> {shares}
    equity_vals = np.empty(len(calendar), dtype=float)

    n_taken = n_skip_cap = n_skip_conc = 0
    realized = 0.0
    trade_pnls: list[float] = []

    def mtm(on_i: int) -> float:
        v = 0.0
        for ti, pos in open_pos.items():
            v += pos["shares"] * close_mat[trades[ti].symbol][on_i]
        return v

    for di, d in enumerate(calendar):
        # 1) exits from positions opened on a prior day
        for ti in by_exit.get(d, []):
            if ti in open_pos and trades[ti].entry_date < d:
                t = trades[ti]
                sh = open_pos[ti]["shares"]
                proceeds = sh * t.exit_price * (1.0 - half)
                cost_paid = open_pos[ti]["entry_cash"]
                pnl = proceeds - cost_paid
                cash += proceeds
                realized += pnl
                trade_pnls.append(pnl)
                del open_pos[ti]

        # 2) equity available for sizing today's entries
        equity_for_sizing = cash + mtm(di)

        # 3) entries starting today
        for ti in by_entry.get(d, []):
            t = trades[ti]
            if cfg.max_concurrent > 0 and len(open_pos) >= cfg.max_concurrent:
                n_skip_conc += 1
                continue
            if sizing == "risk":
                per_share_risk = t.entry_price - t.initial_stop
                if per_share_risk <= 0:
                    n_skip_cap += 1
                    continue
                target_shares = math.floor((equity_for_sizing * cfg.risk_per_trade) / per_share_risk)
            else:  # equal
                target_shares = math.floor((equity_for_sizing * cfg.equal_weight_frac) / t.entry_price)

            buy_px = t.entry_price * (1.0 + half)
            affordable = math.floor(cash / buy_px) if buy_px > 0 else 0
            shares = min(target_shares, affordable)
            if shares <= 0:
                n_skip_cap += 1
                continue
            entry_cash = shares * t.entry_price * (1.0 + half)
            cash -= entry_cash
            open_pos[ti] = {"shares": shares, "entry_cash": entry_cash}
            n_taken += 1

        # 4) same-day round trips (entered today, exit today)
        for ti in by_exit.get(d, []):
            if ti in open_pos and trades[ti].entry_date == d:
                t = trades[ti]
                sh = open_pos[ti]["shares"]
                proceeds = sh * t.exit_price * (1.0 - half)
                pnl = proceeds - open_pos[ti]["entry_cash"]
                cash += proceeds
                realized += pnl
                trade_pnls.append(pnl)
                del open_pos[ti]

        # 5) mark to market
        equity_vals[di] = cash + mtm(di)

    eq = pd.Series(equity_vals, index=calendar, name="equity")
    res = PortfolioResult(
        sizing=sizing, with_costs=with_costs, equity_curve=eq,
        n_taken=n_taken, n_skipped_capital=n_skip_cap,
        n_skipped_concurrency=n_skip_conc, realized_pnl=realized,
        trade_pnls=trade_pnls,
    )
    compute_metrics(res, cfg)
    return res


def compute_metrics(res: PortfolioResult, cfg: Config) -> None:
    eq = res.equity_curve.dropna()
    if len(eq) < 2:
        return
    start, end = eq.iloc[0], eq.iloc[-1]
    res.total_return_pct = (end / start - 1.0) * 100.0

    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    res.cagr_pct = ((end / start) ** (1.0 / years) - 1.0) * 100.0 if start > 0 else float("nan")

    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    res.max_drawdown_pct = dd.min() * 100.0
    mdd = abs(res.max_drawdown_pct)
    res.return_over_maxdd = (res.total_return_pct / mdd) if mdd > 1e-9 else float("nan")

    daily_ret = eq.pct_change().dropna()
    if daily_ret.std(ddof=1) > 0:
        res.sharpe = (daily_ret.mean() / daily_ret.std(ddof=1)) * math.sqrt(252.0)
    else:
        res.sharpe = float("nan")
