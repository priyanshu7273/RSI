"""Reporting: metrics, distributions, equity curves, trade log, verdict.

Everything here is descriptive -- it computes and prints the numbers; it does
NOT tune anything. Results are reported both gross and net of costs, with
sample sizes shown so small-n results can be caveated.
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .engine import Trade, StockResult, R_TARGET
from . import portfolio as pf


# --------------------------------------------------------------------------
# per-trade statistics
# --------------------------------------------------------------------------
def _risk_frac(t: Trade) -> float:
    rf = (t.entry_price - t.initial_stop) / t.entry_price
    return rf if rf > 0 else float("nan")


def trade_stats(trades: list[Trade], cfg: Config, with_costs: bool) -> dict:
    cost = cfg.round_trip_cost if with_costs else 0.0
    n = len(trades)
    if n == 0:
        return {"n": 0}
    net_ret = np.array([t.return_pct / 100.0 - cost for t in trades])     # fraction
    risk = np.array([_risk_frac(t) for t in trades])
    net_R = net_ret / risk
    wins = net_ret > 0
    losses = ~wins
    out = {
        "n": n,
        "win_rate": 100.0 * wins.mean(),
        "avg_win_pct": 100.0 * net_ret[wins].mean() if wins.any() else 0.0,
        "avg_loss_pct": 100.0 * net_ret[losses].mean() if losses.any() else 0.0,
        "avg_R": float(np.nanmean(net_R)),
        "expectancy_R": float(np.nanmean(net_R)),            # per-trade expectancy in R
        "expectancy_pct": 100.0 * net_ret.mean(),            # per-trade expectancy in %
        "avg_days_held": float(np.mean([t.days_held for t in trades])),
        "n_wins": int(wins.sum()),
        "n_losses": int(losses.sum()),
    }
    # profit factor
    gross_win = net_ret[wins].sum()
    gross_loss = -net_ret[losses].sum()
    out["profit_factor"] = (gross_win / gross_loss) if gross_loss > 1e-12 else float("inf")
    return out


def exit_counts(trades: list[Trade]) -> Counter:
    return Counter(t.exit_reason for t in trades)


def winners_distribution(trades: list[Trade], cfg: Config, with_costs: bool) -> dict:
    cost = cfg.round_trip_cost if with_costs else 0.0
    rets = np.array([t.return_pct / 100.0 - cost for t in trades])
    win_rets = 100.0 * rets[rets > 0]
    if len(win_rets) == 0:
        return {"n_winners": 0}
    pcts = [10, 25, 50, 75, 90, 95, 100]
    dist = {f"p{p}": float(np.percentile(win_rets, p)) for p in pcts}
    # histogram buckets
    edges = [0, 2, 5, 10, 15, 20, 25, 30, 50, 100, np.inf]
    labels = ["0-2", "2-5", "5-10", "10-15", "15-20", "20-25", "25-30",
              "30-50", "50-100", "100+"]
    hist = np.histogram(win_rets, bins=edges)[0]
    return {
        "n_winners": int(len(win_rets)),
        "percentiles": dist,
        "hist_labels": labels,
        "hist_counts": hist.tolist(),
        "mean": float(win_rets.mean()),
    }


def cap_analysis(trades: list[Trade], cfg: Config) -> dict:
    """Quantify what the +20% hard cap costs.

    For each trade that exited via the +20% target, compare the capped result
    (+20%) with the hypothetical uncapped result (same rules, target disabled).
    """
    capped = [t for t in trades if t.exit_reason == R_TARGET]
    n_cap = len(capped)
    res = {
        "n_trades": len(trades),
        "n_cap_hits": n_cap,
        "cap_hit_rate": (100.0 * n_cap / len(trades)) if trades else 0.0,
    }
    if n_cap == 0:
        return res
    unc = np.array([t.uncapped_return_pct for t in capped])
    still_rising = unc > cfg.hard_target_pct * 100.0 + 1e-9
    res["n_still_rising_at_cap"] = int(still_rising.sum())
    res["pct_still_rising_at_cap"] = 100.0 * still_rising.mean()
    # how much extra the uncapped versions captured, on the capped trades
    res["avg_uncapped_return_on_cap_hits"] = float(unc.mean())
    res["avg_cap_cost_pct"] = float((unc - cfg.hard_target_pct * 100.0).mean())
    res["median_uncapped_return_on_cap_hits"] = float(np.median(unc))
    res["max_uncapped_return_on_cap_hits"] = float(unc.max())
    return res


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_variant(variant: str, port_results: dict, winners: dict,
                 outputs_dir: str) -> str:
    os.makedirs(outputs_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for key, pr in port_results.items():          # key like "risk/net"
        ax.plot(pr.equity_curve.index, pr.equity_curve.values, label=key, linewidth=1.2)
    ax.set_title(f"Equity curve - trail={variant}")
    ax.set_ylabel("Equity (INR)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if winners.get("n_winners", 0) > 0:
        ax2.bar(winners["hist_labels"], winners["hist_counts"], color="#2a7")
        ax2.set_title(f"Winning-trade return distribution (net) - trail={variant}")
        ax2.set_xlabel("return bucket (%)")
        ax2.set_ylabel("count")
        ax2.tick_params(axis="x", rotation=45)
    else:
        ax2.text(0.5, 0.5, "no winners", ha="center")
    fig.tight_layout()
    path = os.path.join(outputs_dir, f"variant_{variant}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# trade log
# --------------------------------------------------------------------------
def trade_log_dataframe(trades: list[Trade], cfg: Config) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "symbol": t.symbol,
            "tminus1_date": t.tminus1_date.date(),
            "signal_T_date": t.t_date.date(),
            "entry_date": t.entry_date.date(),
            "entry_price": round(t.entry_price, 4),
            "initial_stop": round(t.initial_stop, 4),
            "exit_date": t.exit_date.date(),
            "exit_price": round(t.exit_price, 4),
            "exit_reason": t.exit_reason,
            "return_pct_gross": round(t.return_pct, 3),
            "return_pct_net": round(t.return_pct - cfg.round_trip_cost * 100.0, 3),
            "r_multiple_gross": round(t.r_multiple, 3),
            "days_held": t.days_held,
            "activated_10pct": t.activated,
            "rsi_at_signal": round(t.rsi_value, 2) if t.rsi_value == t.rsi_value else None,
            "uncapped_return_pct": round(t.uncapped_return_pct, 3),
            "possible_corp_action_near_signal": t.ca_flag,
        })
    return pd.DataFrame(rows)


def write_trade_log(trades: list[Trade], cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    trade_log_dataframe(trades, cfg).to_csv(path, index=False)


# --------------------------------------------------------------------------
# text report
# --------------------------------------------------------------------------
def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:,.{nd}f}"


def format_variant_report(variant: str, trades: list[Trade], cfg: Config,
                          port_results: dict) -> str:
    L = []
    L.append(f"\n{'='*78}\nTRAIL VARIANT: {variant}\n{'='*78}")
    n = len(trades)
    L.append(f"Total trades (signals that entered): {n}")
    if n == 0:
        L.append("No trades generated for this variant.")
        return "\n".join(L)
    if n < 30:
        L.append(f"  !! SMALL SAMPLE (n={n}) -- statistics below are NOT reliable.")

    for wc, tag in [(False, "GROSS (no costs)"), (True, f"NET (costs {cfg.round_trip_cost*100:.2f}% round-trip)")]:
        s = trade_stats(trades, cfg, with_costs=wc)
        L.append(f"\n-- Per-trade stats [{tag}] --")
        L.append(f"   win rate      : {_fmt(s['win_rate'])}%   ({s['n_wins']}W / {s['n_losses']}L)")
        L.append(f"   avg win       : {_fmt(s['avg_win_pct'])}%")
        L.append(f"   avg loss      : {_fmt(s['avg_loss_pct'])}%")
        L.append(f"   avg R         : {_fmt(s['avg_R'])}")
        L.append(f"   expectancy    : {_fmt(s['expectancy_R'])} R  ({_fmt(s['expectancy_pct'])}% per trade)")
        L.append(f"   profit factor : {_fmt(s['profit_factor'])}")
        L.append(f"   avg days held : {_fmt(s['avg_days_held'],1)}")

    # exit breakdown
    ec = exit_counts(trades)
    L.append("\n-- Exit types fired --")
    for reason in sorted(ec):
        L.append(f"   {reason:16s}: {ec[reason]}")

    # cap analysis
    ca = cap_analysis(trades, cfg)
    L.append("\n-- +20% cap analysis (what the cap costs) --")
    L.append(f"   cap hits (+20% target): {ca['n_cap_hits']} / {ca['n_trades']} "
             f"({_fmt(ca['cap_hit_rate'])}% of trades)")
    if ca.get("n_cap_hits", 0) > 0:
        L.append(f"   of those, still rising past +20% if uncapped: "
                 f"{ca['n_still_rising_at_cap']} ({_fmt(ca['pct_still_rising_at_cap'])}%)")
        L.append(f"   avg uncapped return on cap-hit trades: {_fmt(ca['avg_uncapped_return_on_cap_hits'])}%")
        L.append(f"   avg return left on table by the cap  : {_fmt(ca['avg_cap_cost_pct'])}% per cap-hit trade")
        L.append(f"   max uncapped return on a cap-hit trade: {_fmt(ca['max_uncapped_return_on_cap_hits'])}%")

    # winners distribution
    wd = winners_distribution(trades, cfg, with_costs=True)
    if wd.get("n_winners", 0) > 0:
        p = wd["percentiles"]
        L.append("\n-- Winning-trade return distribution (net) --")
        L.append(f"   winners: {wd['n_winners']}   mean: {_fmt(wd['mean'])}%")
        L.append("   percentiles: " + "  ".join(f"{k}={_fmt(v)}%" for k, v in p.items()))
        L.append("   histogram: " + "  ".join(
            f"[{lab}]={cnt}" for lab, cnt in zip(wd["hist_labels"], wd["hist_counts"]) if cnt))

    # portfolio metrics
    L.append("\n-- Portfolio metrics (capital-aware simulation) --")
    for key, pr in port_results.items():
        L.append(f"   [{key}] totRet={_fmt(pr.total_return_pct)}%  CAGR={_fmt(pr.cagr_pct)}%  "
                 f"maxDD={_fmt(pr.max_drawdown_pct)}%  ret/DD={_fmt(pr.return_over_maxdd)}  "
                 f"Sharpe={_fmt(pr.sharpe)}")
        L.append(f"        taken={pr.n_taken}  skipped(capital)={pr.n_skipped_capital}  "
                 f"skipped(concurrency)={pr.n_skipped_concurrency}")
    return "\n".join(L)


def signal_funnel(stock_results: list[StockResult]) -> dict:
    tot = sum(r.n_signals for r in stock_results)
    ent = sum(r.n_entered for r in stock_results)
    exp = sum(r.n_expired for r in stock_results)
    skp = sum(r.n_skipped_overlap for r in stock_results)
    return {"signals": tot, "entered": ent, "expired_no_entry": exp, "skipped_overlap": skp}
