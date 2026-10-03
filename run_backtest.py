#!/usr/bin/env python3
"""Run the 52-week-low reversal backtest across the Nifty 500 universe.

Usage examples
--------------
  # spec default: fetch from Yahoo (needs internet), run all three variants
  python run_backtest.py

  # run on already-cached data (offline), cap the universe size
  python run_backtest.py --provider cache --max-symbols 200

  # just verify the data layer on a few symbols (build-order step 1)
  python run_backtest.py --provider cache --verify-only --symbols RELIANCE INFY TCS
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from rsi_backtest.config import Config, TrailConfig, TRAIL_VARIANTS
from rsi_backtest import data as datamod
from rsi_backtest import engine as enginemod
from rsi_backtest import portfolio as pf
from rsi_backtest import reporting as rep
from rsi_backtest import universe as uni


def verify_data(frames: dict, cfg: Config, n: int = 3) -> None:
    """Build-order step 1: eyeball a few symbols before trusting the engine."""
    print("\n" + "#" * 78)
    print("# DATA VERIFICATION (sample)")
    print("#" * 78)
    for sym in list(frames.keys())[:n]:
        df = frames[sym]
        print(f"\n--- {sym}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
        print(df.head(3).to_string())
        print("  ...")
        print(df.tail(3).to_string())
        lo = df["low"].rolling(cfg.week52_lookback, min_periods=cfg.week52_lookback).min()
        last = df.iloc[-1]
        print(f"  last close={last['close']:.2f}  "
              f"52w-low(as of last)={lo.iloc[-1]:.2f}  "
              f"all-time-min low={df['low'].min():.2f}")
        # crude split-adjustment sanity: flag any single-day move > 35%
        rets = df["close"].pct_change().abs()
        big = rets[rets > 0.35]
        if len(big):
            print(f"  !! {len(big)} day(s) with >35% close-to-close move "
                  f"(possible split/adjustment artifact): "
                  f"{[d.date().isoformat() for d in big.index[:5]]}")
        else:
            print("  no >35% single-day moves (adjustment looks sane)")


def verdict(variant_summaries: dict, funnel: dict, n_symbols: int,
            source_label: str) -> str:
    """Plain-English, conservative read of whether an edge is shown."""
    L = ["\n" + "=" * 78, "VERDICT", "=" * 78]
    total_trades = max((v["n_trades"] for v in variant_summaries.values()), default=0)

    # pull net expectancy per variant
    exps = {k: v["net_expectancy_R"] for k, v in variant_summaries.items()}
    best = max(exps, key=lambda k: exps[k]) if exps else None

    small = total_trades < 100
    edge_signs = [e > 0.05 for e in exps.values()]
    positive_everywhere = all(edge_signs) and len(edge_signs) > 0
    positive_somewhere = any(edge_signs)

    if total_trades == 0:
        call = "UNCLEAR (no trades generated)"
    elif small and positive_somewhere:
        call = "UNCLEAR"
    elif positive_everywhere:
        call = "WEAK YES (with heavy caveats)"
    elif positive_somewhere:
        call = "MIXED / UNCLEAR"
    else:
        call = "NO"

    L.append(f"Does this strategy show a real, tradeable edge on this data? -> {call}\n")
    L.append("Per-variant net per-trade expectancy (R):")
    for k, e in exps.items():
        L.append(f"   {k:10s}: {e:+.3f} R   (trades={variant_summaries[k]['n_trades']})")
    L.append("")
    L.append("Why, and the caveats that dominate this read:")
    L.append(f"  * Universe source: {source_label}; symbols actually tested: {n_symbols}.")
    L.append(f"  * Signals: {funnel['signals']} -> entered {funnel['entered']}, "
             f"expired {funnel['expired_no_entry']}, skipped(overlap) {funnel['skipped_overlap']}.")
    L.append(f"  * Total trades (max across variants): {total_trades}"
             + ("  -- SMALL SAMPLE; treat all stats as indicative only." if small else ""))
    L.append("  * SURVIVORSHIP BIAS: only current/surviving constituents are tested. This")
    L.append("    strategy buys stocks making new 52-week lows -- exactly where delisted")
    L.append("    losers are missing -- so win rate and returns are biased UPWARD. Any")
    L.append("    positive edge here is an optimistic upper bound, not a realistic one.")
    L.append("  * The +20% hard cap truncates winners (see per-variant cap analysis) and")
    L.append("    mechanically lowers avg win and expectancy vs an uncapped version.")
    L.append("  * Costs matter: compare GROSS vs NET per variant above; a gross edge that")
    L.append("    vanishes net of ~0.2% round-trip is not tradeable.")
    if best is not None:
        L.append(f"  * Least-bad variant by net expectancy: {best} ({exps[best]:+.3f} R).")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="cache", choices=["yfinance", "cache"],
                    help="data source (default: cache; use yfinance where Yahoo is reachable)")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="explicit symbol list (overrides the universe)")
    ap.add_argument("--max-symbols", type=int, default=0, help="cap universe size (0=all)")
    ap.add_argument("--variants", nargs="*", default=list(TRAIL_VARIANTS))
    ap.add_argument("--years", type=int, default=7)
    ap.add_argument("--cost", type=float, default=0.002, help="round-trip cost fraction")
    ap.add_argument("--no-live-universe", action="store_true",
                    help="use the bundled list, do not fetch NSE")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--data-dir", default="data_cache")
    args = ap.parse_args()

    base = Config(provider=args.provider, years=args.years,
                  round_trip_cost=args.cost, outputs_dir=args.outputs,
                  data_dir=args.data_dir)

    # 1) universe
    if args.symbols:
        symbols, source_label = args.symbols, "explicit --symbols"
    else:
        symbols, source_label = uni.get_universe(prefer_live=not args.no_live_universe)
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[:args.max_symbols]
    print(f"Universe: {len(symbols)} symbols from {source_label}")

    # 2) data
    print(f"Loading data via provider='{base.provider}' ...")
    frames, skipped = datamod.load_universe_data(symbols, base)
    print(f"Loaded {len(frames)} symbols; skipped {len(skipped)} "
          f"(sparse/missing/too-short).")
    if skipped[:10]:
        print("  examples of skipped:", skipped[:10])
    if not frames:
        print("No data loaded. (In this sandbox Yahoo is firewalled -- populate the\n"
              "cache via the Kite ingestion path, or run with --provider yfinance where\n"
              "Yahoo is reachable.)")
        sys.exit(1)

    verify_data(frames, base, n=3)
    if args.verify_only:
        return

    # 3) run each trail variant
    variant_summaries = {}
    funnel = None
    for variant in args.variants:
        cfg = base.with_trail(variant)
        results = enginemod.run_all(frames, cfg)
        trades = [t for r in results for t in r.trades]
        funnel = rep.signal_funnel(results)

        port_results = {}
        for sizing in ("risk", "equal"):
            for wc, tag in ((False, "gross"), (True, "net")):
                c2 = cfg.replace(sizing=sizing)
                port_results[f"{sizing}/{tag}"] = pf.simulate(trades, frames, c2, sizing, wc)

        print(rep.format_variant_report(variant, trades, cfg, port_results))
        img = rep.plot_variant(variant, port_results, rep.winners_distribution(trades, cfg, True), cfg.outputs_dir)
        print(f"   [plot] {img}")

        # trade log per variant
        log_path = os.path.join(cfg.outputs_dir, f"trades_{variant}.csv")
        rep.write_trade_log(trades, cfg, log_path)
        print(f"   [trade log] {log_path}")

        ca = rep.cap_analysis(trades, cfg)
        s_net = rep.trade_stats(trades, cfg, with_costs=True)
        pr_rn = port_results["risk/net"]
        variant_summaries[variant] = {
            "n_trades": len(trades),
            "net_expectancy_R": s_net.get("expectancy_R", 0.0) if trades else 0.0,
            "net_win_rate": s_net.get("win_rate", 0.0) if trades else 0.0,
            "cap_hit_rate": ca.get("cap_hit_rate", 0.0),
            "risknet_totret": pr_rn.total_return_pct,
            "risknet_cagr": pr_rn.cagr_pct,
            "risknet_maxdd": pr_rn.max_drawdown_pct,
            "risknet_sharpe": pr_rn.sharpe,
        }

    # 3b) side-by-side comparison of the trail variants
    print("\n" + "=" * 92)
    print("TRAIL-VARIANT COMPARISON  (portfolio = risk-sizing, net of costs)")
    print("=" * 92)
    hdr = (f"{'variant':10s} {'trades':>7s} {'winRate':>8s} {'exp(R)':>8s} "
           f"{'capHit%':>8s} {'totRet%':>9s} {'CAGR%':>7s} {'maxDD%':>8s} {'Sharpe':>7s}")
    print(hdr); print("-" * len(hdr))
    for v, s in variant_summaries.items():
        print(f"{v:10s} {s['n_trades']:>7d} {s['net_win_rate']:>7.1f}% {s['net_expectancy_R']:>8.3f} "
              f"{s['cap_hit_rate']:>7.1f}% {s['risknet_totret']:>9.1f} {s['risknet_cagr']:>7.1f} "
              f"{s['risknet_maxdd']:>8.1f} {s['risknet_sharpe']:>7.2f}")

    # 4) signal funnel + verdict
    print("\n" + "=" * 78 + "\nSIGNAL FUNNEL (shared across variants' signal detection)\n" + "=" * 78)
    print(f"  signals detected : {funnel['signals']}")
    print(f"  -> entered       : {funnel['entered']}")
    print(f"  -> expired (buy-stop never triggered in window): {funnel['expired_no_entry']}")
    print(f"  -> skipped (position/order already open)       : {funnel['skipped_overlap']}")

    print(verdict(variant_summaries, funnel, len(frames), source_label))


if __name__ == "__main__":
    main()
