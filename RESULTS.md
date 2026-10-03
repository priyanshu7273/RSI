# Results — 52-Week-Low Reversal on the Nifty 500 (genuine run)

**TL;DR verdict: NO — this strategy does not show a real, tradeable edge on this
data, and the result is negative even before the survivorship bias (which flatters
it) is accounted for.**

---

## What was actually run

- **Data:** 433 Nifty-500 names, real **split/bonus-adjusted daily OHLCV from the
  Kite Connect (Zerodha) NSE feed**, ~Oct 2018 → Oct 2026 (7-year window + warm-up).
  (Yahoo/`yfinance` is firewalled in the build sandbox, so the live NSE broker feed
  was used instead; the code runs identically on `yfinance` where Yahoo is reachable.)
- **Universe coverage:** 433 of 453 bundled symbols loaded. 20 were dropped —
  unresolvable renames/demergers/delistings on Kite's current master (e.g.
  **TATAMOTORS**, which demerged into TMCV/TMPV in Nov 2025; LTIM, PEL, GSPL,
  INTERGLOBE→INDIGO, etc.).
- **Rules:** exactly as specified. No parameter tuning. Costs 0.2% round-trip.
- **Signals:** 1,202 detected → **846 entered** → 238 expired (buy-stop never
  triggered in the 5-day window) → 118 skipped (a position/order was already open).
- Sample size (846 trades) is healthy, so the statistics are meaningful (for this
  period and this biased universe — see caveats).

## Headline numbers (net of 0.2% costs; portfolio = 1%-risk sizing)

| variant  | trades | win rate | avg win | avg loss | exp (R) | profit factor | total ret | CAGR | max DD | Sharpe |
|----------|-------:|---------:|--------:|---------:|--------:|--------------:|----------:|-----:|-------:|-------:|
| ema10    |    846 |   30.4%  | +10.4%  |  −4.96%  | −0.158  |     0.91      |  −63.2%   | −11.8% | −68.1% | −0.86 |
| lastlow  |    846 |   27.5%  | +11.2%  |  −4.80%  | −0.156  |     0.88      |  −62.6%   | −11.6% | −67.6% | −0.86 |
| atr      |    846 |   26.6%  | +11.2%  |  −4.74%  | −0.183  |     0.85      |  −66.2%   | −12.7% | −69.3% | −0.98 |

Equal-weight sizing is less bad but still clearly negative (−42% to −53% total
return). **Gross of costs the per-trade expectancy is still negative** (−0.10 to
−0.13 R), so this is not merely a cost problem — the raw edge is absent.

## Why it loses (the mechanism)

Exit mix (ema10, representative):

| exit reason  | share |
|--------------|------:|
| initial_stop | 63.6% |
| trail_ema10  | 22.7% |
| target_20pct |  9.0% |
| breakeven    |  4.6% |

Almost **two-thirds of trades are stopped out at the initial stop** for ~−5%. The
winners (27–30% of trades) average only ~+10%. The arithmetic
(0.30 × 10.4% − 0.70 × 5.0% ≈ −0.3%/trade) is negative. In plain terms: buying the
first green candle after a capitulation 52-week low mostly catches **falling knives**
— the "bottom" keeps dropping and takes out the stop.

## What the +20% cap costs (it is NOT the problem)

- Only **8–10% of trades** reach the +20% cap.
- Of those, ~35–46% were still rising if left uncapped, but the average return left
  on the table is just **~+1.0 to +1.7%** per cap-hit trade (for `atr` it was slightly
  *negative* — uncapped would have given back gains via the trail).
- Removing the cap would add a fraction of a percent to a few percent of trades — it
  **cannot rescue** a strategy losing ~0.3–0.5% per trade across 846 trades. The cap
  is a minor drag, not the cause of failure.

## Data quality

- Kite data is properly adjusted: **no symbol showed rampant split artifacts**, and
  only **3 of 846 trades** were flagged for a large single-day move near the signal
  (INDUSINDBK, BANDHANBNK) — both genuine price crashes, not adjustment errors. So
  the 52-week lows here are real, not manufactured.
- The one isolated artifact found (RAINBOW's sub-₹1 placeholder candles from 2019,
  pre-dating its 2022 listing) is removed automatically by the pre-listing-fragment
  cleaner.

## The caveat that dominates everything: SURVIVORSHIP BIAS

The universe is **current/surviving constituents only**. This strategy specifically
buys stocks cratering to new 52-week lows — exactly the population where the worst
names keep falling and get **delisted or removed from the index**. Those are entirely
absent here. So the measured result is an **optimistic upper bound**: the true,
point-in-time result would be **worse than −63%**. We already get a clear NO on the
flattering version; the honest version is unambiguously NO.

Additional caveats: 20 names excluded for corporate-action discontinuities (a source
of their own bias); results shown for one fixed parameter set (no optimization —
by design); past period (2018–2026) only.

## Bottom line

Across all three trail variants, both sizing methods, and gross or net of costs, the
52-week-low reversal strategy as specified produced **negative expectancy and large
drawdowns** on 846 real trades over 7 years of Nifty-500 data — and that is before
correcting for a survivorship bias that can only make it worse. **It does not show a
tradeable edge on this data.** If pursued further, the next honest step would be a
point-in-time (survivorship-free) universe; tuning parameters on this biased,
already-negative sample would just be fitting noise.
