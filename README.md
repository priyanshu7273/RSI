# NSE 52-Week-Low Reversal — Swing Strategy Backtester

A modular, event-driven backtester that measures whether a fully mechanical
"52-week-low reversal" swing strategy has a real, tradeable edge on the Nifty 500
universe over ~7 years of daily data.

The design goal is **correctness and honesty about data quality over a clean-looking
result.** If the data is biased or the edge is unproven, the output says so loudly.

---

## The strategy (fully mechanical, no discretion)

A stock grinds down to repeated 52-week lows; then one day it fails to make a new
low and closes green — that green candle is the *signal candle*. We buy only if
price then breaks the signal candle's high.

**Signal (day T)** — both must hold:
1. **Oversold at a fresh low coming in.** A new 252-day low occurred on T-1 (strict,
   `low_lookback_days=1`) or within the last `low_lookback_days` days, **and** RSI(14)
   (Wilder) is in `[rsi_min, rsi_max)` on the capitulation day (T-1 by default). RSI is
   tested on T-1, not T: by the time T prints green, RSI has already ticked up off the
   bottom, so testing it on T would wrongly reject good signals.
2. **The green break.** Day T's low does **not** undercut the running 252-day low, **and**
   day T closes green (`close > open`).

**Entry** — buy-stop at `signal_high + 1 tick`, live for `entry_window` (default 5)
trading days after T. Fills at the stop level, or at the day's **open** if it gaps above.
One open position per stock; new signals while a position/order is live are ignored.

**Stops & management**
- Initial stop = `signal_low − 1 tick`, held from entry.
- At +10% (intraday touch): activate the trailing stop **and** raise the stop floor to
  breakeven. The trail only ratchets up.
- Trail method (parameter, all three implemented):
  - `ema10` (default): exit on a daily **close** below the 10-day EMA.
  - `lastlow`: trail below the low of the last `N` completed candles (default 2).
  - `atr`: stop = running high − `atr_mult`×ATR(14) (default 2×).
- **Hard cap (FIRM):** at +20% (intraday touch) exit immediately at +20%, regardless of
  the trail. This caps winners on purpose; the report quantifies what the cap costs.

**Exit priority each day:** (1) +20% target → (2) trailing stop → (3) initial/breakeven.

All parameters live in `rsi_backtest/config.py`.

---

## Install

```bash
pip install -r requirements.txt    # yfinance, pandas, numpy, matplotlib, pyarrow, lxml
```

## Run

```bash
# Spec default: fetch adjusted OHLCV from Yahoo (needs Yahoo reachable), all 3 variants
python run_backtest.py

# Run on already-cached data (offline), optionally cap the universe
python run_backtest.py --provider cache --data-dir data_cache

# Just verify the data layer on a few symbols (build-order step 1)
python run_backtest.py --provider cache --symbols RELIANCE INFY TCS --verify-only

# Run the full tests
python tests/test_indicators.py && python tests/test_signal_engine.py
```

Key CLI flags: `--provider {yfinance,cache}`, `--symbols ...`, `--max-symbols N`,
`--variants ema10 lastlow atr`, `--years 7`, `--cost 0.002`, `--no-live-universe`,
`--data-dir`, `--outputs`.

---

## Data layer

Two interchangeable providers return the same normalized frame
(`open, high, low, close, volume`, tz-naive daily index):

- **`yfinance` (spec default).** Pulls split/bonus-**adjusted** OHLCV (`auto_adjust=True`)
  with the NSE `.NS` suffix (e.g. `RELIANCE.NS`), in polite throttled batches
  (`yf_batch_size`, `yf_pause_sec`), and **caches each symbol to `data_cache/<SYM>.parquet`**
  so re-runs never re-hit Yahoo and you don't get rate-limited.
- **`cache`.** Reads only the local cache (parquet/csv); never touches the network. Used
  for offline runs and for data ingested from other sources.

No fabrication, no forward-filling of OHLCV. Symbols with fewer than
`min_history_days` rows are skipped and counted.

### ⚠️ The yfinance NSE-adjustment caveat
yfinance's NSE split/bonus adjustments are sometimes wrong, which can *manufacture a fake
52-week low* — and this strategy's trigger **is** the 52-week low. The engine flags any
trade whose signal sits near a suspiciously large single-day move
(`possible_corp_action_near_signal` in the trade log) so you can spot-check it. Don't
silently trust every low.

### How the bundled cache in this repo was produced
This project was built in a sandbox where **Yahoo Finance is firewalled**, so `yfinance`
could not fetch here. The cached data was instead pulled from the **Kite Connect**
(Zerodha) historical API — the real NSE broker feed, already split/bonus-adjusted — and
ingested with `tools/ingest_kite.py`:

```bash
python tools/ingest_kite.py --symbol RELIANCE --file <kite_chunkA.json> <kite_chunkB.json>
```

On your own machine with Yahoo reachable, just run `python run_backtest.py` (provider
`yfinance`) to reproduce the full Nifty 500 run end-to-end from Yahoo; the strategy,
engine, and reporting are identical regardless of provider.

---

## Outputs (per trail variant)

Printed report + `outputs/variant_<trail>.png` (equity curves + winner histogram) +
`outputs/trades_<trail>.csv` (full trade log).

- **Per-trade stats** (gross and net of costs): total trades, win rate, avg win %,
  avg loss %, avg R, expectancy (R and %), profit factor, avg days held.
- **Exit-type counts**: `target_20pct` / `trail_*` / `initial_stop` / `breakeven`.
- **+20% cap analysis**: how many trades hit the cap, how many were *still rising* past
  +20% if left uncapped, and the average return the cap left on the table (this is the
  cost of the cap, measured directly against each trade's hypothetical uncapped exit).
- **Winner distribution**: percentiles + histogram of winning-trade returns.
- **Portfolio metrics** (capital-aware forward simulation, both sizing modes × gross/net):
  total return, CAGR, max drawdown, return/drawdown, Sharpe, plus counts of trades skipped
  for capital or concurrency limits.
- **Signal funnel**: signals detected → entered → expired (buy-stop never triggered) →
  skipped (position already open).

### Trade-log columns
`symbol, tminus1_date, signal_T_date, entry_date, entry_price, initial_stop, exit_date,
exit_price, exit_reason, return_pct_gross, return_pct_net, r_multiple_gross, days_held,
activated_10pct, rsi_at_signal, uncapped_return_pct, possible_corp_action_near_signal`.

### Position sizing (both reported)
- **risk**: risk `risk_per_trade` (1%) of equity on the entry-to-initial-stop distance.
- **equal**: deploy `equal_weight_frac` (10%) of equity per position.

Both are run through a shared-capital forward simulation that respects available cash and
a concurrent-position cap — important because 52-week-low signals **cluster** in crashes,
so a real portfolio cannot take every one. Ignoring that would flatter the result.

---

## Honesty / caveats (read before trusting any number)

- **SURVIVORSHIP BIAS (the big one).** The universe is *current* constituents only. This
  strategy buys stocks cratering to new 52-week lows — exactly the population where some
  names keep falling, get removed, or delist. Those losers are missing, so win rate and
  returns are biased **upward**. Any positive result is an optimistic upper bound. A truly
  honest version needs point-in-time (survivorship-free) constituent history, which neither
  yfinance nor a current index list provides.
- **Sample size.** Variants with < 30 trades are flagged as unreliable; < 100 is still
  thin. The verdict is caveated accordingly.
- **Costs.** Everything is reported gross and net of a configurable round-trip cost
  (default 0.2%). A gross edge that dies net of costs is not tradeable.
- **The +20% cap** truncates winners by design and mechanically lowers expectancy vs an
  uncapped version — see the cap analysis for the magnitude.
- **No parameter tuning.** The rules run exactly as specified. Optimization is a separate,
  later step (and invites overfitting).

The run ends with a plain-English verdict: does this show an edge on this data, with these
caveats — yes / no / unclear, and why.

---

## Module layout

```
rsi_backtest/
  config.py      all tunable parameters (one place)
  indicators.py  RSI (Wilder), EMA, ATR (Wilder), rolling 252-day low  [hand-checked]
  signal.py      mechanical signal detection
  engine.py      per-stock event-driven engine, realistic fills, all exits + uncapped sim
  portfolio.py   shared-capital simulation -> equity curve, CAGR, maxDD, Sharpe
  reporting.py   metrics, distributions, cap analysis, plots, trade log, verdict
  data.py        pluggable data layer (yfinance + cache) with parquet caching
  universe.py    Nifty 500 constituents (live NSE fetch + bundled fallback)
run_backtest.py  CLI entry point
tools/ingest_kite.py   ingest Kite historical JSON into the cache
tests/           hand-checked indicator tests + signal/engine scenario tests
```
