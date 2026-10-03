"""52-week-low reversal swing-strategy backtester for NSE equities.

Modules:
  config      -- all tunable parameters in one place
  indicators  -- RSI (Wilder), EMA, ATR (Wilder), rolling 252-day low
  signal      -- mechanical signal detection
  engine      -- event-driven per-stock backtest with realistic fills
  reporting   -- metrics, distributions, equity curves, trade log
  data        -- pluggable data layer (yfinance default + local cache)
  universe    -- Nifty 500 constituent handling
"""
__version__ = "1.0.0"
