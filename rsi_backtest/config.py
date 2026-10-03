"""Central configuration for the 52-week-low reversal backtest.

Every tunable parameter lives here. Nothing in the strategy, engine, or
reporting layers should hard-code a number that belongs in this block.

The defaults encode the strategy EXACTLY as specified. Do not tune these to
flatter results -- parameter optimisation is a separate, later exercise and
must be done with proper out-of-sample discipline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


# --------------------------------------------------------------------------
# Trail-method sub-configs
# --------------------------------------------------------------------------
TrailMethod = Literal["ema10", "lastlow", "atr"]


@dataclass(frozen=True)
class TrailConfig:
    """Parameters for each trailing-stop method.

    Only the fields relevant to the active `method` are used, but all are kept
    here so a single Config can describe any variant.
    """
    method: TrailMethod = "ema10"

    # ema10: exit on a daily CLOSE below the N-day EMA.
    ema_period: int = 10

    # lastlow: trail below the low of the last N completed candles.
    lastlow_n: int = 2

    # atr: stop = running_high - atr_mult * ATR(atr_period).
    atr_period: int = 14
    atr_mult: float = 2.0


# --------------------------------------------------------------------------
# Main config
# --------------------------------------------------------------------------
RsiOnDay = Literal["T-1", "T"]
SizingMode = Literal["risk", "equal"]


@dataclass(frozen=True)
class Config:
    # ---- Universe & data ------------------------------------------------
    years: int = 7                      # lookback window of daily history
    # Data provider: "yfinance" (spec default) or "cache" (read pre-fetched
    # CSV/parquet from data_dir, e.g. populated from the Kite MCP).
    provider: str = "yfinance"
    data_dir: str = "data_cache"
    outputs_dir: str = "outputs"
    yf_batch_size: int = 25             # symbols per yfinance batch download
    yf_pause_sec: float = 1.5           # polite pause between batches
    min_history_days: int = 300         # skip symbols with fewer rows than this

    # ---- 52-week low / RSI signal ----------------------------------------
    week52_lookback: int = 252          # trading days in the "52-week" window
    # Require the fresh 52w low STRICTLY on T-1 (low_lookback_days == 1), OR
    # allow it to have occurred within the last `low_lookback_days` days.
    low_lookback_days: int = 1
    rsi_period: int = 14
    rsi_min: float = 0.0                # lower band (0 => pure "< rsi_max")
    rsi_max: float = 25.0               # oversold threshold
    rsi_on_day: RsiOnDay = "T-1"        # which day the RSI test is evaluated on

    # ---- Entry -----------------------------------------------------------
    tick: float = 0.05                  # NSE tick size (rupees)
    entry_window: int = 5               # buy-stop live for N trading days after T

    # ---- Trade management ------------------------------------------------
    activate_pct: float = 0.10          # +10% intraday touch activates trail+BE
    hard_target_pct: float = 0.20       # +20% intraday touch -> firm exit (CAP)
    trail: TrailConfig = field(default_factory=TrailConfig)

    # ---- Costs -----------------------------------------------------------
    # Round-trip cost as a fraction of notional (brokerage + STT + slippage).
    round_trip_cost: float = 0.002      # ~0.2%

    # ---- Position sizing -------------------------------------------------
    sizing: SizingMode = "risk"         # "risk" (1% equity on stop distance) or "equal"
    risk_per_trade: float = 0.01        # fraction of equity risked (risk sizing)
    equal_weight_frac: float = 0.10     # fraction of equity per position (equal sizing)
    starting_equity: float = 1_000_000.0  # rupees
    max_concurrent: int = 0             # 0 => unlimited concurrent positions

    def with_trail(self, method: TrailMethod, **kwargs) -> "Config":
        """Return a copy of this config with a different trail method."""
        new_trail = TrailConfig(method=method, **{
            **{k: getattr(self.trail, k) for k in
               ("ema_period", "lastlow_n", "atr_period", "atr_mult")},
            **kwargs,
        })
        return self.replace(trail=new_trail)

    def replace(self, **kwargs) -> "Config":
        from dataclasses import replace as _replace
        return _replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


# A ready-to-use default instance.
DEFAULT = Config()

# The three trail variants the spec asks to run and compare.
TRAIL_VARIANTS: tuple[TrailMethod, ...] = ("ema10", "lastlow", "atr")
