"""Declarative watchlist for the daily notifier.

Edit this file to change what gets tracked. `daily_signals.py` reads these
lists and reuses the LETF Lab engine to evaluate them, so the numbers always
match the app.
"""

# The 4 standard indicators, mirroring backend/scripts/seed.py. Applied to each
# benchmark below to produce a vote-of-k signal.
_STANDARD_INDICATORS = [
    {"name": "SMA250", "type": "SMA_GATE", "params": {"period": 250, "threshold": 0.05}},
    {"name": "SMA100", "type": "SMA_GATE", "params": {"period": 100, "threshold": 0.05}},
    {"name": "Vol21d", "type": "VOL_GATE", "params": {"window": 21, "threshold": 0.40}},
    {"name": "AR(1)", "type": "AR1_GATE", "params": {"window": 30, "threshold": 0.0}},
]

# Standard vote-of-2 signals to report. Signals compute on the benchmark ticker.
STRATEGIES = [
    {"name": "SPY", "benchmark": "SPY", "k": 2, "indicators": _STANDARD_INDICATORS},
    {"name": "QQQ", "benchmark": "QQQ", "k": 2, "indicators": _STANDARD_INDICATORS},
]

# 3-state SMA-200 "traffic light" strategies. State is decided by the latest
# close vs SMA200 bands:
#   BUY  (green)  price > SMA200 * upper
#   SELL (red)    price < SMA200 * lower
#   HOLD (yellow) otherwise (inside the band)
TRAFFIC_LIGHTS = [
    {"name": "SPY 200SMA", "asset": "SPY", "key": "SPY_200sma", "upper": 1.04, "lower": 0.97},
    {"name": "QQQ 200SMA", "asset": "QQQ", "key": "QQQ_200sma", "upper": 1.04, "lower": 0.97},
]

# AND-combined multi-asset gates: risk-on only when EVERY indicator passes,
# each on its own asset. This is the r/LETFs "Golden Ratio" de-lever signal
# (see research/golden_ratio_delever.py, variant C, and the robustness-grid
# follow-up) — SPY 200SMA with a +/-1% hysteresis band AND TIP 200SMA with a
# tighter +/-0.15% band. SPY's band was widened from 0.5% -> 1% after testing
# showed 1% strictly dominates on CAGR/MaxDD/Sortino/trade-count. TIP's band
# was widened from 0.1% -> 0.15% after research/golden_ratio_tip_band_compare.py
# showed 0.15% strictly dominates 0.1% (same Sortino, slightly higher CAGR,
# ~12% fewer trades, byte-identical 2022-bear behavior) — a free improvement,
# not a trade-off. TIP still stays comparatively tight because its signal is
# the primary regime-read for slow bear markets (0.25%+ trades bear-market
# protection for fewer whipsaws; 0.5%+ measurably hurts it).
DUAL_GATES = [
    {
        "name": "Golden Ratio (SPY+TIP)",
        "key": "golden_ratio_signal",
        "indicators": [
            {"asset": "SPY", "name": "SPY_SMA200", "type": "SMA_GATE", "params": {"period": 200, "threshold": 0.01}},
            {"asset": "TIP", "name": "TIP_SMA200", "type": "SMA_GATE", "params": {"period": 200, "threshold": 0.0015}},
        ],
    },
]

# Emergency euphoria-valve checks — a blow-off-top guard from the dot-com
# backtest (research/synth_200sma_spy_tqqq_1995_euphoria.py): force attention
# whenever price runs unusually far above its own 200SMA, regardless of what
# the regular gates say. Normally hidden from the message; when any of these
# trip, an EMERGENCY banner is shown at the very top of the alert.
EMERGENCY = [
    {"name": "QQQ euphoria", "asset": "QQQ", "threshold": 0.30},
    {"name": "SPY euphoria", "asset": "SPY", "threshold": 0.30},
]

# Assets to print raw values for (price, day change, SMA250/100/200, +4%/-3% bands).
RAW_ASSETS = ["SPY", "QQQ"]
