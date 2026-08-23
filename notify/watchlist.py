"""Declarative watchlist for the daily notifier.

Edit this file to change what gets tracked. Nothing else needs to change:
`daily_signals.py` reads these two lists and reuses the LETF Lab engine to
evaluate them, so the numbers always match the app.
"""

# The 4 standard indicators, mirroring backend/scripts/seed.py. Applied to each
# benchmark below to produce a vote-of-k signal.
_STANDARD_INDICATORS = [
    {"name": "SMA250", "type": "SMA_GATE", "params": {"period": 250, "threshold": 0.0}},
    {"name": "SMA100", "type": "SMA_GATE", "params": {"period": 100, "threshold": 0.0}},
    {"name": "Vol21d", "type": "VOL_GATE", "params": {"window": 21, "threshold": 0.40}},
    {"name": "AR(1)", "type": "AR1_GATE", "params": {"window": 30, "threshold": 0.0}},
]

# Standard vote-of-2 signals to report. Signals compute on the benchmark ticker.
STRATEGIES = [
    {"name": "SPY", "benchmark": "SPY", "k": 2, "indicators": _STANDARD_INDICATORS},
    {"name": "QQQ", "benchmark": "QQQ", "k": 2, "indicators": _STANDARD_INDICATORS},
]

# Standalone SMA-200 boolean checks. Each compares the latest close to
# SMA200 * factor. op "lt" fires when price < threshold; "gt" when price > threshold.
SMA200_CHECKS = [
    {"key": "SPY_below_sma200",    "asset": "SPY", "label": "SPY < 200SMA",     "op": "lt", "factor": 1.00},
    {"key": "SPY_above_sma200_p4", "asset": "SPY", "label": "SPY > 200SMA +4%", "op": "gt", "factor": 1.04},
    {"key": "SPY_below_sma200_m3", "asset": "SPY", "label": "SPY < 200SMA −3%", "op": "lt", "factor": 0.97},
    {"key": "QQQ_below_sma200",    "asset": "QQQ", "label": "QQQ < 200SMA",     "op": "lt", "factor": 1.00},
    {"key": "QQQ_above_sma200_p4", "asset": "QQQ", "label": "QQQ > 200SMA +4%", "op": "gt", "factor": 1.04},
    {"key": "QQQ_below_sma200_m3", "asset": "QQQ", "label": "QQQ < 200SMA −3%", "op": "lt", "factor": 0.97},
]
