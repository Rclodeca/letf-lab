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

# Assets to print raw values for (price, day change, SMA250/100/200, +4%/-3% bands).
RAW_ASSETS = ["SPY", "QQQ"]
