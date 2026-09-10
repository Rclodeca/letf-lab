"""Archived watchlist entries — removed from the live notifier on 2026-09-05
(commit 8cf9230, "refactor: simplify Telegram notifier to Triplet + HAA
only") to slim the Telegram message down to just Triplet + HAA. Kept here
for reference in case any of these need to come back.

(The Golden Ratio SPY+TIP dual gate was resurrected back into watchlist.py
on 2026-09-10 — see DUAL_GATES there — so it's no longer listed below.)

Not imported by watchlist.py or daily_signals.py — this is inert config.
To resurrect one:
  1. Copy the relevant entry(ies) back into watchlist.py and re-add them to
     the corresponding import list in daily_signals.py.
  2. Restore the matching compute()/format_message() handling in
     daily_signals.py — the full pre-removal logic (vote-of-k gate
     evaluation, AND-combined dual-gate evaluation, 3-state traffic lights,
     and their ladder-style raw-values rendering incl. `_ladder()`) is at
     commit 023e14c, i.e. `git show 023e14c:notify/daily_signals.py`.
"""

# The 4 standard indicators, mirroring backend/scripts/seed.py. Applied to each
# benchmark below to produce a vote-of-k signal.
_STANDARD_INDICATORS = [
    {"name": "SMA250", "type": "SMA_GATE", "params": {"period": 250, "threshold": 0.05}},
    {"name": "SMA100", "type": "SMA_GATE", "params": {"period": 100, "threshold": 0.05}},
    {"name": "Vol21d", "type": "VOL_GATE", "params": {"window": 21, "threshold": 0.40}},
    {"name": "AR(1)", "type": "AR1_GATE", "params": {"window": 30, "threshold": 0.0}},
]

# Standard vote-of-2 signals. Signals compute on the benchmark ticker.
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
