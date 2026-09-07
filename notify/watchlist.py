"""Declarative watchlist for the daily notifier.

Edit this file to change what gets tracked. `daily_signals.py` reads these
lists and reuses the LETF Lab engine to evaluate them, so the numbers always
match the app.
"""

# Emergency euphoria-valve checks — a blow-off-top guard from the dot-com
# backtest (research/synth_200sma_spy_tqqq_1995_euphoria.py): force attention
# whenever price runs unusually far above its own 200SMA, regardless of what
# the regular gates say. Normally hidden from the message; when any of these
# trip, an EMERGENCY banner is shown at the very top of the alert.
EMERGENCY = [
    {"name": "QQQ euphoria", "asset": "QQQ", "threshold": 0.30},
    {"name": "SPY euphoria", "asset": "SPY", "threshold": 0.30},
]

# Assets to print the full raw snapshot for: day change, price, SMA100,
# SMA200, and % above/below each SMA.
RAW_ASSETS = ["SPY", "QQQ"]

# Assets to print the same raw snapshot for, minus the SMA100 line.
RAW_ASSETS_200_ONLY = ["TIP"]

# Monthly "Triplet" momentum rotation (3-of-5 selection from a 14-asset
# universe). Unlike the gate strategies this replaced, it doesn't reduce to a
# risk-on/off boolean — it's a 14-asset momentum rank, dual-momentum filter
# vs. BIL, then least-correlated-trio selection. The actual algorithm lives
# in ai_swing.scoring.rotation_3of5 (shared with
# research/momentum_rotation_3of5.py so the numbers match exactly); this
# entry just tells the notifier to compute and display it. Recomputed daily
# from the same trailing-return windows, but only "actionable" (banner-worthy)
# on days the selected tickers/allocation actually change.
ROTATION_STRATEGIES = [
    {"name": "Triplet", "key": "rotation_3of5_signal"},
]

# Monthly Hybrid Asset Allocation (HAA): TIP canary decides risk-on/off; when
# on, a 9-asset offensive universe is filtered by absolute momentum vs. BIL
# and the top 4 survivors held equal-weight via their leveraged/substitute
# funds; when off (or a slot's unfilled), allocated to whichever of IEF/BIL
# scores higher. Algorithm lives in ai_swing.scoring.haa (shared with
# research/haa.py, verified against 7 known reference months); this entry
# just tells the notifier to compute and display it. Same month-end cadence
# as the Triplet rotation above (a staggered mid-month cadence was tested and
# rejected — see research/haa_triplet_combined.py — it hurt the combined
# portfolio's COVID drawdown rather than adding resilience).
HAA_STRATEGIES = [
    {"name": "HAA", "key": "haa_signal"},
]

# Daily SPY 200SMA "Switch": a buffered TQQQ/QQQ band strategy. Unlike the
# retired 3-state traffic light (see archived_strategies.py), the zone
# between the bands HOLDS the previous trading day's position instead of
# resolving to a third neutral state (buffer against whipsaws). A two-tier
# QQQ-euphoria guard overrides the SPY read whenever QQQ has itself run too
# far above its own 200SMA: 30% -> deleverage TQQQ down to QQQ, 40% -> cash.
# Rules and +4%/-3% bands mirror research/synth_200sma_spy_tqqq_1995_euphoria.py
# (single-tier 30% cash valve); the second 30%/40% tier here is a variant not
# yet backtested in research/.
SWITCH_STRATEGIES = [
    {
        "name": "SPY 200SMA Switch",
        "key": "spy_switch_signal",
        "spy_asset": "SPY",
        "qqq_asset": "QQQ",
        "upper": 1.04,
        "lower": 0.97,
        "qqq_delever_threshold": 0.30,
        "qqq_cash_threshold": 0.40,
    },
]
