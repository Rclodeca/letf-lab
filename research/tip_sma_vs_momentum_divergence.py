"""How closely does TIP price-vs-SMA200 track TIP 1/3/6/12m avg momentum?

Not a backtest -- a direct diagnostic on the two underlying continuous
signals themselves: overall correlation, whether that correlation is stable
over time (rolling), and the specific episodes where their *sign* disagrees
(price above SMA200 while momentum is negative, or vice versa) for more than
a couple weeks, since those sustained-disagreement stretches are exactly
where golden_ratio_tip_sma_vs_momentum.py's backtest showed the two signals
producing different risk-on/off calls.
"""
import numpy as np
import pandas as pd

from ai_swing.data import get_price_service

ps = get_price_service()
tip = ps.get_close_series("TIP")

SMA_PERIOD = 200
# Frozen at the trading-day-offset convention this analysis was actually run
# with -- ai_swing.scoring.rotation_3of5's own RETURN_OFFSETS has since been
# replaced by calendar-month lookback (MONTHS_BACK), which isn't what this
# script's momentum_score() implements below.
MOM_OFFSETS = (21, 63, 126, 252)  # 1/3/6/12m trading days


def momentum_score(prices, offsets):
    return sum(prices / prices.shift(k) - 1 for k in offsets) / len(offsets)


sma = tip.rolling(SMA_PERIOD, min_periods=SMA_PERIOD).mean()
price_vs_sma = (tip / sma - 1).dropna()
mom = momentum_score(tip, MOM_OFFSETS).dropna()

idx = price_vs_sma.index.intersection(mom.index)
price_vs_sma, mom = price_vs_sma.loc[idx], mom.loc[idx]

print(f"Window: {idx[0].date()}..{idx[-1].date()}  (n={len(idx)})\n")
print(f"Full-history correlation (price-vs-SMA200 %, momentum score): {price_vs_sma.corr(mom):.3f}")

roll_corr = price_vs_sma.rolling(252).corr(mom)
print(f"Rolling 252d correlation: min {roll_corr.min():.2f}, "
      f"median {roll_corr.median():.2f}, max {roll_corr.max():.2f}")
print("\nRolling 252d correlation by year (year-end value):")
for yr, v in roll_corr.resample("YE").last().items():
    if not np.isnan(v):
        print(f"  {yr.year}: {v:5.2f}")

# Sign-disagreement episodes lasting >= 10 trading days.
disagree = np.sign(price_vs_sma) != np.sign(mom)
print(f"\nOverall sign disagreement: {disagree.mean()*100:.1f}% of days")

episodes = []
in_ep, start = False, None
for i, (dt, d) in enumerate(disagree.items()):
    if d and not in_ep:
        in_ep, start = True, dt
    elif not d and in_ep:
        in_ep = False
        episodes.append((start, disagree.index[i - 1]))
if in_ep:
    episodes.append((start, disagree.index[-1]))

long_episodes = [(s, e) for s, e in episodes if (e - s).days >= 14]
print(f"\nSign-disagreement episodes lasting >= ~2 weeks ({len(long_episodes)} of {len(episodes)} total):")
print(f"{'Start':<12}{'End':<12}{'Days':>6}{'Price-vs-SMA200':>18}{'Mom(1/3/6/12m)':>16}")
for s, e in long_episodes:
    mid = price_vs_sma.loc[s:e].index[len(price_vs_sma.loc[s:e]) // 2]
    print(f"{s.date()!s:<12}{e.date()!s:<12}{(e-s).days:>6}"
          f"{price_vs_sma.loc[mid]*100:17.2f}%{mom.loc[mid]*100:15.2f}%")
