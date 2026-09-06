"""Hybrid Asset Allocation (HAA) strategy — logic only, no backtest yet.

At the end of each month:
  1. Canary: is TIP's 1/3/6/12-month equal-weighted momentum score positive?
  2. If yes -> drop any offensive-universe asset that doesn't beat BIL's
     score (IEF can independently survive this on its own momentum), rank
     the rest, take the top 4, equal-weight 25% each, trade each via its
     substitute/leveraged fund at face value (no leverage-based
     reweighting). Unfilled slots backfill to the best-of-IEF/BIL asset's
     own substitute (e.g. IEF backfills as UST).
  3. If no -> allocate 100% to whichever of IEF/BIL scores higher, held
     unlevered.

The selection algorithm itself lives in `ai_swing.scoring.haa` so it's
shared with any live notifier that later wires this in — this file is just
the manual-run / trailing-history CLI on top of it, same pattern as
`research/momentum_rotation_3of5.py`.

Usage:
    python research/haa.py              # current allocation
    python research/haa.py --history 12 # trailing N months
"""
import sys

import pandas as pd

from ai_swing.data import get_price_service
from ai_swing.scoring.haa import CANARY, DEFENSIVE_CANDIDATES, OFFENSIVE_UNIVERSE, compute_allocation
from ai_swing.scoring.rotation_3of5 import alloc_str


def load_closes() -> pd.DataFrame:
    ps = get_price_service()
    all_tickers = sorted(set(OFFENSIVE_UNIVERSE) | {CANARY} | set(DEFENSIVE_CANDIDATES))
    for t in all_tickers:
        ps.refresh(t, days=14)
    return pd.concat({t: ps.get_close_series(t) for t in all_tickers}, axis=1, sort=True).dropna()


def print_report(r: dict) -> None:
    print(f"As of: {r['as_of']}")
    print("Canary (TIP) Score")
    print(f"{r['canary_score']:.4f}")
    print("Defensive Scores")
    for t, v in r["defensive_scores"].items():
        print(f"  {t:<6} {v:.4f}")
    print(f"Best Defensive: {r['best_defensive']}")
    print("Abs Survivors")
    print(f"{r['abs_survivors']:.0f}")
    print("Offensive Ranking")
    for row in r["offensive_ranking"]:
        print(f"  {row['ticker']:<6} {row['score']:.4f}")
    print(f"Risk On: {r['risk_on']}")
    print("Selected")
    print(r["selected"])
    print(f"Allocation: {alloc_str(r['allocation'])}")


def month_end_dates(closes: pd.DataFrame, n_months: int):
    by_month = closes.index.to_series().groupby(closes.index.to_period("M")).max()
    return list(by_month.tail(n_months))


def run_history(n_months: int) -> None:
    closes = load_closes()
    dates = month_end_dates(closes, n_months)
    print(f"{'Month':<10} {'As of':<12} {'TIP score':<10} {'Risk On':<8} {'Selected':<32} Allocation")
    for d in dates:
        r = compute_allocation(closes.loc[:d])
        print(f"{d.strftime('%Y-%m'):<10} {str(r['as_of']):<12} {r['canary_score']:<10.4f} "
              f"{str(r['risk_on']):<8} {','.join(r['selected']) or '-':<32} {alloc_str(r['allocation'])}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--history":
        n_months = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        run_history(n_months)
        return
    closes = load_closes()
    print_report(compute_allocation(closes))


if __name__ == "__main__":
    main()
