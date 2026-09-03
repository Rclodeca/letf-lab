"""Monthly "3-of-5" LETF rotation strategy — logic only, no backtest yet.

At the end of each month:
  1. Score 14 candidate assets by avg(3m, 6m, 12m trailing returns).
  2. Dual-momentum filter: drop anything that doesn't beat BIL (cash) on the
     same score, then take the top 5 of whatever survives.
  3. Enumerate every C(n,3) combination of the survivors (n = len(candidates),
     up to 5) and pick the trio with the lowest average pairwise correlation
     of trailing daily returns.
  4. Equal-weight the trio at 1/3 each. Any of the 3 slots that couldn't be
     filled (fewer than 3 survivors) goes to BIL instead — this is the
     "catch" (partial/full risk-off), realized here as a side effect of
     filtering candidates by absolute momentum *before* the combination
     step, rather than substituting after the fact.
  5. Size the final allocation so leverage doesn't change net exposure: each
     of the 3 slots is worth 1/3 of *unlevered* exposure, so a slot filled by
     a 2x substitute gets half the dollar weight, and that freed-up capital
     is redistributed pro rata across the other slots (solved below as
     `w = 1 / sum(1/leverage_i)`).

Universe intentionally excludes BIL from ranking — BIL is the cash benchmark
used for the absolute-momentum filter and the risk-off destination, not a
candidate for the trio itself.

The selection algorithm itself lives in `ai_swing.scoring.rotation_3of5` so
it's shared byte-for-byte with the Telegram notifier (notify/daily_signals.py)
— this file is just the manual-run / trailing-history CLI on top of it.

Usage:
    python research/momentum_rotation_3of5.py              # current allocation
    python research/momentum_rotation_3of5.py --history 12 # trailing N months
"""
import sys

import pandas as pd

from ai_swing.data import get_price_service
from ai_swing.scoring.rotation_3of5 import CASH, UNIVERSE, alloc_str, compute_allocation


def load_closes() -> pd.DataFrame:
    """Full aligned close-price history for the universe + cash benchmark.

    get_history() is cache-first and only re-primes an empty/near-empty
    cache — a populated-but-stale cache (e.g. a ticker not touched in a
    week) is served as-is. Force a recent-days refresh so callers get the
    latest actual close, not whatever was cached last.
    """
    ps = get_price_service()
    all_tickers = UNIVERSE + [CASH]
    for t in all_tickers:
        ps.refresh(t, days=14)
    return pd.concat({t: ps.get_close_series(t) for t in all_tickers}, axis=1, sort=True).dropna()


def print_report(r: dict) -> None:
    print(f"As of: {r['as_of']}")
    print("Momentum Ranking")
    print(r["ranking"])
    print("Cash Momentum")
    print(f"{r['cash_score']:.4f}")
    print("Abs Survivors")
    print(f"{r['abs_survivors']:.3f}")
    print("Rel Candidates")
    print(r["candidates"])
    print("Selected")
    print(r["selected"])
    print("Avg Pairwise Corr")
    print(f"{r['avg_corr']:.4f}" if r["avg_corr"] is not None else "-")
    print("Weights")
    for t, w in sorted(r["weights"].items(), key=lambda kv: -kv[1]):
        print(f"{t:<6} {w:.4f}")
    print(f"Allocation: {alloc_str(r['allocation'])}")


def month_end_dates(closes: pd.DataFrame, n_months: int):
    """Last available trading date in each of the trailing `n_months`
    calendar months (including the current, possibly partial, month)."""
    by_month = closes.index.to_series().groupby(closes.index.to_period("M")).max()
    return list(by_month.tail(n_months))


def run_history(n_months: int) -> None:
    closes = load_closes()
    dates = month_end_dates(closes, n_months)
    print(f"{'Month':<10} {'As of':<12} {'Abs Surv':<9} {'Selected':<28} {'Avg Corr':<9} Allocation")
    for d in dates:
        r = compute_allocation(closes.loc[:d])
        corr_str = f"{r['avg_corr']:.3f}" if r["avg_corr"] is not None else "-"
        print(f"{d.strftime('%Y-%m'):<10} {str(r['as_of']):<12} {r['abs_survivors']:<9.0f} "
              f"{','.join(r['selected']) or '-':<28} {corr_str:<9} "
              f"{alloc_str(r['allocation'])}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--history":
        n_months = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        run_history(n_months)
        return
    closes = load_closes()
    print_report(compute_allocation(closes))


if __name__ == "__main__":
    main()
