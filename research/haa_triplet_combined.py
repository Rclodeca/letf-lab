"""Does staggering HAA to mid-month actually help the COMBINED Triplet+HAA
book, or does it just move risk around?

The single-strategy HAA backtest (research/haa_backtest.py) showed mid-month
checkpointing hurting HAA's OWN COVID number badly -- but that's testing the
wrong thing. The staggering hypothesis is about portfolio-level resilience
when running Triplet (month-end) and HAA (mid-month) TOGETHER, not about
whether mid-month is better for HAA in isolation. This builds all three
equity curves (Triplet month-end, HAA month-end, HAA mid-month) and compares
two 50/50 combined portfolios: Triplet+HAA both month-end, vs. Triplet
month-end + HAA mid-month (the actual staggered proposal).

No cross-rebalancing between the two strategy legs (each is independently
already monthly-rebalanced internally; the 50/50 split between strategies
is buy-and-hold, dollar-drift between legs).
"""
import pandas as pd

from ai_swing.backtest.metrics import cagr, max_drawdown, sortino
from ai_swing.data import get_price_service
from ai_swing.scoring import haa
from ai_swing.scoring import rotation_3of5 as rot

BENCHMARK = "SPY"


def series(ps, ticker):
    ps.refresh(ticker, days=14)
    return ps.get_close_series(ticker)


def rebalance_dates(index, anchor_shift_days):
    shifted = (index - pd.Timedelta(days=anchor_shift_days)).to_period("M")
    return index.to_series().groupby(shifted).max()


def run_strategy(closes, trade_rets, score_universe, compute_allocation_fn, anchor_shift_days):
    score_closes = closes[score_universe]
    checkpoints = rebalance_dates(score_closes.index, anchor_shift_days)

    alloc_by_date = {}
    for d in checkpoints:
        sub = score_closes.loc[:d]
        if not rot.has_enough_history(sub):
            continue
        alloc_by_date[d] = compute_allocation_fn(sub)["allocation"]

    sleeves = None
    equity_vals, equity_dates = [], []
    for d in closes.index:
        if d not in alloc_by_date and sleeves is None:
            continue
        if sleeves is not None:
            sleeves = sleeves * (1.0 + trade_rets.loc[d, sleeves.index])
        if d in alloc_by_date:
            total = sleeves.sum() if sleeves is not None else 1.0
            sleeves = pd.Series({t: total * w for t, w in alloc_by_date[d].items()})
        equity_vals.append(sleeves.sum())
        equity_dates.append(d)

    return pd.Series(equity_vals, index=equity_dates)


def main():
    ps = get_price_service()

    triplet_universe = sorted(set(rot.UNIVERSE) | {rot.CASH})
    triplet_tradable = sorted(set(triplet_universe) | {v[0] for v in rot.LEVERAGE_MAP.values()} | {BENCHMARK})

    haa_universe = sorted(set(haa.OFFENSIVE_UNIVERSE) | {haa.CANARY} | set(haa.DEFENSIVE_CANDIDATES))
    haa_tradable = sorted(set(haa_universe) | set(haa.SUBSTITUTE.values()) | {BENCHMARK})

    all_tickers = sorted(set(triplet_tradable) | set(haa_tradable))
    closes = pd.concat({t: series(ps, t) for t in all_tickers}, axis=1, sort=True).dropna()
    trade_rets = closes.pct_change().fillna(0.0)

    MID_MONTH_ANCHOR_SHIFT_DAYS = 15  # frozen experiment param -- HAA no longer runs mid-month live, see conversation history

    triplet_eq = run_strategy(closes, trade_rets, triplet_universe, rot.compute_allocation, anchor_shift_days=0)
    haa_eq_monthend = run_strategy(closes, trade_rets, haa_universe, haa.compute_allocation, anchor_shift_days=0)
    haa_eq_midmonth = run_strategy(closes, trade_rets, haa_universe, haa.compute_allocation, anchor_shift_days=MID_MONTH_ANCHOR_SHIFT_DAYS)

    common_start = max(triplet_eq.index[0], haa_eq_monthend.index[0], haa_eq_midmonth.index[0])
    triplet_eq = triplet_eq.loc[common_start:] / triplet_eq.loc[common_start]
    haa_eq_monthend = haa_eq_monthend.loc[common_start:] / haa_eq_monthend.loc[common_start]
    haa_eq_midmonth = haa_eq_midmonth.loc[common_start:] / haa_eq_midmonth.loc[common_start]

    def combine(eq_a, eq_b):
        idx = eq_a.index.intersection(eq_b.index)
        return 0.5 * eq_a.loc[idx] + 0.5 * eq_b.loc[idx]

    combined_synced = combine(triplet_eq, haa_eq_monthend)
    combined_staggered = combine(triplet_eq, haa_eq_midmonth)

    print(f"Common window: {common_start.date()} .. {closes.index[-1].date()}\n")

    def report(name, eq):
        rets = eq.pct_change().dropna()
        print(f"{name:<42}CAGR {cagr(eq)*100:6.1f}%   MaxDD {max_drawdown(eq)*100:6.1f}%   Sortino {sortino(rets):5.2f}")

    print("Individual legs:")
    report("Triplet (month-end)", triplet_eq)
    report("HAA (month-end)", haa_eq_monthend)
    report("HAA (mid-month)", haa_eq_midmonth)

    print("\n50/50 combined portfolio:")
    report("Triplet + HAA, both month-end (synced)", combined_synced)
    report("Triplet + HAA, HAA mid-month (staggered)", combined_staggered)

    WIN = {"COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}
    print("\nCrash-window check, combined portfolio:")
    for wname, (lo, hi) in WIN.items():
        for label, eq in [("synced", combined_synced), ("staggered", combined_staggered)]:
            rs = eq.pct_change().fillna(0.0).loc[lo:hi]
            if len(rs) < 5:
                continue
            es = (1 + rs).cumprod()
            dd = ((es / es.cummax() - 1).min()) * 100
            print(f"  {wname:<12} {label:<11} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%")


if __name__ == "__main__":
    main()
