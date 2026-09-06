"""Backtest of the Hybrid Asset Allocation (HAA) strategy (see
ai_swing.scoring.haa / research/haa.py for the algorithm derivation and its
verification against 7 known reference months).

Same methodology as research/momentum_rotation_3of5_backtest.py: at each
month-end with >=1yr of trailing history, decide the target allocation
using that day's close (scoring always runs on the UNLEVERED universe/
canary/defensive tickers). The new weights take effect starting the NEXT
trading day's return -- the old weights still earn the rebalance day's own
return before the switch -- so there's no lookahead. Between rebalances,
sleeve dollar-weights drift with each holding's actual daily return (no
interim rebalancing).

Trades the REAL substitute tickers (UST/UBT/QLD/SSO/EET/EFO/URE/UWM/PDBC),
same as research/haa.py -- this means the backtest window is bound by
PDBC's inception (2014-11-07), the newest of the tradable substitutes.

Run:
    python research/haa_backtest.py [years_back]
"""
import sys

import pandas as pd

from ai_swing.backtest.metrics import cagr, max_drawdown, sortino
from ai_swing.data import get_price_service
from ai_swing.scoring.haa import CANARY, DEFENSIVE_CANDIDATES, OFFENSIVE_UNIVERSE, SUBSTITUTE, compute_allocation
from ai_swing.scoring.rotation_3of5 import alloc_str, has_enough_history

BENCHMARK = "SPY"


def series(ps, ticker):
    ps.refresh(ticker, days=14)
    return ps.get_close_series(ticker)


def run_backtest():
    ps = get_price_service()
    score_universe = sorted(set(OFFENSIVE_UNIVERSE) | {CANARY} | set(DEFENSIVE_CANDIDATES))
    tradable = sorted(set(score_universe) | set(SUBSTITUTE.values()) | {BENCHMARK})
    closes = pd.concat({t: series(ps, t) for t in tradable}, axis=1, sort=True).dropna()

    score_closes = closes[score_universe]
    trade_rets = closes.pct_change().fillna(0.0)

    by_month = score_closes.index.to_series().groupby(score_closes.index.to_period("M")).max()

    alloc_by_date = {}
    log_rows = []
    for d in by_month:
        sub = score_closes.loc[:d]
        if not has_enough_history(sub):
            continue
        r = compute_allocation(sub)
        alloc_by_date[d] = r["allocation"]
        log_rows.append((d, r["risk_on"], r["selected"], r["allocation"]))

    sleeves = None
    equity_vals, equity_dates = [], []
    for d in closes.index:
        if d not in alloc_by_date and sleeves is None:
            continue
        if sleeves is not None:
            sleeves = sleeves * (1.0 + trade_rets.loc[d, sleeves.index])
        if d in alloc_by_date:
            total = sleeves.sum() if sleeves is not None else 1.0
            target = alloc_by_date[d]
            sleeves = pd.Series({t: total * w for t, w in target.items()})
        equity_vals.append(sleeves.sum())
        equity_dates.append(d)

    equity = pd.Series(equity_vals, index=equity_dates)
    bh_equity = (1 + closes[BENCHMARK].pct_change().fillna(0.0)).cumprod()
    bh_equity = bh_equity.loc[equity.index[0]:] / bh_equity.loc[equity.index[0]]

    return equity, bh_equity, log_rows


def trailing_window(equity: pd.Series, years: float):
    end = equity.index[-1]
    start_target = end - pd.DateOffset(years=years)
    if start_target < equity.index[0]:
        return None
    sub = equity.loc[start_target:]
    total_return = sub.iloc[-1] / sub.iloc[0] - 1
    return {
        "start": sub.index[0].date(), "end": sub.index[-1].date(),
        "total_return": total_return, "cagr": cagr(sub), "max_dd": max_drawdown(sub),
        "sortino": sortino(sub.pct_change().dropna()),
    }


def main():
    years_back = float(sys.argv[1]) if len(sys.argv) > 1 else 5

    equity, bh_equity, log_rows = run_backtest()
    print(f"Backtest window: {equity.index[0].date()} -> {equity.index[-1].date()} "
          f"({len(log_rows)} monthly rebalances, PDBC-inception-bound)")
    risk_on_pct = sum(1 for _, ro, _, _ in log_rows if ro) / len(log_rows) * 100
    print(f"Risk-on at {risk_on_pct:.0f}% of rebalance dates\n")

    print(f"{'Window':<8} {'Strategy return':<18} {'Strategy CAGR':<16} {'MaxDD':<10} {'Sortino':<10} {'SPY B&H return':<16}")
    for label, yrs in [("1yr", 1), ("3yr", 3), (f"{years_back:g}yr", years_back)]:
        w = trailing_window(equity, yrs)
        if w is None:
            print(f"{label:<8} insufficient history")
            continue
        bh_w = trailing_window(bh_equity, yrs)
        bh_ret = f'{bh_w["total_return"] * 100:+.1f}%' if bh_w else "n/a"
        print(f'{label:<8} {w["total_return"] * 100:>+7.1f}%          '
              f'{w["cagr"] * 100:>+6.1f}%          {w["max_dd"] * 100:>+6.1f}%    '
              f'{w["sortino"]:>+6.2f}     {bh_ret}')

    def full_window(eq: pd.Series):
        return {
            "total_return": eq.iloc[-1] / eq.iloc[0] - 1, "cagr": cagr(eq), "max_dd": max_drawdown(eq),
            "sortino": sortino(eq.pct_change().dropna()),
        }

    full = full_window(equity)
    bh_full = full_window(bh_equity.loc[equity.index[0]:])
    print(f'{"Full":<8} {full["total_return"] * 100:>+7.1f}%          '
          f'{full["cagr"] * 100:>+6.1f}%          {full["max_dd"] * 100:>+6.1f}%    '
          f'{full["sortino"]:>+6.2f}     {bh_full["total_return"] * 100:+.1f}%')

    WIN = {"COVID 2020": ("2020-02-15", "2020-04-30"), "2022 bear": ("2022-01-01", "2022-12-31")}
    print("\nCrash-window check:")
    strat_ret = equity.pct_change().fillna(0.0)
    for wname, (lo, hi) in WIN.items():
        rs = strat_ret.loc[lo:hi]
        if len(rs) < 5:
            print(f"  {wname:<12} insufficient data")
            continue
        es = (1 + rs).cumprod()
        dd = ((es / es.cummax() - 1).min()) * 100
        print(f"  {wname:<12} return {(es.iloc[-1]-1)*100:6.1f}%   maxDD {dd:6.1f}%")

    print("\nMonthly rebalance log (last 12):")
    for d, risk_on, selected, allocation in log_rows[-12:]:
        state = "ON " if risk_on else "OFF"
        print(f'  {d.date()}  {state}  {",".join(selected) or "-":<24} {alloc_str(allocation)}')


if __name__ == "__main__":
    main()
