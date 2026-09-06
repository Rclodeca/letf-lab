"""Core selection logic for the monthly "3-of-5" LETF rotation strategy.

Shared by `research/momentum_rotation_3of5.py` (manual/backtest-lite runs)
and `notify/daily_signals.py` (Telegram notification) so both compute the
identical numbers. See `research/momentum_rotation_3of5.py`'s module
docstring for the full algorithm write-up.
"""
from itertools import combinations

import pandas as pd

UNIVERSE = [
    "BWX", "EWJ", "GLDM", "IEF", "IEMG", "PDBC", "QQQ", "SCZ",
    "SPY", "TIP", "TLT", "VGK", "VNQ", "VNQI",
]
CASH = "BIL"

# 2x substitutes used at allocation time only — scoring/ranking always runs
# on the plain unlevered series. Anything not listed here (incl. BIL) trades
# as itself at 1x.
LEVERAGE_MAP = {
    "GLDM": ("UGL", 2),
    "IEF": ("UST", 2),
    "IEMG": ("EET", 2),
    "QQQ": ("QLD", 2),
    "SPY": ("SSO", 2),
    "TLT": ("UBT", 2),
    "VNQ": ("URE", 2),
}

MONTHS_BACK = (1, 3, 6, 12)  # calendar months, not trading days -- see score()
CORR_LOOKBACK_DAYS = 252
TOP_N_CANDIDATES = 5
PORTFOLIO_SLOTS = 3

# Minimum calendar-day span of history needed to compute every leg in
# MONTHS_BACK (12 months + a buffer for the lookup below to find a trading
# day on/before the 12-month-ago target even across long holiday gaps).
MIN_HISTORY_DAYS = 366


def has_enough_history(closes) -> bool:
    """True if `closes` (a Series or DataFrame with a DatetimeIndex) spans
    enough calendar time to compute every MONTHS_BACK leg."""
    return (closes.index[-1] - closes.index[0]).days >= MIN_HISTORY_DAYS


def price_months_ago(prices: pd.Series, months: int) -> float:
    """Price on the last trading day on/before `months` calendar months
    before the last date in `prices`. Reverse engineered from a third-party
    TAA backtesting site's numbers (see conversation history) -- it uses
    actual calendar-month lookback, not a fixed trading-day count. Shared
    with ai_swing.scoring.haa, which builds its own differently-aggregated
    score() on top of the same lookback mechanics -- see that module's
    docstring for why the two strategies' formulas deliberately diverge."""
    target = prices.index[-1] - pd.DateOffset(months=months)
    idx = prices.index[prices.index <= target]
    return prices.loc[idx[-1]]


def score(prices: pd.Series) -> float:
    """avg(1m, 3m, 6m, 12m trailing return) as of the last row of `prices`,
    each leg measured over actual calendar months (not a fixed trading-day
    count) -- confirmed by matching a third-party TAA site's published
    numbers to within rounding across two independent checkpoint dates."""
    last = prices.iloc[-1]
    return sum(last / price_months_ago(prices, m) - 1 for m in MONTHS_BACK) / len(MONTHS_BACK)


def avg_pairwise_corr(returns: pd.DataFrame, tickers) -> float:
    sub = returns[list(tickers)]
    corr = sub.corr()
    pairs = list(combinations(tickers, 2))
    return sum(corr.loc[a, b] for a, b in pairs) / len(pairs)


def leverage_for(ticker: str):
    """Return (tradable_ticker, leverage) for `ticker`."""
    return LEVERAGE_MAP.get(ticker, (ticker, 1))


def resolve_leveraged_allocation(weights: dict) -> dict:
    """Given raw equal-weight-per-slot `weights` (ticker -> weight, summing to
    1.0), rescale so leveraged substitutes carry proportionally less dollar
    weight while keeping net unlevered exposure equal across slots, and the
    freed-up capital fully deployed (dollars sum back to 1.0)."""
    n_slots = {t: w * PORTFOLIO_SLOTS for t, w in weights.items()}
    leverages = {t: leverage_for(t)[1] for t in weights}
    # w_per_slot solves sum_i(n_slots_i * w_per_slot / leverage_i) == 1.0
    w_per_slot = 1 / sum(n_slots[t] / leverages[t] for t in weights)

    allocation = {}
    for t in weights:
        tradable, lev = leverage_for(t)
        dollars = n_slots[t] * w_per_slot / lev
        allocation[tradable] = allocation.get(tradable, 0.0) + dollars
    return allocation


def compute_allocation(closes: pd.DataFrame) -> dict:
    """Run the full ranking -> filter -> select -> weight -> allocate pipeline
    as of the LAST row of `closes` (caller controls the cutoff, so this same
    function serves both "today" and "as of some past month-end"). `closes`
    must have one column per ticker in UNIVERSE + [CASH]."""
    if not has_enough_history(closes):
        span = (closes.index[-1] - closes.index[0]).days
        raise ValueError(f"need >= {MIN_HISTORY_DAYS} days of history, got {span}")

    as_of = closes.index.max().date()
    scores = {t: float(score(closes[t])) for t in closes.columns}
    cash_score = scores[CASH]

    ranking = sorted(({"ticker": t, "score": round(scores[t], 4)} for t in UNIVERSE),
                      key=lambda r: r["score"], reverse=True)

    survivors = [r["ticker"] for r in ranking if scores[r["ticker"]] > cash_score]
    abs_survivors = len(survivors)
    candidates = survivors[:TOP_N_CANDIDATES]

    returns = closes.tail(CORR_LOOKBACK_DAYS + 1).pct_change().dropna()

    avg_corr = None
    if len(candidates) >= PORTFOLIO_SLOTS:
        best_combo, best_corr = None, None
        for combo in combinations(candidates, PORTFOLIO_SLOTS):
            c = avg_pairwise_corr(returns, combo)
            if best_corr is None or c < best_corr:
                best_combo, best_corr = combo, c
        selected, avg_corr = list(best_combo), best_corr
    else:
        selected = candidates
        if len(selected) == 2:
            avg_corr = avg_pairwise_corr(returns, selected)

    weights = {}
    for t in selected:
        weights[t] = weights.get(t, 0.0) + 1 / PORTFOLIO_SLOTS
    n_empty_slots = PORTFOLIO_SLOTS - len(selected)
    if n_empty_slots:
        weights[CASH] = weights.get(CASH, 0.0) + n_empty_slots / PORTFOLIO_SLOTS

    allocation = resolve_leveraged_allocation(weights)

    return {
        "as_of": as_of, "ranking": ranking, "cash_score": cash_score,
        "abs_survivors": abs_survivors, "candidates": candidates,
        "selected": selected, "avg_corr": avg_corr, "weights": weights,
        "allocation": allocation,
    }


def alloc_str(allocation: dict) -> str:
    return ", ".join(f"{t} {round(w * 100)}%" for t, w in sorted(allocation.items(), key=lambda kv: -kv[1]))


def last_completed_rebalance_date(closes: pd.DataFrame):
    """Last trading day strictly before the current calendar month — i.e. the
    month-end close that decided the allocation still active today. None if
    `closes` doesn't yet span a prior month."""
    current_period = closes.index.max().to_period("M")
    prior = closes.index[closes.index.to_period("M") < current_period]
    return prior.max() if len(prior) else None
