"""Core selection logic for the Hybrid Asset Allocation (HAA) strategy.

Uses the same four legs (1/3/6/12 calendar months) and lookback mechanics
as the 3-of-5 rotation strategy, but a DIFFERENT aggregation: this module's
own score() SUMS the four legs, where rotation_3of5.score() AVERAGES them.
Both conventions were reverse engineered independently against a
third-party TAA site's published numbers for each strategy and confirmed
to a decisive margin (see conversation history) -- the site genuinely uses
two different momentum formulas for these two strategies, not the same one
inconsistently. Referred to here as "13612 (unweighted sum)" to distinguish
from Keller & Keuning's original WEIGHTED 13612W formula (12/4/2/1 on
1/3/6/12mo), which this does NOT use.

Algorithm, evaluated at each month-end:
  1. Canary: is TIP's score positive?
  2. If yes -> risk-on. Absolute-momentum filter: drop any offensive-universe
     asset whose own score doesn't beat BIL's score -- BIL is always the
     fixed hurdle here (same role BIL plays as the cash benchmark in the
     3-of-5 rotation strategy), regardless of which of IEF/BIL separately
     wins the defensive-pair comparison below. IEF, being dual-purpose, can
     independently survive this filter as an offensive pick on its own
     momentum. Take the top 4 of whatever survives, equal-weight 25% each,
     traded via each pick's designated substitute/leveraged fund
     (SUBSTITUTE) at face value -- unlike the 3-of-5 rotation strategy,
     there is NO leverage-based reweighting; a 25% slot filled by a 2x fund
     is still a flat 25% dollar allocation. Any slot left unfilled (fewer
     than 4 survivors) backfills to the best-defensive asset's OWN
     substitute (e.g. IEF backfills as UST, not plain IEF) -- backfill is
     still part of the risk-on allocation mechanics, so it gets levered the
     same as a direct pick. If IEF is both a direct survivor AND the
     backfill destination, the two contributions add into the same
     substitute ticker.
  3. If no -> risk-off. Allocate 100% to whichever of the defensive pair
     (IEF, BIL) has the higher score, held UNLEVERED (this step's rule text
     never mentions substitution, unlike step 2's).

IEF is dual-purpose: it's both a candidate in the offensive ranking (step 2)
and one of the two defensive candidates (steps 2's backfill and step 3) --
TLT is offensive-only, never a defensive destination.
"""
import pandas as pd

from .rotation_3of5 import (
    MIN_HISTORY_DAYS,
    MONTHS_BACK,
    alloc_str,
    has_enough_history,
    last_completed_rebalance_date,
    price_months_ago,
)

CANARY = "TIP"
CASH = "BIL"  # fixed absolute-momentum hurdle for the offensive filter
DEFENSIVE_CANDIDATES = ["IEF", "BIL"]
OFFENSIVE_UNIVERSE = ["VWO", "VEA", "DBC", "QQQ", "SPY", "TLT", "VNQ", "IEF", "IWM"]
TOP_N = 4

# Tradable substitute for each offensive-universe ticker. Most are 2x
# leveraged; DBC -> PDBC is a like-for-like substitution (both 1x optimum-
# yield commodities), not leverage.
SUBSTITUTE = {
    "VWO": "EET",
    "VEA": "EFO",
    "DBC": "PDBC",
    "QQQ": "QLD",
    "SPY": "SSO",
    "TLT": "UBT",
    "VNQ": "URE",
    "IEF": "UST",
    "IWM": "UWM",
}


def score(prices: pd.Series) -> float:
    """sum(1m, 3m, 6m, 12m trailing return) as of the last row of `prices`,
    each leg measured over actual calendar months -- deliberately a SUM,
    not rotation_3of5.score()'s average; see module docstring."""
    last = prices.iloc[-1]
    return sum(last / price_months_ago(prices, m) - 1 for m in MONTHS_BACK)


def compute_allocation(closes: pd.DataFrame) -> dict:
    """Run the canary -> rank -> select -> allocate pipeline as of the LAST
    row of `closes`. `closes` must have one column per ticker in
    OFFENSIVE_UNIVERSE + [CANARY] + DEFENSIVE_CANDIDATES (IEF is shared
    between the offensive universe and the defensive pair, so it only needs
    one column)."""
    if not has_enough_history(closes):
        span = (closes.index[-1] - closes.index[0]).days
        raise ValueError(f"need >= {MIN_HISTORY_DAYS} days of history, got {span}")

    as_of = closes.index.max().date()

    canary_score = float(score(closes[CANARY]))
    defensive_scores = {t: float(score(closes[t])) for t in DEFENSIVE_CANDIDATES}
    best_defensive = max(defensive_scores, key=defensive_scores.get)
    cash_score = defensive_scores[CASH] if CASH in defensive_scores else float(score(closes[CASH]))

    offensive_scores = {t: float(score(closes[t])) for t in OFFENSIVE_UNIVERSE}
    ranking = sorted(
        ({"ticker": t, "score": round(offensive_scores[t], 4)} for t in OFFENSIVE_UNIVERSE),
        key=lambda r: r["score"], reverse=True,
    )

    risk_on = canary_score > 0
    abs_survivors = sum(1 for t in OFFENSIVE_UNIVERSE if offensive_scores[t] > cash_score)

    if risk_on:
        survivors = [r["ticker"] for r in ranking if offensive_scores[r["ticker"]] > cash_score]
        selected = survivors[:TOP_N]
        n_empty_slots = TOP_N - len(selected)

        weights = {t: 1 / TOP_N for t in selected}
        allocation = {SUBSTITUTE[t]: w for t, w in weights.items()}
        if n_empty_slots:
            backfill_ticker = SUBSTITUTE.get(best_defensive, best_defensive)
            allocation[backfill_ticker] = allocation.get(backfill_ticker, 0.0) + n_empty_slots / TOP_N
    else:
        selected = [best_defensive]
        weights = {best_defensive: 1.0}
        allocation = {best_defensive: 1.0}

    return {
        "as_of": as_of,
        "canary_score": round(canary_score, 4),
        "defensive_scores": {t: round(v, 4) for t, v in defensive_scores.items()},
        "best_defensive": best_defensive,
        "offensive_ranking": ranking,
        "risk_on": risk_on,
        "abs_survivors": abs_survivors,
        "selected": selected,
        "weights": weights,
        "allocation": allocation,
    }
