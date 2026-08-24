# LETF strategy research — findings

Backtests run through the LETF Lab engine (real ETF price history, adjusted
close). Rotation metrics are **net-of-tax** (Brazilian DARF); static-hold
metrics are **gross** (the app's tax layer only applies to rotation strategies —
but holds have far lower turnover, so real tax drag is small).

## Windows & the history limit

We are **not** capped at 15 years by a setting — the limit is when the ETFs were
created (the engine uses real prices, no synthetic reconstruction):

| Ticker | Inception | | Ticker | Inception |
|---|---|---|---|---|
| GLD | 2004-11 | | UPRO (3× SPY) | 2009-06 |
| SSO (2× SPY) | 2006-06 | | **ZROZ** | **2009-11** ← binds the hold |
| BIL | 2007-05 | | TQQQ (3× QQQ) | 2010-02 |

So the SSO/GLD/ZROZ hold can go back to ~Nov 2009 (~16.8y). Testing 2008/2000
would require **simulating** the leveraged ETFs from their underlying index
(daily-reset leverage minus financing + expense) — not yet built; would need
calibration against the real-ETF overlap before trusting it.

## Rotation strategies (risk-on/off), 15y, net-of-tax, BIL risk-off

Trigger types: **k≥2** = vote-of-2 across SMA250/SMA100/Vol21d<40%/AR(1)>0;
**200SMA** = symmetric ±3.5% band around the 200-day SMA.

| Strategy | CAGR | Max DD | Sortino | Deploy (10y) |
|---|--:|--:|--:|---|
| SPY 2x · k≥2 | 21.6% | −36.6% | 1.19 | **64 PROMISING** |
| SPY 3x · k≥2 | 30.1% | −50.1% | 1.19 | 55 MARGINAL |
| QQQ 2x · k≥2 | 26.1% | −56.5% | 1.18 | 58 MARGINAL |
| QQQ 3x · k≥2 | 35.6% | −73.3% | 1.19 | 55 MARGINAL |
| SPY 2x · 200SMA | 15.5% | −40.6% | 0.97 | 42 MARGINAL |
| SPY 3x · 200SMA | 21.1% | −55.9% | 0.97 | 44 MARGINAL |
| QQQ 2x · 200SMA | 19.6% | −49.7% | 0.99 | 52 MARGINAL |
| QQQ 3x · 200SMA | 26.0% | −65.9% | 1.00 | 52 MARGINAL |
| 200SMA→TQQQ (SPY trend times 3× QQQ) | 30.9% | −55.6% | 1.12 | 55 MARGINAL |

Notes:
- **k≥2 beats the 200SMA buffer** on every cell (return and Sortino).
- **2x is the risk-adjusted sweet spot**; 3x adds CAGR but drawdowns blow out
  with little Sortino gain.
- **Risk-off asset matters:** switching ZROZ→BIL added ~1pp CAGR, ~10pp shallower
  drawdown, and ~18 deploy points (ZROZ long bonds cratered in 2022).
- Deploy score's 30-pt **Sortino-edge** criterion is the ceiling: leverage adds
  return and downside vol together, so risk-adjusted edge vs the plain index is
  thin (best case +0.07). That's why nothing reaches STRONG/WINNER.

## Static hold: SSO/GLD/ZROZ, weight × rebalance (gross, ~16.8y)

| Weights | Rebalance | CAGR | Max DD | Sortino |
|---|---|--:|--:|--:|
| 60/20/20 | annual | 17.7% | −38.1% | 1.35 |
| 60/20/20 | **quarterly** | **18.9%** | −38.9% | 1.41 |
| 60/20/20 | monthly | 18.3% | −38.6% | 1.35 |
| 50/25/25 | annual | 15.9% | −36.0% | 1.40 |
| 50/25/25 | **quarterly** | 17.2% | −37.1% | **1.47** |
| 50/25/25 | monthly | 16.5% | −36.7% | 1.41 |

Notes:
- **Quarterly rebalancing won everywhere** (monthly over-trades; annual lags).
- **50/25/25 quarterly = best risk-adjusted of everything tested (Sortino 1.47).**
  **60/20/20 quarterly = best return among holds (18.9%).**
- The diversified holds beat every rotation strategy on **Sortino**, at a
  fraction of the complexity — but always fully invested (no crash exit), so the
  −38% drawdown is optimistic for a window with no 2008.

## Standing caveats

- Test window has **no 2008-scale crash** — drawdowns are optimistic.
- Holds are gross; rotations are net-of-tax (not perfectly apples-to-apples).
- This is analysis of backtests, **not financial advice**.

## Synthetic-LETF crash backtest (2000 & 2008)

Leveraged ETFs reconstructed from total-return indices (^SP500TR, ^NDX) via the
daily-reset model: `daily = L*idx_ret - (L-1)*(short_rate+0.5% spread)/252 - expense/252`.
Calibrated against real ETFs over their overlap: daily-return correlation 0.995-0.999
(drawdowns trustworthy); CAGR ~0.5-1.9pp/yr optimistic (so real results slightly worse).
Scripts: `synth_calibrate.py` (validation), `synth_scenarios.py` (crashes).

Max drawdowns:

| Instrument | Dot-com 00-02 | GFC 07-09 | Full 2000-26 CAGR / maxDD |
|---|--:|--:|--:|
| S&P 1x | -47% | -55% | 8.4% / -55% |
| SSO 2x | -79% | -84% | 9.5% / -88% |
| UPRO 3x | -92% | -96% | 7.4% / -98% |
| QLD 2x | -99% | -83% | 6.3% / -99% |
| TQQQ 3x | -100% | -95% | -2.6% / -100% |
| k>=2 SPY->SSO | -61% | -40% | 12.2% / -71% |
| 200SMA QQQ->TQQQ | -92% | -57% | 12.5% / -93% |
| 60/20/20 hold* | (n/a, GLD from 2004) | -60% | 14.2% / -60% |

*Hold 2004-2026, ZROZ proxied by TLT.

Key conclusions:
- Real-crash drawdowns are ~2x the bull-only (2009-2026) window's.
- 3x (esp. TQQQ) is catastrophic in a tech crash; buy-hold TQQQ from 2000 lost money over 25y.
- Trend signals protect in orderly crashes (2008) but fail in choppy bears (2000).
- Leverage doesn't reliably beat 1x once crashes are included; only timed 2x (k>=2 SPY->SSO) did.
