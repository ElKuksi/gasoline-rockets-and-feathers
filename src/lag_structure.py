"""Cross-correlation of retail and upstream price *changes* across a range of lags — the
diagnostic that informs how many lags (K) Stage 3's distributed-lag model should carry,
by showing which lag actually holds the strongest week-to-week relationship.
"""

from __future__ import annotations

import pandas as pd

from src.point_in_time import align_retail_to_upstream


def cross_correlation(retail: pd.DataFrame, upstream: pd.DataFrame, max_lag: int = 12, mode: str = "weekly") -> pd.DataFrame:
    """Cross-correlation of retail and upstream price changes, at lags 0..max_lag.

    Aligns `retail` to `upstream` via `align_retail_to_upstream` (`src.point_in_time`) —
    the only join this function performs, and per ADR-0002 the only join a retail series
    is ever allowed to have with an upstream one anywhere in this codebase. Then takes
    first differences of both aligned series and, for each k in 0..max_lag, correlates
    `delta_retail_t` against `delta_upstream_{t-k}`.

    Why first differences, not levels
    ----------------------------------
    Retail and crude price *levels* are both highly persistent — each trends and drifts
    for reasons that have nothing to do with week-to-week pass-through (broad energy-
    market cycles, inflation, seasonal demand). Two persistent series correlated in
    levels produce a high correlation driven by that shared drift, not by any real
    relationship between a given week's changes — the classic spurious-regression
    problem (Granger & Newbold, 1974). Differencing removes the shared trend and asks
    the question a lag-structure study actually needs answered: does a *change* in
    upstream price around lag k coincide with a *change* in retail price this week?
    That's the relationship Stage 3's distributed-lag model is built on, so it's the
    relationship this diagnostic has to measure — not a levels correlation that would
    look strong for the wrong reason.

    Parameters
    ----------
    retail, upstream : DataFrames with `date` (datetime64) and `value` (float64), exactly
        as `align_retail_to_upstream` expects.
    max_lag : largest lag (in weeks) to compute, inclusive. Default 12.
    mode : passed straight through to `align_retail_to_upstream` — `"weekly"` or
        `"daily_pit"`.

    Returns
    -------
    DataFrame with one row per lag 0..max_lag and columns:
    - `lag` — the lag k, in weeks.
    - `correlation` — corr(delta_retail_t, delta_upstream_{t-k}), pairwise NaN-dropped.
    - `n_obs` — how many (delta_retail_t, delta_upstream_{t-k}) pairs were actually valid
      at that lag. This shrinks as k grows: shifting delta_upstream by k rows introduces
      k additional leading NaNs on top of the one first-differencing already loses, so a
      later lag's correlation rests on a smaller sample than an earlier one. Reported
      explicitly, alongside the correlation, rather than left implicit — a correlation
      computed on a much smaller n is weaker evidence, and a lag-structure choice (K)
      should never be made on numbers whose sample sizes silently vary without saying so.
    """
    merged = align_retail_to_upstream(retail, upstream, mode=mode)

    delta_retail = merged["value_retail"].diff()
    delta_upstream = merged["value_upstream"].diff()

    rows = []
    for k in range(max_lag + 1):
        shifted_upstream = delta_upstream.shift(k)
        n_obs = int((delta_retail.notna() & shifted_upstream.notna()).sum())
        correlation = delta_retail.corr(shifted_upstream)
        rows.append({"lag": k, "correlation": correlation, "n_obs": n_obs})

    return pd.DataFrame(rows)
