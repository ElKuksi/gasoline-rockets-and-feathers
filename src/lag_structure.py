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


def up_down_response(retail: pd.DataFrame, upstream: pd.DataFrame, horizon: int = 8, mode: str = "weekly") -> pd.DataFrame:
    """Mean retail response over the weeks following an up-week vs. a down-week upstream.

    Aligns `retail` to `upstream` via `align_retail_to_upstream` (the same single
    permitted join, per ADR-0002) and takes first differences of both, exactly as
    `cross_correlation` does. Every week is then classified by the *sign* of
    `delta_upstream` alone — an "up-week" (`delta_upstream > 0`) or a "down-week"
    (`delta_upstream < 0`); a week with `delta_upstream` exactly 0 falls into neither
    group. For each horizon h in 0..horizon, this computes the mean, count, and standard
    error of `delta_retail_{t+h}` separately across the up-week group and the down-week
    group — i.e. "h weeks after an up-week, how much did retail typically move; h weeks
    after a down-week, how much did it typically move."

    THIS IS DESCRIPTIVE ONLY — it does not, and cannot, establish asymmetry on its own.
    It compares the two groups purely by the *direction* of the upstream move, never its
    *size*. If up-weeks in this sample happen to be systematically larger or smaller
    swings than down-weeks (a real possibility, not accounted for here at all), that
    size difference alone could produce a gap between `mean_after_up` and
    `mean_after_down` even if retail responded with perfect symmetry to every dollar of
    crude movement regardless of direction. Establishing genuine "rockets and feathers"
    asymmetry requires controlling for move size — the asymmetric distributed-lag model
    in Stage 3, with separate coefficients on positive and negative `delta_upstream`, is
    what actually does that. This function is a first look, not a test, and its output
    should never be read as evidence of asymmetry by itself.

    It's also worth checking, before trusting even the descriptive picture, whether
    `n_up` and `n_down` are roughly balanced — if upstream rose in (say) 60% of weeks in
    this sample, the up/down comparison is already working from a lopsided base rate,
    and that skew belongs in any interpretation of the numbers below.

    Parameters
    ----------
    retail, upstream : DataFrames with `date` (datetime64) and `value` (float64), exactly
        as `align_retail_to_upstream` expects.
    horizon : largest number of weeks ahead to compute, inclusive. Default 8.
    mode : passed straight through to `align_retail_to_upstream` — `"weekly"` or
        `"daily_pit"`.

    Returns
    -------
    DataFrame with one row per horizon 0..horizon and columns:
    - `horizon` — h, in weeks after the up/down week.
    - `mean_after_up`, `mean_after_down` — mean `delta_retail_{t+h}` within each group.
    - `n_up`, `n_down` — how many valid observations each mean rests on. Shrinks as h
      grows, the same way `cross_correlation`'s `n_obs` does: shifting `delta_retail` by
      h rows loses its last h observations.
    - `se_up`, `se_down` — standard error of each mean (sample standard deviation,
      `ddof=1`, over sqrt(n)). `NaN` wherever the corresponding n is 0.
    """
    merged = align_retail_to_upstream(retail, upstream, mode=mode)

    delta_retail = merged["value_retail"].diff()
    delta_upstream = merged["value_upstream"].diff()

    up_mask = delta_upstream > 0
    down_mask = delta_upstream < 0

    rows = []
    for h in range(horizon + 1):
        future_delta_retail = delta_retail.shift(-h)

        up_values = future_delta_retail[up_mask & future_delta_retail.notna()]
        down_values = future_delta_retail[down_mask & future_delta_retail.notna()]
        n_up = len(up_values)
        n_down = len(down_values)

        rows.append(
            {
                "horizon": h,
                "mean_after_up": up_values.mean(),
                "mean_after_down": down_values.mean(),
                "n_up": n_up,
                "n_down": n_down,
                "se_up": up_values.std(ddof=1) / n_up**0.5 if n_up > 0 else float("nan"),
                "se_down": down_values.std(ddof=1) / n_down**0.5 if n_down > 0 else float("nan"),
            }
        )

    return pd.DataFrame(rows)
