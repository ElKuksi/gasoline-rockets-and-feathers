"""Unit-root and cointegration tests for the crude/retail price-level relationship.

Stage 3 works in first differences to avoid spurious regression: price levels both
trend upward over the sample, so regressing one on the other inflates R^2 even if the
series are unrelated (Granger & Newbold, 1974). Two trending series can still be
cointegrated, though -- individually non-stationary but never drifting far apart. If
retail and crude are cointegrated there's a real long-run relationship worth modelling
directly (the ECM); if not, that model would be spurious and Stage 4 can't proceed as
planned. `adf_test` checks that each series is individually I(1); `engle_granger` tests
whether they share a common stochastic trend.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from src.point_in_time import align_retail_to_upstream


def adf_test(series: pd.Series, name: str) -> dict:
    """Augmented Dickey-Fuller test for a unit root in `series`, at levels (not
    differenced).

    H0: unit root (I(1), random walk). H1: stationary around a constant. Uses
    statsmodels' defaults, `regression="c"` and `autolag="AIC"`. Constant-only (no
    trend term) matches `coint()`'s own default below -- crude and retail are
    conventionally treated as a random walk with drift rather than trend-stationary, so
    both tests should assume the same thing about the deterministic part.

    We expect this to fail to reject H0 on both crude and retail -- that's the
    precondition for `engle_granger` below. A clean rejection on either series would
    mean it's already stationary in levels, and cointegration is the wrong framework
    for it.

    Parameters
    ----------
    series : level series to test, e.g. `retail.set_index("date")["value"]`. NaNs
        aren't dropped here; a gap in a level series is a data problem upstream of
        this test.
    name : label carried through into the returned dict.

    Returns
    -------
    dict with `name`, `stat`, `p_value`, `crit_1pct`/`crit_5pct`/`crit_10pct`, and
    `likely_unit_root` (`p_value > 0.05`).
    """
    stat, p_value, _, _, crit_values, _ = adfuller(series, regression="c", autolag="AIC")

    return {
        "name": name,
        "stat": stat,
        "p_value": p_value,
        "crit_1pct": crit_values["1%"],
        "crit_5pct": crit_values["5%"],
        "crit_10pct": crit_values["10%"],
        "likely_unit_root": p_value > 0.05,
    }


def engle_granger(retail: pd.DataFrame, upstream: pd.DataFrame, mode: str = "weekly") -> dict:
    """Engle-Granger cointegration test between retail and upstream price levels.

    Aligns via `align_retail_to_upstream` (`src.point_in_time`), same as every other
    retail/upstream join in this codebase (ADR-0002). Works on levels, not
    differences -- that's the whole point of a long-run test.

    Uses `coint()` rather than fitting OLS on levels and running plain ADF on the
    residual. The residual from a cointegrating regression is estimated from the same
    two series being tested, which shifts the ADF statistic's null distribution --
    Engle & Granger (1987) derived the correct asymptotic critical values for this
    case, and they're more negative than plain ADF's. Using the wrong ones biases the
    test toward finding cointegration that isn't there.

    Parameters
    ----------
    retail, upstream : DataFrames with `date` and `value`, as `align_retail_to_upstream`
        expects -- levels, not the diffed inputs `asymmetry.build_design_matrix` takes.
    mode : passed through to `align_retail_to_upstream` (`"weekly"` or `"daily_pit"`).

    Returns
    -------
    dict with `coint_stat`, `coint_pvalue` (from `coint()`, H0 = no cointegration),
    `crit_1pct`/`crit_5pct`/`crit_10pct` (`coint()`'s own critical values), `gamma0`,
    `gamma1` (intercept and long-run pass-through slope from
    `retail = gamma0 + gamma1*upstream + u_t`), and `residuals`.
    """
    merged = align_retail_to_upstream(retail, upstream, mode=mode)
    retail_level = merged["value_retail"]
    upstream_level = merged["value_upstream"]

    coint_stat, coint_pvalue, crit_values = coint(retail_level, upstream_level)

    ols_res = sm.OLS(retail_level, sm.add_constant(upstream_level)).fit()
    gamma0, gamma1 = ols_res.params.iloc[0], ols_res.params.iloc[1]
    residuals = pd.Series(ols_res.resid.to_numpy(), index=pd.Index(merged["date"], name="date"), name="u_t")

    return {
        "coint_stat": coint_stat,
        "coint_pvalue": coint_pvalue,
        "crit_1pct": crit_values[0],
        "crit_5pct": crit_values[1],
        "crit_10pct": crit_values[2],
        "gamma0": gamma0,
        "gamma1": gamma1,
        "residuals": residuals,
    }
