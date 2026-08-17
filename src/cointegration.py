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

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

from src.asymmetry import DEFAULT_HAC_MAXLAGS, _restriction_vector, build_design_matrix
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


def build_ecm_design_matrix(
    retail: pd.DataFrame, upstream: pd.DataFrame, K: int, gamma0: float, gamma1: float, mode: str = "weekly"
) -> pd.DataFrame:
    """Design matrix for the asymmetric error-correction model: `build_design_matrix`'s
    short-run `d_up_lag*`/`d_down_lag*` columns, plus the lagged equilibrium error.

    `gamma0`/`gamma1` (from `engle_granger`) define the long-run relationship
    `retail_level = gamma0 + gamma1 * upstream_level + u`. We compute `u` at each aligned
    date, then use `u_{t-1}` one period before since the error-correction term has
    to be predetermined, not contemporaneous with the `d_retail_t` it's predicting.
    `u_pos_lag1`/`u_neg_lag1` split it into its positive and negative parts the same way
    `d_up`/`d_down` split `delta_upstream`: not a sample split, two views of one series
    that sum back to `u_{t-1}`, so the fitted coefficients on each (`lambda+`/`lambda-`)
    are separate speed-of-adjustment estimates for "priced too high" vs. "priced too low".

    Parameters
    ----------
    retail, upstream : as `build_design_matrix` expects.
    K : largest short-run lag, passed through to `build_design_matrix`.
    gamma0, gamma1 : long-run intercept and slope.
    mode : passed through to `build_design_matrix`/`align_retail_to_upstream`.

    Returns
    -------
    `build_design_matrix`'s output with `u_pos_lag1`, `u_neg_lag1` added.
    """
    design = build_design_matrix(retail, upstream, K, mode=mode)

    merged = align_retail_to_upstream(retail, upstream, mode=mode)
    u_lag1 = (merged["value_retail"] - gamma0 - gamma1 * merged["value_upstream"]).shift(1)
    levels = pd.DataFrame({"date": merged["date"], "u_pos_lag1": u_lag1.clip(lower=0), "u_neg_lag1": u_lag1.clip(upper=0)})

    design = design.merge(levels, on="date", how="left")

    assert design[["u_pos_lag1", "u_neg_lag1"]].notna().all().all(), (
        "u_pos_lag1/u_neg_lag1 has NaN after merge -- build_design_matrix's own lag-based "
        "row-dropping should already exclude the one leading NaN row the u_{t-1} shift produces"
    )

    return design


def fit_ecm(design: pd.DataFrame, maxlags: int = DEFAULT_HAC_MAXLAGS):
    """Fit `d_retail` on the short-run lags plus `u_pos_lag1`/`u_neg_lag1`, with HAC
    (Newey-West) standard errors same pattern as `fit_distributed_lag`, same
    reasoning for `maxlags` defaulting to `DEFAULT_HAC_MAXLAGS`.

    Parameters
    ----------
    design : a `build_ecm_design_matrix()` output.
    maxlags : HAC lag window.

    Returns
    -------
    The fitted statsmodels results object, unwrapped.
    """
    regressor_columns = [col for col in design.columns if col not in ("date", "d_retail")]
    y = design["d_retail"]
    X = sm.add_constant(design[regressor_columns])

    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def test_adjustment_asymmetry(res) -> dict:
    """Test whether the two speed-of-adjustment coefficients differ: lambda+
    (`u_pos_lag1`) vs lambda- (`u_neg_lag1`). Reuses `asymmetry._restriction_vector` and
    `t_test`, same technique `test_asymmetry` uses for beta+ vs beta-, same reason 
    the two coefficients are correlated, so their SEs can't just be added.

    Parameters
    ----------
    res : a fitted `fit_ecm` results object.

    Returns
    -------
    dict with `estimate` (lambda+ minus lambda-), `se`, `ci_lo`, `ci_hi`, `p_value`.
    """
    restriction = _restriction_vector(res, positive_names={"u_pos_lag1"}, negative_names={"u_neg_lag1"})
    t = res.t_test(restriction)
    ci = np.asarray(t.conf_int())

    return {
        "estimate": float(np.asarray(t.effect).item()),
        "se": float(np.asarray(t.sd).item()),
        "ci_lo": float(ci[0, 0]),
        "ci_hi": float(ci[0, 1]),
        "p_value": float(np.asarray(t.pvalue).item()),
    }
