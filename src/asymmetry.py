"""Design matrix for the asymmetric distributed-lag model: decomposes upstream
price *changes* into positive and negative parts, each carried across lags 0..K, so a
regression on the result can estimate a different per-dollar retail response to a crude
increase than to a crude decrease — the actual test of "rockets and feathers."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.point_in_time import align_retail_to_upstream


def build_design_matrix(retail: pd.DataFrame, upstream: pd.DataFrame, K: int, mode: str = "weekly") -> pd.DataFrame:
    """Build the regressor matrix for an asymmetric distributed-lag model.

    Aligns `retail` to `upstream` via `align_retail_to_upstream` (`src.point_in_time`) —
    the only join this function performs, and per ADR-0002 the only join a retail series
    is ever allowed to have with an upstream one anywhere in this codebase. Takes first
    differences of both aligned series, splits `delta_upstream` into its positive and
    negative parts, and builds lags 0..K of each part.

    Why this is a decomposition of one variable, not a split of the sample:
    `src.lag_structure.up_down_response` answers a *descriptive* question by splitting
    the SAMPLE: every week is routed into one of two disjoint groups (an "up-week" or a
    "down-week"), and each group's mean is computed separately. That throws away two
    things — the magnitude of each week's move (only its sign is used), and half the
    sample's worth of statistical power for whichever group a given week didn't land in.

    This function does something different on purpose: it decomposes `delta_upstream`
    itself into two components that sum back to the original,

        d_up   = max(delta_upstream, 0)
        d_down = min(delta_upstream, 0)
        d_up + d_down == delta_upstream   (identically, every row)

    Every single observation contributes a real, non-discarded value to *both* `d_up`
    and `d_down` (one of the two is exactly 0 on any given row, not missing) — so a
    regression of `d_retail` on `[d_up_lag0..K, d_down_lag0..K]` uses the *full* sample
    at every lag, and because `d_up`/`d_down` are continuous (not a 0/1 group label), the
    fitted coefficients express the retail response *per dollar* of an upstream increase
    vs. an upstream decrease — controlling for move size, which is exactly the limitation
    `up_down_response`'s own docstring names as the reason it "cannot on its own
    establish asymmetry." A design built this way can: if the fitted coefficient on
    `d_up_lag0` differs from `d_down_lag0`, that is the actual, size-controlled asymmetry
    test this whole project exists to run. This function only builds the regressors; it
    does not fit anything (see the module docstring).

    This is a property of the *construction* (`clip(lower=0)` / `clip(upper=0)` always
    summing back to the original), not of any particular data — it holds identically
    whichever pair this is called on (crude->wholesale, wholesale->retail, or the primary
    crude->retail) and whatever `K` is given. The assertion below re-checks it on every
    single call for exactly that reason: a construction-level guarantee is still only as
    trustworthy as the code that implements it, on every actual run, not just the runs
    someone happened to test by hand.

    Parameters
    ----------
    retail, upstream : DataFrames with `date` (datetime64) and `value` (float64), exactly
        as `align_retail_to_upstream` expects.
    K : largest lag (in weeks) to build, inclusive. Must be >= 0.
    mode : passed straight through to `align_retail_to_upstream` — `"weekly"` or
        `"daily_pit"`.

    Returns
    -------
    DataFrame with columns `date`, `d_retail`, `d_up_lag0..d_up_lagK`,
    `d_down_lag0..d_down_lagK` (2(K+1) regressor columns total). Rows with any NaN —
    introduced by the initial `.diff()` and by each additional lag's leading NaNs — are
    dropped; how many rows were dropped, out of how many before dropping, is printed.
    """
    if K < 0:
        raise ValueError(f"K must be >= 0, got {K}")

    merged = align_retail_to_upstream(retail, upstream, mode=mode)

    d_retail = merged["value_retail"].diff()
    d_upstream = merged["value_upstream"].diff()

    # The decomposition itself: d_up and d_down are two views of the SAME series, not a
    # partition of rows — clip() does not introduces a NaN that wasn't already in
    # d_upstream, so a row's presence/absence here tracks d_upstream's own, unchanged.
    d_up = d_upstream.clip(lower=0)
    d_down = d_upstream.clip(upper=0)

    columns = {"date": merged["date"], "d_retail": d_retail}
    for lag in range(K + 1):
        columns[f"d_up_lag{lag}"] = d_up.shift(lag)
    for lag in range(K + 1):
        columns[f"d_down_lag{lag}"] = d_down.shift(lag)

    design = pd.DataFrame(columns)

    n_before = len(design)
    design = design.dropna().reset_index(drop=True)
    n_dropped = n_before - len(design)
    print(f"build_design_matrix: dropped {n_dropped} of {n_before} rows to NaN (differencing + {K} lag(s)); {len(design)} rows remain")

    # The decomposition identity: d_up_lag0 + d_down_lag0 must equal the raw (unsplit)
    # upstream change, exactly, on every surviving row — lag 0 applies no shift, and one
    # of d_up/d_down is always precisely 0 by construction, so this is a definitional
    # identity of this function's own arithmetic, not an empirical claim about the data.
    # A bare `assert` is deliberate here — this checks whether this function's own code
    # did what it claims to do, which is exactly what `assert` is for; contrast with
    # `src.point_in_time`'s `raise ValueError` guards, which exist to catch bad DATA a
    # *caller* might pass in, not a bug in the function itself.
    raw_change_by_date = pd.Series(d_upstream.to_numpy(), index=merged["date"])
    raw_change_at_surviving_rows = raw_change_by_date.loc[design["date"]].to_numpy()
    identity_lhs = (design["d_up_lag0"] + design["d_down_lag0"]).to_numpy()
    assert np.allclose(identity_lhs, raw_change_at_surviving_rows, atol=1e-10), (
        "decomposition identity failed: d_up_lag0 + d_down_lag0 != raw upstream change"
    )

    return design


# Newey-West's own rule-of-thumb lag window, floor(4*(T/100)**(2/9)), evaluated at
# T=860 (this project's typical weekly sample size across all three supply-chain links,
# 2010-present): 4*(8.6)**(2/9) = 6.45 -> 6. This is a SAMPLE-SIZE-based constant, not
# tied to any particular model's K — see fit_distributed_lag's docstring for why those
# two things must not be conflated.
DEFAULT_HAC_MAXLAGS = 6


def fit_distributed_lag(design: pd.DataFrame, maxlags: int | None = None):
    """Fit `d_retail` on every lagged up/down regressor in `design`, with a constant,
    using HAC (Newey-West) standard errors.

    Why HAC, not OLS's default standard errors:
    a distributed-lag model's residuals are almost certainly autocorrelated across nearby weeks (this
    week's unexplained retail move is not independent of last week's — persistence in
    price-setting, overlapping demand/supply conditions), and OLS's default standard
    errors assume independence, so without HAC the model would understate its own
    uncertainty and make the up-vs-down coefficient difference look more significant than
    the data actually supports.

    `maxlags` (the HAC correction's own lag window) defaults to `DEFAULT_HAC_MAXLAGS`
    (6) — the standard Newey-West rule-of-thumb, `floor(4*(T/100)**(2/9))`, evaluated at
    this project's typical sample size (T~860 weekly observations). It does NOT default
    to `design`'s own K, and that's deliberate, not an oversight: `maxlags` is about how
    far RESIDUAL autocorrelation plausibly reaches, a property of the data's own
    time-series structure (persistence, overlapping conditions week to week) — while K is
    about how many REGRESSOR lags a particular model happens to include, a modelling
    choice that differs link to link (e.g. this project's crude->wholesale link uses
    K=1, wholesale->retail uses K=6). Tying maxlags to K would silently under-correct
    every short-K model — a K=1 model would get maxlags=1, nowhere near enough lags to
    catch real autocorrelation in weekly price-change residuals, regardless of how few
    regressor lags that particular link needed. Pass an explicit `maxlags` to override
    the default (e.g. to sensitivity-check it, as notebooks/05_asymmetry.ipynb does).

    Parameters
    ----------
    design : a `build_design_matrix()` output — must contain `d_retail` and one or more
        `d_up_lag*`/`d_down_lag*` columns (any other columns, e.g. `date`, are ignored as
        regressors).
    maxlags : the HAC lag window. Defaults to `DEFAULT_HAC_MAXLAGS` if not given.

    Returns
    -------
    The fitted `statsmodels` results object, returned exactly as `.fit()` produces it —
    no wrapping, no partial extraction — so `.summary()`, `.params`, `.bse`, etc. all
    work normally on it.
    """
    if maxlags is None:
        maxlags = DEFAULT_HAC_MAXLAGS

    regressor_columns = [col for col in design.columns if col not in ("date", "d_retail")]
    y = design["d_retail"]
    X = sm.add_constant(design[regressor_columns])

    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def restriction_vector(res, positive_names: set[str] = frozenset(), negative_names: set[str] = frozenset()) -> np.ndarray:
    """Build a `t_test` restriction row-vector for `res`: +1 at each name in
    `positive_names`, -1 at each in `negative_names`, 0 everywhere else (including the
    constant).

    Built by looking up `res.params.index` (the fitted model's own parameter names).
    """
    r = pd.Series(0.0, index=res.params.index)
    r[sorted(positive_names)] = 1.0
    r[sorted(negative_names)] = -1.0
    return r.to_numpy()


def cumulative_passthrough(res, K: int) -> pd.DataFrame:
    """Running-sum pass-through of an up-move and of a down-move, through each horizon
    h = 0..K, with a standard error and 95% interval at each h.

    `cum_up(h) = sum(d_up_lag0..d_up_lag{h})` is the fraction of a crude *increase* that
    has reached the pump by h weeks out (dimensionless: $/gal retail per $/gal crude —
    see `test_asymmetry`'s docstring for the units discussion), and `cum_down(h)` is the
    same for a decrease. Watching how the gap between the two changes as h grows from 0
    to K is the actual "is the difference about speed or about a permanently different
    total" question this whole model exists to answer.

    Standard errors are NOT computed by adding the individual lag coefficients' SEs —
    `Var(sum) = sum(Var) + 2*sum(Cov)` for correlated estimators, and adjacent-lag
    coefficients in a distributed-lag model are correlated (they're estimated from
    overlapping, autocorrelated regressors). Adding SEs directly ignores every Cov term
    and is wrong in general. `res.t_test(r)` with a restriction vector `r` instead uses
    `res`'s full coefficient covariance matrix (`res.cov_params()`), which is the way
    to get the variance of a sum of correlated coefficients, this is why `t_test`, 
    not manual arithmetic on `res.params`/`res.bse`, is used throughout.

    Parameters
    ----------
    res : a fitted `statsmodels` results object — the output of `fit_distributed_lag`,
        with parameter names `d_up_lag0..d_up_lagK`, `d_down_lag0..d_down_lagK`.
    K : the largest lag to accumulate through. Must match (or be <=) the lags actually
        present in `res`'s parameters — `restriction_vector` will raise a `KeyError`
        naming the missing parameter if it doesn't.

    Returns
    -------
    DataFrame with one row per horizon 0..K and columns `horizon`, `cum_up`,
    `cum_up_se`, `cum_up_lo`, `cum_up_hi`, `cum_down`, `cum_down_se`, `cum_down_lo`,
    `cum_down_hi` (the `_lo`/`_hi` pair is the 95% CI from `t_test`'s default alpha=0.05).
    """
    rows = []
    for h in range(K + 1):
        up_names = {f"d_up_lag{lag}" for lag in range(h + 1)}
        down_names = {f"d_down_lag{lag}" for lag in range(h + 1)}

        t_up = res.t_test(restriction_vector(res, positive_names=up_names))
        t_down = res.t_test(restriction_vector(res, positive_names=down_names))

        ci_up = np.asarray(t_up.conf_int())
        ci_down = np.asarray(t_down.conf_int())

        rows.append(
            {
                "horizon": h,
                "cum_up": float(np.asarray(t_up.effect).item()),
                "cum_up_se": float(np.asarray(t_up.sd).item()),
                "cum_up_lo": float(ci_up[0, 0]),
                "cum_up_hi": float(ci_up[0, 1]),
                "cum_down": float(np.asarray(t_down.effect).item()),
                "cum_down_se": float(np.asarray(t_down.sd).item()),
                "cum_down_lo": float(ci_down[0, 0]),
                "cum_down_hi": float(ci_down[0, 1]),
            }
        )

    return pd.DataFrame(rows)


def test_asymmetry(res, K: int, horizon: int | None = None) -> dict:
    """Test whether cumulative up-pass-through and down-pass-through differ, through a
    given horizon: estimate, standard error, 95% CI, and p-value for (sum(beta_up_0..h) -
    sum(beta_down_0..h)), via `res.t_test` on a single +1/-1 restriction vector.

    Both `d_retail` and `d_upstream` are in $/gallon (retail's native unit; crude's
    $/barrel is already converted to $/gal upstream of this module, see
    `build_tidy_tables.py`), so every coefficient here is a $/gal-per-$/gal ratio, i.e.
    dimensionless: `cum_up(h) = 0.8` means 80% of a crude increase has reached the pump
    by week h. That makes the difference tested here directly interpretable the same way
    — a positive estimate means up-pass-through is running ahead of down-pass-through
    through that horizon (the "rockets" side of rockets-and-feathers).

    Parameters
    ----------
    res : a fitted `statsmodels` results object (e.g. from `fit_distributed_lag`), with
        parameter names `d_up_lag0..d_up_lagK`, `d_down_lag0..d_down_lagK`.
    K : the largest lag present in `res`'s parameters.
    horizon : how many lags (0..horizon) to accumulate before comparing. Defaults to K
        (the full cumulative pass-through). Must be between 0 and K.

    Returns
    -------
    dict with `horizon`, `estimate` (the point estimate of the up-minus-down
    difference), `se`, `ci_lo`, `ci_hi` (95% CI, `t_test`'s default alpha=0.05), and
    `p_value` (two-sided, testing the difference against 0).
    """
    if horizon is None:
        horizon = K
    if not 0 <= horizon <= K:
        raise ValueError(f"horizon must be between 0 and K={K}, got {horizon}")

    up_names = {f"d_up_lag{lag}" for lag in range(horizon + 1)}
    down_names = {f"d_down_lag{lag}" for lag in range(horizon + 1)}
    restriction = restriction_vector(res, positive_names=up_names, negative_names=down_names)

    t = res.t_test(restriction)
    ci = np.asarray(t.conf_int())

    return {
        "horizon": horizon,
        "estimate": float(np.asarray(t.effect).item()),
        "se": float(np.asarray(t.sd).item()),
        "ci_lo": float(ci[0, 0]),
        "ci_hi": float(ci[0, 1]),
        "p_value": float(np.asarray(t.pvalue).item()),
    }
