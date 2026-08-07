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


def _infer_k(design: pd.DataFrame) -> int:
    """The largest lag K implied by a design matrix's own `d_up_lag*`/`d_down_lag*`
    column names — read back from the DataFrame rather than passed in separately, so
    `fit_distributed_lag` can't be called with a `maxlags` default that silently
    disagrees with the design it's actually fitting.

    Raises `ValueError` if the up-lags and down-lags present don't match — `design`
    isn't a valid `build_design_matrix()` output if they don't.
    """
    up_lags = sorted(int(col.removeprefix("d_up_lag")) for col in design.columns if col.startswith("d_up_lag"))
    down_lags = sorted(int(col.removeprefix("d_down_lag")) for col in design.columns if col.startswith("d_down_lag"))
    if up_lags != down_lags:
        raise ValueError(f"design matrix's up-lags {up_lags} and down-lags {down_lags} don't match — not a valid build_design_matrix() output")
    return up_lags[-1]


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

    `maxlags` (the HAC correction's own lag window) defaults to K — the design's own
    largest regressor lag, read back from its column names via `_infer_k` — on the
    reasoning that if the model itself claims price effects can persist K weeks out, the
    correction for autocorrelation in its errors should look at least that far too. Pass
    an explicit `maxlags` to override this.

    Parameters
    ----------
    design : a `build_design_matrix()` output — must contain `d_retail` and one or more
        `d_up_lag*`/`d_down_lag*` columns (any other columns, e.g. `date`, are ignored as
        regressors).
    maxlags : the HAC lag window. Defaults to `design`'s own K if not given.

    Returns
    -------
    The fitted `statsmodels` results object, returned exactly as `.fit()` produces it —
    no wrapping, no partial extraction — so `.summary()`, `.params`, `.bse`, etc. all
    work normally on it.
    """
    if maxlags is None:
        maxlags = _infer_k(design)

    regressor_columns = [col for col in design.columns if col not in ("date", "d_retail")]
    y = design["d_retail"]
    X = sm.add_constant(design[regressor_columns])

    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
