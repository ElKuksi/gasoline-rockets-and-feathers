"""Region-versus-region comparison of asymmetric pass-through.

`05` and `08` fit one distributed-lag model per region and produce one gap estimate each.
Those estimates cannot be compared to one another directly: each carries its own standard
error, and neither knows how the two regions' residuals covary, so the difference between
two of them has no standard error attached and no test can be run on it.

This module supplies the missing piece. `build_region_difference` forms the series
`retail_a - retail_b`; fitting the *same* distributed-lag model to that difference yields
coefficients that are exactly the per-lag differences between the two regions' coefficients,
with a HAC standard error computed on the differenced residual. `src.asymmetry.test_asymmetry`
applied to that fit therefore returns `gap_a - gap_b` with a valid interval and p-value.

Why the equality is exact rather than approximate
-------------------------------------------------
Both regional models are regressed on the *same* design matrix X, built from the same crude
series under the same point-in-time alignment. OLS is linear in the dependent variable:

    beta_hat(y) = (X'X)^-1 X' y

so for any two outcome vectors y_a and y_b sharing that X,

    beta_hat(y_a - y_b) = (X'X)^-1 X' (y_a - y_b)
                        = (X'X)^-1 X' y_a - (X'X)^-1 X' y_b
                        = beta_hat(y_a) - beta_hat(y_b)

identically, not to within sampling error. `tests/test_regional.py` pins this against real
fits rather than taking the algebra on trust -- a misaligned join or a dropped row would break
the shared-X premise and the identity with it, silently, which is precisely the failure this
module needs guarding against.

The standard errors are *not* differences of the separate standard errors, and that is the
point of the exercise: they are estimated from the differenced residual, which carries the
covariance between the two regions that two separately fitted intervals cannot express.

Two further consequences of differencing, both useful here: anything common to both regions
(nationwide demand shocks, federal tax, seasonal blend timing) cancels out of the comparison,
and no cross-region error-correlation term needs modelling, as it would in a panel stacking
all regions into a single regression.
"""

from __future__ import annotations

import pandas as pd


def build_region_difference(retail_a: pd.DataFrame, retail_b: pd.DataFrame) -> pd.DataFrame:
    """Form the region-difference series `retail_a - retail_b`, ready to pass to
    `src.asymmetry.build_design_matrix` in place of a single region's retail series.

    Parameters
    ----------
    retail_a, retail_b : DataFrames with `date` (datetime64) and `value` ($/gal), each
        sorted ascending by `date`, as returned by `scripts.verify_alignment.load_series`.
        Both must cover exactly the same dates: the exactness of the coefficient identity
        documented in this module's docstring depends on both regions entering the same
        design matrix, and a silently truncated overlap would break that while still
        producing plausible-looking output.

    Returns
    -------
    DataFrame with `date` and `value`, where `value` is `retail_a.value - retail_b.value`
    on the shared date grid. Column names match what `build_design_matrix` expects, so the
    result is a drop-in substitute for a regional retail series.

    Raises
    ------
    ValueError
        If either frame is missing a required column, if either contains duplicate dates,
        or if the two date grids are not identical.
    """
    for name, frame in (("retail_a", retail_a), ("retail_b", retail_b)):
        missing = {"date", "value"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing required column(s) {sorted(missing)}")
        if frame["date"].duplicated().any():
            dupes = frame.loc[frame["date"].duplicated(), "date"].tolist()
            raise ValueError(f"{name}: duplicate dates {dupes[:5]}")

    dates_a = retail_a["date"].reset_index(drop=True)
    dates_b = retail_b["date"].reset_index(drop=True)

    if not dates_a.equals(dates_b):
        only_a = sorted(set(dates_a) - set(dates_b))
        only_b = sorted(set(dates_b) - set(dates_a))
        raise ValueError(
            "retail_a and retail_b must cover identical dates; "
            f"{len(only_a)} date(s) only in retail_a (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in retail_b (e.g. {only_b[:3]})"
        )

    a = retail_a.sort_values("date").reset_index(drop=True)
    b = retail_b.sort_values("date").reset_index(drop=True)

    return pd.DataFrame({"date": a["date"], "value": a["value"] - b["value"]})
