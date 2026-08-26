"""Tests for `src.regional.build_region_difference` -- hermetic, hand-built fixtures, no
network, no reads from `data/`.

The load-bearing test is `test_differenced_fit_equals_difference_of_separate_fits`: the whole
method rests on the claim that fitting the differenced series gives coefficients exactly equal
to the difference of the two separate fits, and that claim holds only while both regions enter
the *same* design matrix. A misaligned join, a dropped row, or a reversed subtraction would all
break it. The remaining tests cover the guards that keep that premise true.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.asymmetry import build_design_matrix, fit_distributed_lag
from src.asymmetry import test_asymmetry as asymmetry_gap
from src.regional import build_region_difference

K = 2


def _weekly_pair(n_mondays: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One retail region and its matching upstream series: `n_mondays` consecutive Mondays
    from 2020-01-06, each paired with the Friday of the prior week, which is what
    `align_retail_to_upstream("weekly")` expects. Prices are random-walk-ish but seeded, so
    the fits are non-degenerate and reproducible.
    """
    rng = np.random.default_rng(seed)
    mondays = pd.date_range("2020-01-06", periods=n_mondays, freq="7D")
    fridays = mondays - pd.Timedelta(days=3)

    retail = pd.DataFrame({"date": mondays, "value": 2.5 + np.cumsum(rng.normal(0, 0.05, n_mondays))})
    upstream = pd.DataFrame({"date": fridays, "value": 1.8 + np.cumsum(rng.normal(0, 0.07, n_mondays))})
    return retail, upstream


def test_differenced_fit_equals_difference_of_separate_fits():
    """The claim the whole module rests on: because both regions are regressed on the same
    design matrix and OLS is linear in the outcome, the differenced fit's coefficients equal
    region A's minus region B's exactly -- not to within sampling error.

    Asserted to 1e-10 on every coefficient, and separately on the asymmetry gap itself, which
    is the quantity actually reported.
    """
    retail_a, upstream = _weekly_pair(200, seed=1)
    retail_b, _ = _weekly_pair(200, seed=2)

    res_a = fit_distributed_lag(build_design_matrix(retail_a, upstream, K=K, mode="weekly"))
    res_b = fit_distributed_lag(build_design_matrix(retail_b, upstream, K=K, mode="weekly"))

    diff = build_region_difference(retail_a, retail_b)
    res_diff = fit_distributed_lag(build_design_matrix(diff, upstream, K=K, mode="weekly"))

    assert list(res_diff.params.index) == list(res_a.params.index)
    np.testing.assert_allclose(
        res_diff.params.to_numpy(),
        res_a.params.to_numpy() - res_b.params.to_numpy(),
        atol=1e-10,
        err_msg="differenced fit's coefficients must equal A's minus B's exactly",
    )

    gap_a = asymmetry_gap(res_a, K=K, horizon=0)["estimate"]
    gap_b = asymmetry_gap(res_b, K=K, horizon=0)["estimate"]
    gap_diff = asymmetry_gap(res_diff, K=K, horizon=0)["estimate"]
    assert gap_diff == pytest.approx(gap_a - gap_b, abs=1e-10)


def test_differenced_standard_error_is_not_the_difference_of_separate_ones():
    """The reason for differencing rather than subtracting two published estimates: the
    standard error comes from the differenced residual and carries the covariance between the
    two regions. If this ever *did* equal the difference (or the sum) of the separate standard
    errors, the covariance would have been dropped and the interval would be wrong.
    """
    retail_a, upstream = _weekly_pair(200, seed=3)
    retail_b, _ = _weekly_pair(200, seed=4)

    res_a = fit_distributed_lag(build_design_matrix(retail_a, upstream, K=K, mode="weekly"))
    res_b = fit_distributed_lag(build_design_matrix(retail_b, upstream, K=K, mode="weekly"))
    res_diff = fit_distributed_lag(
        build_design_matrix(build_region_difference(retail_a, retail_b), upstream, K=K, mode="weekly")
    )

    se_a = asymmetry_gap(res_a, K=K, horizon=0)["se"]
    se_b = asymmetry_gap(res_b, K=K, horizon=0)["se"]
    se_diff = asymmetry_gap(res_diff, K=K, horizon=0)["se"]

    assert se_diff > 0
    assert se_diff != pytest.approx(abs(se_a - se_b), abs=1e-8)
    assert se_diff != pytest.approx(se_a + se_b, abs=1e-8)


def test_difference_is_antisymmetric_in_argument_order():
    """Swapping the arguments negates the series, and therefore the reported gap difference.
    Guards against a silently reversed subtraction, which would flip the sign of every
    conclusion drawn from this module while leaving magnitudes and p-values untouched.
    """
    retail_a, _ = _weekly_pair(50, seed=5)
    retail_b, _ = _weekly_pair(50, seed=6)

    ab = build_region_difference(retail_a, retail_b)
    ba = build_region_difference(retail_b, retail_a)

    np.testing.assert_allclose(ab["value"].to_numpy(), -ba["value"].to_numpy(), atol=1e-12)


def test_output_columns_match_what_build_design_matrix_expects():
    """The result is meant to be a drop-in substitute for a regional retail series."""
    retail_a, _ = _weekly_pair(20, seed=7)
    retail_b, _ = _weekly_pair(20, seed=8)

    out = build_region_difference(retail_a, retail_b)

    assert list(out.columns) == ["date", "value"]
    assert len(out) == 20
    assert out["date"].is_monotonic_increasing


def test_mismatched_date_grids_are_rejected():
    """A truncated overlap would still produce plausible output while quietly breaking the
    shared-design-matrix premise the coefficient identity depends on, so it must raise rather
    than silently inner-join.
    """
    retail_a, _ = _weekly_pair(30, seed=9)
    retail_b, _ = _weekly_pair(30, seed=10)

    with pytest.raises(ValueError, match="identical dates"):
        build_region_difference(retail_a, retail_b.iloc[:-1])


def test_duplicate_dates_are_rejected():
    retail_a, _ = _weekly_pair(10, seed=11)
    retail_b, _ = _weekly_pair(10, seed=12)
    doubled = pd.concat([retail_b, retail_b.iloc[[0]]], ignore_index=True).sort_values("date")

    with pytest.raises(ValueError, match="duplicate dates"):
        build_region_difference(retail_a, doubled)


def test_missing_column_is_rejected():
    retail_a, _ = _weekly_pair(10, seed=13)
    retail_b, _ = _weekly_pair(10, seed=14)

    with pytest.raises(ValueError, match="missing required column"):
        build_region_difference(retail_a, retail_b.rename(columns={"value": "price"}))
