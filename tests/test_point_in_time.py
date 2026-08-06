"""Tests for `src/point_in_time.align_retail_to_upstream` — hermetic, no network, no
files on disk. Fixtures below are hand-built (a handful of Mondays/Fridays), not pulled
from `data/`, so these run in the fast default suite alongside the rest of the unit tests.

Each test targets one specific claim from ADR-0002 / the `point_in_time.py` docstring:
- the correct-alignment rule itself (prior-week Friday, same-day exclusion for daily),
- that each of the three guards actually *fires* on input built to violate it (not just
  that it exists in the source, per ADR-0002: "a safety check never observed to fire is
  not known to work"),
- and `gap_days`'s arithmetic, including the holiday case where the naive "prior Friday"
  phrasing and the actual "most recent observation before the survey" rule diverge.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.point_in_time import align_retail_to_upstream

# Four consecutive Monday retail dates, Jan 2024.
_RETAIL_MONDAYS = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"])


def _retail(dates) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": [3.00, 3.10, 3.20, 3.30][: len(dates)]})


def test_weekly_picks_prior_week_friday():
    """A retail Monday must pair with the Friday of the *prior* week, not its own —
    the exact leak ADR-0002 measured and fixed (same-week Friday postdates the survey).
    """
    retail = _retail(_RETAIL_MONDAYS)
    # One weekly-Friday upstream row per week, including retail's own week's Friday, so a
    # bug that picked the same-week Friday would have a real (wrong) row to match onto.
    weekly_upstream = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-22", "2023-12-29", "2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"]),
            "value": [70.0, 71.0, 72.0, 73.0, 74.0, 75.0],
        }
    )

    result = align_retail_to_upstream(retail, weekly_upstream, mode="weekly")

    # 2024-01-01 (Mon, week of Jan 1-5) -> the prior week's Friday, 2023-12-29 — NOT
    # 2024-01-05, which is the Friday of retail's *own* week and postdates the Monday.
    expected_upstream_dates = pd.to_datetime(["2023-12-29", "2024-01-05", "2024-01-12", "2024-01-19"])
    assert list(result["upstream_date"]) == list(expected_upstream_dates)


def test_daily_pit_excludes_same_day():
    """A daily upstream row dated exactly on the retail Monday must be skipped in favour
    of the previous trading day's close — `allow_exact_matches=False` is what makes
    `daily_pit` different from `weekly` (a same-day match is structurally impossible for
    `weekly` since Fridays never equal Mondays, but very possible for daily data).
    """
    retail = _retail([_RETAIL_MONDAYS[0]])  # 2024-01-01
    daily_upstream = pd.DataFrame(
        {
            # 2023-12-29 (Fri, the correct prior trading day) and 2024-01-01 itself (the
            # retail Monday) both present — a bug that allowed exact matches would pick
            # the same-day row instead of falling back to the Friday.
            "date": pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-01"]),
            "value": [69.0, 70.0, 71.0],
        }
    )

    result = align_retail_to_upstream(retail, daily_upstream, mode="daily_pit")

    assert result["upstream_date"].iloc[0] == pd.Timestamp("2023-12-29")
    assert result["value_upstream"].iloc[0] == 70.0


def test_leaky_alignment_raises():
    """Feed a deliberately leaky alignment — an upstream_date on or after the retail
    date — straight into the guard and assert it raises. This proves the guard actually
    *fires*, not merely that it's present in the source.

    `_check_no_look_ahead` operates on an already-merged frame (`date`, `upstream_date`
    columns), so it's called directly here rather than routed through
    `align_retail_to_upstream` + `merge_asof` — `merge_asof(direction="backward")` makes
    a genuinely leaky *merge* output essentially impossible to construct honestly; the
    guard itself is the thing under test, and a hand-built leaky frame is the direct way
    to prove it fires.
    """
    from src.point_in_time import _check_no_look_ahead

    leaky = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-08"]),
            "upstream_date": pd.to_datetime(["2024-01-12"]),  # AFTER the retail date — leaky
        }
    )

    with pytest.raises(ValueError, match="not strictly before"):
        _check_no_look_ahead(leaky)


def test_duplicate_upstream_raises():
    """Two retail rows resolving to the same upstream observation must raise — that
    would otherwise produce an identical value_upstream on both rows, and thus a fake
    zero in first differences downstream.
    """
    from src.point_in_time import _check_no_duplicate_upstream_date

    duplicated = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "upstream_date": pd.to_datetime(["2023-12-29", "2023-12-29"]),  # same upstream date twice
        }
    )

    with pytest.raises(ValueError, match="shared by more than one retail row"):
        _check_no_duplicate_upstream_date(duplicated)


def test_gap_days_computed():
    """gap_days is 3 for an ordinary week (Fri -> Mon) and 4 when the prior Friday is
    missing — simulating a market holiday by omitting that row, exactly as Good Friday
    is genuinely absent from FRED's real daily series (see point_in_time.py's docstring).
    """
    retail = _retail([_RETAIL_MONDAYS[0], _RETAIL_MONDAYS[1]])  # 2024-01-01, 2024-01-08
    daily_upstream = pd.DataFrame(
        {
            # 2023-12-29 (Fri before the first Monday) present.
            # 2024-01-05 (Fri before the second Monday) DELETED — simulated holiday —
            # so the second row must fall back to 2024-01-04 (Thu), a 4-day gap.
            "date": pd.to_datetime(["2023-12-29", "2024-01-03", "2024-01-04"]),
            "value": [70.0, 71.0, 72.0],
        }
    )

    result = align_retail_to_upstream(retail, daily_upstream, mode="daily_pit")

    assert result["gap_days"].iloc[0] == 3  # 2023-12-29 (Fri) -> 2024-01-01 (Mon)
    assert result["upstream_date"].iloc[1] == pd.Timestamp("2024-01-04")
    assert result["gap_days"].iloc[1] == 4  # 2024-01-04 (Thu, Friday missing) -> 2024-01-08 (Mon)
