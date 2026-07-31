"""Print every weekly series in FRED releases 183 and 212, with ID, title, and date range.

Read-only catalogue exploration: no analysis, no plotting, no data files written.
"""

from src.fred_client import list_release_series

_RELEASE_IDS = [183, 212]


def _is_weekly(frequency: str) -> bool:
    """True if a FRED `frequency` string denotes a weekly-cadence series.

    FRED's human-readable `frequency` field for weekly series is rarely the bare word
    "Weekly" — it's usually "Weekly, Ending Friday" (or Monday/Wednesday/etc., depending
    on which day the underlying survey lands on). All such strings start with "Weekly",
    so that prefix is what's matched here; an exact match against "Weekly" would
    silently drop most real weekly series.
    """
    return frequency.strip().lower().startswith("weekly")


def print_weekly_series(release_id: int) -> None:
    """Print every weekly series in a FRED release, one line each, sorted by series ID."""
    series = list_release_series(release_id)
    weekly = series[series["frequency"].apply(_is_weekly)].sort_values("series_id")

    print(f"\nRelease {release_id} — {len(weekly)} weekly series")
    for _, row in weekly.iterrows():
        print(f"  {row['series_id']:<15} {row['title']:<70} {row['observation_start']} to {row['observation_end']}")


if __name__ == "__main__":
    for release_id in _RELEASE_IDS:
        print_weekly_series(release_id)
