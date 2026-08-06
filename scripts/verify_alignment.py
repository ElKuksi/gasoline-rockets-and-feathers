"""Reproduce ADR-0002's evidence table: the contemporaneous correlation of weekly first
differences between retail (GASREGW) and crude (WCOILWTICO), under the clean point-in-time
alignment vs. a deliberately leaky one.

This is the empirical case ADR-0002 makes for `src/point_in_time.py`'s guard in the first
place — pairing a retail Monday with its *own* week's upstream Friday leaks four days of
future crude into that row, and ADR-0002 measured what that leak does to the correlation:
0.209 (leaky) vs. 0.591 (clean). This script re-measures both numbers directly against
whatever is currently in `data/processed/`, so the ADR's claim stays checkable rather than
just asserted.

Read-only against `data/processed/{crude,retail}.csv`; writes nothing.

Run: `python -m scripts.verify_alignment`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.point_in_time import align_retail_to_upstream

_REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"

RETAIL_SERIES_ID = "GASREGW"
UPSTREAM_SERIES_ID = "WCOILWTICO"


def load_series(path: Path, series_id: str, freq: str) -> pd.DataFrame:
    """Load one series' `date`/`value` columns from a tidy table in `data/processed/`.

    Filters to `series_id` and `freq`, keeps `$/gallon` (ADR-0002's table is stated in
    $/gal for both sides, so crude's `price_usd_per_gallon` is used here, not
    `price_usd_per_barrel`), and renames the price column to `value` — the column name
    `point_in_time.align_retail_to_upstream` expects.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    subset = df[(df["series_id"] == series_id) & (df["freq"] == freq)]
    return subset[["date", "price_usd_per_gallon"]].rename(columns={"price_usd_per_gallon": "value"}).sort_values("date").reset_index(drop=True)


def leaky_merge(retail: pd.DataFrame, upstream: pd.DataFrame) -> pd.DataFrame:
    """Build the deliberately leaky alignment ADR-0002 measured against: each retail
    Monday paired with the *next* upstream Friday at or after it — i.e. the Friday of
    retail's own week, which is later than the survey and therefore leaks four days of
    future crude into that row.

    This does NOT go through `align_retail_to_upstream` — that function's guard would
    (correctly) raise `ValueError` on exactly this input, since `upstream_date` ends up
    *after* `date` on every row. That's the point: this script's whole purpose is to
    reproduce the leak the guard exists to prevent, so it has to be built directly with
    `pd.merge_asof(direction="forward")` here. This is the one place in the codebase a
    retail-to-upstream join deliberately bypasses `point_in_time.py`, and only for this
    diagnostic comparison — every other join still must route through it per ADR-0002.
    """
    upstream_renamed = upstream.rename(columns={"date": "upstream_date"})
    return pd.merge_asof(
        retail,
        upstream_renamed,
        left_on="date",
        right_on="upstream_date",
        direction="forward",
        suffixes=("_retail", "_upstream"),
    )


def contemporaneous_corr_of_first_differences(merged: pd.DataFrame) -> float:
    """corr(delta_retail_t, delta_upstream_t) — the exact quantity in ADR-0002's table.

    `Series.corr` drops NaN pairwise, so the leading NaN from each `.diff()` (no prior
    observation for the first row) doesn't need to be dropped explicitly first.
    """
    delta_retail = merged["value_retail"].diff()
    delta_upstream = merged["value_upstream"].diff()
    return delta_retail.corr(delta_upstream)


def print_correlation_table(clean_corr: float, leaky_corr: float, n_obs: int) -> None:
    """Print the ADR-0002 comparison as a small table — the two numbers this script
    exists to check, side by side with the ADR's originally reported values.
    """
    rows = [
        ("clean (prior week's Friday)", clean_corr, 0.591),
        ("leaky (same week's Friday)", leaky_corr, 0.209),
    ]
    print(f"contemporaneous correlation of weekly first differences, n={n_obs} paired observations")
    print(f"{'alignment':<32} {'measured':>10} {'ADR-0002':>10}")
    print("-" * 54)
    for label, measured, adr_value in rows:
        print(f"{label:<32} {measured:>10.3f} {adr_value:>10.3f}")


def print_gap_days_report(clean_merged: pd.DataFrame) -> None:
    """Print the clean alignment's gap_days distribution, then every row where the gap
    isn't the ordinary 3 days (Friday -> Monday) — each such row is a market holiday
    that shifted the upstream date, nameable directly from its date column.
    """
    print()
    print("gap_days distribution (clean alignment):")
    print(clean_merged["gap_days"].value_counts().sort_index().to_string())

    irregular = clean_merged[clean_merged["gap_days"] != 3]
    print()
    print(f"rows where gap_days != 3 ({len(irregular)} rows):")
    if irregular.empty:
        print("(none)")
    else:
        print(irregular[["date", "upstream_date", "gap_days"]].to_string(index=False))


def main() -> None:
    retail = load_series(PROCESSED_DIR / "retail.csv", RETAIL_SERIES_ID, "weekly-mon")
    upstream = load_series(PROCESSED_DIR / "crude.csv", UPSTREAM_SERIES_ID, "weekly-fri")

    clean_merged = align_retail_to_upstream(retail, upstream, mode="weekly")
    leaky_merged = leaky_merge(retail, upstream)

    clean_corr = contemporaneous_corr_of_first_differences(clean_merged)
    leaky_corr = contemporaneous_corr_of_first_differences(leaky_merged)

    print_correlation_table(clean_corr, leaky_corr, n_obs=len(clean_merged))
    print_gap_days_report(clean_merged)


if __name__ == "__main__":
    main()
