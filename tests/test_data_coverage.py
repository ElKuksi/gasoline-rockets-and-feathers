"""Coverage/sanity checks on the tidy tables in `data/processed/` — no network, but they
require `fetch_series.py` and `build_tidy_tables.py` to have already been run, so they're
marked `@pytest.mark.data` and excluded from the default `pytest` run (see pyproject.toml).

Every assertion here prints the actual offending rows on failure, not just a bare
True/False — the point of these tests is to tell you *which* series/date broke, not just
that something did.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fetch_series import START_DATE

_REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"

TABLE_NAMES = ["crude", "spot", "retail"]

# Weekday each weekly freq should land on (Monday=0 ... Friday=4, per date.dt.dayofweek).
EXPECTED_WEEKDAY = {"weekly-fri": 4, "weekly-mon": 0}

_MAX_PRINTED_ROWS = 20


def _format_offending(df: pd.DataFrame) -> str:
    """Render up to `_MAX_PRINTED_ROWS` offending rows for an assertion message, so a
    failure with hundreds of bad rows doesn't dump an unreadable wall of text."""
    if df.empty:
        return "(no rows)"
    if len(df) > _MAX_PRINTED_ROWS:
        shown = df.head(_MAX_PRINTED_ROWS).to_string()
        return f"{shown}\n... ({len(df) - _MAX_PRINTED_ROWS} more rows not shown, {len(df)} total)"
    return df.to_string()


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    """Load the three tidy tables `build_tidy_tables.py` produces.

    Skips (rather than erroring) if a table is missing, with an actionable message —
    these tests need the pipeline to have run, and a fresh clone without that shouldn't
    look like a broken test, just an unmet precondition.
    """
    loaded = {}
    for name in TABLE_NAMES:
        path = PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            pytest.skip(f"{path} not found — run `python -m src.fetch_series` then `python -m src.build_tidy_tables` first")
        loaded[name] = pd.read_csv(path, parse_dates=["date"])
    return loaded


@pytest.mark.data
def test_weekly_dates_land_on_expected_weekday(tables):
    """Every weekly-fri row must date to a Friday; every weekly-mon row to a Monday."""
    for name, df in tables.items():
        for freq, weekday in EXPECTED_WEEKDAY.items():
            subset = df[df["freq"] == freq]
            if subset.empty:
                continue
            bad = subset[subset["date"].dt.dayofweek != weekday]
            assert bad.empty, f"{name}.csv: {freq} rows not landing on the expected weekday:\n{_format_offending(bad)}"


@pytest.mark.data
def test_no_daily_row_with_a_real_price_falls_on_a_weekend(tables):
    """FRED's daily series are trading-day series — markets are closed on weekends. A
    weekend row carrying an actual price would mean something upstream is broken (a
    shifted date, a timezone bug, silent interpolation). A weekend row that's already NaN
    can't be any of those — it claims no price at all, so there's nothing to have been
    shifted or invented. Confirmed real: two NaN-valued Saturday rows exist in FRED's own
    `DDFUELNYH`/`DDFUELUSGULF` data (2010-11-13, 2010-11-20) — a source-data quirk, not a
    bug in this pipeline — so only weekend rows with a non-null price are flagged.
    """
    for name, df in tables.items():
        daily = df[df["freq"] == "daily"]
        if daily.empty:
            continue
        price_col = "price_usd_per_barrel" if name == "crude" else "price_usd_per_gallon"
        bad = daily[(daily["date"].dt.dayofweek >= 5) & daily[price_col].notna()]
        assert bad.empty, f"{name}.csv: daily rows with a real price landing on a weekend:\n{_format_offending(bad)}"


@pytest.mark.data
def test_each_series_first_date_within_a_week_of_expected_start(tables):
    """Every series was pulled starting at `fetch_series.START_DATE` — its first actual
    observation should land within a week of that (weekends/holidays can shift it by a
    few days, but not further).
    """
    expected_start = pd.Timestamp(START_DATE)
    offenders = []
    for name, df in tables.items():
        first_dates = df.groupby("series_id")["date"].min()
        too_far = first_dates[(first_dates - expected_start).abs() > pd.Timedelta(days=7)]
        for series_id, first_date in too_far.items():
            offenders.append({"table": name, "series_id": series_id, "first_date": first_date})

    bad = pd.DataFrame(offenders)
    assert bad.empty, f"series starting more than 7 days from expected {expected_start.date()}:\n{_format_offending(bad)}"


@pytest.mark.data
def test_no_duplicate_series_id_and_date(tables):
    """No (series_id, date) pair should appear twice in any table."""
    for name, df in tables.items():
        dupes = df[df.duplicated(subset=["series_id", "date"], keep=False)].sort_values(["series_id", "date"])
        assert dupes.empty, f"{name}.csv: duplicate (series_id, date) rows:\n{_format_offending(dupes)}"


@pytest.mark.data
def test_prices_are_finite(tables):
    """Every non-missing price must be finite (no `inf`/`-inf` from a bad division).

    `NaN` (FRED's `"."` marker, preserved by `build_tidy_tables.py`'s "missing stays
    missing" rule) is excluded here on purpose — it's expected, not a defect.
    """
    price_columns = {
        "crude": ["price_usd_per_barrel", "price_usd_per_gallon"],
        "spot": ["price_usd_per_gallon"],
        "retail": ["price_usd_per_gallon"],
    }
    for name, df in tables.items():
        for col in price_columns[name]:
            present = df[df[col].notna()]
            bad = present[~np.isfinite(present[col])]
            assert bad.empty, f"{name}.csv: non-finite {col}:\n{_format_offending(bad[['series_id', 'date', col]])}"


@pytest.mark.data
def test_spot_and_retail_prices_are_positive(tables):
    """Spot and retail prices must be > 0 — refined products can't structurally trade at
    a negative price the way a futures-settled physical commodity can.

    Crude is deliberately excluded: WTI settled at -$36.98 on 2020-04-20, the real,
    well-documented first-ever negative-price day (COVID demand collapse collided with
    storage capacity running out, so holders paid to avoid taking delivery). That's a
    genuine market state, not corrupted data, so `crude.csv` only gets the finiteness
    check above, not this one.
    """
    for name in ["spot", "retail"]:
        df = tables[name]
        present = df[df["price_usd_per_gallon"].notna()]
        bad = present[present["price_usd_per_gallon"] <= 0]
        assert bad.empty, f"{name}.csv: non-positive price_usd_per_gallon:\n{_format_offending(bad[['series_id', 'date', 'price_usd_per_gallon']])}"


@pytest.mark.data
def test_crude_gallon_price_matches_barrel_conversion(tables):
    """crude.csv's price_usd_per_gallon must equal price_usd_per_barrel / 42, to 6 dp.

    `equal_nan=True` treats a (NaN barrel price, NaN gallon price) pair as consistent —
    a missing barrel price should produce a missing gallon price, not a mismatch.
    """
    crude = tables["crude"]
    expected_gallon_price = crude["price_usd_per_barrel"] / 42
    matches = np.isclose(crude["price_usd_per_gallon"], expected_gallon_price, atol=1e-6, equal_nan=True)
    bad = crude[~matches]
    assert bad.empty, f"crude.csv: price_usd_per_gallon != price_usd_per_barrel/42 within 6dp:\n{_format_offending(bad)}"


@pytest.mark.data
def test_weekly_series_row_counts_are_in_expected_range(tables):
    """Each weekly series (weekly-fri or weekly-mon) should carry one row per week from
    `fetch_series.START_DATE` to the newest date in the table, give or take a few. The
    bounds are derived from the data's own span rather than hard-coded, so a later data
    refresh doesn't fail this test purely for having more weeks in it — what's actually
    being caught is a truncated pull, a duplicated pull, or a freq mislabeled in the
    manifest, none of which scale with the sample.
    """
    newest = max(df["date"].max() for df in tables.values())
    expected = (newest - pd.Timestamp(START_DATE)).days // 7
    low, high = expected - 15, expected + 15

    offenders = []
    for name, df in tables.items():
        weekly = df[df["freq"].isin(["weekly-fri", "weekly-mon"])]
        if weekly.empty:
            continue
        counts = weekly.groupby("series_id").size()
        out_of_range = counts[(counts < low) | (counts > high)]
        for series_id, count in out_of_range.items():
            offenders.append({"table": name, "series_id": series_id, "rows": count})

    bad = pd.DataFrame(offenders)
    assert bad.empty, f"weekly series with row count outside [{low}, {high}]:\n{_format_offending(bad)}"
