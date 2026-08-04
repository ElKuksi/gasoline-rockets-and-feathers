"""Pull every series in the manifest from FRED, 2010-01-01 through today, into
`data/raw/{series_id}.csv` — unmodified, unparsed, one file per series.

No cleaning, no joins, no plots here: this script's only job is getting FRED's raw
response onto disk. Re-running it is cheap and safe by default — a series already saved
is skipped unless `--refresh` is passed, so you're not re-hitting FRED for 29 series every
time you touch this file.

Run: `python -m src.fetch_series` (add `--refresh` to force every series to re-download).
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import pandas as pd

from src.fred_client import get_series_raw_observations
from src.series_manifest import SERIES

_REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _REPO_ROOT / "data" / "raw"

START_DATE = "2010-01-01"
SLEEP_SECONDS = 0.5  # small pause between live FRED requests — 29 series, ~15s added, polite to their API


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: only `--refresh`, which forces every series to re-download
    even if its raw CSV already exists on disk."""
    parser = argparse.ArgumentParser(description="Pull every manifest series from FRED into data/raw/*.csv.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download every series even if data/raw/<series_id>.csv already exists.",
    )
    return parser.parse_args()


def raw_path(series_id: str) -> Path:
    """The on-disk path a series' raw CSV lives at (whether or not it exists yet)."""
    return RAW_DIR / f"{series_id}.csv"


def fetch_one_series(series_id: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one series' observations from FRED, unparsed.

    Raises `RuntimeError`, naming `series_id`, if the request itself fails (network/HTTP
    error, propagated from `fred_client`) or if FRED returns zero observations for the
    requested range (a silent failure otherwise — e.g. a typo'd series ID or a range with
    no data — that would only surface much later as a mysteriously-missing series).

    Returns a DataFrame with FRED's observation fields exactly as given
    (`realtime_start`, `realtime_end`, `date`, `value`), `value` still a raw string —
    including `"."` for FRED's missing-observation marker. No parsing, no type coercion.
    """
    try:
        body = get_series_raw_observations(series_id, start=start, end=end)
    except RuntimeError as exc:
        raise RuntimeError(f"{series_id}: fetch failed — {exc}") from exc

    observations = body["observations"]
    if not observations:
        raise RuntimeError(f"{series_id}: FRED returned zero rows for range {start} to {end}")

    return pd.DataFrame(observations)


def save_raw_csv(series_id: str, df: pd.DataFrame) -> Path:
    """Write one series' raw observations to `data/raw/{series_id}.csv`, unmodified."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = raw_path(series_id)
    df.to_csv(path, index=False)
    return path


def load_raw_csv(series_id: str) -> pd.DataFrame:
    """Read an already-saved raw CSV back from disk, as plain strings.

    `dtype=str` guarantees `value` comes back exactly as written (including `"."`) rather
    than pandas guessing a numeric dtype and silently reinterpreting anything.
    """
    return pd.read_csv(raw_path(series_id), dtype=str)


def summarize(series_id: str, df: pd.DataFrame) -> dict:
    """Build one summary row for a series: row count, date range, and how many
    observations are FRED's `"."` missing-value marker.

    Dates are compared as plain ISO-8601 strings (`YYYY-MM-DD`), which sort identically
    to their true chronological order — so `min`/`max` need no date parsing at all,
    keeping this script's "no cleaning" promise even for its own summary stats.
    """
    return {
        "series_id": series_id,
        "rows": len(df),
        "first_date": df["date"].min(),
        "last_date": df["date"].max(),
        "nan_count": int((df["value"] == ".").sum()),
    }


def print_summary_table(summary_rows: list[dict]) -> None:
    """Print the accumulated per-series summary rows as a fixed-width table."""
    header = f"{'series_id':<14} {'rows':>6} {'first_date':>12} {'last_date':>12} {'nan_count':>9}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(f"{row['series_id']:<14} {row['rows']:>6} {row['first_date']:>12} {row['last_date']:>12} {row['nan_count']:>9}")


def fetch_all(refresh: bool) -> list[dict]:
    """Ensure every manifest series has a raw CSV on disk, then return one summary row
    per series (`series_id`, `rows`, `first_date`, `last_date`, `nan_count`).

    A series already saved to `data/raw/{series_id}.csv` is skipped (its existing file is
    read back for the summary, not re-fetched) unless `refresh` is True. Stops immediately
    — via the `RuntimeError`s raised by `fetch_one_series` — on the first series that
    errors or comes back empty, naming that series in the message.
    """
    end = date.today().isoformat()
    summary_rows = []

    for series_id in sorted(SERIES):
        if raw_path(series_id).exists() and not refresh:
            print(f"{series_id:<14} skipped (already on disk)")
            df = load_raw_csv(series_id)
        else:
            df = fetch_one_series(series_id, start=START_DATE, end=end)
            save_raw_csv(series_id, df)
            print(f"{series_id:<14} fetched  ({len(df)} rows)")
            time.sleep(SLEEP_SECONDS)

        summary_rows.append(summarize(series_id, df))

    return summary_rows


def main() -> None:
    """Entry point: parse `--refresh`, fetch/skip every manifest series, print the
    summary table."""
    args = parse_args()
    summary_rows = fetch_all(refresh=args.refresh)
    print()
    print_summary_table(summary_rows)


if __name__ == "__main__":
    main()
