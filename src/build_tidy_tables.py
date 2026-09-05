"""Build three tidy, long-format price tables from the raw FRED CSVs in `data/raw/`.

Reads every raw CSV `fetch_series.py` produced, parses `date`/`value` with
`fred_client.parse_observations` (the same `"."` -> `NaN` logic used everywhere else in
this project), and routes each series into one of three tables by its `kind` in
`series_manifest.SERIES`, with that kind's metadata fields attached as columns:

- `crude.csv`  — date, series_id, benchmark, freq, price_usd_per_barrel, price_usd_per_gallon
- `spot.csv`   — date, series_id, hub, fuel, formulation, freq, price_usd_per_gallon
- `retail.csv` — date, series_id, region, fuel, formulation, freq, price_usd_per_gallon

No resampling, reindexing, or filling: a `"."` observation becomes a `NaN` value in its
row, never a dropped or filled-in one. The three tables are never joined to each other —
crude/spot trade on a Friday grid, retail is a Monday survey, and merging them here would
be exactly the kind of look-ahead-bias risk this project's rules call out; that alignment
work is a deliberate later stage, not this script's.

Run: `python -m src.build_tidy_tables`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.fred_client import parse_observations
from src.series_manifest import SERIES, write_manifest_csv

_REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _REPO_ROOT / "data" / "raw"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"

# Unit conversion for price comparison ONLY — not a refinery yield. A 42-gallon barrel of
# crude oil yields roughly 19-20 gallons of gasoline once refined, not 42; this constant
# exists purely to put crude's $/barrel quote on the same $/gallon scale as spot and
# retail prices, so the three levels of the supply chain can be compared side by side.
GALLONS_PER_BARREL = 42

_CRUDE_COLUMNS = ["date", "series_id", "benchmark", "freq", "price_usd_per_barrel", "price_usd_per_gallon"]
_SPOT_COLUMNS = ["date", "series_id", "hub", "fuel", "formulation", "freq", "price_usd_per_gallon"]
_RETAIL_COLUMNS = ["date", "series_id", "region", "fuel", "formulation", "freq", "price_usd_per_gallon"]


def load_raw_series(series_id: str) -> pd.DataFrame:
    """Read one series' raw CSV from `data/raw/` and parse it with `fred_client`'s shared
    date/value logic (`"."` -> `NaN`, `value` -> `float64`).

    No rows are dropped or filled here — a `NaN` value stays exactly where FRED put it.
    """
    raw = pd.read_csv(RAW_DIR / f"{series_id}.csv", dtype=str)
    return parse_observations(raw.to_dict("records"))


def crude_row(series_id: str, entry: dict, parsed: pd.DataFrame) -> pd.DataFrame:
    """Build one crude series' rows for `crude.csv`: attaches `benchmark`/`freq` from the
    manifest, keeps the original $/barrel price, and adds its $/gallon conversion.
    """
    df = parsed.rename(columns={"value": "price_usd_per_barrel"})
    df["series_id"] = series_id
    df["benchmark"] = entry["benchmark"]
    df["freq"] = entry["freq"]
    df["price_usd_per_gallon"] = df["price_usd_per_barrel"] / GALLONS_PER_BARREL
    return df[_CRUDE_COLUMNS]


def spot_row(series_id: str, entry: dict, parsed: pd.DataFrame) -> pd.DataFrame:
    """Build one spot series' rows for `spot.csv`: attaches `hub`/`fuel`/`formulation`/
    `freq` from the manifest. Spot prices are already quoted in $/gallon, so `value` is
    used as-is — `formulation` is `None` for the diesel spot series, which don't carry
    that field in the manifest (diesel has no conventional/RBOB-style split).
    """
    df = parsed.rename(columns={"value": "price_usd_per_gallon"})
    df["series_id"] = series_id
    df["hub"] = entry["hub"]
    df["fuel"] = entry["fuel"]
    df["formulation"] = entry.get("formulation")
    df["freq"] = entry["freq"]
    return df[_SPOT_COLUMNS]


def retail_row(series_id: str, entry: dict, parsed: pd.DataFrame) -> pd.DataFrame:
    """Build one retail series' rows for `retail.csv`: attaches `region`/`fuel`/
    `formulation`/`freq` from the manifest. Retail prices are already quoted in $/gallon,
    so `value` is used as-is.
    """
    df = parsed.rename(columns={"value": "price_usd_per_gallon"})
    df["series_id"] = series_id
    df["region"] = entry["region"]
    df["fuel"] = entry["fuel"]
    df["formulation"] = entry["formulation"]
    df["freq"] = entry["freq"]
    return df[_RETAIL_COLUMNS]


_ROW_BUILDERS = {"crude": crude_row, "spot": spot_row, "retail": retail_row}


def build_tables() -> dict[str, pd.DataFrame]:
    """Read every raw CSV in `data/raw/`, join each to its manifest entry, and return the
    three tidy tables (`{"crude": df, "spot": df, "retail": df}`).

    Raises `RuntimeError`, naming the file, if a raw CSV's filename doesn't match any
    `series_manifest.SERIES` entry — an untracked file silently joined to nothing would
    otherwise just vanish from every output table with no explanation. A manifest series
    with no raw CSV yet is not an error (run `fetch_series.py` first if that's unexpected)
    but is reported so it isn't silently missing from a table without you knowing.
    """
    rows_by_kind: dict[str, list[pd.DataFrame]] = {"crude": [], "spot": [], "retail": []}
    seen_series_ids: set[str] = set()

    for path in sorted(RAW_DIR.glob("*.csv")):
        series_id = path.stem
        if series_id not in SERIES:
            raise RuntimeError(f"{series_id}: raw file '{path.name}' has no series_manifest.SERIES entry")

        entry = SERIES[series_id]
        parsed = load_raw_series(series_id)
        row_builder = _ROW_BUILDERS[entry["kind"]]
        rows_by_kind[entry["kind"]].append(row_builder(series_id, entry, parsed))
        seen_series_ids.add(series_id)

    missing = sorted(set(SERIES) - seen_series_ids)
    if missing:
        print(f"NOTE: {len(missing)} manifest series have no raw file yet, excluded from output: {missing}")

    return {kind: pd.concat(rows, ignore_index=True) for kind, rows in rows_by_kind.items()}


def print_summary(tables: dict[str, pd.DataFrame]) -> None:
    """Print row counts per table, and a breakdown by `freq` within each table."""
    for name, df in tables.items():
        print(f"{name}.csv: {len(df)} rows")
        for freq, count in df["freq"].value_counts().sort_index().items():
            print(f"  {freq:<12} {count}")


def main() -> None:
    """Entry point: build the three tables, write them to `data/processed/`, print the
    row-count summary."""
    tables = build_tables()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    # series_manifest.csv is the manifest as `02_data_pipeline.ipynb` reads it back. Written
    # here rather than left to a manual call, so editing SERIES and re-running the pipeline
    # can't leave a stale catalogue on disk that still looks current.
    write_manifest_csv(PROCESSED_DIR / "series_manifest.csv")

    print()
    print_summary(tables)


if __name__ == "__main__":
    main()
