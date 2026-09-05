"""Static catalogue of every FRED series this project pulls, and how each one fits the
crude → wholesale spot → retail transmission chain the study is built around.

This module holds no fetching logic and makes no API calls — it's a lookup table other
code (data-pull scripts, the eventual model) reads to know *what* to fetch and *why each
series exists* (its role in the analysis), not *how* to fetch it. That's `fred_client.py`'s
job.

Every entry's `kind` determines which extra fields it carries:
- `crude`  → `benchmark` (which crude oil marker, e.g. WTI, Brent)
- `spot`   → `hub` (the wholesale trading hub) + `fuel` (what's being priced there)
- `retail` → `region` (US or a PADD) + `fuel` + `formulation`

`role` records why a series is in the study, not just that it is:
- `core`        — the primary series for its slot in the transmission chain.
- `regional`    — a hub/PADD variant used for the regional-comparison stage.
- `robustness`  — spec-matched alternative (e.g. conventional vs. RBOB/all-formulations,
                   or Brent for WTI) used to check a result isn't an artifact of one
                   series definition.
- `comparison`  — a different refined product (diesel), refined from the same crude but
                   sold through a separate retail market. Not a placebo: diesel shares
                   crude with gasoline, so an asymmetry appearing here is expected to be
                   informative about how broad the pattern is, not evidence that the
                   gasoline design picks up noise. `09` finds diesel's same-week gap is
                   in fact *larger* than gasoline's.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SERIES: dict[str, dict] = {
    # --- Crude benchmarks (weekly-fri / daily pairs) ------------------------------------
    "WCOILWTICO": {"kind": "crude", "freq": "weekly-fri", "role": "core", "benchmark": "WTI", "fuel": None},
    "DCOILWTICO": {"kind": "crude", "freq": "daily", "role": "core", "benchmark": "WTI", "fuel": None},
    "WCOILBRENTEU": {"kind": "crude", "freq": "weekly-fri", "role": "robustness", "benchmark": "Brent", "fuel": None},
    "DCOILBRENTEU": {"kind": "crude", "freq": "daily", "role": "robustness", "benchmark": "Brent", "fuel": None},

    # --- Wholesale spot gasoline (weekly-fri / daily pairs) ------------------------------
    # WRGASLA prices RBOB (the LA hub's actual conventional-gasoline benchmark contract),
    # not conventional gasoline like the other two hubs — recorded via `formulation`, not
    # glossed over. `fuel` stays the coarse "gasoline" value shared with retail so that
    # filtering on fuel == "gasoline" catches every gasoline series, spot and retail alike.
    "WGASUSGULF": {"kind": "spot", "freq": "weekly-fri", "role": "core", "hub": "Gulf Coast", "fuel": "gasoline", "formulation": "conventional"},
    "DGASUSGULF": {"kind": "spot", "freq": "daily", "role": "core", "hub": "Gulf Coast", "fuel": "gasoline", "formulation": "conventional"},
    "WGASNYH": {"kind": "spot", "freq": "weekly-fri", "role": "regional", "hub": "NY Harbor", "fuel": "gasoline", "formulation": "conventional"},
    "DGASNYH": {"kind": "spot", "freq": "daily", "role": "regional", "hub": "NY Harbor", "fuel": "gasoline", "formulation": "conventional"},
    "WRGASLA": {"kind": "spot", "freq": "weekly-fri", "role": "regional", "hub": "Los Angeles", "fuel": "gasoline", "formulation": "RBOB"},
    "DRGASLA": {"kind": "spot", "freq": "daily", "role": "regional", "hub": "Los Angeles", "fuel": "gasoline", "formulation": "RBOB"},

    # --- Wholesale spot diesel (weekly-fri / daily pairs) — all comparison ---------------
    "WDFUELUSGULF": {"kind": "spot", "freq": "weekly-fri", "role": "comparison", "hub": "Gulf Coast", "fuel": "diesel"},
    "DDFUELUSGULF": {"kind": "spot", "freq": "daily", "role": "comparison", "hub": "Gulf Coast", "fuel": "diesel"},
    "WDFUELNYH": {"kind": "spot", "freq": "weekly-fri", "role": "comparison", "hub": "NY Harbor", "fuel": "diesel"},
    "DDFUELNYH": {"kind": "spot", "freq": "daily", "role": "comparison", "hub": "NY Harbor", "fuel": "diesel"},
    "WDFUELLA": {"kind": "spot", "freq": "weekly-fri", "role": "comparison", "hub": "Los Angeles", "fuel": "diesel"},
    "DDFUELLA": {"kind": "spot", "freq": "daily", "role": "comparison", "hub": "Los Angeles", "fuel": "diesel"},

    # --- Retail gasoline, regular grade, all formulations (weekly-mon) -------------------
    "GASREGW": {"kind": "retail", "freq": "weekly-mon", "role": "core", "region": "US", "fuel": "gasoline", "formulation": "all formulations"},
    "GASREGECW": {"kind": "retail", "freq": "weekly-mon", "role": "regional", "region": "PADD 1", "fuel": "gasoline", "formulation": "all formulations"},
    "GASREGMWW": {"kind": "retail", "freq": "weekly-mon", "role": "regional", "region": "PADD 2", "fuel": "gasoline", "formulation": "all formulations"},
    "GASREGGCW": {"kind": "retail", "freq": "weekly-mon", "role": "regional", "region": "PADD 3", "fuel": "gasoline", "formulation": "all formulations"},
    "GASREGRMW": {"kind": "retail", "freq": "weekly-mon", "role": "regional", "region": "PADD 4", "fuel": "gasoline", "formulation": "all formulations"},
    "GASREGWCW": {"kind": "retail", "freq": "weekly-mon", "role": "regional", "region": "PADD 5", "fuel": "gasoline", "formulation": "all formulations"},

    # --- Retail gasoline, conventional (weekly-mon) — spec-matched to the conventional spot series
    "GASREGCOVW": {"kind": "retail", "freq": "weekly-mon", "role": "robustness", "region": "US", "fuel": "gasoline", "formulation": "conventional"},

    # --- Retail diesel (weekly-mon) — all comparison. No formulation subtype applies to
    # diesel the way conventional/all-formulations does to gasoline, so `formulation`
    # is explicitly None rather than a guessed value. -------------------------------------
    "GASDESW": {"kind": "retail", "freq": "weekly-mon", "role": "comparison", "region": "US", "fuel": "diesel", "formulation": None},
    "GASDESECW": {"kind": "retail", "freq": "weekly-mon", "role": "comparison", "region": "PADD 1", "fuel": "diesel", "formulation": None},
    "GASDESMWW": {"kind": "retail", "freq": "weekly-mon", "role": "comparison", "region": "PADD 2", "fuel": "diesel", "formulation": None},
    "GASDESGCW": {"kind": "retail", "freq": "weekly-mon", "role": "comparison", "region": "PADD 3", "fuel": "diesel", "formulation": None},
    "GASDESRMW": {"kind": "retail", "freq": "weekly-mon", "role": "comparison", "region": "PADD 4", "fuel": "diesel", "formulation": None},
    "GASDESWCW": {"kind": "retail", "freq": "weekly-mon", "role": "comparison", "region": "PADD 5", "fuel": "diesel", "formulation": None},
}


# Which wholesale hub feeds each PADD's retail price, for joining retail to its matching
# spot series. PADD 2 (Midwest) and PADD 4 (Rocky Mountain) are landlocked with no FRED
# wholesale-hub series of their own — mapped to None rather than a nearby hub, because a
# substituted hub would silently misrepresent the actual local supply chain for those regions.
HUB_BY_REGION: dict[str, str | None] = {
    "PADD 1": "NY Harbor",
    "PADD 2": None,
    "PADD 3": "Gulf Coast",
    "PADD 4": None,
    "PADD 5": "Los Angeles",
}


def write_manifest_csv(path: str | Path) -> None:
    """Write `SERIES` to a CSV at `path`, one row per series ID.

    Columns are the union of every field used across all three `kind`s (`series_id`,
    `kind`, `freq`, `role`, `benchmark`, `hub`, `fuel`, `region`, `formulation`); a field
    that doesn't apply to a given series' `kind` (e.g. `benchmark` for a retail row) is
    left blank in that row rather than raising an error.
    """
    columns = ["series_id", "kind", "freq", "role", "benchmark", "hub", "fuel", "region", "formulation"]
    df = pd.DataFrame.from_dict(SERIES, orient="index").rename_axis("series_id").reset_index()
    df = df.reindex(columns=columns)
    df.to_csv(path, index=False)
