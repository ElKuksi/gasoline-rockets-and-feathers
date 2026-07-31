"""Structural tests for `series_manifest.SERIES` — no network calls.

These check the manifest is internally consistent (every entry has the fields its `kind`
promises, `freq`/`role` are drawn from the allowed sets, no series ID is accidentally
duplicated) — not that the data is *correct* against FRED. That's `test_live_fred.py`'s job.
"""

import ast
from collections import Counter
from pathlib import Path

from src.series_manifest import SERIES, HUB_BY_REGION

ALLOWED_FREQ = {"daily", "weekly-fri", "weekly-mon"}
ALLOWED_ROLE = {"core", "regional", "robustness", "placebo"}

# The fields every entry of a given `kind` must carry, beyond the common `kind`/`freq`/`role`.
REQUIRED_KEYS_BY_KIND = {
    "crude": {"kind", "freq", "role", "benchmark", "fuel"},
    "spot": {"kind", "freq", "role", "hub", "fuel"},
    "retail": {"kind", "freq", "role", "region", "fuel", "formulation"},
}


def test_every_entry_has_required_keys_for_its_kind():
    for series_id, entry in SERIES.items():
        required = REQUIRED_KEYS_BY_KIND[entry["kind"]]
        missing = required - entry.keys()
        assert not missing, f"{series_id} ({entry['kind']}) missing keys: {missing}"


def test_freq_is_one_of_the_allowed_values():
    for series_id, entry in SERIES.items():
        assert entry["freq"] in ALLOWED_FREQ, f"{series_id} has unexpected freq {entry['freq']!r}"


def test_role_is_one_of_the_allowed_values():
    for series_id, entry in SERIES.items():
        assert entry["role"] in ALLOWED_ROLE, f"{series_id} has unexpected role {entry['role']!r}"


def test_no_duplicate_series_ids():
    """A duplicated key in the `SERIES` dict literal silently overwrites the earlier entry
    at runtime — no exception, no warning, just a quietly missing series. Checking the
    *source* (via `ast`) for repeated literal keys catches that even though `SERIES` itself,
    already deduplicated by the time Python builds the dict, cannot reveal it.
    """
    import src.series_manifest as manifest_module

    source = Path(manifest_module.__file__).read_text()
    tree = ast.parse(source)

    def is_series_binding(node) -> bool:
        # `SERIES = {...}` is an ast.Assign (a list of targets); `SERIES: dict[...] = {...}`
        # (what series_manifest.py actually uses) is an ast.AnnAssign (a single target) —
        # both are handled since either form could plausibly appear here.
        if isinstance(node, ast.Assign):
            return any(getattr(t, "id", None) == "SERIES" for t in node.targets)
        if isinstance(node, ast.AnnAssign):
            return getattr(node.target, "id", None) == "SERIES"
        return False

    series_dict_node = next(node.value for node in ast.walk(tree) if is_series_binding(node))
    keys = [k.value for k in series_dict_node.keys]

    counts = Counter(keys)
    duplicates = [key for key, count in counts.items() if count > 1]
    assert not duplicates, f"duplicate series IDs in SERIES literal: {duplicates}"


def test_hub_by_region_padd2_and_padd4_are_none():
    assert HUB_BY_REGION["PADD 2"] is None
    assert HUB_BY_REGION["PADD 4"] is None


def test_fuel_has_exactly_three_values():
    assert {v.get("fuel") for v in SERIES.values()} == {"gasoline", "diesel", None}
