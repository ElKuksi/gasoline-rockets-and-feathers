"""Integration tests against the live FRED API — excluded from the default `pytest` run
(see the `integration` marker registered in `pyproject.toml`); run explicitly with
`pytest -m integration`. These check real data against known values, not just plumbing.
"""

import pytest

from src.fred_client import get_series, get_series_info
from src.series_manifest import SERIES

# frequency_short is FRED's reliable, consistent code (D/W/M/...); every manifest freq
# maps to exactly one of these.
_FRED_FREQUENCY_SHORT_BY_MANIFEST_FREQ = {"daily": "D", "weekly-fri": "W", "weekly-mon": "W"}

# For the two weekly manifest freqs, which weekday (Monday=0 .. Sunday=6) their observations
# should land on.
_EXPECTED_WEEKDAY_BY_MANIFEST_FREQ = {"weekly-fri": 4, "weekly-mon": 0}


@pytest.mark.integration
def test_wcoilwtico_known_values():
    df = get_series("WCOILWTICO", start="2026-04-24", end="2026-05-01")
    values = df.set_index(df["date"].dt.strftime("%Y-%m-%d"))["value"]
    assert values["2026-05-01"] == pytest.approx(105.57)
    assert values["2026-04-24"] == pytest.approx(95.43)


@pytest.mark.integration
def test_gasregw_and_gasdesw_known_values():
    gas = get_series("GASREGW", start="2026-06-08", end="2026-06-08")
    diesel = get_series("GASDESW", start="2026-06-08", end="2026-06-08")
    assert gas["value"].iloc[0] == pytest.approx(4.146)
    assert diesel["value"].iloc[0] == pytest.approx(5.210)


@pytest.mark.integration
def test_every_wcoilwtico_date_is_a_friday():
    df = get_series("WCOILWTICO")
    non_fridays = df.loc[df["date"].dt.dayofweek != 4, "date"]
    assert non_fridays.empty, f"non-Friday WCOILWTICO dates: {list(non_fridays)}"


@pytest.mark.integration
def test_every_gasregw_date_is_a_monday():
    df = get_series("GASREGW")
    non_mondays = df.loc[df["date"].dt.dayofweek != 0, "date"]
    assert non_mondays.empty, f"non-Monday GASREGW dates: {list(non_mondays)}"


@pytest.mark.integration
def test_every_manifest_series_resolves_with_matching_frequency():
    """Every manifest series ID must resolve on FRED, and FRED's reported cadence must
    match what the manifest claims.

    FRED's `frequency_short` code (D/W/...) is checked first — it's a consistent,
    reliable field. For the weekly series, `freq` further claims a specific weekday
    (Friday vs. Monday); FRED's human-readable `frequency` string usually spells that
    out too (e.g. "Weekly, Ending Friday"), but it's not reliable — `GASREGCOVW` is
    labeled bare "Weekly" despite its data landing on Mondays exactly like its sibling
    series. So the weekday claim is checked against one real observation's actual date
    instead of trusting that label.
    """
    mismatches = []
    for series_id, entry in SERIES.items():
        info = get_series_info(series_id)  # raises RuntimeError if the ID doesn't resolve

        expected_short = _FRED_FREQUENCY_SHORT_BY_MANIFEST_FREQ[entry["freq"]]
        if info["frequency_short"] != expected_short:
            mismatches.append((series_id, entry["freq"], "frequency_short", expected_short, info["frequency_short"]))
            continue

        expected_weekday = _EXPECTED_WEEKDAY_BY_MANIFEST_FREQ.get(entry["freq"])
        if expected_weekday is not None:
            latest = get_series(series_id, start=info["observation_end"], end=info["observation_end"])
            actual_weekday = latest["date"].iloc[0].dayofweek
            if actual_weekday != expected_weekday:
                mismatches.append((series_id, entry["freq"], "weekday", expected_weekday, actual_weekday))

    assert not mismatches, (
        "manifest freq vs FRED-reported cadence mismatches "
        "(series_id, manifest freq, check, expected, actual): "
        f"{mismatches}"
    )
