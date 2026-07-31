"""Tests for `fred_client.get_series`'s response-parsing logic — no network calls.

Rather than hitting FRED, these monkeypatch `fred_client._get` to return a real response
body saved to disk (`tests/fixtures/dcoilwtico_observations.json`), so the actual parsing
code (date/value conversion, the "." -> NaN rule) runs against real FRED output shape,
without a network dependency.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import fred_client

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dcoilwtico_observations.json"


@pytest.fixture
def parsed_series(monkeypatch) -> pd.DataFrame:
    """`get_series` run against the saved fixture instead of a live FRED call."""
    body = json.loads(FIXTURE_PATH.read_text())
    monkeypatch.setattr(fred_client, "_get", lambda endpoint, params: body)
    return fred_client.get_series("DCOILWTICO")


def test_value_dtype_is_float64_not_object(parsed_series):
    assert parsed_series["value"].dtype == np.float64


def test_missing_value_dot_becomes_nan(parsed_series):
    # The fixture has "." on 2019-12-25 (Christmas) and 2020-01-01 (New Year's Day).
    missing_dates = parsed_series.loc[parsed_series["value"].isna(), "date"]
    assert set(missing_dates.dt.strftime("%Y-%m-%d")) == {"2019-12-25", "2020-01-01"}


def test_dates_parse_to_datetime(parsed_series):
    assert pd.api.types.is_datetime64_any_dtype(parsed_series["date"])


def test_columns_are_exactly_date_and_value(parsed_series):
    assert list(parsed_series.columns) == ["date", "value"]


def test_no_duplicate_dates(parsed_series):
    assert not parsed_series["date"].duplicated().any()
