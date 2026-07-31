"""Thin client for the subset of the FRED (Federal Reserve Economic Data) API this project
needs: pulling one series' observations, and listing every series in a release.

API docs: https://fred.stlouisfed.org/docs/api/fred/
"""

from __future__ import annotations

import os

import pandas as pd
import requests
from dotenv import load_dotenv

_BASE_URL = "https://api.stlouisfed.org/fred"
_TIMEOUT_SECONDS = 30
_PAGE_LIMIT = 1000  # FRED's documented maximum for `limit` on paginated endpoints


def _api_key() -> str:
    """Read the FRED API key from the environment.

    Loads `.env` (via python-dotenv, if present) then reads `FRED_API_KEY`. Raises a
    `RuntimeError` with a clear, actionable message if the key is missing. There is
    nothing to leak when the key is absent, but by convention this function's error
    message — and every error message in this module — never interpolates an actual
    key value, so callers can safely print or log any exception raised here.
    """
    load_dotenv()
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free key: https://fred.stlouisfed.org/docs/api/api_key.html)."
        )
    return key


def _get(endpoint: str, params: dict) -> dict:
    """Call a FRED endpoint and return the parsed JSON body.

    Attaches the API key and `file_type=json` to every request, applies a timeout, and
    translates transport/HTTP errors into a `RuntimeError`.

    The API key travels as a query parameter, and `requests` embeds the *full request
    URL* (query string included) in its exception objects and in Python's default
    chained-traceback output. So on failure this deliberately does not re-raise or
    stringify the original exception — it builds a new message from only the endpoint
    name and HTTP status code, and raises `from None` to suppress the automatic
    exception chaining that would otherwise print the original, key-bearing exception.
    """
    query = {**params, "api_key": _api_key(), "file_type": "json"}
    try:
        response = requests.get(f"{_BASE_URL}/{endpoint}", params=query, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else "no response"
        raise RuntimeError(f"FRED request to '{endpoint}' failed ({type(exc).__name__}, status={status}).") from None
    return response.json()


def get_series(series_id: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch one FRED series' observations as a tidy DataFrame.

    Parameters
    ----------
    series_id : the FRED series ID, e.g. "WCOILWTICO".
    start, end : optional "YYYY-MM-DD" bounds (inclusive), passed through to FRED's
        `observation_start` / `observation_end`. Omit either to get FRED's full history
        on that side.

    Returns
    -------
    DataFrame with exactly two columns, `date` (datetime64[ns]) and `value` (float64),
    one row per observation, in the chronological order FRED returns them.

    FRED encodes a missing observation as the literal string "." rather than omitting
    the row. `pd.to_numeric(..., errors="coerce")` turns that (and anything else
    non-numeric) into NaN, so `value` ends up float64 rather than a mixed-type object
    column with "." sitting in it.
    """
    params: dict = {"series_id": series_id}
    if start is not None:
        params["observation_start"] = start
    if end is not None:
        params["observation_end"] = end

    body = _get("series/observations", params)
    df = pd.DataFrame(body["observations"])[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def list_release_series(release_id: int) -> pd.DataFrame:
    """List every series belonging to a FRED release.

    The `release/series` endpoint caps each response at `_PAGE_LIMIT` series (FRED's
    documented maximum is 1000) and reports the total series count in the response
    body. This loops on `offset`, accumulating pages, until every series has been
    collected — a release can hold far more series than fit in one page, so a single
    call is never assumed to be complete.

    Returns
    -------
    DataFrame with columns `series_id`, `title`, `frequency`, `observation_start`,
    `observation_end` — one row per series, in whatever order FRED returns them
    (unsorted; callers sort as needed).
    """
    all_series: list[dict] = []
    offset = 0
    while True:
        body = _get("release/series", {"release_id": release_id, "limit": _PAGE_LIMIT, "offset": offset})
        page = body["seriess"]  # FRED's actual (misspelled) key for this list — not "series"
        all_series.extend(page)
        offset += len(page)
        if not page or offset >= body["count"]:
            break

    columns = ["series_id", "title", "frequency", "observation_start", "observation_end"]
    if not all_series:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(all_series).rename(columns={"id": "series_id"})
    return df[columns]
