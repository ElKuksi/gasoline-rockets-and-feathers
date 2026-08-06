"""Point-in-time alignment of a retail price series to an upstream (crude or spot) series.

This is the **only** place in this codebase a retail row is ever joined to an upstream
row — per ADR-0002 in `decisions.md`. No other join of retail to upstream should exist
anywhere else, notebooks included; route every such join through
`align_retail_to_upstream` so the look-ahead guard below always runs.

Background (ADR-0002): retail is a Monday ~08:00 survey snapshot. The weekly upstream
series is Friday-dated, and its value is the *mean of that week's Monday-through-Friday
daily prices* — so naively pairing a retail Monday with the upstream Friday of its own
week bakes 4 days of future upstream data into that row (the Monday itself plus
Tue/Wed/Thu/Fri, none of which had happened yet at 08:00 that Monday). ADR-0002 measured
this directly: same-week alignment gives corr(delta_retail, delta_crude) = 0.209; the
correct prior-observation alignment gives 0.591. The leak doesn't just add noise — it
shifts the entire estimated lag structure by a week, which is fatal for a study that is
specifically about the timing of pass-through.
"""

from __future__ import annotations

import pandas as pd

_VALID_MODES = {"weekly", "daily_pit"}
_MAX_PRINTED_ROWS = 20  # keep a raised error's row dump readable, not a wall of text


def _format_offending_rows(df: pd.DataFrame) -> str:
    """Render up to `_MAX_PRINTED_ROWS` offending rows for an error message."""
    if df.empty:
        return "(no rows)"
    if len(df) > _MAX_PRINTED_ROWS:
        shown = df.head(_MAX_PRINTED_ROWS).to_string()
        return f"{shown}\n... ({len(df) - _MAX_PRINTED_ROWS} more rows not shown, {len(df)} total)"
    return df.to_string()


def _check_no_look_ahead(merged: pd.DataFrame) -> None:
    """Guard 1 (ADR-0002's mandatory guardrail): every row's `upstream_date` must be
    strictly before its retail `date`.

    This is the exact invariant the whole module exists to enforce — a violation means an
    upstream observation from on-or-after the survey date entered that row, the same
    look-ahead bug ADR-0002 measured and fixed. `direction="backward"` in `merge_asof`
    should make this structurally impossible to violate; the guard exists anyway because,
    per ADR-0002, a safety check never observed to fire is not known to work.

    A row where `merge_asof` found no eligible upstream observation at all (e.g. a retail
    date earlier than the upstream slice's coverage) comes back with `upstream_date =
    NaT`. `NaT < date` evaluates to `False` in pandas, so that row is caught here too —
    correctly: a retail date with no valid point-in-time upstream match is a real
    problem, not something that should silently pass through as a row full of NaN.
    """
    is_strictly_before = merged["upstream_date"] < merged["date"]
    bad = merged[~is_strictly_before]
    if not bad.empty:
        raise ValueError(
            "point_in_time.align_retail_to_upstream: found rows where upstream_date is "
            "not strictly before the retail date — either a look-ahead leak or no "
            f"upstream match at all (NaT):\n{_format_offending_rows(bad[['date', 'upstream_date']])}"
        )


def _check_no_duplicate_upstream_date(merged: pd.DataFrame) -> None:
    """Guard 2: no `upstream_date` may be reused across more than one retail row.

    Two retail Mondays landing on the same upstream observation would carry an identical
    `value_upstream`. Since this alignment exists to feed first-differenced series
    downstream (Stage 3+), an identical `value_upstream` on two rows produces a
    first-difference of exactly zero — a fake "no change" observation that never
    happened, not a real one.
    """
    duplicated_mask = merged["upstream_date"].duplicated(keep=False)
    bad = merged[duplicated_mask].sort_values("upstream_date")
    if not bad.empty:
        raise ValueError(
            "point_in_time.align_retail_to_upstream: found upstream_date values shared "
            f"by more than one retail row:\n{_format_offending_rows(bad[['date', 'upstream_date']])}"
        )


def _check_gap_days_in_range(merged: pd.DataFrame) -> None:
    """Guard 3: `gap_days` must be between 1 and 10.

    Below 1 restates guard 1 (a gap of 0 or negative days means `upstream_date` isn't
    strictly before `date`); `NaN` (from a `NaT` `upstream_date`) is caught the same way,
    explicitly, so this guard is correct standing alone and not merely because guard 1
    already ran first. Above 10 means the upstream series has a hole wider than any
    ordinary market closure explains — a run of missing upstream data or a calendar bug,
    not a normal weekend/holiday gap. An ordinary week gives `gap_days = 3` (Fri -> Mon);
    a holiday shifting the upstream date back by one trading day (Good Friday is the
    recurring case) gives 4 — both comfortably inside [1, 10].
    """
    gap = merged["gap_days"]
    bad = merged[(gap < 1) | (gap > 10) | gap.isna()]
    if not bad.empty:
        raise ValueError(
            "point_in_time.align_retail_to_upstream: found rows with gap_days outside "
            f"[1, 10]:\n{_format_offending_rows(bad[['date', 'upstream_date', 'gap_days']])}"
        )


def align_retail_to_upstream(retail: pd.DataFrame, upstream: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Join each retail observation to **the most recent upstream observation strictly
    before the survey date** — never the same-week Friday, never the survey date itself.

    That phrasing is deliberate (see ADR-0002 in `decisions.md`, which this function
    implements): describing the rule as "the Friday of the previous week" is *usually*
    the same thing, but the two descriptions diverge on a market holiday. Good Friday is
    the recurring case — no upstream observation is dated that Friday at all, so "most
    recent observation strictly before the survey date" correctly falls back to Thursday
    (or further), while "the Friday of the previous week" describes a date that doesn't
    exist in the data. The code below implements the first description; anyone defending
    this function by the second description is describing a bug it doesn't have.

    Parameters
    ----------
    retail : DataFrame with `date` (datetime64, every value a Monday) and `value`,
        sorted ascending by `date`.
    upstream : DataFrame with `date` (datetime64) and `value`, sorted ascending by
        `date` — either the weekly (Friday-dated) or daily crude/spot series, matching
        `mode`.
    mode : `"weekly"` — `upstream` is the weekly series. Each retail Monday pairs with
        the most recent weekly-Friday strictly before it; the same-week Friday is
        *later* than that Monday (it's the Friday *after*, not before), so a
        backward-direction `merge_asof` always skips over it without any extra logic.
        `"daily_pit"` — `upstream` is the daily series. Same rule, plus
        `allow_exact_matches=False`: the daily series has a row for every trading
        weekday, including Mondays, so without this a retail Monday could pair with a
        same-day daily close that hadn't happened yet when the 08:00 survey was taken.
        With it, that row is excluded and the join falls back to the prior trading day's
        close instead.

    Returns
    -------
    DataFrame: `retail`'s columns (`value` -> `value_retail`), `upstream`'s columns
    renamed (`date` -> `upstream_date`, `value` -> `value_upstream`), plus `gap_days`
    (`(date - upstream_date).days`).

    Raises
    ------
    ValueError
        If `mode` isn't `"weekly"` or `"daily_pit"`, or if any of the three ADR-0002
        guards fails (see `_check_no_look_ahead`, `_check_no_duplicate_upstream_date`,
        `_check_gap_days_in_range`) — each raises with the actual offending rows in the
        message, not just a bare failure.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")

    # Rename upstream's date column before merging (rather than pd.merge_asof(..., on="date")
    # as ADR-0002 sketches it) specifically so upstream_date survives the merge as its own
    # output column — on="date" would leave only one shared "date" column, with no way to
    # tell, after the fact, which upstream observation actually got matched to which row.
    upstream_renamed = upstream.rename(columns={"date": "upstream_date"})

    # allow_exact_matches=False only for daily_pit: the weekly series is Friday-dated and a
    # retail Monday can never equal a Friday date, so the flag would be a no-op there — it's
    # the daily series, which has a row on every weekday including Mondays, where a same-day
    # match is actually possible and needs to be excluded.
    allow_exact_matches = mode != "daily_pit"

    merged = pd.merge_asof(
        retail,
        upstream_renamed,
        left_on="date",
        right_on="upstream_date",
        direction="backward",
        allow_exact_matches=allow_exact_matches,
        suffixes=("_retail", "_upstream"),
    )

    merged["gap_days"] = (merged["date"] - merged["upstream_date"]).dt.days

    _check_no_look_ahead(merged)
    _check_no_duplicate_upstream_date(merged)
    _check_gap_days_in_range(merged)

    return merged
