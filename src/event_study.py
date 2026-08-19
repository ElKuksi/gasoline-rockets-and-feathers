"""Interaction design matrix for testing whether pass-through behaved differently during
a specific price episode (e.g. the 2026 crude spike) than in the rest of the sample.

Builds on `src.asymmetry.build_design_matrix`'s short-run up/down decomposition by adding
an event-window dummy and its interaction with each lag, so a regression on the result can
estimate a separate coefficient for "response to a shock that arrived during the event"
versus the ordinary response the rest of the sample identifies.
"""

from __future__ import annotations

import pandas as pd

from src.asymmetry import build_design_matrix


def build_event_interaction_design(
    retail: pd.DataFrame,
    upstream: pd.DataFrame,
    K: int,
    event_start,
    event_end,
    mode: str = "weekly",
    interact_lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add an event-window dummy and its lagged interactions to `build_design_matrix`'s
    short-run design.

    `event` is 1 for rows whose `date` falls in `[event_start, event_end]`, else 0 — the
    main effect, included so a plain level shift in `d_retail` during the event isn't
    forced into the interaction coefficients (see `fit`'s caller for why that matters).

    For each lag `k` in `interact_lags` (default `0..K`), `d_up_lag{k}_event` and
    `d_down_lag{k}_event` are `d_up_lag{k}` / `d_down_lag{k}` multiplied by `event`
    shifted `k` weeks: `event[t - k]`, not `event[t]`. `d_up_lag{k}[t]` already *is* the
    crude shock from `k` weeks ago, so the question of whether that shock arrived during
    the event has to be asked about week `t - k`, not week `t`. Indexing by `event[t]`
    instead would tag an ordinary pre-event shock as an event shock merely because its
    delayed response happened to land inside the window, and the resulting coefficient
    would no longer mean "extra response to an event-period shock."

    Concretely: `d_up_lag3_event` marks rows whose crude move landed inside the window
    three weeks earlier, even where the retail response itself falls outside it.
    `tests/test_event_study.py::test_interaction_lags_the_dummy_with_the_regressor`
    pins this.

    This only produces the right rows if the design matrix has no missing weeks between
    them — `event.shift(k)` walks back `k` *rows*, which is only the same thing as `k`
    *weeks* if every row is exactly 7 days after the previous one. That's checked
    explicitly below and raises rather than assumed, the same guard philosophy as
    `src.point_in_time`'s alignment checks: not expected to ever fire on real data, but a
    silent gap would misalign every interaction column past it without any other symptom.

    Parameters
    ----------
    retail, upstream, K, mode : passed straight through to `build_design_matrix`.
    event_start, event_end : inclusive window bounds, anything `pd.Timestamp` accepts.
    interact_lags : which lags get an interaction column. Defaults to every lag `0..K`.
        A restricted list (e.g. `[0, 1]`) fits fewer interaction parameters against the
        same number of event-window observations — useful when the window is short
        relative to K.

    Returns
    -------
    `build_design_matrix`'s output, plus `event` and, for each `k` in `interact_lags`,
    `d_up_lag{k}_event` / `d_down_lag{k}_event`.
    """
    design = build_design_matrix(retail, upstream, K, mode=mode).copy()

    gaps = design["date"].diff()
    non_weekly = gaps.iloc[1:] != pd.Timedelta(days=7)
    if non_weekly.any():
        offending_dates = design.loc[non_weekly[non_weekly].index, "date"].tolist()
        raise ValueError(
            "build_event_interaction_design: design dates are not contiguous weekly "
            f"observations -- gap before these dates is not 7 days: {offending_dates}"
        )

    if interact_lags is None:
        interact_lags = list(range(K + 1))

    event_start = pd.Timestamp(event_start)
    event_end = pd.Timestamp(event_end)
    event = ((design["date"] >= event_start) & (design["date"] <= event_end)).astype(int)
    design["event"] = event

    for k in interact_lags:
        event_lag = event.shift(k).fillna(0).astype(int)
        up_col, down_col = f"d_up_lag{k}_event", f"d_down_lag{k}_event"
        design[up_col] = design[f"d_up_lag{k}"] * event_lag
        design[down_col] = design[f"d_down_lag{k}"] * event_lag

    n_window = int(event.sum())
    n_up = int(((design["event"] == 1) & (design["d_up_lag0"] > 0)).sum())
    n_down = int(((design["event"] == 1) & (design["d_down_lag0"] < 0)).sum())

    sorted_lags = sorted(interact_lags)
    if len(sorted_lags) > 1 and sorted_lags == list(range(sorted_lags[0], sorted_lags[-1] + 1)):
        lag_label = f"{sorted_lags[0]}-{sorted_lags[-1]}"
    else:
        lag_label = ",".join(str(k) for k in sorted_lags)

    lag_flag_counts = {
        k: int(((design[f"d_up_lag{k}_event"] != 0) | (design[f"d_down_lag{k}_event"] != 0)).sum()) for k in interact_lags
    }
    deviating = {k: c for k, c in lag_flag_counts.items() if c != n_window}
    status = f"all lags flag {n_window}" if not deviating else f"lag counts vary (expected {n_window})"

    print(
        f"build_event_interaction_design: {n_window} event rows ({n_up} up, {n_down} down), "
        f"interactions at lags {lag_label}, {status}"
    )
    for k, c in deviating.items():
        print(f"  ! lag {k}: {c} flagged rows (expected {n_window})")

    return design
