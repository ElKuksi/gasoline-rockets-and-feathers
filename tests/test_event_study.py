"""Tests for `src.event_study.build_event_interaction_design` -- hermetic, hand-built
fixtures, no network, no reads from `data/`. Each test targets one specific claim from the
function's docstring: the dummy is lagged with the regressor (not the response week), the
interaction columns are exactly zero outside the lagged window, the up/down decomposition
identity survives the interaction, non-contiguous input is rejected, and a restricted
`interact_lags` produces exactly the requested columns.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.event_study import build_event_interaction_design


def _weekly_fixture(n_mondays: int, prices: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`n_mondays` consecutive retail Mondays starting 2024-01-01, each paired with the
    Friday of the prior week -- exactly what `align_retail_to_upstream("weekly")`
    expects -- upstream priced by `prices` (one value per Friday, in date order).
    """
    mondays = pd.date_range("2024-01-01", periods=n_mondays, freq="7D")
    fridays = mondays - pd.Timedelta(days=3)
    retail = pd.DataFrame({"date": mondays, "value": [1.0] * n_mondays})  # retail level unused by this module
    upstream = pd.DataFrame({"date": fridays, "value": prices[:n_mondays]})
    return retail, upstream


def test_interaction_lags_the_dummy_with_the_regressor():
    """The load-bearing test: `d_up_lag2_event` must be non-zero exactly on the rows TWO
    WEEKS AFTER the event window, not on the window rows themselves. This is the test
    that fails if `event[t-k]` is simplified to `event[t]`.
    """
    # 13 Mondays, monotonically rising upstream price (+1/week) so every week has a real
    # up-move and d_up_lag2 is nonzero on every surviving row -- isolating the shift
    # logic from "does this week even have a move."
    retail, upstream = _weekly_fixture(13, [10.0 + i for i in range(13)])

    design = build_event_interaction_design(retail, upstream, K=2, event_start="2024-02-05", event_end="2024-02-12")

    # K=2 drops the first 3 rows (diff + 2 lags); 10 rows remain, 2024-01-22..2024-03-25.
    assert list(design["date"]) == list(pd.date_range("2024-01-22", periods=10, freq="7D"))

    expected_nonzero = pd.to_datetime(["2024-02-19", "2024-02-26"])  # window shifted 2 weeks later
    actual_nonzero = design.loc[design["d_up_lag2_event"] != 0, "date"]
    assert list(actual_nonzero) == list(expected_nonzero)


def test_interaction_zero_outside_window():
    """Every `*_event` column must be exactly 0 wherever that lag's shifted dummy is 0,
    checked against an independently recomputed shifted dummy for every lag.
    """
    retail, upstream = _weekly_fixture(13, [10.0 + i for i in range(13)])
    design = build_event_interaction_design(retail, upstream, K=2, event_start="2024-02-05", event_end="2024-02-12")

    event = design["event"]
    for k in (0, 1, 2):
        outside = event.shift(k).fillna(0) == 0
        assert (design.loc[outside, f"d_up_lag{k}_event"] == 0).all()
        assert (design.loc[outside, f"d_down_lag{k}_event"] == 0).all()


def test_decomposition_identity_survives_interaction():
    """`d_up_lag{k}_event + d_down_lag{k}_event` must equal the raw (unsplit) Δupstream
    at lag k, times that lag's shifted event dummy -- the up/down decomposition identity,
    carried through the interaction.
    """
    # Alternating deltas so both up and down weeks are exercised, not just up.
    prices = [10.0]
    for i in range(12):
        prices.append(prices[-1] + (1.0 if i % 2 == 0 else -0.5))
    retail, upstream = _weekly_fixture(13, prices)

    design = build_event_interaction_design(retail, upstream, K=2, event_start="2024-02-05", event_end="2024-02-19")

    event = design["event"]
    for k in (0, 1, 2):
        raw_delta_lag_k = design[f"d_up_lag{k}"] + design[f"d_down_lag{k}"]
        event_lag = event.shift(k).fillna(0)
        lhs = design[f"d_up_lag{k}_event"] + design[f"d_down_lag{k}_event"]
        rhs = raw_delta_lag_k * event_lag
        assert (lhs - rhs).abs().max() < 1e-10


def test_non_contiguous_dates_raise():
    """A design with a missing week (a real gap between two retail Mondays) must raise --
    per ADR-0002's standing rule, a safety check never observed to fire is not known to
    work.
    """
    retail, upstream = _weekly_fixture(13, [10.0 + i for i in range(13)])
    retail = retail[retail["date"] != pd.Timestamp("2024-02-12")].reset_index(drop=True)  # drop one Monday

    with pytest.raises(ValueError, match="not contiguous weekly"):
        build_event_interaction_design(retail, upstream, K=2, event_start="2024-02-05", event_end="2024-02-12")


def test_interact_lags_subset():
    """With `interact_lags=[0, 1]` and `K=4`, exactly four interaction columns exist and
    `d_up_lag2_event` (a lag outside the restricted set) is absent entirely.
    """
    retail, upstream = _weekly_fixture(12, [10.0 + i for i in range(12)])

    design = build_event_interaction_design(
        retail, upstream, K=4, event_start="2024-02-19", event_end="2024-02-26", interact_lags=[0, 1]
    )

    interaction_cols = [c for c in design.columns if c.endswith("_event") and c != "event"]
    assert sorted(interaction_cols) == ["d_down_lag0_event", "d_down_lag1_event", "d_up_lag0_event", "d_up_lag1_event"]
    assert "d_up_lag2_event" not in design.columns
