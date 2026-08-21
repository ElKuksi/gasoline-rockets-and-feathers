"""Interaction design matrix for testing whether pass-through behaved differently during
a specific price episode (e.g. the 2026 crude spike) than in the rest of the sample.

Builds on `src.asymmetry.build_design_matrix`'s short-run up/down decomposition by adding
an event-window dummy and its interaction with each lag, so a regression on the result can
estimate a separate coefficient for "response to a shock that arrived during the event"
versus the ordinary response the rest of the sample identifies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.asymmetry import DEFAULT_HAC_MAXLAGS, build_design_matrix, fit_distributed_lag, restriction_vector


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
    main effect, included so a plain level shift in `d_retail` during the event (a drift
    unrelated to crude) isn't forced into the interaction coefficients and misread as
    changed pass-through. `fit_distributed_lag` picks it up automatically along with the
    interaction columns, but a caller should confirm `event` actually appears in
    `res.params` rather than assume it.

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


def test_event_interaction_joint(res) -> dict:
    """Joint F-test that every `*_event` interaction coefficient in `res` is zero.

    Interaction names are discovered from `res.params.index` (suffix `_event`, excluding
    the bare `event` main effect) rather than reconstructed from K, so this works
    unchanged whether `res` came from Spec A (10 interaction columns) or Spec B (4) --
    a K-derived list would test coefficients Spec B doesn't have.

    This is the gatekeeper before reading any individual delta coefficient: an omnibus
    test with lower power than `test_event_asymmetry_change`, but reading individual
    coefficients without it reintroduces the multiple-comparisons problem.

    Returns
    -------
    dict with `f_stat`, `p_value`, `df_num`, `df_denom`, `n_restrictions`, and
    `interaction_names` (the parameter names actually tested).
    """
    interaction_names = [name for name in res.params.index if name.endswith("_event") and name != "event"]
    restrictions = np.vstack([restriction_vector(res, positive_names={name}) for name in interaction_names])
    f_test = res.f_test(restrictions)

    return {
        "f_stat": float(np.asarray(f_test.fvalue).item()),
        "p_value": float(np.asarray(f_test.pvalue).item()),
        "df_num": float(f_test.df_num),
        "df_denom": float(f_test.df_denom),
        "n_restrictions": len(interaction_names),
        "interaction_names": interaction_names,
    }


def test_event_asymmetry_change(res) -> dict:
    """Test whether the up-versus-down gap itself changed during the event: `+1` on
    every `d_up_lag{k}_event`, `-1` on every `d_down_lag{k}_event`, `0` elsewhere, via
    `res.t_test` -- same technique as `src.asymmetry.test_asymmetry` and for the same
    reason, the interaction coefficients are correlated, so summing their standard
    errors would drop the covariance terms.

    Sign convention: a **positive** estimate means the up-versus-down gap was *wider*
    during the event than normally -- rockets and feathers intensified under stress. A
    **negative** estimate means the gap *narrowed* -- pass-through became more
    symmetric. Neither is the expected answer.

    Returns
    -------
    dict with `estimate` (Sum(delta+) - Sum(delta-)), `se`, `ci_lo`, `ci_hi`, `p_value`.
    """
    up_names = {name for name in res.params.index if name.startswith("d_up_lag") and name.endswith("_event")}
    down_names = {name for name in res.params.index if name.startswith("d_down_lag") and name.endswith("_event")}
    restriction = restriction_vector(res, positive_names=up_names, negative_names=down_names)

    t = res.t_test(restriction)
    ci = np.asarray(t.conf_int())

    return {
        "estimate": float(np.asarray(t.effect).item()),
        "se": float(np.asarray(t.sd).item()),
        "ci_lo": float(ci[0, 0]),
        "ci_hi": float(ci[0, 1]),
        "p_value": float(np.asarray(t.pvalue).item()),
    }


def cumulative_passthrough_by_regime(res, K: int) -> pd.DataFrame:
    """Cumulative pass-through through each horizon h = 0..K, separately for normal
    weeks and event weeks: `cum_up_normal(h) = Sum_{k<=h} beta+_k`,
    `cum_up_event(h) = Sum_{k<=h} (beta+_k + delta+_k)`, and the two down equivalents --
    each via `res.t_test` on a restriction vector, for the same correlated-coefficients
    reason `src.asymmetry.cumulative_passthrough` uses it rather than summing SEs by
    hand.

    An event-lag `k` with no `d_up_lag{k}_event` / `d_down_lag{k}_event` column in
    `res.params` (e.g. Spec B beyond lag 1) contributes no delta term at that lag, so
    `cum_*_event` and `cum_*_normal` converge past the last interacted lag by
    construction -- expected, not a finding.

    Parameters
    ----------
    res : a fitted model with `d_up_lag0..K` / `d_down_lag0..K` and some subset of
        `d_up_lag{k}_event` / `d_down_lag{k}_event`.
    K : the largest baseline lag present in `res`.

    Returns
    -------
    DataFrame with one row per horizon 0..K: `horizon`, `cum_up_normal`,
    `cum_up_normal_lo`, `cum_up_normal_hi`, `cum_up_event`, `cum_up_event_lo`,
    `cum_up_event_hi`, and the four `cum_down_*` equivalents (95% CI bounds throughout).
    """
    rows = []
    for h in range(K + 1):
        up_base = {f"d_up_lag{k}" for k in range(h + 1)}
        down_base = {f"d_down_lag{k}" for k in range(h + 1)}
        up_event = {name for k in range(h + 1) if (name := f"d_up_lag{k}_event") in res.params.index}
        down_event = {name for k in range(h + 1) if (name := f"d_down_lag{k}_event") in res.params.index}

        t_up_normal = res.t_test(restriction_vector(res, positive_names=up_base))
        t_up_event = res.t_test(restriction_vector(res, positive_names=up_base | up_event))
        t_down_normal = res.t_test(restriction_vector(res, positive_names=down_base))
        t_down_event = res.t_test(restriction_vector(res, positive_names=down_base | down_event))

        ci_up_normal = np.asarray(t_up_normal.conf_int())
        ci_up_event = np.asarray(t_up_event.conf_int())
        ci_down_normal = np.asarray(t_down_normal.conf_int())
        ci_down_event = np.asarray(t_down_event.conf_int())

        rows.append(
            {
                "horizon": h,
                "cum_up_normal": float(np.asarray(t_up_normal.effect).item()),
                "cum_up_normal_lo": float(ci_up_normal[0, 0]),
                "cum_up_normal_hi": float(ci_up_normal[0, 1]),
                "cum_up_event": float(np.asarray(t_up_event.effect).item()),
                "cum_up_event_lo": float(ci_up_event[0, 0]),
                "cum_up_event_hi": float(ci_up_event[0, 1]),
                "cum_down_normal": float(np.asarray(t_down_normal.effect).item()),
                "cum_down_normal_lo": float(ci_down_normal[0, 0]),
                "cum_down_normal_hi": float(ci_down_normal[0, 1]),
                "cum_down_event": float(np.asarray(t_down_event.effect).item()),
                "cum_down_event_lo": float(ci_down_event[0, 0]),
                "cum_down_event_hi": float(ci_down_event[0, 1]),
            }
        )

    return pd.DataFrame(rows)


def run_event_study(
    retail: pd.DataFrame,
    upstream: pd.DataFrame,
    K: int,
    event_start,
    event_end,
    label: str,
    interact_lags: list[int] | None = None,
    maxlags: int = DEFAULT_HAC_MAXLAGS,
) -> dict:
    """Build the interaction design, fit it, and run the joint and asymmetry-change tests
    for one window/spec combination -- composition of `build_event_interaction_design`,
    `fit_distributed_lag`, `test_event_interaction_joint`, and `test_event_asymmetry_change`,
    no new statistics. Exists so a notebook comparing several windows (e.g. 2026 against
    placebo episodes) loops over this once instead of repeating the same four calls per
    window.

    Parameters
    ----------
    retail, upstream, K, event_start, event_end, interact_lags : passed straight through
        to `build_event_interaction_design`.
    label : a name for this window/spec combination, carried through into the result so a
        list of these dicts can be turned into a table without re-deriving it.
    maxlags : passed straight through to `fit_distributed_lag`.

    Returns
    -------
    dict with `label`, `n_event`, `n_up`, `n_down` (the event window's size and up/down
    split), `joint` (`test_event_interaction_joint`'s result), `asymmetry_change`
    (`test_event_asymmetry_change`'s result), and `res` (the fitted model, unwrapped, for
    anything else a caller needs from it).
    """
    design = build_event_interaction_design(
        retail, upstream, K=K, event_start=event_start, event_end=event_end, interact_lags=interact_lags
    )
    res = fit_distributed_lag(design, maxlags=maxlags)

    n_up = int(((design["event"] == 1) & (design["d_up_lag0"] > 0)).sum())
    n_down = int(((design["event"] == 1) & (design["d_down_lag0"] < 0)).sum())

    return {
        "label": label,
        "n_event": int(design["event"].sum()),
        "n_up": n_up,
        "n_down": n_down,
        "joint": test_event_interaction_joint(res),
        "asymmetry_change": test_event_asymmetry_change(res),
        "res": res,
    }
