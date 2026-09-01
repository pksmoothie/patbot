from __future__ import annotations

import pandas as pd

from patbot.final_call import candidate_shortlist, run_final_call
from patbot.final_call_stability import _checkpoint_futility_reason


class FakeEngine:
    def __init__(self, config=None):
        self.config = config or {
            "simulation": {"through_round": 8},
            "final_call": {
                "min_candidates": 3,
                "max_candidates": 4,
                "score_gap": 8.0,
                "initial_runs": 30,
                "refine_runs": 100,
                "stability_checkpoint_runs": 200,
                "stability_runs": 500,
                "overturn_probe_margin": 2.5,
                "overturn_required_margin": 10.0,
                "overturn_min_paired_win_pct": 55.0,
                "overturn_require_positive_ci": True,
                "future_rounds": 3,
                "max_sim_round": 13,
                "bypass_round": 14,
            },
        }
        self.league = {"teams": 12}


def board_frame():
    return pd.DataFrame(
        [
            {"player_id": "a", "name": "Alpha", "score": 90.0, "proj_points": 300.0, "adp": 10.0},
            {"player_id": "b", "name": "Beta", "score": 87.0, "proj_points": 295.0, "adp": 11.0},
            {"player_id": "c", "name": "Gamma", "score": 84.0, "proj_points": 290.0, "adp": 12.0},
            {"player_id": "d", "name": "Delta", "score": 70.0, "proj_points": 280.0, "adp": 13.0},
        ]
    )


def _details(summary):
    lookup = {"Alpha": "a", "Beta": "b", "Gamma": "c", "Delta": "d"}
    return [
        {"candidate": name, "candidate_id": lookup[name]}
        for name in summary["Candidate"].tolist()
    ]


def _screen(*, beta: float, alpha: float, gamma: float = 400.0):
    summary = pd.DataFrame(
        [
            {"Candidate": "Beta", "Avg Lineup Score": beta},
            {"Candidate": "Alpha", "Avg Lineup Score": alpha},
            {"Candidate": "Gamma", "Avg Lineup Score": gamma},
        ]
    ).sort_values("Avg Lineup Score", ascending=False).reset_index(drop=True)
    return summary, _details(summary)


def _stability_result(
    *,
    challenger_avg: float,
    base_avg: float,
    paired_mean: float,
    win_pct: float,
    ci_low: float,
    ci_high: float,
    runs: int,
    stop_stage: str = "full",
    stop_reason: str = "",
):
    summary = pd.DataFrame(
        [
            {
                "Candidate": "Beta",
                "Avg Lineup Score": challenger_avg,
                "10th %ile": challenger_avg - 40,
                "25th %ile": challenger_avg - 20,
                "75th %ile": challenger_avg + 20,
                "90th %ile": challenger_avg + 40,
                "League Winner Score": 60.0,
                "Runs": runs,
            },
            {
                "Candidate": "Alpha",
                "Avg Lineup Score": base_avg,
                "10th %ile": base_avg - 40,
                "25th %ile": base_avg - 20,
                "75th %ile": base_avg + 20,
                "90th %ile": base_avg + 40,
                "League Winner Score": 58.0,
                "Runs": runs,
            },
        ]
    ).sort_values("Avg Lineup Score", ascending=False).reset_index(drop=True)
    details = _details(summary)
    paired = {
        "runs": runs,
        "requested_runs": 500,
        "challenger": "Beta",
        "base": "Alpha",
        "mean_delta": paired_mean,
        "paired_win_pct": win_pct,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "stop_stage": stop_stage,
        "stopped_early": runs < 500,
        "stop_reason": stop_reason,
    }
    return summary, details, paired


def test_shortlist_uses_score_neighborhood_but_keeps_minimum():
    short = candidate_shortlist(board_frame(), FakeEngine().config)
    assert short["name"].tolist() == ["Alpha", "Beta", "Gamma"]


def test_base_winner_stops_after_fast_initial_screen():
    calls = []

    def fake_compare(engine, **kwargs):
        calls.append(kwargs["runs"])
        summary = pd.DataFrame(
            [
                {"Candidate": "Alpha", "Avg Lineup Score": 410.0},
                {"Candidate": "Beta", "Avg Lineup Score": 408.0},
                {"Candidate": "Gamma", "Avg Lineup Score": 400.0},
            ]
        )
        return summary, _details(summary)

    result = run_final_call(
        FakeEngine(), current_pick=3, drafted_ids=set(), my_roster_ids=[],
        board=board_frame(), draft_history=[], compare_fn=fake_compare,
    )
    assert calls == [30]
    assert result["recommendation"] == "Alpha"
    assert result["stage"] == "initial"


def test_tiny_initial_challenger_edge_does_not_burn_clock_or_overturn():
    calls = []

    def fake_compare(engine, **kwargs):
        calls.append(kwargs["runs"])
        return _screen(beta=411.0, alpha=409.0)

    result = run_final_call(
        FakeEngine(), current_pick=3, drafted_ids=set(), my_roster_ids=[],
        board=board_frame(), draft_history=[], compare_fn=fake_compare,
    )
    assert calls == [30]
    assert result["recommendation"] == "Alpha"
    assert result["sim_winner"] == "Beta"
    assert result["base_agrees"] is True


def test_strong_challenger_uses_one_continuous_stability_stream_and_can_overturn():
    calls = []
    stability_calls = []

    def fake_compare(engine, **kwargs):
        calls.append(kwargs["runs"])
        return _screen(beta=423.0, alpha=409.0)

    def fake_stability(engine, **kwargs):
        stability_calls.append(kwargs)
        return _stability_result(
            challenger_avg=422.0, base_avg=410.0, paired_mean=12.0,
            win_pct=58.0, ci_low=2.2, ci_high=21.8, runs=500,
        )

    result = run_final_call(
        FakeEngine(), current_pick=3, drafted_ids=set(), my_roster_ids=[],
        board=board_frame(), draft_history=[], compare_fn=fake_compare,
        stability_fn=fake_stability,
    )
    assert calls == [30]
    assert len(stability_calls) == 1
    call = stability_calls[0]
    assert call["runs"] == 500
    assert call["confirmation_runs"] == 100
    assert call["checkpoint_runs"] == 200
    assert call["challenger_id"] == "b"
    assert call["base_id"] == "a"
    assert result["recommendation"] == "Beta"
    assert result["stage"] == "stabilized"
    assert result["runs"] == 500
    assert result["paired_evidence_pass"] is True


def test_challenger_can_stop_at_100_run_confirmation():
    def fake_compare(engine, **kwargs):
        return _screen(beta=420.0, alpha=409.0)

    def fake_stability(engine, **kwargs):
        return _stability_result(
            challenger_avg=416.2, base_avg=410.0, paired_mean=6.2,
            win_pct=53.0, ci_low=-4.0, ci_high=16.4, runs=100,
            stop_stage="confirmation",
            stop_reason="100-run mean did not clear +10",
        )

    result = run_final_call(
        FakeEngine(), current_pick=3, drafted_ids=set(), my_roster_ids=[],
        board=board_frame(), draft_history=[], compare_fn=fake_compare,
        stability_fn=fake_stability,
    )
    assert result["recommendation"] == "Alpha"
    assert result["stage"] == "refined"
    assert result["runs"] == 100
    assert result["paired_evidence_pass"] is False
    assert "100-run paired confirmation" in result["reason"]


def test_futile_challenger_stops_at_200_run_checkpoint():
    def fake_compare(engine, **kwargs):
        return _screen(beta=425.0, alpha=405.0)

    def fake_stability(engine, **kwargs):
        return _stability_result(
            challenger_avg=406.0, base_avg=409.0, paired_mean=-3.0,
            win_pct=48.0, ci_low=-14.0, ci_high=8.0, runs=200,
            stop_stage="checkpoint",
            stop_reason="95% CI upper bound +8.00 is below the +10.0 mean edge required for an overturn",
        )

    result = run_final_call(
        FakeEngine(), current_pick=3, drafted_ids=set(), my_roster_ids=[],
        board=board_frame(), draft_history=[], compare_fn=fake_compare,
        stability_fn=fake_stability,
    )
    assert result["recommendation"] == "Alpha"
    assert result["stage"] == "stabilized"
    assert result["runs"] == 200
    assert result["paired_stop_stage"] == "checkpoint"
    assert result["paired_evidence_pass"] is False
    assert "became futile" in result["reason"]


def test_noisy_challenger_is_rejected_when_full_500_run_check_reverses():
    def fake_compare(engine, **kwargs):
        return _screen(beta=425.0, alpha=405.0)

    def fake_stability(engine, **kwargs):
        return _stability_result(
            challenger_avg=403.4, base_avg=407.0, paired_mean=-3.6,
            win_pct=48.5, ci_low=-9.2, ci_high=1.9, runs=500,
        )

    result = run_final_call(
        FakeEngine(), current_pick=3, drafted_ids=set(), my_roster_ids=[],
        board=board_frame(), draft_history=[], compare_fn=fake_compare,
        stability_fn=fake_stability,
    )
    assert result["recommendation"] == "Alpha"
    assert result["runs"] == 500
    assert result["paired_mean_delta"] == -3.6
    assert result["paired_win_pct"] == 48.5
    assert result["paired_evidence_pass"] is False
    assert "did not survive" in result["reason"]


def test_full_sample_challenger_must_clear_win_rate_and_ci_gate():
    def fake_compare(engine, **kwargs):
        return _screen(beta=425.0, alpha=405.0)

    def fake_stability(engine, **kwargs):
        return _stability_result(
            challenger_avg=422.0, base_avg=410.0, paired_mean=12.0,
            win_pct=53.0, ci_low=-0.5, ci_high=24.5, runs=500,
        )

    result = run_final_call(
        FakeEngine(), current_pick=3, drafted_ids=set(), my_roster_ids=[],
        board=board_frame(), draft_history=[], compare_fn=fake_compare,
        stability_fn=fake_stability,
    )
    assert result["recommendation"] == "Alpha"
    assert result["edge_label"] == "UNSTABLE"
    assert result["paired_evidence_pass"] is False
    assert "paired wins 53.0% < 55.0%" in result["reason"]
    assert "95% CI crosses zero" in result["reason"]


def test_checkpoint_futility_is_conservative_and_never_accepts_early():
    bad = {
        "mean_delta": -3.0,
        "paired_win_pct": 48.0,
        "ci_low": -14.0,
        "ci_high": 8.0,
    }
    good = {
        "mean_delta": 12.0,
        "paired_win_pct": 58.0,
        "ci_low": 2.0,
        "ci_high": 22.0,
    }
    assert _checkpoint_futility_reason(
        bad, required_margin=10.0, min_win_pct=55.0, require_positive_ci=True
    ) is not None
    assert _checkpoint_futility_reason(
        good, required_margin=10.0, min_win_pct=55.0, require_positive_ci=True
    ) is None


def test_round_14_bypasses_room_sim_and_uses_forced_base_board():
    called = False

    def fake_compare(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("should not simulate")

    result = run_final_call(
        FakeEngine(), current_pick=159, drafted_ids=set(), my_roster_ids=[],
        board=board_frame(), draft_history=[], compare_fn=fake_compare,
    )
    assert called is False
    assert result["recommendation"] == "Alpha"
    assert result["stage"] == "base"
