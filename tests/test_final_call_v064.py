from __future__ import annotations

import pandas as pd

from patbot.final_call import candidate_shortlist, run_final_call


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
    return [{"candidate": name, "candidate_id": lookup[name]} for name in summary["Candidate"].tolist()]


def _stability_result(
    *,
    challenger_avg: float,
    base_avg: float,
    paired_mean: float,
    win_pct: float,
    ci_low: float,
    ci_high: float,
    runs: int = 500,
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
        "challenger": "Beta",
        "base": "Alpha",
        "mean_delta": paired_mean,
        "paired_win_pct": win_pct,
        "ci_low": ci_low,
        "ci_high": ci_high,
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
        FakeEngine(),
        current_pick=3,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
    )
    assert calls == [30]
    assert result["recommendation"] == "Alpha"
    assert result["stage"] == "initial"


def test_tiny_initial_challenger_edge_does_not_burn_clock_or_overturn():
    calls = []

    def fake_compare(engine, **kwargs):
        calls.append(kwargs["runs"])
        summary = pd.DataFrame(
            [
                {"Candidate": "Beta", "Avg Lineup Score": 411.0},
                {"Candidate": "Alpha", "Avg Lineup Score": 409.0},
                {"Candidate": "Gamma", "Avg Lineup Score": 400.0},
            ]
        )
        return summary, _details(summary)

    result = run_final_call(
        FakeEngine(),
        current_pick=3,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
    )
    assert calls == [30]
    assert result["recommendation"] == "Alpha"
    assert result["sim_winner"] == "Beta"
    assert result["base_agrees"] is True


def test_strong_challenger_gets_large_sample_stability_check_and_can_overturn():
    calls = []
    stability_calls = []

    def fake_compare(engine, **kwargs):
        calls.append((kwargs["runs"], list(kwargs["candidate_ids"])))
        if kwargs["runs"] == 30:
            summary = pd.DataFrame(
                [
                    {"Candidate": "Beta", "Avg Lineup Score": 423.0},
                    {"Candidate": "Alpha", "Avg Lineup Score": 409.0},
                    {"Candidate": "Gamma", "Avg Lineup Score": 400.0},
                ]
            )
        else:
            summary = pd.DataFrame(
                [
                    {"Candidate": "Beta", "Avg Lineup Score": 421.0},
                    {"Candidate": "Alpha", "Avg Lineup Score": 410.0},
                ]
            )
        return summary, _details(summary)

    def fake_stability(engine, **kwargs):
        stability_calls.append(kwargs)
        return _stability_result(
            challenger_avg=422.0,
            base_avg=410.0,
            paired_mean=12.0,
            win_pct=58.0,
            ci_low=2.2,
            ci_high=21.8,
            runs=kwargs["runs"],
        )

    result = run_final_call(
        FakeEngine(),
        current_pick=3,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
        stability_fn=fake_stability,
    )
    assert [x[0] for x in calls] == [30, 100]
    assert calls[1][1] == ["b", "a"]
    assert len(stability_calls) == 1
    assert stability_calls[0]["runs"] == 500
    assert stability_calls[0]["challenger_id"] == "b"
    assert stability_calls[0]["base_id"] == "a"
    assert result["recommendation"] == "Beta"
    assert result["base_agrees"] is False
    assert result["stage"] == "stabilized"
    assert result["runs"] == 500
    assert result["paired_win_pct"] == 58.0
    assert result["paired_evidence_pass"] is True


def test_moderate_confirmed_challenger_edge_does_not_overturn_base_prior():
    calls = []

    def fake_compare(engine, **kwargs):
        calls.append(kwargs["runs"])
        if kwargs["runs"] == 30:
            rows = [
                {"Candidate": "Beta", "Avg Lineup Score": 420.0},
                {"Candidate": "Alpha", "Avg Lineup Score": 409.0},
                {"Candidate": "Gamma", "Avg Lineup Score": 400.0},
            ]
        else:
            rows = [
                {"Candidate": "Beta", "Avg Lineup Score": 416.2},
                {"Candidate": "Alpha", "Avg Lineup Score": 410.0},
            ]
        summary = pd.DataFrame(rows)
        return summary, _details(summary)

    result = run_final_call(
        FakeEngine(),
        current_pick=3,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
    )
    assert calls == [30, 100]
    assert result["sim_winner"] == "Beta"
    assert result["recommendation"] == "Alpha"
    assert result["base_agrees"] is True
    assert "+10.0 threshold" in result["reason"]


def test_initial_challenge_must_survive_confirmation():
    calls = []

    def fake_compare(engine, **kwargs):
        calls.append(kwargs["runs"])
        if kwargs["runs"] == 30:
            rows = [
                {"Candidate": "Beta", "Avg Lineup Score": 420.0},
                {"Candidate": "Alpha", "Avg Lineup Score": 409.0},
                {"Candidate": "Gamma", "Avg Lineup Score": 400.0},
            ]
        else:
            rows = [
                {"Candidate": "Alpha", "Avg Lineup Score": 412.0},
                {"Candidate": "Beta", "Avg Lineup Score": 410.0},
            ]
        summary = pd.DataFrame(rows)
        return summary, _details(summary)

    result = run_final_call(
        FakeEngine(),
        current_pick=3,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
    )
    assert calls == [30, 100]
    assert result["recommendation"] == "Alpha"
    assert result["stage"] == "refined"


def test_noisy_100_run_overturn_is_rejected_when_500_run_check_reverses():
    calls = []

    def fake_compare(engine, **kwargs):
        calls.append(kwargs["runs"])
        if kwargs["runs"] == 30:
            rows = [
                {"Candidate": "Beta", "Avg Lineup Score": 425.0},
                {"Candidate": "Alpha", "Avg Lineup Score": 405.0},
                {"Candidate": "Gamma", "Avg Lineup Score": 395.0},
            ]
        else:
            rows = [
                {"Candidate": "Beta", "Avg Lineup Score": 422.0},
                {"Candidate": "Alpha", "Avg Lineup Score": 402.0},
            ]
        summary = pd.DataFrame(rows)
        return summary, _details(summary)

    def fake_stability(engine, **kwargs):
        return _stability_result(
            challenger_avg=403.4,
            base_avg=407.0,
            paired_mean=-3.6,
            win_pct=48.5,
            ci_low=-9.2,
            ci_high=1.9,
            runs=kwargs["runs"],
        )

    result = run_final_call(
        FakeEngine(),
        current_pick=3,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
        stability_fn=fake_stability,
    )
    assert calls == [30, 100]
    assert result["recommendation"] == "Alpha"
    assert result["stage"] == "stabilized"
    assert result["paired_mean_delta"] == -3.6
    assert result["paired_win_pct"] == 48.5
    assert result["paired_evidence_pass"] is False
    assert "did not survive" in result["reason"]


def test_large_sample_challenger_must_clear_win_rate_and_ci_gate():
    def fake_compare(engine, **kwargs):
        if kwargs["runs"] == 30:
            rows = [
                {"Candidate": "Beta", "Avg Lineup Score": 425.0},
                {"Candidate": "Alpha", "Avg Lineup Score": 405.0},
                {"Candidate": "Gamma", "Avg Lineup Score": 395.0},
            ]
        else:
            rows = [
                {"Candidate": "Beta", "Avg Lineup Score": 421.0},
                {"Candidate": "Alpha", "Avg Lineup Score": 410.0},
            ]
        summary = pd.DataFrame(rows)
        return summary, _details(summary)

    def fake_stability(engine, **kwargs):
        return _stability_result(
            challenger_avg=422.0,
            base_avg=410.0,
            paired_mean=12.0,
            win_pct=53.0,
            ci_low=-0.5,
            ci_high=24.5,
            runs=kwargs["runs"],
        )

    result = run_final_call(
        FakeEngine(),
        current_pick=3,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
        stability_fn=fake_stability,
    )
    assert result["recommendation"] == "Alpha"
    assert result["edge_label"] == "UNSTABLE"
    assert result["paired_evidence_pass"] is False
    assert "paired wins 53.0% < 55.0%" in result["reason"]
    assert "95% CI crosses zero" in result["reason"]


def test_round_14_bypasses_room_sim_and_uses_forced_base_board():
    called = False

    def fake_compare(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("should not simulate")

    result = run_final_call(
        FakeEngine(),
        current_pick=159,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
    )
    assert called is False
    assert result["recommendation"] == "Alpha"
    assert result["stage"] == "base"
