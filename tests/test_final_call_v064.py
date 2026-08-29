from __future__ import annotations

import pandas as pd

from patbot.final_call import candidate_shortlist, run_final_call


class FakeEngine:
    def __init__(self, config=None):
        self.config = config or {
            "simulation": {"through_round": 8},
            "final_call": {
                "min_candidates": 3,
                "max_candidates": 6,
                "score_gap": 10.0,
                "initial_runs": 100,
                "refine_runs": 300,
                "final_runs": 600,
                "refine_margin": 8.0,
                "final_margin": 2.5,
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


def _details(summary, ids):
    lookup = {"Alpha": "a", "Beta": "b", "Gamma": "c", "Delta": "d"}
    return [{"candidate": name, "candidate_id": lookup[name]} for name in summary["Candidate"].tolist()]


def test_shortlist_uses_score_neighborhood_but_keeps_minimum():
    short = candidate_shortlist(board_frame(), FakeEngine().config)
    assert short["name"].tolist() == ["Alpha", "Beta", "Gamma"]


def test_final_call_refines_any_base_board_overturn():
    calls = []

    def fake_compare(engine, **kwargs):
        calls.append((kwargs["runs"], list(kwargs["candidate_ids"])))
        if kwargs["runs"] == 100:
            summary = pd.DataFrame(
                [
                    {"Candidate": "Beta", "Avg Lineup Score": 420.0},
                    {"Candidate": "Alpha", "Avg Lineup Score": 409.0},
                    {"Candidate": "Gamma", "Avg Lineup Score": 400.0},
                ]
            )
        else:
            summary = pd.DataFrame(
                [
                    {"Candidate": "Beta", "Avg Lineup Score": 418.0},
                    {"Candidate": "Alpha", "Avg Lineup Score": 410.0},
                    {"Candidate": "Gamma", "Avg Lineup Score": 399.0},
                ]
            )
        return summary, _details(summary, kwargs["candidate_ids"])

    result = run_final_call(
        FakeEngine(),
        current_pick=3,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
    )
    assert [x[0] for x in calls] == [100, 300]
    assert result["recommendation"] == "Beta"
    assert result["base_winner"] == "Alpha"
    assert result["base_agrees"] is False
    assert result["stage"] == "refined"


def test_final_call_uses_final_confirmation_when_refined_margin_is_tiny():
    calls = []

    def fake_compare(engine, **kwargs):
        calls.append(kwargs["runs"])
        runs = kwargs["runs"]
        if runs == 100:
            rows = [
                {"Candidate": "Alpha", "Avg Lineup Score": 410.0},
                {"Candidate": "Beta", "Avg Lineup Score": 405.0},
                {"Candidate": "Gamma", "Avg Lineup Score": 390.0},
            ]
        elif runs == 300:
            rows = [
                {"Candidate": "Beta", "Avg Lineup Score": 411.0},
                {"Candidate": "Alpha", "Avg Lineup Score": 410.0},
                {"Candidate": "Gamma", "Avg Lineup Score": 389.0},
            ]
        else:
            rows = [
                {"Candidate": "Alpha", "Avg Lineup Score": 412.0},
                {"Candidate": "Beta", "Avg Lineup Score": 410.0},
            ]
        summary = pd.DataFrame(rows)
        return summary, _details(summary, kwargs["candidate_ids"])

    result = run_final_call(
        FakeEngine(),
        current_pick=3,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
    )
    assert calls == [100, 300, 600]
    assert result["recommendation"] == "Alpha"
    assert result["stage"] == "final"
    assert result["runs"] == 600


def test_round_14_bypasses_room_sim_and_uses_forced_base_board():
    called = False

    def fake_compare(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("should not simulate")

    result = run_final_call(
        FakeEngine(),
        current_pick=159,  # Round 14 in a 12-team league.
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
    )
    assert called is False
    assert result["recommendation"] == "Alpha"
    assert result["stage"] == "base"
