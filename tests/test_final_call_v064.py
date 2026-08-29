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
                "overturn_probe_margin": 2.5,
                "overturn_required_margin": 10.0,
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


def test_strong_challenger_gets_confirmation_and_can_overturn():
    calls = []

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

    result = run_final_call(
        FakeEngine(),
        current_pick=3,
        drafted_ids=set(),
        my_roster_ids=[],
        board=board_frame(),
        draft_history=[],
        compare_fn=fake_compare,
    )
    assert [x[0] for x in calls] == [30, 100]
    assert calls[1][1] == ["b", "a"]
    assert result["recommendation"] == "Beta"
    assert result["base_agrees"] is False
    assert result["stage"] == "refined"


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
