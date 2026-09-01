from __future__ import annotations

import pytest

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.draft_state import drafted_ids_from_history, make_pick_record
from patbot.mock_draft import (
    DEFAULT_MOCK_SEED,
    simulate_next_opponent_pick,
    simulate_opponents_until_patbot,
)


def _engine():
    cfg = load_config()
    players = load_players("data/example_players.csv")
    players["player_id"] = players["player_id"].astype(str)
    return DraftEngine(players, cfg)


def test_mock_next_pick_is_deterministic_for_same_room_state():
    engine = _engine()
    kwargs = dict(
        current_pick=1,
        drafted_ids=set(),
        draft_history=[],
        seed=DEFAULT_MOCK_SEED,
    )
    first = simulate_next_opponent_pick(engine, **kwargs)
    second = simulate_next_opponent_pick(engine, **kwargs)
    assert first["owner_slot"] == 1
    assert first["player_id"] == second["player_id"]
    assert first["archetype"] == second["archetype"]


def test_mock_advances_opponents_and_stops_at_patbot_103():
    engine = _engine()
    history, simulated = simulate_opponents_until_patbot(
        engine,
        current_pick=1,
        drafted_ids=set(),
        draft_history=[],
        make_record=make_pick_record,
        seed=DEFAULT_MOCK_SEED,
    )
    assert len(history) == 2
    assert len(simulated) == 2
    assert [x["owner_slot"] for x in simulated] == [1, 2]
    assert len(drafted_ids_from_history(history)) == 2
    assert history[0]["player_id"] != history[1]["player_id"]


def test_mock_refuses_to_simulate_patbot_pick():
    engine = _engine()
    with pytest.raises(ValueError, match="belongs to PatBot"):
        simulate_next_opponent_pick(
            engine,
            current_pick=3,
            drafted_ids=set(),
            draft_history=[],
            seed=DEFAULT_MOCK_SEED,
        )
