import pandas as pd

from patbot.config import load_config
from patbot.draft import snake_pick, next_team_pick, survive_probability, DraftEngine

def test_snake_slot_3():
    assert snake_pick(1, 12, 3) == 3
    assert snake_pick(2, 12, 3) == 22
    assert snake_pick(3, 12, 3) == 27

def test_next_pick():
    assert next_team_pick(4, 12, 3) == 22
    assert next_team_pick(23, 12, 3) == 27

def test_survival_bounds():
    p = survive_probability(20, 30)
    assert 0 <= p <= 1

def test_engine_board():
    cfg = load_config("config/league.yaml")
    df = pd.read_csv("data/example_players.csv")
    df["player_id"] = df["player_id"].astype(str)
    engine = DraftEngine(df, cfg)
    board = engine.recommend(3, set(), [], top_n=5)
    assert len(board) == 5
    assert "score" in board.columns
