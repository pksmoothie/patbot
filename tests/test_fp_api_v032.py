import pandas as pd
import patbot.market as market

def test_fp_ecr_parser(monkeypatch):
    monkeypatch.setattr(
        market,
        "_fp_get",
        lambda path, params: {
            "players": [
                {"player_name": "Jahmyr Gibbs", "rank_ecr": 3},
                {"player_name": "Bijan Robinson", "rank_ecr": 4},
            ]
        },
    )
    df = market.fetch_fantasypros_api_ecr(
        ["Jahmyr Gibbs", "Bijan Robinson"], 2026
    )
    assert len(df) == 2
    assert df.loc[df["name"] == "Jahmyr Gibbs", "fp_ecr"].iloc[0] == 3

def test_fp_adp_parser_accepts_rank_ecr_for_adp_type(monkeypatch):
    monkeypatch.setattr(
        market,
        "_fp_get",
        lambda path, params: {
            "players": [
                {"player_name": "Jahmyr Gibbs", "rank_ecr": 1},
                {"player_name": "Bijan Robinson", "rank_ecr": 2},
            ]
        },
    )
    df = market.fetch_fantasypros_api_adp(
        ["Jahmyr Gibbs", "Bijan Robinson"], 2026
    )
    assert df.loc[df["name"] == "Jahmyr Gibbs", "fp_adp"].iloc[0] == 1
