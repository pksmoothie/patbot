import patbot.market as market


def test_fp_ecr_parser_prefers_ppr_field(monkeypatch):
    monkeypatch.setattr(
        market,
        "_fp_get",
        lambda path, params: {
            "players": [
                {"player_name": "Jahmyr Gibbs", "rank_ecr_ppr": 3, "rank_ecr": 5, "rank_adp_ppr": 4},
                {"player_name": "Bijan Robinson", "rank_ecr_ppr": 4, "rank_ecr": 6, "rank_adp_ppr": 3},
            ]
        },
    )
    df = market.fetch_fantasypros_api_ecr(
        ["Jahmyr Gibbs", "Bijan Robinson"], 2026
    )
    assert len(df) == 2
    assert df.loc[df["name"] == "Jahmyr Gibbs", "fp_ecr"].iloc[0] == 3


def test_fp_adp_parser_prefers_ppr_field(monkeypatch):
    monkeypatch.setattr(
        market,
        "_fp_get",
        lambda path, params: {
            "players": [
                {"player_name": "Jahmyr Gibbs", "rank_ecr_ppr": 1, "rank_adp_ppr": 2},
                {"player_name": "Bijan Robinson", "rank_ecr_ppr": 2, "rank_adp_ppr": 1},
            ]
        },
    )
    df = market.fetch_fantasypros_api_adp(
        ["Jahmyr Gibbs", "Bijan Robinson"], 2026
    )
    assert df.loc[df["name"] == "Jahmyr Gibbs", "fp_adp"].iloc[0] == 2


def test_fp_full_board_uses_players_endpoint_once(monkeypatch):
    calls = []

    def fake_get(path, params):
        calls.append((path, params))
        return {
            "players": [
                {"player_name": "Jahmyr Gibbs", "rank_ecr_ppr": 3, "rank_adp_ppr": 4},
                {"player_name": "Bijan Robinson", "rank_ecr_ppr": 4, "rank_adp_ppr": 3},
            ]
        }

    monkeypatch.setattr(market, "_fp_get", fake_get)
    board = market.fetch_fantasypros_api_board(
        ["Jahmyr Gibbs", "Bijan Robinson"], 2026
    )
    assert len(board) == 2
    assert calls == [("nfl/players", {"ecr": "included", "show": "pos_rank"})]
