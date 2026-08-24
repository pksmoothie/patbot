import pandas as pd

from patbot.sim import FastDraftSimulator, compare_candidates


class FakeEngine:
    def __init__(self):
        rows = []
        pid = 1
        for pos, n, top, decline in [
            ("RB", 40, 315, 2.3),
            ("WR", 50, 320, 2.0),
            ("QB", 24, 415, 4.2),
            ("TE", 24, 255, 3.0),
            ("K", 14, 150, 1.0),
            ("DEF", 14, 145, 1.0),
        ]:
            for i in range(n):
                rows.append({
                    "player_id": str(pid),
                    "name": f"{pos}{i + 1}",
                    "team": "X",
                    "pos": pos,
                    "adp": float(pid),
                    "proj_points": float(top - i * decline),
                    "injury_risk": 0.0,
                    "expert_rank": float(pid),
                    "is_rookie": False,
                })
                pid += 1
        self.players = pd.DataFrame(rows)
        self.config = {
            "league": {"teams": 12, "draft_slot": 3},
            "roster": {
                "QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DEF": 1,
                "FLEX": 1, "flex_eligible": ["RB", "WR", "TE"],
            },
            "draft_engine": {
                "weights": {
                    "vorp": 0.35, "projection": 0.25, "urgency": 0.20,
                    "scarcity": 0.12, "roster_fit": 0.08,
                },
                "injury_risk_penalty": 8.0,
                "min_round_k": 13,
                "min_round_def": 13,
                "bench_position_caps": {"QB": 2, "TE": 2, "K": 1, "DEF": 1},
            },
            "simulation": {
                "opponent_adp_sd_floor": 5.0,
                "opponent_adp_sd_pct": 0.14,
                "comparison_seed": 20260818,
                "patbot_lookahead": {
                    "enabled": True,
                    "rounds": [2, 3],
                    "branch_width": 4,
                    "future_pick_weight": 0.90,
                    "pair_vorp_weight": 0.02,
                    "max_gap_picks": 24,
                },
            },
            "opponent_archetypes": {
                "counts": {"casual": 4, "market": 3, "league_aware": 2, "sharp": 1, "extremely_sharp": 1},
                "casual": {"market_weight": 0.95, "custom_weight": 0.05, "roster_need_strength": 0.55, "randomness": 1.20},
                "market": {"market_weight": 0.82, "custom_weight": 0.18, "roster_need_strength": 0.75, "randomness": 1.00},
                "league_aware": {"market_weight": 0.55, "custom_weight": 0.45, "roster_need_strength": 0.95, "randomness": 0.85},
                "sharp": {"market_weight": 0.32, "custom_weight": 0.68, "roster_need_strength": 1.15, "randomness": 0.70},
                "extremely_sharp": {"market_weight": 0.15, "custom_weight": 0.85, "roster_need_strength": 1.30, "randomness": 0.55},
            },
            "roster_evaluation": {
                "bench_vorp_discount": 0.20,
                "empty_group_penalty": {"RB": 18, "WR": 12},
                "missing_starter_penalty": {"QB": 7, "RB": 12, "WR": 9, "TE": 6, "FLEX": 7},
                "construction_bonus": {"has_qb": 2, "has_te": 2, "has_two_rb": 6, "has_three_wr": 6, "has_flex_depth": 3},
            },
        }
        self.league = self.config["league"]
        self.roster_cfg = self.config["roster"]
        self.engine_cfg = self.config["draft_engine"]

    def replacement_levels(self):
        ranks = {"QB": 12, "RB": 30, "WR": 42, "TE": 14, "K": 12, "DEF": 12}
        levels = {}
        for pos, rank in ranks.items():
            group = self.players[self.players["pos"] == pos].sort_values("proj_points", ascending=False)
            levels[pos] = float(group.iloc[min(rank - 1, len(group) - 1)]["proj_points"])
        return levels


def test_v038_lookahead_is_on_for_rounds_two_and_three():
    sim = FastDraftSimulator(FakeEngine())
    assert sim.lookahead_enabled is True
    assert sim.lookahead_rounds == {2, 3}
    assert sim.lookahead_branch_width == 4


def test_candidate_comparison_uses_common_random_numbers():
    engine = FakeEngine()
    _, details = compare_candidates(
        engine,
        current_pick=3,
        drafted_ids={"1", "2"},
        my_roster_ids=[],
        candidate_ids=["3", "3"],
        runs=8,
        through_round=5,
    )
    assert details[0]["avg_lineup_score"] == details[1]["avg_lineup_score"]
    assert details[0]["most_common_second_pick"] == details[1]["most_common_second_pick"]
    assert details[0]["most_common_third_pick"] == details[1]["most_common_third_pick"]
