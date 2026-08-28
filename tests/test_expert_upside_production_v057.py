from datetime import date

import pandas as pd

from patbot.expert_upside_production import production_bonus_for_round
from patbot.upside_intel import score_upside_evidence


def _bonus_players():
    return pd.DataFrame(
        {
            "name": ["A", "B"],
            "expert_upside_lws_bonus": [4.0, 8.0],
        }
    )


def test_expert_upside_bonus_is_zero_before_round_eight():
    out = production_bonus_for_round(_bonus_players(), {}, 7)
    assert out.tolist() == [0.0, 0.0]


def test_expert_upside_bonus_is_available_from_round_eight():
    out = production_bonus_for_round(_bonus_players(), {}, 8)
    assert out.tolist() == [4.0, 8.0]


def test_expert_upside_bonus_can_be_disabled_without_touching_other_inputs():
    cfg = {
        "championship_strategy": {
            "expert_upside_intel": {
                "production_enabled": False,
            }
        }
    }
    out = production_bonus_for_round(_bonus_players(), cfg, 13)
    assert out.tolist() == [0.0, 0.0]


def test_two_narrative_sources_without_objective_support_are_not_eligible(tmp_path):
    evidence = tmp_path / "intel.yaml"
    evidence.write_text(
        """
sources:
  one:
    publisher: One
    independence_key: one
    quality: 1.0
    published: '2026-08-26'
  two:
    publisher: Two
    independence_key: two
    quality: 1.0
    published: '2026-08-26'
evidence:
  - player: Player A
    source: one
    strength: 1.0
    specificity: 1.0
    objective_support: false
  - player: Player A
    source: two
    strength: 1.0
    specificity: 1.0
    objective_support: false
""",
        encoding="utf-8",
    )
    players = pd.DataFrame(
        {
            "name": ["Player A"],
            "pos": ["WR"],
            "team": ["AAA"],
            "adp": [120.0],
            "proj_points": [160.0],
        }
    )
    cfg = {
        "championship_strategy": {
            "expert_upside_intel": {
                "evidence_path": str(evidence),
                "minimum_independent_sources": 2,
                "minimum_objective_sources": 1,
                "max_lws_bonus": 8.0,
            }
        }
    }
    out, status = score_upside_evidence(players, cfg, as_of=date(2026, 8, 27))
    row = out.iloc[0]
    assert row["expert_upside_sources"] == 2
    assert row["expert_upside_objective_sources"] == 0
    assert not bool(row["expert_upside_eligible"])
    assert row["expert_upside_lws_bonus"] == 0.0
    assert status["eligible_players"] == 0
