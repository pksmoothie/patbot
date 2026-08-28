from datetime import date
from pathlib import Path

import pandas as pd

from patbot.upside_intel import score_upside_evidence


def _players():
    return pd.DataFrame(
        {
            "name": ["Player A", "Player B"],
            "pos": ["WR", "RB"],
            "team": ["AAA", "BBB"],
            "adp": [110.0, 125.0],
            "proj_points": [180.0, 170.0],
        }
    )


def _cfg(path: Path):
    return {
        "championship_strategy": {
            "expert_upside_intel": {
                "enabled": True,
                "evidence_path": str(path),
                "minimum_independent_sources": 2,
                "max_lws_bonus": 8.0,
                "recency_half_life_days": 21,
            }
        }
    }


def test_single_publisher_is_visible_but_not_bonus_eligible(tmp_path):
    path = tmp_path / "intel.yaml"
    path.write_text(
        """
sources:
  one:
    publisher: Source One
    independence_key: publisher-one
    quality: 1.0
    published: '2026-08-20'
evidence:
  - player: Player A
    source: one
    strength: 1.0
    specificity: 1.0
    objective_support: true
""",
        encoding="utf-8",
    )
    out, status = score_upside_evidence(_players(), _cfg(path), as_of=date(2026, 8, 27))
    row = out[out["name"].eq("Player A")].iloc[0]
    assert row["expert_upside_sources"] == 1
    assert row["expert_upside_score"] > 0
    assert not bool(row["expert_upside_eligible"])
    assert row["expert_upside_lws_bonus"] == 0.0
    assert status["eligible_players"] == 0


def test_two_independent_publishers_unlock_small_bonus(tmp_path):
    path = tmp_path / "intel.yaml"
    path.write_text(
        """
sources:
  one:
    publisher: Source One
    independence_key: publisher-one
    quality: 1.0
    published: '2026-08-25'
  two:
    publisher: Source Two
    independence_key: publisher-two
    quality: 0.9
    published: '2026-08-24'
evidence:
  - player: Player A
    source: one
    strength: 0.9
    specificity: 0.9
    objective_support: true
  - player: Player A
    source: two
    strength: 0.8
    specificity: 0.8
    objective_support: true
""",
        encoding="utf-8",
    )
    out, _ = score_upside_evidence(_players(), _cfg(path), as_of=date(2026, 8, 27))
    row = out[out["name"].eq("Player A")].iloc[0]
    assert row["expert_upside_sources"] == 2
    assert bool(row["expert_upside_eligible"])
    assert 0.0 < row["expert_upside_lws_bonus"] <= 8.0


def test_multiple_articles_same_publisher_count_once(tmp_path):
    path = tmp_path / "intel.yaml"
    path.write_text(
        """
sources:
  one:
    publisher: Same Publisher
    independence_key: same
    quality: 1.0
    published: '2026-08-25'
  two:
    publisher: Same Publisher
    independence_key: same
    quality: 1.0
    published: '2026-08-26'
evidence:
  - player: Player A
    source: one
    strength: 0.7
    specificity: 0.7
  - player: Player A
    source: two
    strength: 1.0
    specificity: 1.0
""",
        encoding="utf-8",
    )
    out, _ = score_upside_evidence(_players(), _cfg(path), as_of=date(2026, 8, 27))
    row = out[out["name"].eq("Player A")].iloc[0]
    assert row["expert_upside_sources"] == 1
    assert not bool(row["expert_upside_eligible"])


def test_recent_evidence_scores_above_equivalent_stale_evidence(tmp_path):
    path = tmp_path / "intel.yaml"
    path.write_text(
        """
sources:
  recent:
    publisher: Recent Source
    independence_key: recent
    quality: 1.0
    published: '2026-08-26'
  stale:
    publisher: Stale Source
    independence_key: stale
    quality: 1.0
    published: '2026-06-01'
evidence:
  - player: Player A
    source: recent
    strength: 1.0
    specificity: 1.0
  - player: Player B
    source: stale
    strength: 1.0
    specificity: 1.0
""",
        encoding="utf-8",
    )
    out, _ = score_upside_evidence(_players(), _cfg(path), as_of=date(2026, 8, 27))
    a = float(out.loc[out["name"].eq("Player A"), "expert_upside_score"].iloc[0])
    b = float(out.loc[out["name"].eq("Player B"), "expert_upside_score"].iloc[0])
    assert a > b
