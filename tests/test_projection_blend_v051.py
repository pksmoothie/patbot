import numpy as np
import pandas as pd

from patbot.projection_blend import blend_projection_sources


def test_blend_uses_configured_weights_for_offense():
    players = pd.DataFrame(
        {
            "name": ["RB A", "WR B"],
            "pos": ["RB", "WR"],
            "proj_points": [300.0, 200.0],
            "fantasypros_proj_points": [330.0, 230.0],
        }
    )
    cfg = {
        "projection_sources": {
            "production_blend": {
                "enabled": True,
                "sleeper_weight": 0.60,
                "fantasypros_weight": 0.40,
            }
        }
    }
    out, status = blend_projection_sources(players, cfg)
    assert out.loc[0, "sleeper_proj_points"] == 300.0
    assert out.loc[0, "proj_points"] == 312.0
    assert out.loc[1, "proj_points"] == 212.0
    assert status["production_projection_blend"]["coverage_pct"] == 100.0


def test_blend_falls_back_to_sleeper_when_fp_missing():
    players = pd.DataFrame(
        {
            "name": ["RB A", "WR B"],
            "pos": ["RB", "WR"],
            "proj_points": [300.0, 200.0],
            "fantasypros_proj_points": [np.nan, 230.0],
        }
    )
    out, _ = blend_projection_sources(players, {})
    assert out.loc[0, "proj_points"] == 300.0
    assert out.loc[0, "projection_blend_source_count"] == 1
    assert out.loc[1, "projection_blend_source_count"] == 2


def test_k_and_def_remain_sleeper_provider_projection():
    players = pd.DataFrame(
        {
            "name": ["K A", "DEF A"],
            "pos": ["K", "DEF"],
            "proj_points": [120.0, 110.0],
            "fantasypros_proj_points": [150.0, 160.0],
        }
    )
    out, _ = blend_projection_sources(players, {})
    assert list(out["proj_points"]) == [120.0, 110.0]


def test_disabled_blend_preserves_sleeper():
    players = pd.DataFrame(
        {
            "name": ["RB A"],
            "pos": ["RB"],
            "proj_points": [300.0],
            "fantasypros_proj_points": [330.0],
        }
    )
    cfg = {"projection_sources": {"production_blend": {"enabled": False}}}
    out, status = blend_projection_sources(players, cfg)
    assert out.loc[0, "proj_points"] == 300.0
    assert status["production_projection_blend"]["enabled"] is False
