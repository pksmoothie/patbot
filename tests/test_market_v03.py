import pandas as pd
from patbot.market import normalize_name, _known_name_match
from patbot.consensus import add_consensus_values
from patbot.config import load_config

def test_name_normalization():
    assert normalize_name("Ja'Marr Chase") == "ja marr chase"
    assert normalize_name("Marvin Harrison Jr.") == "marvin harrison"

def test_known_name_match_handles_cell_noise():
    known = {
        normalize_name("Jahmyr Gibbs"): "Jahmyr Gibbs",
        normalize_name("Bijan Robinson"): "Bijan Robinson",
    }
    assert _known_name_match("Jahmyr Gibbs J. Gibbs DET (6)", known) == "Jahmyr Gibbs"

def test_consensus_values_and_tiers():
    cfg = load_config("config/league.yaml")
    df = pd.DataFrame([
        {"player_id":"1","name":"A","team":"X","pos":"RB","adp":1,"proj_points":330,"expert_rank":1},
        {"player_id":"2","name":"B","team":"X","pos":"RB","adp":2,"proj_points":325,"expert_rank":2},
        {"player_id":"3","name":"C","team":"X","pos":"RB","adp":8,"proj_points":280,"expert_rank":8},
        {"player_id":"4","name":"D","team":"X","pos":"RB","adp":12,"proj_points":270,"expert_rank":12},
    ])
    out = add_consensus_values(df, cfg)
    assert "consensus_value" in out
    assert "consensus_tier" in out
    assert out.loc[out["name"]=="A", "consensus_value"].iloc[0] > out.loc[out["name"]=="D", "consensus_value"].iloc[0]
