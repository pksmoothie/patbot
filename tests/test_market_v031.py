import pandas as pd
from patbot.market import normalize_name, _best_player_table, _rank_column

def test_best_player_table_with_generic_headers():
    known = {
        normalize_name("Jahmyr Gibbs"): "Jahmyr Gibbs",
        normalize_name("Bijan Robinson"): "Bijan Robinson",
        normalize_name("Ja'Marr Chase"): "Ja'Marr Chase",
    }
    junk = pd.DataFrame({"A": ["hello"], "B": ["world"]})
    real = pd.DataFrame({
        "RK": [1,2,3],
        "Player Name": [
            "Jahmyr Gibbs (DET)",
            "Bijan Robinson (ATL)",
            "Ja'Marr Chase (CIN)",
        ],
        "AVG": [1.0,2.0,3.2],
    })
    df, player_col = _best_player_table([junk, real], known)
    assert player_col == "Player Name"
    assert _rank_column(df, player_col) == "RK"
