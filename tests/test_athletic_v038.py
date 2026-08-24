from patbot.athletic import extract_rankings_from_rows


def _rows():
    # Two OVERALL PLAYER blocks mirror the real workbook. The first is grouped
    # by position and has non-sequential overall ranks; the second is the true
    # 1..N overall board that PatBot should select.
    return [
        {
            34: "OVR RK", 35: "OVERALL PLAYER", 36: "POS RK", 37: "BYE", 38: "Custom", 39: "VORP",
            41: "RK", 42: "OVERALL PLAYER", 43: "POS RK", 44: "BYE", 45: "FPS", 46: "VORP",
        },
        {34: 35, 35: "Josh Allen", 36: "QB1", 37: 7, 38: 457.6, 39: 117.4,
         41: 1, 42: "Ja'Marr Chase", 43: "WR1", 44: 6, 45: 368.0, 46: 259.9},
        {34: 63, 35: "Drake Maye", 36: "QB2", 37: 11, 38: 431.8, 39: 86.5,
         41: 2, 42: "Puka Nacua", 43: "WR2", 44: 11, 45: 349.6, 46: 241.5},
        {34: 80, 35: "Some QB", 36: "QB3", 37: 8, 38: 400.0, 39: 70.0,
         41: 3, 42: "Buffalo Bills", 43: "DST1", 44: 7, 45: 150.0, 46: 20.0},
    ]


def test_athletic_uses_true_sequential_overall_board():
    df = extract_rankings_from_rows(
        _rows(),
        ["Josh Allen", "Ja'Marr Chase", "Puka Nacua", "Buffalo Bills"],
    )
    assert df["name"].tolist()[:2] == ["Ja'Marr Chase", "Puka Nacua"]
    assert df.iloc[0]["athletic_rank"] == 1.0
    assert round(df.iloc[0]["athletic_vorp"], 1) == 259.9


def test_athletic_source_is_offense_only():
    df = extract_rankings_from_rows(
        _rows(),
        ["Ja'Marr Chase", "Puka Nacua", "Buffalo Bills"],
    )
    assert "Buffalo Bills" not in set(df["name"])
    assert set(df["athletic_pos_rank"]) == {"WR1", "WR2"}
