import pandas as pd

from patbot.yahoo_adp import (
    _request_params,
    behavioral_adp,
    manager_yahoo_weight,
    parse_yahoo_adp_tables,
)


def test_parse_yahoo_basic_all_drafts_table():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Fantasy", "Player"),
            ("Fantasy", "Rank"),
            ("Fantasy", "%Drafted"),
            ("Basic ADP", "Preseason"),
            ("Basic ADP", "All Drafts"),
            ("Basic ADP", "Last 7 Days"),
        ]
    )
    table = pd.DataFrame(
        [
            ["Jahmyr Gibbs Det - RB", 1, "100%", 1.0, 1.5, 1.4],
            ["Ja'Marr Chase Cin - WR", 3, "100%", 3.0, 3.4, 3.2],
        ],
        columns=columns,
    )
    out = parse_yahoo_adp_tables([table], ["Jahmyr Gibbs", "Ja'Marr Chase"])
    assert out.set_index("name").loc["Jahmyr Gibbs", "yahoo_adp"] == 1.5
    assert out.set_index("name").loc["Ja'Marr Chase", "yahoo_adp"] == 3.4


def test_parser_prefers_basic_all_drafts_over_plus():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Fantasy", "Player"),
            ("Basic ADP", "All Drafts"),
            ("Plus ADP", "All Drafts 💎"),
        ]
    )
    table = pd.DataFrame(
        [
            ["Player A AAA - WR", 20.0, 5.0],
            ["Player B BBB - RB", 30.0, 7.0],
        ],
        columns=columns,
    )
    out = parse_yahoo_adp_tables([table], ["Player A", "Player B"])
    assert out.set_index("name").loc["Player A", "yahoo_adp"] == 20.0


def test_snake_adp_request_uses_sd_not_auction_ad_tab():
    params = _request_params(pos="ALL", count=50)
    assert params["tab"] == "SD"
    assert params["sort"] == "DA_AP"
    assert params["count"] == 50


def test_behavioral_adp_is_manager_weighted_and_falls_back():
    market = pd.Series([100.0, 50.0, 80.0])
    yahoo = pd.Series([80.0, None, 100.0])
    out = behavioral_adp(market, yahoo, 0.80)
    assert round(float(out.iloc[0]), 2) == 84.0
    assert round(float(out.iloc[1]), 2) == 50.0
    assert round(float(out.iloc[2]), 2) == 96.0


def test_casuals_are_more_yahoo_anchored_than_sharp_managers():
    assert manager_yahoo_weight("casual") > manager_yahoo_weight("market")
    assert manager_yahoo_weight("market") > manager_yahoo_weight("league_aware")
    assert manager_yahoo_weight("league_aware") > manager_yahoo_weight("sharp")
    assert manager_yahoo_weight("sharp") > manager_yahoo_weight("extremely_sharp")
