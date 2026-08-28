from __future__ import annotations

import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.expert_upside_production import expert_upside_settings
from patbot.strategy import strategy_phase


CSV_PATH = "data/players_2026_live.csv"


def main():
    cfg = load_config()
    players = load_players(CSV_PATH)
    engine = DraftEngine(players, cfg)
    settings = expert_upside_settings(cfg)
    enriched = engine.players.copy()

    bonus = pd.to_numeric(
        enriched.get("expert_upside_lws_bonus", pd.Series(0.0, index=enriched.index)),
        errors="coerce",
    ).fillna(0.0)
    mentioned = enriched[bonus.gt(0)].copy()
    mentioned["LWS Bonus"] = bonus[bonus.gt(0)].values if not mentioned.empty else []

    upside_phase = strategy_phase(8, cfg)
    lottery_phase = strategy_phase(12, cfg)
    upside_weight = float(upside_phase.get("upside_weight", 0.0))
    lottery_weight = float(lottery_phase.get("upside_weight", 0.0))

    if not mentioned.empty:
        mentioned["R8-11 Score Increment"] = mentioned["LWS Bonus"] * upside_weight
        mentioned["R12-13 Score Increment"] = mentioned["LWS Bonus"] * lottery_weight
        mentioned["Approx Round"] = (
            (pd.to_numeric(mentioned["adp"], errors="coerce") - 1.0) / 12.0 + 1.0
        ).round(1)
        mentioned = mentioned.sort_values(
            ["R12-13 Score Increment", "adp"], ascending=[False, True]
        )

    print("\nPatBot v0.5.7 expert upside production check")
    print("Corroborated expert intel is now production-enabled only from Round 8 onward.")
    print("It does not modify projections, VORP, expert rank, market ADP, or injury risk.\n")
    print(
        f"Guardrails: min round {settings['minimum_round']} | "
        f"min independent sources {settings['minimum_independent_sources']} | "
        f"min objective sources {settings['minimum_objective_sources']} | "
        f"max raw LWS bonus {settings['max_lws_bonus']:.1f}"
    )
    print(
        f"Phase weights: R8-11 {upside_weight:.2f} | R12-13 {lottery_weight:.2f}. "
        f"Therefore the absolute configured max pick-score impact is "
        f"{settings['max_lws_bonus'] * lottery_weight:.2f} points."
    )
    print("Rounds 1-7 expert-upside production impact: exactly 0.00 by rule.\n")

    print("=== CURRENT ELIGIBLE PLAYER IMPACT ===\n")
    if mentioned.empty:
        print("No current players are production-eligible for an expert-upside increment.")
    else:
        cols = [
            "name", "pos", "team", "adp", "Approx Round", "proj_points",
            "expert_upside_sources", "expert_upside_objective_sources",
            "expert_upside_score", "LWS Bonus", "R8-11 Score Increment",
            "R12-13 Score Increment",
        ]
        print(mentioned[cols].round(2).to_string(index=False))

    print("\nInterpretation:")
    print("- This is a close-call late-round nudge, not a sleeper list that overrides PatBot's football model.")
    print("- The maximum possible contribution is bounded by the existing round-phase upside weight.")
    print("- Two independent narrative-only takes are still ineligible; at least one objective-support source is required.")
    print("- Evidence will decay automatically with age through the existing recency half-life.\n")


if __name__ == "__main__":
    main()
