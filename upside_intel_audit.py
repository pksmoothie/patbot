from __future__ import annotations

import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.sim import FastDraftSimulator
from patbot.upside_intel import score_upside_evidence


CSV_PATH = "data/players_2026_live.csv"


def main():
    cfg = load_config()
    players = load_players(CSV_PATH)
    enriched, status = score_upside_evidence(players, cfg)

    # Base LWS comes from the current production model. v0.5.6 does not modify it.
    engine = DraftEngine(enriched, cfg)
    sim = FastDraftSimulator(engine)
    base_lws = pd.Series(sim.league_winner_score, index=enriched.index, dtype=float)

    report = enriched[[
        "name", "pos", "team", "adp", "proj_points", "expert_upside_score",
        "expert_upside_sources", "expert_upside_objective_sources",
        "expert_upside_eligible", "expert_upside_lws_bonus", "expert_upside_note",
    ]].copy()
    report["Base LWS"] = base_lws
    report["Diagnostic LWS"] = (report["Base LWS"] + report["expert_upside_lws_bonus"]).clip(upper=100.0)
    report["LWS Delta"] = report["Diagnostic LWS"] - report["Base LWS"]
    report["Approx Round"] = ((pd.to_numeric(report["adp"], errors="coerce") - 1.0) / 12.0 + 1.0).round(1)

    mentioned = report[report["expert_upside_sources"] > 0].copy()
    mentioned = mentioned.sort_values(
        ["expert_upside_eligible", "expert_upside_score", "adp"],
        ascending=[False, False, True],
    )

    print("\nPatBot v0.5.6 expert sleeper/upside intelligence audit")
    print("DIAGNOSTIC ONLY: this does not change projections, VORP, expert rank, ADP, or production LWS.\n")
    print(
        f"Evidence items: {status['evidence_items']} | matched items: {status['matched_items']} | "
        f"players mentioned: {status['matched_players']} | corroborated/eligible: {status['eligible_players']}"
    )
    print(
        f"Rule: at least {status['minimum_independent_sources']} independent publishers before any future bonus; "
        f"maximum possible LWS bonus {status['max_lws_bonus']:.1f} points."
    )
    if status["unmatched_players"]:
        print("Unmatched evidence names: " + ", ".join(status["unmatched_players"]))

    print("\n=== CURATED EXPERT UPSIDE EVIDENCE ===\n")
    if mentioned.empty:
        print("No evidence matched the current player snapshot.")
    else:
        cols = [
            "name", "pos", "team", "adp", "Approx Round", "proj_points",
            "Base LWS", "expert_upside_score", "expert_upside_sources",
            "expert_upside_objective_sources", "expert_upside_eligible",
            "expert_upside_lws_bonus", "Diagnostic LWS", "LWS Delta",
        ]
        print(mentioned[cols].round(2).to_string(index=False))

    print("\n=== SOURCE / CORROBORATION NOTES ===\n")
    for _, row in mentioned.iterrows():
        print(
            f"{row['name']}: {int(row['expert_upside_sources'])} independent source(s) | "
            f"score {row['expert_upside_score']:.1f} | {row['expert_upside_note']}"
        )

    print("\nGuardrails:")
    print("- This is a positive-only sleeper/upside signal, not a replacement projection or ranking source.")
    print("- Multiple FantasyPros articles still count as one independent publisher; same-source repetition cannot manufacture corroboration.")
    print("- Recency, source quality, specificity and objective usage/role evidence all affect the score.")
    print("- A one-source hot take is visible but gets zero production-eligible bonus.")
    print("- If promoted later, the bonus will flow only through late-round LWS weighting, so it cannot overpower early-round value.\n")


if __name__ == "__main__":
    main()
