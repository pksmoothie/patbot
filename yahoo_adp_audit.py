from __future__ import annotations

from pathlib import Path

import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.yahoo_adp import behavioral_adp, fetch_yahoo_adp, manager_yahoo_weight

CSV_PATH = Path("data/players_2026_live.csv")
CACHE_PATH = Path("data/yahoo_adp_2026.csv")


def main():
    cfg = load_config()
    players = load_players(str(CSV_PATH))
    names = players["name"].dropna().astype(str).tolist()

    yahoo, status = fetch_yahoo_adp(names)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    yahoo.to_csv(CACHE_PATH, index=False)

    frame = players.merge(yahoo, on="name", how="left")
    frame["existing_market_adp"] = pd.to_numeric(frame.get("adp"), errors="coerce")
    frame["Yahoo minus Existing"] = frame["yahoo_adp"] - frame["existing_market_adp"]

    print("\nPatBot v0.5.9.2 Yahoo ADP / room-behavior audit")
    print("DIAGNOSTIC ONLY: Yahoo ADP is not yet changing PatBot production picks.\n")
    print(
        f"Yahoo rows matched: {status['matched']} | pages loaded: {status.get('pages_loaded', status['requests'])} | "
        f"transport: {status.get('transport', 'requests')} | cached locally: {CACHE_PATH}"
    )
    if status.get("browser_pages"):
        print(
            f"Yahoo served plain requests without the ADP table, so PatBot rendered "
            f"{status['browser_pages']} page(s) headlessly with the installed Edge/Chrome browser."
        )
    print("Source: Yahoo Draft Analysis -> snake/standard draft -> Avg Pick.")
    print("Purpose: help predict what this room is likely to do; Yahoo is a supporting behavioral signal, not player valuation and not the whole room model.\n")

    coverage = 100.0 * frame["yahoo_adp"].notna().mean()
    offense = frame[frame["pos"].isin(["QB", "RB", "WR", "TE"])]
    offense_coverage = 100.0 * offense["yahoo_adp"].notna().mean() if len(offense) else 0.0
    print(f"Coverage: {coverage:.1f}% all draftable players | {offense_coverage:.1f}% QB/RB/WR/TE")

    print("\n=== YAHOO ADP TOP 40 ===\n")
    top = frame[frame["yahoo_adp"].notna()].sort_values("yahoo_adp").head(40)
    print(
        top[["name", "pos", "team", "yahoo_adp", "existing_market_adp", "Yahoo minus Existing"]]
        .round(2)
        .to_string(index=False)
    )

    print("\n=== BIGGEST YAHOO VS CURRENT-MARKET DISAGREEMENTS (TOP 180 YAHOO) ===\n")
    disagreements = frame[
        frame["yahoo_adp"].notna()
        & frame["existing_market_adp"].notna()
        & frame["yahoo_adp"].le(180)
    ].copy()
    disagreements["abs_delta"] = disagreements["Yahoo minus Existing"].abs()
    disagreements = disagreements.sort_values("abs_delta", ascending=False).head(35)
    print(
        disagreements[["name", "pos", "team", "yahoo_adp", "existing_market_adp", "Yahoo minus Existing"]]
        .round(2)
        .to_string(index=False)
    )

    print("\n=== PROPOSED MANAGER-SPECIFIC YAHOO BOARD INFLUENCE ===\n")
    archetypes = ["casual", "market", "league_aware", "sharp", "extremely_sharp"]
    for archetype in archetypes:
        weight = manager_yahoo_weight(archetype)
        print(f"{archetype:17s}: Yahoo {weight:>5.0%} | existing market/custom behavior remains the balance")

    fixed = cfg.get("opponent_archetypes", {}).get("fixed_by_slot", {})
    weights = []
    for slot, archetype in fixed.items():
        if int(slot) == int(cfg["league"]["draft_slot"]):
            continue
        weights.append(manager_yahoo_weight(str(archetype)))
    if weights:
        print(f"\nRoom-average Yahoo anchoring across the 11 modeled opponents: {sum(weights)/len(weights):.1%}")

    print("\n=== EXAMPLE BEHAVIORAL ADP SHIFTS FOR CASUAL VS SHARP MANAGERS ===\n")
    examples = disagreements.head(20).copy()
    examples["casual_behavior_adp"] = behavioral_adp(
        examples["existing_market_adp"], examples["yahoo_adp"], manager_yahoo_weight("casual")
    )
    examples["sharp_behavior_adp"] = behavioral_adp(
        examples["existing_market_adp"], examples["yahoo_adp"], manager_yahoo_weight("sharp")
    )
    print(
        examples[[
            "name", "pos", "yahoo_adp", "existing_market_adp",
            "casual_behavior_adp", "sharp_behavior_adp",
        ]].round(2).to_string(index=False)
    )

    print("\nGuardrails for production integration:")
    print("- Yahoo ADP will never enter proj_points, VORP, expert rank, or Athletic/FantasyPros player-quality signals.")
    print("- It will be only one component of simulated opponent selection and survival estimates.")
    print("- Existing market signals, manager profile, roster needs and randomness remain in the opponent model.")
    print("- Casual managers receive more Yahoo influence than sharp managers, but Yahoo is not a majority input even for casuals.")
    print("- If Yahoo is unavailable or stale, behavior falls back to the existing market model rather than breaking the draft engine.\n")


if __name__ == "__main__":
    main()
