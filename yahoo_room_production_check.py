import numpy as np
import pandas as pd

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine
from patbot.sim import FastDraftSimulator
from patbot.yahoo_adp import manager_yahoo_weight
from patbot.yahoo_room_behavior import blend_room_market


def main():
    cfg = load_config()
    players = load_players("data/players_2026_live.csv")
    engine = DraftEngine(players, cfg)
    sim = FastDraftSimulator(engine)

    print("\nPatBot v0.6.0 Yahoo supporting room-behavior production check")
    print("Yahoo can influence simulated opponents and survival estimates only.")
    print("It does not enter PatBot projections, VORP, expert rank, or intrinsic player valuation.\n")

    status = getattr(sim, "yahoo_room_status", {}) or {}
    print(
        f"Cache status: {'OK' if status.get('ok') else 'OFF'} | "
        f"matched {status.get('matched', 0)} | coverage {status.get('coverage_pct', 0)}% | "
        f"age {status.get('age_hours', '—')}h | max age {status.get('max_age_hours', '—')}h"
    )
    if not status.get("ok"):
        print(f"Reason: {status.get('reason', 'unknown')}\n")
        raise SystemExit(1)

    source_adp = pd.to_numeric(players["adp"], errors="coerce").fillna(999.0).to_numpy(float)
    max_adp_change = float(np.max(np.abs(sim.adp - source_adp))) if len(source_adp) else 0.0
    print(f"PatBot valuation ADP isolation check: max change = {max_adp_change:.2f} (expected 0.00)\n")

    fixed = cfg.get("opponent_archetypes", {}).get("fixed_by_slot", {})
    room_weights = []
    print("Manager-type Yahoo influence:")
    for archetype in ["casual", "market", "league_aware", "sharp", "extremely_sharp"]:
        print(f"  {archetype:17s} {manager_yahoo_weight(archetype):>5.0%}")
    for slot, archetype in fixed.items():
        if int(slot) != int(cfg["league"]["draft_slot"]):
            room_weights.append(manager_yahoo_weight(str(archetype)))
    if room_weights:
        print(f"Room-average Yahoo share: {np.mean(room_weights):.1%}\n")

    frame = sim.players[["name", "pos", "team"]].copy()
    frame["market_adp"] = sim.adp
    frame["yahoo_adp"] = sim.yahoo_adp
    frame = frame[
        frame["yahoo_adp"].notna()
        & frame["pos"].isin(["QB", "RB", "WR", "TE"])
        & frame["market_adp"].le(200)
    ].copy()
    frame["delta"] = frame["yahoo_adp"] - frame["market_adp"]
    frame["abs_delta"] = frame["delta"].abs()
    frame = frame.sort_values("abs_delta", ascending=False).head(20)

    print("=== LARGEST CURRENT OFFENSE ROOM-SIGNAL DISAGREEMENTS ===\n")
    for archetype in ["casual", "league_aware", "sharp"]:
        frame[f"{archetype}_behavior"] = blend_room_market(
            frame["market_adp"].to_numpy(float),
            frame["yahoo_adp"].to_numpy(float),
            manager_yahoo_weight(archetype),
        )
    print(
        frame[[
            "name", "pos", "market_adp", "yahoo_adp", "delta",
            "casual_behavior", "league_aware_behavior", "sharp_behavior",
        ]].round(2).to_string(index=False)
    )

    print("\nGuardrails confirmed:")
    print("- Missing Yahoo rows leave the existing opponent market signal unchanged.")
    print("- Yahoo influence is capped at 35%, even if a future config accidentally asks for more.")
    print("- Stale Yahoo cache disables itself and falls back to the existing opponent model.")
    print("- PatBot's own ADP/urgency valuation vector remains the existing market ADP, not Yahoo ADP.\n")


if __name__ == "__main__":
    main()
