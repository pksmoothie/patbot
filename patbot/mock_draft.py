from __future__ import annotations

import numpy as np
import pandas as pd

from .draft_state import team_slot_for_pick
from .sim import FastDraftSimulator


DEFAULT_MOCK_SEED = 20260901


def simulate_next_opponent_pick(
    engine,
    *,
    current_pick: int,
    drafted_ids: set[str],
    draft_history: list[dict] | None = None,
    seed: int = DEFAULT_MOCK_SEED,
) -> dict:
    """Choose one opponent pick using PatBot's installed production room model.

    The room latents are regenerated from one fixed mock seed on every call, so
    advancing a mock one click at a time reproduces the same coherent room path
    that would result from keeping one latent room draw for the whole draft.
    Live draft history supplies current roster counts and ownership identity.
    """
    sim = FastDraftSimulator(engine)
    pick = int(current_pick)
    owner_slot = int(team_slot_for_pick(pick, sim.teams))
    if owner_slot == sim.slot:
        raise ValueError(f"Pick {pick} belongs to PatBot; opponent simulation must stop here.")

    available = np.ones(sim.n, dtype=bool)
    drafted_idx = [
        sim.id_to_idx[str(pid)]
        for pid in drafted_ids
        if str(pid) in sim.id_to_idx
    ]
    if drafted_idx:
        available[drafted_idx] = False
    if not available.any():
        raise RuntimeError("No players remain available for the mock room.")

    # _seed_opponent_counts is patched in production to restore manager rosters
    # and PatBot/opponent identity sets from the recorded draft history.
    opp_counts = sim._seed_opponent_counts(draft_history)

    rng = np.random.default_rng(int(seed))
    archetypes = sim._archetype_assignments(rng)
    latent_sd = np.maximum(sim.sd_floor, sim.adp * sim.sd_pct)
    market_latent = np.maximum(1.0, rng.normal(sim.adp, latent_sd))
    custom_noise_base = rng.normal(
        0.0,
        np.maximum(3.0, sim.custom_rank * 0.06),
    )

    idx, archetype = sim._take_opponent_pick(
        pick,
        available,
        opp_counts,
        archetypes,
        market_latent,
        custom_noise_base,
    )
    idx = int(idx)
    row = sim.players.iloc[idx]
    profile = sim._manager_profile(owner_slot, str(archetype))

    return {
        "overall_pick": pick,
        "owner_slot": owner_slot,
        "archetype": str(archetype),
        "player_id": str(sim.ids[idx]),
        "player_name": str(sim.names[idx]),
        "team": str(row.get("team", "")),
        "pos": str(sim.pos[idx]),
        "adp": float(sim.adp[idx]),
        "profile": dict(profile),
        "seed": int(seed),
    }


def simulate_opponents_until_patbot(
    engine,
    *,
    current_pick: int,
    drafted_ids: set[str],
    draft_history: list[dict],
    make_record,
    seed: int = DEFAULT_MOCK_SEED,
    max_picks: int = 24,
) -> tuple[list[dict], list[dict]]:
    """Advance opponents until PatBot is on the clock, returning history + metadata."""
    history = [dict(x) for x in draft_history]
    drafted = {str(x) for x in drafted_ids}
    simulated: list[dict] = []
    pick = int(current_pick)

    for _ in range(max(1, int(max_picks))):
        owner_slot = int(team_slot_for_pick(pick, int(engine.league["teams"])))
        if owner_slot == int(engine.league["draft_slot"]):
            break
        result = simulate_next_opponent_pick(
            engine,
            current_pick=pick,
            drafted_ids=drafted,
            draft_history=history,
            seed=int(seed),
        )
        history.append(
            make_record(
                overall_pick=pick,
                teams=int(engine.league["teams"]),
                player_id=result["player_id"],
                player_name=result["player_name"],
                nfl_team=result["team"],
                pos=result["pos"],
            )
        )
        drafted.add(result["player_id"])
        simulated.append(result)
        pick += 1

    return history, simulated
