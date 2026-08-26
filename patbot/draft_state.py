from __future__ import annotations

from collections import defaultdict
import pandas as pd


# Local app context for strategy layers that need player identity (not just
# position counts). The Streamlit draft room calls roster_ids_for_slot directly
# before building the recommendation board.
_LAST_ROSTER_IDS: list[str] = []


def round_for_pick(overall_pick: int, teams: int) -> int:
    return ((int(overall_pick) - 1) // int(teams)) + 1


def pick_in_round(overall_pick: int, teams: int) -> int:
    return ((int(overall_pick) - 1) % int(teams)) + 1


def team_slot_for_pick(overall_pick: int, teams: int) -> int:
    """Return the draft slot that owns an overall pick in a snake draft."""
    rnd = round_for_pick(overall_pick, teams)
    within = pick_in_round(overall_pick, teams)
    return within if rnd % 2 else int(teams) + 1 - within


def drafted_ids_from_history(history: list[dict]) -> set[str]:
    return {
        str(p["player_id"])
        for p in history
        if p.get("player_id") is not None
    }


def roster_ids_for_slot(history: list[dict], slot: int) -> list[str]:
    global _LAST_ROSTER_IDS
    result = [
        str(p["player_id"])
        for p in history
        if int(p.get("owner_slot", -1)) == int(slot)
        and p.get("player_id") is not None
    ]
    _LAST_ROSTER_IDS = list(result)
    return result


def last_roster_ids() -> list[str]:
    return list(_LAST_ROSTER_IDS)


def make_pick_record(
    overall_pick: int,
    teams: int,
    player_id: str,
    player_name: str,
    nfl_team: str,
    pos: str,
) -> dict:
    owner_slot = team_slot_for_pick(overall_pick, teams)
    return {
        "overall_pick": int(overall_pick),
        "round": round_for_pick(overall_pick, teams),
        "pick_in_round": pick_in_round(overall_pick, teams),
        "owner_slot": int(owner_slot),
        "player_id": str(player_id),
        "player": str(player_name),
        "nfl_team": str(nfl_team),
        "pos": str(pos),
    }


def roster_summary(
    history: list[dict],
    teams: int,
    team_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """One row per draft slot, with the real drafted players by position."""
    team_names = team_names or {}
    by_slot = defaultdict(lambda: defaultdict(list))

    for pick in history:
        slot = int(pick["owner_slot"])
        pos = str(pick.get("pos", ""))
        name = str(pick.get("player", ""))
        by_slot[slot][pos].append(name)

    rows = []
    for slot in range(1, int(teams) + 1):
        pdata = by_slot[slot]
        rows.append({
            "Slot": slot,
            "Manager": team_names.get(slot, f"Slot {slot}"),
            "QB": ", ".join(pdata.get("QB", [])) or "—",
            "RB": ", ".join(pdata.get("RB", [])) or "—",
            "WR": ", ".join(pdata.get("WR", [])) or "—",
            "TE": ", ".join(pdata.get("TE", [])) or "—",
            "K": ", ".join(pdata.get("K", [])) or "—",
            "DEF": ", ".join(pdata.get("DEF", [])) or "—",
            "Picks": sum(len(v) for v in pdata.values()),
        })

    return pd.DataFrame(rows)


def history_frame(
    history: list[dict],
    team_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    team_names = team_names or {}
    if not history:
        return pd.DataFrame(columns=[
            "Pick", "Round", "Slot", "Manager", "Player", "NFL", "Pos"
        ])

    rows = []
    for p in history:
        slot = int(p["owner_slot"])
        rows.append({
            "Pick": int(p["overall_pick"]),
            "Round": int(p["round"]),
            "Slot": slot,
            "Manager": team_names.get(slot, f"Slot {slot}"),
            "Player": p["player"],
            "NFL": p.get("nfl_team", ""),
            "Pos": p.get("pos", ""),
        })
    return pd.DataFrame(rows)
