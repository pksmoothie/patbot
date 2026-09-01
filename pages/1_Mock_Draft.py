from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from patbot import __version__
from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine, all_team_picks
from patbot.draft_persistence import clear_draft_session, load_draft_session, save_draft_session
from patbot.draft_state import (
    drafted_ids_from_history,
    history_frame,
    make_pick_record,
    roster_ids_for_slot,
    roster_summary,
    team_slot_for_pick,
)
from patbot.final_call import run_final_call
from patbot.mock_draft import DEFAULT_MOCK_SEED, simulate_next_opponent_pick, simulate_opponents_until_patbot
from patbot.yahoo_room_behavior import load_yahoo_room_cache


LIVE_CSV = Path("data/players_2026_live.csv")
EXAMPLE_CSV = Path("data/example_players.csv")
MOCK_SESSION = Path("data/mock_draft_session_2026.json")


def _clear_mock_cache() -> None:
    for key in (
        "mock_final_signature",
        "mock_final_result",
        "mock_last_simulated",
    ):
        st.session_state.pop(key, None)


st.set_page_config(page_title="PatBot Mock Draft", layout="wide")
st.title("PatBot — Controlled Mock Draft")
st.caption(
    f"v{__version__} • isolated rehearsal mode • production opponent model • "
    "manual chaos overrides • separate crash-safe mock session"
)

cfg = load_config()
teams = int(cfg["league"]["teams"])
slot = int(cfg["league"]["draft_slot"])
draft_order = {int(k): str(v) for k, v in cfg["league"].get("draft_order", {}).items()}
team_names = {
    i: draft_order.get(i, "PatBot" if i == slot else f"Slot {i}")
    for i in range(1, teams + 1)
}
team_names[slot] = "PatBot"

player_path = LIVE_CSV if LIVE_CSV.exists() else EXAMPLE_CSV
if not LIVE_CSV.exists():
    st.warning("Live player snapshot is missing; this rehearsal is using example data.")
players = load_players(str(player_path))
players["player_id"] = players["player_id"].astype(str)
engine = DraftEngine(players, cfg)
player_by_id = players.set_index("player_id", drop=False)
id_to_pos = dict(zip(players["player_id"], players["pos"]))

yahoo_rows, yahoo_status = load_yahoo_room_cache(players["name"].astype(str).tolist())

history = load_draft_session(MOCK_SESSION)
drafted_ids = drafted_ids_from_history(history)
my_roster_ids = roster_ids_for_slot(history, slot)
current_pick = len(history) + 1
current_owner_slot = team_slot_for_pick(current_pick, teams)
current_owner = team_names.get(current_owner_slot, f"Slot {current_owner_slot}")
my_picks = set(all_team_picks(teams, slot, rounds=20))
is_my_pick = current_pick in my_picks

available = players[~players["player_id"].isin(drafted_ids)].copy()
available = available.sort_values(["adp", "name"], na_position="last")
available_ids = available["player_id"].astype(str).tolist()


def player_label(pid: str) -> str:
    row = player_by_id.loc[str(pid)]
    adp = pd.to_numeric(pd.Series([row.get("adp")]), errors="coerce").iloc[0]
    adp_text = f" • ADP {adp:.1f}" if pd.notna(adp) else ""
    return f"{row['name']} — {row['team']} {row['pos']}{adp_text}"


def record_player(pid: str) -> None:
    row = player_by_id.loc[str(pid)]
    new_history = list(history)
    new_history.append(
        make_pick_record(
            overall_pick=current_pick,
            teams=teams,
            player_id=str(pid),
            player_name=row["name"],
            nfl_team=row["team"],
            pos=row["pos"],
        )
    )
    save_draft_session(new_history, MOCK_SESSION)
    _clear_mock_cache()


# Sidebar controls are intentionally separate from the production draft page.
st.sidebar.header("Controlled mock")
st.sidebar.metric("Current overall pick", current_pick)
st.sidebar.write(f"**On the clock:** {current_owner} (slot {current_owner_slot})")
st.sidebar.caption(f"Mock room seed: {DEFAULT_MOCK_SEED}")
st.sidebar.caption(f"Persistence: `{MOCK_SESSION}`")
if yahoo_status.get("ok"):
    st.sidebar.success(
        f"Yahoo room signal ON • {yahoo_status.get('matched', len(yahoo_rows))} matches"
    )
else:
    st.sidebar.warning("Yahoo room signal unavailable; base opponent model remains active.")

if not is_my_pick and available_ids:
    if st.sidebar.button("Simulate next opponent pick", type="primary", use_container_width=True):
        result = simulate_next_opponent_pick(
            engine,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            draft_history=history,
            seed=DEFAULT_MOCK_SEED,
        )
        record_player(result["player_id"])
        st.session_state.mock_last_simulated = [result]
        st.rerun()

    if st.sidebar.button("Simulate opponents to PatBot", use_container_width=True):
        new_history, simulated = simulate_opponents_until_patbot(
            engine,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            draft_history=history,
            make_record=make_pick_record,
            seed=DEFAULT_MOCK_SEED,
        )
        save_draft_session(new_history, MOCK_SESSION)
        _clear_mock_cache()
        st.session_state.mock_last_simulated = simulated
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Manual / chaos override")
manual_pid = st.sidebar.selectbox(
    "Force the current pick",
    options=["—"] + available_ids,
    format_func=lambda x: "—" if x == "—" else player_label(x),
    key="mock_manual_pid",
)
if st.sidebar.button(
    f"Record for {current_owner}",
    use_container_width=True,
    disabled=(manual_pid == "—"),
):
    record_player(str(manual_pid))
    st.rerun()

u1, u2 = st.sidebar.columns(2)
if u1.button("Undo", use_container_width=True, disabled=(len(history) == 0)):
    save_draft_session(history[:-1], MOCK_SESSION)
    _clear_mock_cache()
    st.rerun()
if u2.button("Reset", use_container_width=True):
    clear_draft_session(MOCK_SESSION)
    _clear_mock_cache()
    st.rerun()

if history:
    st.info(
        f"Mock session has {len(history)} persisted picks. Closing/restarting PatBot will reload this rehearsal without touching the live draft session."
    )
else:
    st.info("Fresh controlled mock. Simulate Paul and Faherty to reach PatBot at 1.03.")

last_simulated = st.session_state.get("mock_last_simulated") or []
if last_simulated:
    text = " • ".join(
        f"{x['overall_pick']}. {team_names.get(int(x['owner_slot']), x['owner_slot'])}: {x['player_name']}"
        for x in last_simulated[-8:]
    )
    st.caption(f"Last simulated opponent picks: {text}")

if is_my_pick:
    st.success(f"PATBOT IS ON THE CLOCK — PICK {current_pick}")
else:
    nxt = next((p for p in sorted(my_picks) if p > current_pick), None)
    if nxt:
        st.info(f"Next PatBot pick: {nxt}")

roster_positions = [id_to_pos[x] for x in my_roster_ids if x in id_to_pos]
board = engine.recommend(
    current_pick=current_pick,
    drafted_ids=drafted_ids,
    roster_positions=roster_positions,
    top_n=18,
)

final_call = None
if is_my_pick and not board.empty:
    data_mtime = player_path.stat().st_mtime_ns if player_path.exists() else 0
    signature = (
        current_pick,
        tuple((int(x.get("overall_pick", 0)), str(x.get("player_id", ""))) for x in history),
        data_mtime,
        tuple(board.head(6)["player_id"].astype(str).tolist()),
    )
    if st.session_state.get("mock_final_signature") != signature:
        with st.spinner("PatBot is making the Final Call for the controlled room..."):
            st.session_state.mock_final_result = run_final_call(
                engine,
                current_pick=current_pick,
                drafted_ids=drafted_ids,
                my_roster_ids=my_roster_ids,
                board=board,
                draft_history=history,
            )
            st.session_state.mock_final_signature = signature
    final_call = st.session_state.get("mock_final_result")

left, right = st.columns([2.6, 1])
with left:
    st.subheader("Final Call")
    if not is_my_pick:
        st.write("Advance the opponent room until PatBot is on the clock.")
    elif not final_call:
        st.warning("No Final Call available.")
    else:
        rec = str(final_call.get("recommendation", board.iloc[0]["name"]))
        st.success(f"FINAL CALL — DRAFT **{rec}**")
        m1, m2, m3, m4 = st.columns(4)
        edge = final_call.get("edge")
        m1.metric("Final sim edge", "—" if edge is None else f"+{float(edge):.2f}")
        m2.metric("Edge strength", str(final_call.get("edge_label", "—")))
        m3.metric("Paired runs", int(final_call.get("runs", 0)))
        m4.metric("Elapsed", f"{float(final_call.get('elapsed_seconds', 0.0)):.1f}s")
        st.write(str(final_call.get("reason", "")))
        st.caption(
            f"Stage: {final_call.get('stage', '—')} • simulation horizon Round {final_call.get('through_round', '—')}"
        )
        if st.button("Draft the Final Call", type="primary"):
            rec_id = str(final_call.get("candidate_id", ""))
            if rec_id not in set(available_ids):
                st.error("Final Call candidate is no longer available; inspect the mock state before continuing.")
            else:
                record_player(rec_id)
                st.rerun()

    st.divider()
    st.subheader("Base score board")
    if board.empty:
        st.write("No available players.")
    else:
        preferred = [
            "name", "team", "pos", "score", "proj_points", "vorp", "risk_score",
            "adp", "expert_rank", "consensus_tier", "survive_next_pct", "scarcity",
            "league_winner_score", "bye",
        ]
        cols = [c for c in preferred if c in board.columns]
        view = board[cols].rename(columns={
            "name": "Player", "team": "Team", "pos": "Pos", "score": "PatBot",
            "proj_points": "Proj", "vorp": "VORP", "risk_score": "Risk",
            "adp": "Market ADP", "expert_rank": "Expert RK", "consensus_tier": "Tier",
            "survive_next_pct": "Market Survive Next %", "scarcity": "4-Player Drop",
            "league_winner_score": "LWS", "bye": "Bye",
        })
        st.dataframe(view, use_container_width=True, hide_index=True)

with right:
    st.subheader("PatBot roster")
    mine = players[players["player_id"].isin(my_roster_ids)].copy()
    if mine.empty:
        st.write("No selections yet.")
    else:
        order = {pid: i for i, pid in enumerate(my_roster_ids)}
        mine["_order"] = mine["player_id"].map(order)
        mine = mine.sort_values("_order")
        st.dataframe(
            mine[[c for c in ["name", "team", "pos", "proj_points", "risk_score", "bye"] if c in mine.columns]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Room snapshot")
    st.dataframe(
        roster_summary(history, teams, team_names),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("Mock draft log")
log = history_frame(history, team_names)
if log.empty:
    st.write("No mock picks recorded yet.")
else:
    st.dataframe(log, use_container_width=True, hide_index=True)
st.caption(
    "This page is diagnostic only. Its history is stored separately from the production Draft Day session."
)
