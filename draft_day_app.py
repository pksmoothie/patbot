from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from patbot import __version__
from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine, all_team_picks
from patbot.draft_persistence import (
    clear_draft_session,
    load_draft_session,
    save_draft_session,
)
from patbot.draft_state import (
    drafted_ids_from_history,
    history_frame,
    make_pick_record,
    roster_ids_for_slot,
    roster_summary,
    team_slot_for_pick,
)
from patbot.fast_refresh_pipeline import run_fast_refresh
from patbot.refresh_pipeline import run_full_refresh
from patbot.sim import compare_candidates
from patbot.yahoo_room_behavior import load_yahoo_room_cache


LIVE_CSV = Path("data/players_2026_live.csv")
LIVE_META = Path("data/players_2026_live.meta.json")
EXAMPLE_CSV = Path("data/example_players.csv")


@st.cache_data(show_spinner=False)
def _load_meta(path_text: str, mtime_ns: int) -> dict:
    del mtime_ns
    path = Path(path_text)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clear_cached_outputs() -> None:
    for key in ["sim_summary", "sim_details"]:
        st.session_state.pop(key, None)


def _age_text(timestamp) -> str:
    if not timestamp:
        return "unknown"
    try:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        now = pd.Timestamp.now(tz="UTC")
        hours = max(0.0, (now - ts).total_seconds() / 3600.0)
        return f"{hours:.1f}h ago"
    except Exception:
        return str(timestamp)


st.set_page_config(page_title="PatBot Draft Day", layout="wide")
st.title("PatBot — 2026 Draft Day")
st.caption(
    f"v{__version__} • production projections • fast injury/news layer • "
    "expert late-round upside • fixed-manager room model • Yahoo supporting room signal"
)

cfg = load_config()
teams = int(cfg["league"]["teams"])
slot = int(cfg["league"]["draft_slot"])
draft_order_cfg = {int(k): str(v) for k, v in cfg["league"].get("draft_order", {}).items()}
manager_cfg = cfg.get("opponent_managers", {})
fixed_archetypes = cfg.get("opponent_archetypes", {}).get("fixed_by_slot", {})

st.sidebar.header("Draft-day controls")
st.sidebar.caption(
    "Yahoo API access is not required for draft recommendations. Record actual Yahoo picks here; "
    "PatBot uses the public Yahoo ADP cache only as a supporting room-behavior input."
)

if st.sidebar.button("Full pre-draft refresh", use_container_width=True):
    with st.spinner("Running the full production refresh chain..."):
        try:
            _, _, refreshed_meta = run_full_refresh(cfg, LIVE_CSV, LIVE_META)
            st.cache_data.clear()
            _clear_cached_outputs()
            st.sidebar.success(f"Full refresh complete: {refreshed_meta.get('draftable_rows', '?')} players")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"{type(exc).__name__}: {exc}")

if st.sidebar.button("Fast injury/news refresh", use_container_width=True):
    with st.spinner("Refreshing current injury/news risk only..."):
        try:
            _, _, alerts, elapsed = run_fast_refresh(cfg, LIVE_CSV, LIVE_META)
            st.cache_data.clear()
            _clear_cached_outputs()
            st.sidebar.success(f"Fast refresh complete in {elapsed:.1f}s • {len(alerts)} material alerts")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"{type(exc).__name__}: {exc}")

player_path = LIVE_CSV if LIVE_CSV.exists() else EXAMPLE_CSV
if not LIVE_CSV.exists():
    st.warning("No live player snapshot is present. Run Full pre-draft refresh before using PatBot for the real draft.")

try:
    players = load_players(str(player_path))
except Exception as exc:
    st.error(f"Could not load player data: {exc}")
    st.stop()
players["player_id"] = players["player_id"].astype(str)

meta_mtime = LIVE_META.stat().st_mtime_ns if LIVE_META.exists() else 0
meta = _load_meta(str(LIVE_META), meta_mtime)
engine = DraftEngine(players, cfg)

# Readiness panel.
st.sidebar.header("Readiness")
st.sidebar.write(f"**PatBot:** v{__version__}")
st.sidebar.write(f"**Player rows:** {len(players)}")
st.sidebar.write(f"**Slow snapshot:** {_age_text(meta.get('snapshot_at_utc'))}")
st.sidebar.write(f"**Fast risk:** {_age_text(meta.get('fast_risk_refreshed_at_utc'))}")

blend_count = pd.to_numeric(
    players.get("projection_blend_source_count", pd.Series([0] * len(players))),
    errors="coerce",
).fillna(0)
offense = players["pos"].astype(str).isin(["QB", "RB", "WR", "TE"])
blend_pct = 100.0 * float(((blend_count >= 2) & offense).sum()) / max(int(offense.sum()), 1)
st.sidebar.write(f"**2-source offense projection coverage:** {blend_pct:.1f}%")

yahoo_rows, yahoo_status = load_yahoo_room_cache(players["name"].astype(str).tolist())
if yahoo_status.get("ok"):
    st.sidebar.success(
        f"Yahoo room signal ON • {yahoo_status.get('matched', len(yahoo_rows))} matches • "
        f"{yahoo_status.get('age_hours', '?')}h old"
    )
else:
    st.sidebar.warning(
        f"Yahoo room signal OFF ({yahoo_status.get('reason', 'unavailable')}); base market room model remains active."
    )

# Crash-safe draft history.
if "draft_history" not in st.session_state:
    restored = load_draft_session()
    st.session_state.draft_history = restored
    st.session_state.restored_draft_session = bool(restored)
if "team_names" not in st.session_state:
    st.session_state.team_names = {
        i: draft_order_cfg.get(i, "PatBot" if i == slot else f"Slot {i}")
        for i in range(1, teams + 1)
    }
for i in range(1, teams + 1):
    if i in draft_order_cfg:
        st.session_state.team_names[i] = draft_order_cfg[i]
st.session_state.team_names[slot] = "PatBot"

if st.session_state.get("restored_draft_session"):
    st.info(
        f"Restored {len(st.session_state.draft_history)} recorded picks from the local draft-session backup."
    )
    st.session_state.restored_draft_session = False

draft_history = st.session_state.draft_history
drafted_ids = drafted_ids_from_history(draft_history)
my_roster_ids = roster_ids_for_slot(draft_history, slot)
current_pick = len(draft_history) + 1
current_owner_slot = team_slot_for_pick(current_pick, teams)
current_owner_name = st.session_state.team_names.get(current_owner_slot, f"Slot {current_owner_slot}")

id_to_pos = dict(zip(players["player_id"], players["pos"]))
player_by_id = players.set_index("player_id", drop=False)
available = players[~players["player_id"].isin(drafted_ids)].copy()
available = available.sort_values(["adp", "name"], na_position="last")
available_ids = available["player_id"].tolist()

st.sidebar.header("Live draft entry")
st.sidebar.metric("Current overall pick", current_pick)
st.sidebar.write(f"**On the clock:** {current_owner_name} (slot {current_owner_slot})")


def player_label(pid: str) -> str:
    row = player_by_id.loc[str(pid)]
    adp = pd.to_numeric(pd.Series([row.get("adp")]), errors="coerce").iloc[0]
    adp_text = f" • ADP {adp:.1f}" if pd.notna(adp) else ""
    return f"{row['name']} — {row['team']} {row['pos']}{adp_text}"


selected_pid = st.sidebar.selectbox(
    "Record player selected",
    options=["—"] + available_ids,
    format_func=lambda x: "—" if x == "—" else player_label(x),
)
record_label = "Draft for PatBot" if current_owner_slot == slot else f"Record for {current_owner_name}"

if st.sidebar.button(
    record_label,
    use_container_width=True,
    type="primary",
    disabled=(selected_pid == "—"),
):
    row = player_by_id.loc[str(selected_pid)]
    st.session_state.draft_history.append(
        make_pick_record(
            overall_pick=current_pick,
            teams=teams,
            player_id=str(selected_pid),
            player_name=row["name"],
            nfl_team=row["team"],
            pos=row["pos"],
        )
    )
    save_draft_session(st.session_state.draft_history)
    _clear_cached_outputs()
    st.rerun()

u1, u2 = st.sidebar.columns(2)
if u1.button("Undo last", use_container_width=True, disabled=(len(draft_history) == 0)):
    st.session_state.draft_history.pop()
    save_draft_session(st.session_state.draft_history)
    _clear_cached_outputs()
    st.rerun()
if u2.button("Reset draft", use_container_width=True):
    st.session_state.draft_history = []
    clear_draft_session()
    _clear_cached_outputs()
    st.rerun()

my_picks = all_team_picks(teams, slot)
is_my_pick = current_pick in my_picks
if is_my_pick:
    st.success(f"PATBOT IS ON THE CLOCK — PICK {current_pick}")
else:
    nxt = next((p for p in my_picks if p > current_pick), None)
    if nxt:
        st.info(f"Next PatBot pick: {nxt}")

roster_positions = [id_to_pos[x] for x in my_roster_ids if x in id_to_pos]
board = engine.recommend(
    current_pick=current_pick,
    drafted_ids=drafted_ids,
    roster_positions=roster_positions,
    top_n=18,
)

tab_board, tab_rosters, tab_log, tab_room, tab_sim = st.tabs(
    ["Draft Board", "Team Rosters", "Draft Log", "Room Model", "Simulation Lab"]
)

with tab_board:
    left, right = st.columns([2.7, 1])
    with left:
        st.subheader("Live recommendations")
        st.caption(
            "Market Survive Next % is the direct-board ADP survival heuristic. Yahoo's supporting signal "
            "is currently used in the opponent simulations in Simulation Lab, not as intrinsic player value."
        )
        if board.empty:
            st.write("No available players.")
        else:
            preferred = [
                "name", "team", "pos", "score", "proj_points", "vorp",
                "risk_score", "adp", "expert_rank", "consensus_tier",
                "survive_next_pct", "scarcity", "league_winner_score",
                "expert_upside_lws_bonus", "expert_upside_score_increment", "bye",
            ]
            cols = [c for c in preferred if c in board.columns]
            view = board[cols].rename(columns={
                "name": "Player", "team": "Team", "pos": "Pos", "score": "PatBot",
                "proj_points": "Proj", "vorp": "VORP", "risk_score": "Risk",
                "adp": "Market ADP", "expert_rank": "Expert RK", "consensus_tier": "Tier",
                "survive_next_pct": "Market Survive Next %", "scarcity": "4-Player Drop",
                "league_winner_score": "LWS", "expert_upside_lws_bonus": "Expert LWS Bonus",
                "expert_upside_score_increment": "Expert Score +", "bye": "Bye",
            })
            st.dataframe(view, use_container_width=True, hide_index=True)
            st.subheader("Current call")
            st.write(engine.explain_row(board.iloc[0]))

    with right:
        st.subheader("PatBot roster")
        mine = players[players["player_id"].isin(my_roster_ids)].copy()
        if mine.empty:
            st.write("No selections yet.")
        else:
            order = {pid: i for i, pid in enumerate(my_roster_ids)}
            mine["_order"] = mine["player_id"].map(order)
            mine = mine.sort_values("_order")
            cols = [c for c in ["name", "team", "pos", "proj_points", "risk_score", "bye"] if c in mine.columns]
            st.dataframe(mine[cols], use_container_width=True, hide_index=True)

        st.subheader("Material risk alerts")
        material = players.get("current_status_material")
        if material is None:
            st.write("No fast-risk status in this snapshot.")
        else:
            alert_view = players[material.fillna(False).astype(bool)].copy()
            alert_cols = [c for c in ["name", "team", "pos", "current_injury_status", "current_status_source", "fast_news_title", "risk_score"] if c in alert_view.columns]
            if alert_view.empty:
                st.write("No material alerts.")
            else:
                st.dataframe(alert_view[alert_cols].head(20), use_container_width=True, hide_index=True)

with tab_rosters:
    st.subheader("Live league rosters")
    st.dataframe(
        roster_summary(draft_history, teams, st.session_state.team_names),
        use_container_width=True,
        hide_index=True,
    )

with tab_log:
    st.subheader("Draft log")
    log = history_frame(draft_history, st.session_state.team_names)
    if log.empty:
        st.write("No selections recorded yet.")
    else:
        st.dataframe(log, use_container_width=True, hide_index=True)
    st.caption("Every recorded pick is also written to data/draft_session_2026.json for crash recovery.")

with tab_room:
    st.subheader("2026 room model")
    room_rows = []
    for i in range(1, teams + 1):
        manager = draft_order_cfg.get(i, st.session_state.team_names.get(i, f"Slot {i}"))
        if i == slot:
            profile = "PatBot"
            notes = "Our draft engine"
        else:
            raw = manager_cfg.get(i) or manager_cfg.get(str(i)) or {}
            profile = raw.get("archetype", fixed_archetypes.get(i, fixed_archetypes.get(str(i), "market")))
            notes = raw.get("notes", "")
        room_rows.append({"Slot": i, "Manager": manager, "Profile": str(profile).replace("_", " ").title(), "Notes": notes})
    st.dataframe(pd.DataFrame(room_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Yahoo is a supporting behavioral nudge inside simulated opponent decisions; manager profile, "
        "existing market signal, roster need, history and randomness remain active."
    )

with tab_sim:
    st.subheader("Yahoo-informed Monte Carlo Draft Lab")
    if not is_my_pick:
        st.write("Record the real picks ahead of PatBot first; run comparisons when PatBot is on the clock.")
    elif board.empty:
        st.write("No candidates available.")
    else:
        available_board = players[~players["player_id"].isin(drafted_ids)].copy()
        defaults = board.head(6)["name"].tolist()
        selected_candidates = st.multiselect(
            "Candidates to compare",
            options=available_board["name"].sort_values().tolist(),
            default=defaults,
        )
        lookup = dict(zip(available_board["name"], available_board["player_id"].astype(str)))
        candidate_ids = [lookup[x] for x in selected_candidates if x in lookup]
        runs = st.slider(
            "Simulations per candidate",
            min_value=100,
            max_value=int(cfg.get("simulation", {}).get("max_runs", 1500)),
            value=int(cfg.get("simulation", {}).get("default_runs", 300)),
            step=100,
        )
        st.caption(
            "These simulations use the fixed manager profiles, live recorded rosters, Yahoo's supporting "
            "room signal when fresh, the production risk model, and common random numbers across candidates."
        )
        if len(candidate_ids) < 2:
            st.warning("Select at least two candidates.")
        elif st.button("Run candidate simulations", type="primary"):
            with st.spinner(f"Running {runs * len(candidate_ids):,} paired draft paths..."):
                summary, details = compare_candidates(
                    engine,
                    current_pick=current_pick,
                    drafted_ids=drafted_ids,
                    my_roster_ids=my_roster_ids,
                    candidate_ids=candidate_ids,
                    runs=runs,
                    through_round=int(cfg.get("simulation", {}).get("through_round", 8)),
                    draft_history=draft_history,
                )
                st.session_state.sim_summary = summary
                st.session_state.sim_details = details

        if "sim_summary" in st.session_state:
            st.dataframe(st.session_state.sim_summary, use_container_width=True, hide_index=True)
            for detail in st.session_state.sim_details:
                with st.expander(detail["candidate"]):
                    st.write(
                        f"Average lineup score: **{detail['avg_lineup_score']:.2f}** • "
                        f"P10: **{detail['p10_lineup_score']:.2f}** • "
                        f"P25–P75: **{detail['p25_lineup_score']:.2f}–{detail['p75_lineup_score']:.2f}**"
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Second PatBot pick**")
                        st.dataframe(pd.DataFrame(detail["most_common_second_pick"]), hide_index=True)
                    with c2:
                        st.write("**Third PatBot pick**")
                        st.dataframe(pd.DataFrame(detail["most_common_third_pick"]), hide_index=True)

st.divider()
st.caption(
    "Draft-day mode is intentionally narrower than the research app: reliable refresh, manual pick capture, "
    "crash recovery, live board, room state and candidate simulations."
)
