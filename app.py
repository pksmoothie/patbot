from pathlib import Path
import json

import pandas as pd
import streamlit as st

from patbot.config import load_config
from patbot.data import load_players
from patbot.draft import DraftEngine, all_team_picks
from patbot.draft_state import (
    drafted_ids_from_history,
    history_frame,
    make_pick_record,
    roster_ids_for_slot,
    roster_summary,
    team_slot_for_pick,
)
from patbot.sleeper import refresh_snapshot, SleeperDataError
from patbot.sim import compare_candidates


st.set_page_config(page_title="PatBot Draft Room", layout="wide")
st.title("PatBot — 2026 Draft Room")
st.caption(
    "v0.3.8 • Athletic custom rankings • paired simulations • early-pick lookahead • "
    "fixed manager profiles"
)

cfg = load_config()
teams = int(cfg["league"]["teams"])
slot = int(cfg["league"]["draft_slot"])
draft_order_cfg = {
    int(k): str(v)
    for k, v in cfg["league"].get("draft_order", {}).items()
}
manager_cfg = cfg.get("opponent_managers", {})
fixed_archetypes = cfg.get("opponent_archetypes", {}).get("fixed_by_slot", {})
source_cfg = cfg.get("v03_consensus", {})

LIVE_CSV = Path("data/players_2026_live.csv")
LIVE_META = Path("data/players_2026_live.meta.json")
EXAMPLE_CSV = Path("data/example_players.csv")
ATHLETIC_PATH = Path(source_cfg.get("athletic_path", "private_sources/athletic.xlsx"))

st.sidebar.header("Data")

athletic_upload = st.sidebar.file_uploader(
    "Athletic custom rankings (.xlsx)",
    type=["xlsx"],
    help=(
        "The workbook stays on this computer in private_sources and is ignored by Git. "
        "Upload a newer copy here whenever The Athletic updates the projections."
    ),
)
if athletic_upload is not None:
    ATHLETIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    ATHLETIC_PATH.write_bytes(athletic_upload.getvalue())
    st.sidebar.success(f"Saved locally as {ATHLETIC_PATH.name}")
    st.sidebar.caption("Click Refresh live 2026 data to merge the new workbook into PatBot.")

if st.sidebar.button("Refresh live 2026 data", use_container_width=True):
    with st.spinner("Refreshing projections, ADP and independent/custom rankings..."):
        try:
            _, _, meta = refresh_snapshot(cfg, LIVE_CSV, LIVE_META)
            st.sidebar.success(f"Updated {meta['draftable_rows']} players.")
            st.rerun()
        except SleeperDataError as exc:
            st.sidebar.error(str(exc))
        except Exception as exc:
            st.sidebar.error(f"{type(exc).__name__}: {exc}")

if LIVE_CSV.exists():
    default_path = str(LIVE_CSV)
    st.sidebar.success("Using live-data snapshot")
else:
    default_path = str(EXAMPLE_CSV)
    st.sidebar.warning("Using synthetic test data — refresh live data first")

player_file = st.sidebar.text_input("Player CSV", default_path)

meta = {}
if LIVE_META.exists():
    try:
        meta = json.loads(LIVE_META.read_text(encoding="utf-8"))
        st.sidebar.caption(
            f"Snapshot: {meta.get('snapshot_at_utc', 'unknown')}\n\n"
            f"Rows: {meta.get('draftable_rows', 'unknown')}"
        )
    except Exception:
        pass

try:
    players = load_players(player_file)
except Exception as exc:
    st.error(f"Could not load player file: {exc}")
    st.stop()

players["player_id"] = players["player_id"].astype(str)
engine = DraftEngine(players, cfg)

st.info(
    f"{teams}-team snake • PatBot at slot {slot} • "
    "QB / 2 RB / 3 WR / TE / FLEX / K / DEF • 5 bench"
)

# ---------------------------------------------------------------------
# Source status
# ---------------------------------------------------------------------
market_status = meta.get("market_sources", {})
if market_status:
    with st.expander("Independent source status"):
        for source, status in market_status.items():
            if status.get("ok"):
                line = f"✅ {source}: matched {status.get('matched', '?')} players"
                if status.get("file"):
                    line += f" • {status['file']}"
                st.write(line)
                if status.get("warning"):
                    st.caption(f"⚠️ {status['warning']}")
            else:
                st.write(f"⚠️ {source}: {status.get('error', 'unavailable')}")

# ---------------------------------------------------------------------
# Persistent session state
# ---------------------------------------------------------------------
if "draft_history" not in st.session_state:
    st.session_state.draft_history = []

if "team_names" not in st.session_state:
    st.session_state.team_names = {
        i: draft_order_cfg.get(i, "PatBot" if i == slot else f"Slot {i}")
        for i in range(1, teams + 1)
    }

for i in range(1, teams + 1):
    if i in draft_order_cfg:
        st.session_state.team_names[i] = draft_order_cfg[i]
st.session_state.team_names[slot] = "PatBot"

draft_history = st.session_state.draft_history
drafted_ids = drafted_ids_from_history(draft_history)
my_roster_ids = roster_ids_for_slot(draft_history, slot)
current_pick = len(draft_history) + 1
current_owner_slot = team_slot_for_pick(current_pick, teams)
current_owner_name = st.session_state.team_names.get(
    current_owner_slot, f"Slot {current_owner_slot}"
)

id_to_pos = dict(zip(players["player_id"], players["pos"]))
player_by_id = players.set_index("player_id", drop=False)

# ---------------------------------------------------------------------
# Sidebar: real draft recording
# ---------------------------------------------------------------------
st.sidebar.header("Draft state")
st.sidebar.metric("Current overall pick", current_pick)
st.sidebar.write(
    f"**On the clock:** {current_owner_name} "
    f"(draft slot {current_owner_slot})"
)

available = players[~players["player_id"].isin(drafted_ids)].copy()
available_ids = available["player_id"].tolist()


def player_label(pid: str) -> str:
    row = player_by_id.loc[str(pid)]
    return f"{row['name']} — {row['team']} {row['pos']}"


selected_pid = st.sidebar.selectbox(
    "Record player selected",
    options=["—"] + available_ids,
    format_func=lambda x: "—" if x == "—" else player_label(x),
)

record_label = (
    "Draft for PatBot"
    if current_owner_slot == slot
    else f"Record for {current_owner_name}"
)

if st.sidebar.button(
    record_label,
    use_container_width=True,
    type="primary",
    disabled=(selected_pid == "—"),
):
    row = player_by_id.loc[str(selected_pid)]
    record = make_pick_record(
        overall_pick=current_pick,
        teams=teams,
        player_id=str(selected_pid),
        player_name=row["name"],
        nfl_team=row["team"],
        pos=row["pos"],
    )
    st.session_state.draft_history.append(record)

    for key in ["sim_summary", "sim_details"]:
        st.session_state.pop(key, None)

    st.rerun()

u1, u2 = st.sidebar.columns(2)

if u1.button(
    "Undo last",
    use_container_width=True,
    disabled=(len(draft_history) == 0),
):
    st.session_state.draft_history.pop()
    for key in ["sim_summary", "sim_details"]:
        st.session_state.pop(key, None)
    st.rerun()

if u2.button("Reset draft", use_container_width=True):
    st.session_state.draft_history = []
    for key in ["sim_summary", "sim_details"]:
        st.session_state.pop(key, None)
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

tab_board, tab_rosters, tab_log, tab_setup, tab_sim = st.tabs([
    "Draft Board",
    "Team Rosters",
    "Draft Log",
    "Team Setup",
    "Simulation Lab",
])

# ---------------------------------------------------------------------
# Draft board
# ---------------------------------------------------------------------
with tab_board:
    left, right = st.columns([2.5, 1])

    with left:
        st.subheader("Live recommendations")

        if board.empty:
            st.write("No available players.")
        else:
            preferred = [
                "name", "team", "pos", "score", "proj_points", "vorp",
                "athletic_rank", "athletic_vorp", "athletic_points",
                "adp", "expert_rank", "consensus_value", "consensus_tier",
                "tier_cliff", "survive_next_pct", "scarcity", "bye",
            ]
            cols = [c for c in preferred if c in board.columns]
            view = board[cols].rename(columns={
                "name": "Player",
                "team": "Team",
                "pos": "Pos",
                "score": "PatBot",
                "proj_points": "PatBot Proj",
                "vorp": "PatBot VORP",
                "athletic_rank": "Athletic RK",
                "athletic_vorp": "Athletic VORP",
                "athletic_points": "Athletic Pts",
                "adp": "Market ADP",
                "expert_rank": "Blended Expert RK",
                "consensus_value": "Consensus",
                "consensus_tier": "Tier",
                "tier_cliff": "Cliff After?",
                "survive_next_pct": "Survive Next %",
                "scarcity": "4-Player Drop",
                "bye": "Bye",
            })
            st.dataframe(view, use_container_width=True, hide_index=True)

            st.subheader("Current call")
            st.write(engine.explain_row(board.iloc[0]))

    with right:
        st.subheader("PatBot roster")
        mine = players[players["player_id"].isin(my_roster_ids)]
        roster_cols = [
            c
            for c in ["name", "team", "pos", "adp", "proj_points", "bye"]
            if c in mine.columns
        ]
        if mine.empty:
            st.write("No PatBot selections yet.")
        else:
            order = {pid: i for i, pid in enumerate(my_roster_ids)}
            mine = mine.copy()
            mine["_draft_order"] = mine["player_id"].map(order)
            mine = mine.sort_values("_draft_order")
            st.dataframe(
                mine[roster_cols],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Roster counts")
        if roster_positions:
            counts = (
                pd.Series(roster_positions)
                .value_counts()
                .rename_axis("Pos")
                .reset_index(name="Count")
            )
            st.dataframe(counts, use_container_width=True, hide_index=True)
        else:
            st.write("—")

# ---------------------------------------------------------------------
# All real rosters
# ---------------------------------------------------------------------
with tab_rosters:
    st.subheader("Live league rosters")
    st.caption(
        "Every recorded selection is assigned automatically to the correct "
        "snake-draft slot. These rosters feed that specific manager's future "
        "simulation behavior."
    )
    summary = roster_summary(
        draft_history,
        teams,
        st.session_state.team_names,
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if current_owner_slot != slot:
        owner_row = summary[summary["Slot"] == current_owner_slot]
        if not owner_row.empty:
            st.info(
                f"Currently on the clock: **{current_owner_name}**. "
                "PatBot will use this manager's profile and existing positional "
                "roster when estimating future selections."
            )

# ---------------------------------------------------------------------
# Pick-by-pick log
# ---------------------------------------------------------------------
with tab_log:
    st.subheader("Draft log")
    log = history_frame(
        draft_history,
        st.session_state.team_names,
    )
    if log.empty:
        st.write("No selections recorded yet.")
    else:
        st.dataframe(log, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Locked 2026 manager room
# ---------------------------------------------------------------------
with tab_setup:
    st.subheader("2026 room model")
    st.write(
        "The actual draft order is locked. Each opponent keeps the same manager "
        "profile in every simulation instead of receiving a random archetype each run."
    )

    room_rows = []
    for i in range(1, teams + 1):
        manager = draft_order_cfg.get(i, st.session_state.team_names.get(i, f"Slot {i}"))
        if i == slot:
            profile = "PatBot"
            notes = "Our draft engine."
        else:
            raw = manager_cfg.get(i) or manager_cfg.get(str(i)) or {}
            profile = raw.get(
                "archetype",
                fixed_archetypes.get(i, fixed_archetypes.get(str(i), "market")),
            )
            notes = raw.get("notes", "")

        room_rows.append({
            "Slot": i,
            "Manager": manager,
            "Profile": str(profile).replace("_", " ").title(),
            "Notes": notes,
        })

    st.dataframe(
        pd.DataFrame(room_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Manager tendencies are starting priors, not permanent judgments. As real "
        "2026 picks are recorded, PatBot also carries each manager's actual roster "
        "state into the simulation."
    )

# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------
with tab_sim:
    st.subheader("Monte Carlo Draft Lab")

    if not is_my_pick:
        st.write(
            "Simulation comparisons are most useful when PatBot is on the clock. "
            "Record the real picks ahead of us first."
        )
    elif board.empty:
        st.write("No candidates available.")
    else:
        st.write(
            "PatBot forces each candidate as the current pick, then simulates the "
            "fixed real-manager room through Round 8. Early PatBot follow-up picks "
            "use lookahead: several choices are tested against the intervening room "
            "before the pick is made."
        )
        st.caption(
            "All candidates now use common random numbers — run #1 is the same room "
            "for every candidate — so close comparisons carry less simulation noise. "
            "The score is a lineup-aware construction score, not championship probability."
        )

        real_opp_picks = sum(
            1
            for p in draft_history
            if int(p["owner_slot"]) != slot
        )
        st.caption(
            f"Simulation seed: {real_opp_picks} recorded opponent selections "
            f"across {teams - 1} fixed opponent slots."
        )

        available_board = players[
            ~players["player_id"].isin(drafted_ids)
        ].copy()

        default_candidates = board.head(6)["name"].tolist()

        selected_candidates = st.multiselect(
            "Candidates to compare",
            options=available_board["name"].sort_values().tolist(),
            default=default_candidates,
            help=(
                "Choose any available players. PatBot will force each one as "
                "the current selection and simulate the rest of the draft."
            ),
        )

        candidate_lookup = dict(
            zip(
                available_board["name"],
                available_board["player_id"].astype(str),
            )
        )
        candidate_ids = [
            candidate_lookup[name]
            for name in selected_candidates
            if name in candidate_lookup
        ]

        runs = st.slider(
            "Simulations per candidate",
            min_value=100,
            max_value=int(cfg.get("simulation", {}).get("max_runs", 1500)),
            value=int(cfg.get("simulation", {}).get("default_runs", 300)),
            step=100,
        )

        if len(candidate_ids) < 2:
            st.warning("Select at least two candidates to compare.")
        elif st.button("Run candidate simulations", type="primary"):
            with st.spinner(
                f"Running {runs * len(candidate_ids):,} paired draft paths with lookahead..."
            ):
                summary, details = compare_candidates(
                    engine,
                    current_pick=current_pick,
                    drafted_ids=drafted_ids,
                    my_roster_ids=my_roster_ids,
                    candidate_ids=candidate_ids,
                    runs=runs,
                    through_round=int(
                        cfg.get("simulation", {}).get("through_round", 8)
                    ),
                    draft_history=draft_history,
                )
                st.session_state.sim_summary = summary
                st.session_state.sim_details = details

        if "sim_summary" in st.session_state:
            st.subheader("Candidate comparison")
            st.dataframe(
                st.session_state.sim_summary,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("What tends to come back")
            for detail in st.session_state.sim_details:
                with st.expander(detail["candidate"]):
                    st.write(
                        f"Average lineup score: **{detail['avg_lineup_score']:.2f}** "
                        f"(25th–75th percentile: "
                        f"{detail['p25_lineup_score']:.2f}–"
                        f"{detail['p75_lineup_score']:.2f})"
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Most common second PatBot pick**")
                        st.dataframe(
                            pd.DataFrame(detail["most_common_second_pick"]),
                            hide_index=True,
                        )
                    with c2:
                        st.write("**Most common third PatBot pick**")
                        st.dataframe(
                            pd.DataFrame(detail["most_common_third_pick"]),
                            hide_index=True,
                        )

st.divider()
st.caption(
    "v0.3.8: local Athletic custom rankings are merged without entering GitHub; "
    "candidate simulations are paired; Rounds 2–3 use one-pick-ahead room simulation."
)
