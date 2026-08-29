from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


_INSTALLED = False
_SKILL = {"RB", "WR", "TE"}
_INVALID_TEAMS = {"", "FA", "NONE", "NAN", "UNK", "UNKNOWN"}

_DEFAULTS = {
    "enabled": True,
    "draft_pair_penalty": {
        "RB|RB": 2.0,
        "RB|WR": 1.75,
        "RB|TE": 1.25,
        "WR|WR": 3.5,
        "WR|TE": 2.75,
        "TE|TE": 2.25,
    },
    "lineup_pair_penalty": {
        "RB|RB": 5.0,
        "RB|WR": 4.0,
        "RB|TE": 3.0,
        "WR|WR": 8.0,
        "WR|TE": 6.0,
        "TE|TE": 5.0,
    },
    "third_plus_draft_penalty": 2.0,
    "third_plus_lineup_penalty": 4.0,
    "starter_bench_weight": 0.40,
    "bench_bench_weight": 0.15,
    "round_multipliers": {
        "1_7": 1.0,
        "8_11": 0.65,
        "12_13": 0.35,
        "14_plus": 0.0,
    },
}


def concentration_settings(config: dict | None) -> dict:
    supplied = (config or {}).get("same_team_concentration", {}) or {}
    out = dict(_DEFAULTS)
    out.update({k: v for k, v in supplied.items() if k not in {"draft_pair_penalty", "lineup_pair_penalty", "round_multipliers"}})
    out["draft_pair_penalty"] = {**_DEFAULTS["draft_pair_penalty"], **(supplied.get("draft_pair_penalty", {}) or {})}
    out["lineup_pair_penalty"] = {**_DEFAULTS["lineup_pair_penalty"], **(supplied.get("lineup_pair_penalty", {}) or {})}
    out["round_multipliers"] = {**_DEFAULTS["round_multipliers"], **(supplied.get("round_multipliers", {}) or {})}
    return out


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((str(a).upper(), str(b).upper())))


def _valid_team(team) -> bool:
    text = str(team or "").strip().upper()
    return text not in _INVALID_TEAMS


def _round_multiplier(round_no: int, settings: dict) -> float:
    m = settings["round_multipliers"]
    r = int(round_no)
    if r <= 7:
        return float(m.get("1_7", 1.0))
    if r <= 11:
        return float(m.get("8_11", 0.65))
    if r <= 13:
        return float(m.get("12_13", 0.35))
    return float(m.get("14_plus", 0.0))


def pair_penalty(pos_a: str, pos_b: str, config: dict | None = None, *, lineup: bool = False) -> float:
    """Return the generic same-team non-QB skill-player pair penalty.

    QB combinations intentionally return zero. The layer is a soft diversification
    preference, not a ban on stacking or on accepting exceptional value.
    """
    a, b = str(pos_a).upper(), str(pos_b).upper()
    if a not in _SKILL or b not in _SKILL:
        return 0.0
    settings = concentration_settings(config)
    table = settings["lineup_pair_penalty"] if lineup else settings["draft_pair_penalty"]
    return float(table.get(_pair_key(a, b), 0.0))


def candidate_concentration_penalty(
    players: pd.DataFrame,
    *,
    candidate_idx: int,
    roster_indices: list[int] | set[int] | tuple[int, ...],
    round_no: int,
    config: dict | None = None,
) -> tuple[float, str]:
    settings = concentration_settings(config)
    if not bool(settings.get("enabled", True)):
        return 0.0, ""

    frame = players.reset_index(drop=True)
    if not (0 <= int(candidate_idx) < len(frame)):
        return 0.0, ""
    candidate = frame.iloc[int(candidate_idx)]
    cpos = str(candidate.get("pos", "")).upper()
    cteam = str(candidate.get("team", "")).strip().upper()
    if cpos not in _SKILL or not _valid_team(cteam):
        return 0.0, ""

    same_team = []
    for idx in roster_indices:
        i = int(idx)
        if not (0 <= i < len(frame)):
            continue
        row = frame.iloc[i]
        pos = str(row.get("pos", "")).upper()
        team = str(row.get("team", "")).strip().upper()
        if pos in _SKILL and team == cteam:
            same_team.append((i, pos, str(row.get("name", ""))))

    if not same_team:
        return 0.0, ""

    raw = sum(pair_penalty(cpos, pos, config, lineup=False) for _, pos, _ in same_team)
    if len(same_team) >= 2:
        raw += float(settings.get("third_plus_draft_penalty", 2.0)) * (len(same_team) - 1)
    penalty = raw * _round_multiplier(round_no, settings)
    if penalty <= 0:
        return 0.0, ""

    pairs = ", ".join(f"{cpos}+{pos}" for _, pos, _ in same_team)
    note = f"same-team skill concentration ({cteam}: {pairs})"
    if len(same_team) >= 2:
        note += f"; becomes skill player #{len(same_team) + 1} from that offense"
    return float(penalty), note


def _starter_set(sim, mine: list[int], projection_override: np.ndarray | None = None) -> set[int]:
    proj = sim.proj if projection_override is None else np.asarray(projection_override, dtype=float)
    rcfg = sim.engine.roster_cfg
    by_pos: dict[str, list[int]] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        idxs = [int(i) for i in mine if sim.pos[int(i)] == pos]
        idxs.sort(key=lambda i: float(proj[i]), reverse=True)
        by_pos[pos] = idxs

    starters: list[int] = []
    used: set[int] = set()
    for pos in ("QB", "RB", "WR", "TE"):
        chosen = by_pos[pos][: int(rcfg.get(pos, 0))]
        starters.extend(chosen)
        used.update(chosen)

    flex_pool = [
        int(i)
        for i in mine
        if int(i) not in used and sim.pos[int(i)] in set(rcfg.get("flex_eligible", []))
    ]
    flex_pool.sort(key=lambda i: float(proj[i]), reverse=True)
    starters.extend(flex_pool[: int(rcfg.get("FLEX", 0))])
    return set(starters)


def roster_concentration_penalty(sim, mine: list[int], projection_override: np.ndarray | None = None) -> float:
    settings = concentration_settings(sim.cfg)
    if not bool(settings.get("enabled", True)):
        return 0.0

    starters = _starter_set(sim, mine, projection_override)
    grouped: dict[str, list[int]] = {}
    for idx in mine:
        i = int(idx)
        pos = str(sim.pos[i]).upper()
        team = str(sim.nfl_team[i]).strip().upper()
        if pos in _SKILL and _valid_team(team):
            grouped.setdefault(team, []).append(i)

    total = 0.0
    for idxs in grouped.values():
        if len(idxs) < 2:
            continue
        for a, b in combinations(idxs, 2):
            base = pair_penalty(sim.pos[a], sim.pos[b], sim.cfg, lineup=True)
            if a in starters and b in starters:
                weight = 1.0
            elif a in starters or b in starters:
                weight = float(settings.get("starter_bench_weight", 0.40))
            else:
                weight = float(settings.get("bench_bench_weight", 0.15))
            total += base * weight

        if len(idxs) >= 3:
            starter_count = sum(1 for i in idxs if i in starters)
            if starter_count >= 2:
                extra_weight = 1.0
            elif starter_count == 1:
                extra_weight = float(settings.get("starter_bench_weight", 0.40))
            else:
                extra_weight = float(settings.get("bench_bench_weight", 0.15))
            total += float(settings.get("third_plus_lineup_penalty", 4.0)) * (len(idxs) - 2) * extra_weight
    return float(total)


def install_team_concentration_patch() -> None:
    """Install the same-team skill concentration preference in board and simulation."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .draft import DraftEngine
    from .sim import FastDraftSimulator

    original_engine_recommend = DraftEngine.recommend
    original_explain = DraftEngine.explain_row
    original_sim_init = FastDraftSimulator.__init__
    original_score_vector = FastDraftSimulator._patbot_score_vector
    original_evaluate = FastDraftSimulator.evaluate_roster

    def engine_recommend(self, current_pick, drafted_ids, roster_positions, top_n=12):
        full_n = max(int(top_n), len(self.players))
        board = original_engine_recommend(
            self,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            roster_positions=roster_positions,
            top_n=full_n,
        )
        if board.empty:
            return board

        roster_ids = [str(x) for x in getattr(self, "_patbot_roster_ids", [])]
        id_to_idx = {str(pid): i for i, pid in enumerate(self.players["player_id"].astype(str))}
        roster_indices = [id_to_idx[x] for x in roster_ids if x in id_to_idx]
        round_no = ((int(current_pick) - 1) // int(self.league["teams"])) + 1

        penalties = []
        notes = []
        for pid in board["player_id"].astype(str):
            idx = id_to_idx.get(pid)
            if idx is None or not roster_indices:
                penalty, note = 0.0, ""
            else:
                penalty, note = candidate_concentration_penalty(
                    self.players,
                    candidate_idx=idx,
                    roster_indices=roster_indices,
                    round_no=round_no,
                    config=self.config,
                )
            penalties.append(float(penalty))
            notes.append(note)

        board["team_concentration_penalty"] = np.round(penalties, 2)
        board["team_concentration_note"] = notes
        board["score"] = pd.to_numeric(board["score"], errors="coerce").fillna(-1e9) - board["team_concentration_penalty"]
        board["score"] = board["score"].round(2)
        return board.sort_values(["score", "proj_points", "adp"], ascending=[False, False, True]).head(int(top_n)).reset_index(drop=True)

    def explain_row(row: pd.Series) -> str:
        text = original_explain(row)
        penalty = float(row.get("team_concentration_penalty") or 0.0)
        if penalty > 0:
            text += f" Same-team skill concentration penalty: -{penalty:.2f} ({row.get('team_concentration_note', '')})."
        return text

    def sim_init(self, engine):
        original_sim_init(self, engine)
        self.nfl_team = self.players.get("team", pd.Series([""] * self.n)).fillna("").astype(str).str.upper().to_numpy()

    def score_vector(self, available: np.ndarray, roster_counts: np.ndarray, pick: int) -> np.ndarray:
        score = np.asarray(original_score_vector(self, available, roster_counts, pick), dtype=float).copy()
        if not concentration_settings(self.cfg).get("enabled", True):
            return score
        owned = set(getattr(self, "_patbot_owned_idxs", set()))
        if not owned:
            return score
        round_no = (int(pick) - 1) // self.teams + 1
        for idx in np.where(np.asarray(available, dtype=bool))[0]:
            penalty, _ = candidate_concentration_penalty(
                self.players,
                candidate_idx=int(idx),
                roster_indices=owned,
                round_no=round_no,
                config=self.cfg,
            )
            if penalty:
                score[int(idx)] -= float(penalty)
        return score

    def evaluate_roster(self, mine: list[int], projection_override: np.ndarray | None = None) -> dict:
        result = original_evaluate(self, mine, projection_override=projection_override)
        penalty = roster_concentration_penalty(self, mine, projection_override=projection_override)
        result["team_concentration_penalty"] = float(penalty)
        result["lineup_score"] = float(result["lineup_score"]) - float(penalty)
        return result

    DraftEngine.recommend = engine_recommend
    DraftEngine.explain_row = staticmethod(explain_row)
    FastDraftSimulator.__init__ = sim_init
    FastDraftSimulator._patbot_score_vector = score_vector
    FastDraftSimulator.evaluate_roster = evaluate_roster
    _INSTALLED = True
