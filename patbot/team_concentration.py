from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


_INSTALLED = False
_SKILL = {"RB", "WR", "TE"}
_POS_ORDER = {"RB": 0, "WR": 1, "TE": 2}
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
    "round_multipliers": {"1_7": 1.0, "8_11": 0.65, "12_13": 0.35, "14_plus": 0.0},
}


def concentration_settings(config: dict | None) -> dict:
    supplied = (config or {}).get("same_team_concentration", {}) or {}
    out = dict(_DEFAULTS)
    out.update({
        k: v for k, v in supplied.items()
        if k not in {"draft_pair_penalty", "lineup_pair_penalty", "round_multipliers"}
    })
    out["draft_pair_penalty"] = {
        **_DEFAULTS["draft_pair_penalty"], **(supplied.get("draft_pair_penalty", {}) or {})
    }
    out["lineup_pair_penalty"] = {
        **_DEFAULTS["lineup_pair_penalty"], **(supplied.get("lineup_pair_penalty", {}) or {})
    }
    out["round_multipliers"] = {
        **_DEFAULTS["round_multipliers"], **(supplied.get("round_multipliers", {}) or {})
    }
    return out


def _pair_key(a: str, b: str) -> str:
    vals = [str(a).upper(), str(b).upper()]
    vals.sort(key=lambda x: _POS_ORDER.get(x, 99))
    return "|".join(vals)


def _valid_team(team) -> bool:
    return str(team or "").strip().upper() not in _INVALID_TEAMS


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


def _pair_lookup(settings: dict, *, lineup: bool = False) -> np.ndarray:
    table = settings["lineup_pair_penalty"] if lineup else settings["draft_pair_penalty"]
    lookup = np.zeros((3, 3), dtype=float)
    for a, ai in _POS_ORDER.items():
        for b, bi in _POS_ORDER.items():
            lookup[ai, bi] = float(table.get(_pair_key(a, b), 0.0))
    return lookup


def pair_penalty(pos_a: str, pos_b: str, config: dict | None = None, *, lineup: bool = False) -> float:
    """Generic same-team non-QB skill-player penalty; QB stacks are exempt."""
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

    frame = players
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        frame = frame.reset_index(drop=True)
    i = int(candidate_idx)
    if not (0 <= i < len(frame)):
        return 0.0, ""

    candidate = frame.iloc[i]
    cpos = str(candidate.get("pos", "")).upper()
    cteam = str(candidate.get("team", "")).strip().upper()
    if cpos not in _SKILL or not _valid_team(cteam):
        return 0.0, ""

    same_team: list[tuple[int, str, str]] = []
    for idx in roster_indices:
        j = int(idx)
        if not (0 <= j < len(frame)):
            continue
        row = frame.iloc[j]
        pos = str(row.get("pos", "")).upper()
        team = str(row.get("team", "")).strip().upper()
        if pos in _SKILL and team == cteam:
            same_team.append((j, pos, str(row.get("name", ""))))
    if not same_team:
        return 0.0, ""

    table = settings["draft_pair_penalty"]
    raw = sum(float(table.get(_pair_key(cpos, pos), 0.0)) for _, pos, _ in same_team)
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
    starters: list[int] = []
    used: set[int] = set()
    for pos in ("QB", "RB", "WR", "TE"):
        idxs = [int(i) for i in mine if sim.pos[int(i)] == pos]
        idxs.sort(key=lambda j: float(proj[j]), reverse=True)
        chosen = idxs[: int(rcfg.get(pos, 0))]
        starters.extend(chosen)
        used.update(chosen)
    flex_pool = [
        int(i) for i in mine
        if int(i) not in used and sim.pos[int(i)] in set(rcfg.get("flex_eligible", []))
    ]
    flex_pool.sort(key=lambda j: float(proj[j]), reverse=True)
    starters.extend(flex_pool[: int(rcfg.get("FLEX", 0))])
    return set(starters)


def roster_concentration_penalty(sim, mine: list[int], projection_override: np.ndarray | None = None) -> float:
    settings = getattr(sim, "_team_concentration_settings", None) or concentration_settings(sim.cfg)
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

    lineup_lookup = getattr(sim, "_team_concentration_lineup_lookup", None)
    if lineup_lookup is None:
        lineup_lookup = _pair_lookup(settings, lineup=True)

    total = 0.0
    for idxs in grouped.values():
        if len(idxs) < 2:
            continue
        for a, b in combinations(idxs, 2):
            ai = _POS_ORDER.get(str(sim.pos[a]).upper(), -1)
            bi = _POS_ORDER.get(str(sim.pos[b]).upper(), -1)
            base = float(lineup_lookup[ai, bi]) if ai >= 0 and bi >= 0 else 0.0
            if a in starters and b in starters:
                weight = 1.0
            elif a in starters or b in starters:
                weight = float(settings.get("starter_bench_weight", 0.40))
            else:
                weight = float(settings.get("bench_bench_weight", 0.15))
            total += base * weight
        if len(idxs) >= 3:
            starter_count = sum(1 for i in idxs if i in starters)
            weight = 1.0 if starter_count >= 2 else (
                float(settings.get("starter_bench_weight", 0.40)) if starter_count == 1
                else float(settings.get("bench_bench_weight", 0.15))
            )
            total += float(settings.get("third_plus_lineup_penalty", 4.0)) * (len(idxs) - 2) * weight
    return float(total)


def _apply_sim_candidate_penalties(sim, score: np.ndarray, available: np.ndarray, pick: int) -> np.ndarray:
    """Vectorized draft-score concentration penalty for the simulator hot path."""
    settings = getattr(sim, "_team_concentration_settings", None) or concentration_settings(sim.cfg)
    multiplier = _round_multiplier((int(pick) - 1) // sim.teams + 1, settings)
    if not bool(settings.get("enabled", True)) or multiplier <= 0:
        return score

    owned = getattr(sim, "_patbot_owned_idxs", set())
    if not owned:
        return score

    owned_idx = np.fromiter((int(i) for i in owned), dtype=np.int32)
    if owned_idx.size == 0:
        return score

    skill_code = sim._team_concentration_skill_code
    team_code = sim._team_concentration_team_code
    owned_valid = (skill_code[owned_idx] >= 0) & (team_code[owned_idx] >= 0)
    owned_skill = owned_idx[owned_valid]
    if owned_skill.size == 0:
        return score

    nteams = int(sim._team_concentration_team_count)
    pos_counts = np.zeros((nteams, 3), dtype=np.int16)
    np.add.at(pos_counts, (team_code[owned_skill], skill_code[owned_skill]), 1)
    team_totals = pos_counts.sum(axis=1)

    available_mask = np.asarray(available, dtype=bool)
    candidate_valid = available_mask & (skill_code >= 0) & (team_code >= 0)
    candidate_idx = np.where(candidate_valid)[0]
    if candidate_idx.size == 0:
        return score

    lookup = sim._team_concentration_draft_lookup
    candidate_teams = team_code[candidate_idx]
    candidate_pos = skill_code[candidate_idx]
    same_team_counts = team_totals[candidate_teams]
    has_exposure = same_team_counts > 0
    if not has_exposure.any():
        return score

    candidate_idx = candidate_idx[has_exposure]
    candidate_teams = candidate_teams[has_exposure]
    candidate_pos = candidate_pos[has_exposure]
    same_team_counts = same_team_counts[has_exposure]

    raw = np.zeros(len(candidate_idx), dtype=float)
    for pcode in range(3):
        raw += pos_counts[candidate_teams, pcode] * lookup[candidate_pos, pcode]
    raw += (
        np.maximum(same_team_counts - 1, 0)
        * float(settings.get("third_plus_draft_penalty", 2.0))
    )

    out = np.asarray(score, dtype=float).copy()
    out[candidate_idx] -= raw * multiplier
    return out


def install_team_concentration_patch() -> None:
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
        board = original_engine_recommend(
            self,
            current_pick=current_pick,
            drafted_ids=drafted_ids,
            roster_positions=roster_positions,
            top_n=max(int(top_n), len(self.players)),
        )
        if board.empty:
            return board
        roster_ids = [str(x) for x in getattr(self, "_patbot_roster_ids", [])]
        id_to_idx = {str(pid): i for i, pid in enumerate(self.players["player_id"].astype(str))}
        roster_indices = [id_to_idx[x] for x in roster_ids if x in id_to_idx]
        round_no = ((int(current_pick) - 1) // int(self.league["teams"])) + 1
        penalties, notes = [], []
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
        board["score"] = (
            pd.to_numeric(board["score"], errors="coerce").fillna(-1e9)
            - board["team_concentration_penalty"]
        ).round(2)
        return board.sort_values(
            ["score", "proj_points", "adp"], ascending=[False, False, True]
        ).head(int(top_n)).reset_index(drop=True)

    def explain_row(row: pd.Series) -> str:
        text = original_explain(row)
        penalty = float(row.get("team_concentration_penalty") or 0.0)
        if penalty > 0:
            text += (
                f" Same-team skill concentration penalty: -{penalty:.2f} "
                f"({row.get('team_concentration_note', '')})."
            )
        return text

    def sim_init(self, engine):
        original_sim_init(self, engine)
        self.nfl_team = (
            self.players.get("team", pd.Series([""] * self.n))
            .fillna("").astype(str).str.upper().to_numpy()
        )
        settings = concentration_settings(self.cfg)
        self._team_concentration_settings = settings
        self._team_concentration_draft_lookup = _pair_lookup(settings, lineup=False)
        self._team_concentration_lineup_lookup = _pair_lookup(settings, lineup=True)
        self._team_concentration_skill_code = np.array(
            [_POS_ORDER.get(str(p).upper(), -1) for p in self.pos],
            dtype=np.int8,
        )
        valid_teams = sorted({
            str(team).strip().upper()
            for team in self.nfl_team
            if _valid_team(team)
        })
        team_lookup = {team: i for i, team in enumerate(valid_teams)}
        self._team_concentration_team_code = np.array(
            [team_lookup.get(str(team).strip().upper(), -1) for team in self.nfl_team],
            dtype=np.int16,
        )
        self._team_concentration_team_count = len(valid_teams)

    def score_vector(self, available: np.ndarray, roster_counts: np.ndarray, pick: int) -> np.ndarray:
        base = np.asarray(original_score_vector(self, available, roster_counts, pick), dtype=float)
        return _apply_sim_candidate_penalties(self, base, available, pick)

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
