from __future__ import annotations
from collections import Counter
import math
import numpy as np
import pandas as pd
from .draft import all_team_picks

def _team_slot_for_pick(pick: int, teams: int) -> int:
    round_no = (pick - 1) // teams + 1
    within = (pick - 1) % teams + 1
    return within if round_no % 2 else teams + 1 - within

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

class FastDraftSimulator:
    POSITIONS = ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')

    def __init__(self, engine):
        self.engine = engine
        self.cfg = engine.config
        self.players = engine.players.reset_index(drop=True).copy()
        self.players['player_id'] = self.players['player_id'].astype(str)
        self.n = len(self.players)
        self.ids = self.players['player_id'].to_numpy()
        self.names = self.players['name'].astype(str).to_numpy()
        self.pos = self.players['pos'].astype(str).to_numpy()
        self.adp = pd.to_numeric(self.players['adp'], errors='coerce').fillna(999.0).to_numpy(float)
        self.proj = pd.to_numeric(self.players['proj_points'], errors='coerce').fillna(0.0).to_numpy(float)
        self.injury = pd.to_numeric(self.players.get('injury_risk', pd.Series([0.0] * self.n)), errors='coerce').fillna(0.0).to_numpy(float)
        if 'expert_rank' in self.players:
            self.expert_rank = pd.to_numeric(self.players['expert_rank'], errors='coerce').to_numpy(float)
        else:
            self.expert_rank = np.full(self.n, np.nan)
        self.id_to_idx = {pid: i for i, pid in enumerate(self.ids)}
        self.pos_to_code = {p: i for i, p in enumerate(self.POSITIONS)}
        self.pos_code = np.array([self.pos_to_code.get(p, -1) for p in self.pos], dtype=int)
        self.teams = int(engine.league['teams'])
        self.slot = int(engine.league['draft_slot'])
        self.my_picks = set(all_team_picks(self.teams, self.slot, rounds=20))
        levels = engine.replacement_levels()
        self.replacement_levels = levels
        self.replacement = np.array([float(levels.get(p, 0.0)) for p in self.pos])
        self.vorp = self.proj - self.replacement
        self.proj_pct = self._percentile(self.proj)
        self.vorp_pct = self._percentile(self.vorp)
        expert_component = np.full(self.n, 0.5)
        if np.isfinite(self.expert_rank).any():
            finite = np.isfinite(self.expert_rank)
            tmp = np.where(finite, self.expert_rank, np.nanmax(self.expert_rank[finite]) + 50)
            expert_rank_pct = self._percentile(tmp)
            expert_component = np.where(finite, 1.0 - expert_rank_pct + 1.0 / max(self.n, 1), 0.5)
        custom_goodness = 0.58 * self.vorp_pct + 0.27 * self.proj_pct + 0.15 * expert_component
        custom_order = np.argsort(custom_goodness)[::-1]
        self.custom_rank = np.empty(self.n, dtype=float)
        self.custom_rank[custom_order] = np.arange(1, self.n + 1, dtype=float)
        if np.isfinite(self.expert_rank).any():
            finite = np.isfinite(self.expert_rank)
            temp = np.where(finite, self.expert_rank, np.nanmax(self.expert_rank[finite]) + 50)
            rank_pct = self._percentile(temp)
            self.expert_pct = 1.0 - rank_pct + 1.0 / max(self.n, 1)
            self.expert_pct = np.where(finite, self.expert_pct, 0.5)
            self.expert_weight = 0.1
        else:
            self.expert_pct = np.full(self.n, 0.5)
            self.expert_weight = 0.0
        w = engine.engine_cfg['weights']
        base_total = sum((float(w[k]) for k in ['vorp', 'projection', 'urgency', 'scarcity', 'roster_fit']))
        scale = (1.0 - self.expert_weight) / base_total
        self.w_vorp = float(w['vorp']) * scale
        self.w_proj = float(w['projection']) * scale
        self.w_urgency = float(w['urgency']) * scale
        self.w_scarcity = float(w['scarcity']) * scale
        self.w_roster = float(w['roster_fit']) * scale
        self.injury_penalty = float(engine.engine_cfg.get('injury_risk_penalty', 0))
        simcfg = self.cfg.get('simulation', {})
        self.sd_floor = float(simcfg.get('opponent_adp_sd_floor', 5.0))
        self.sd_pct = float(simcfg.get('opponent_adp_sd_pct', 0.14))
        self.min_round_k = int(engine.engine_cfg.get('min_round_k', 13))
        self.min_round_def = int(engine.engine_cfg.get('min_round_def', 13))
        self.bench_caps = engine.engine_cfg.get('bench_position_caps', {})
        self.archetype_cfg = self.cfg.get('opponent_archetypes', {})
        self.roster_eval_cfg = self.cfg.get('roster_evaluation', {})
        counts = self.archetype_cfg.get('counts', {})
        if counts:
            total = sum((int(v) for v in counts.values()))
            if total != self.teams - 1:
                raise ValueError(f'Opponent archetype counts sum to {total}; expected {self.teams - 1}.')

    @staticmethod
    def _percentile(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind='mergesort')
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1, dtype=float)
        return ranks / max(len(values), 1)

    def _archetype_assignments(self, rng: np.random.Generator) -> dict[int, str]:
        counts = self.archetype_cfg.get('counts', {})
        bag = []
        for name, count in counts.items():
            bag.extend([name] * int(count))
        if not bag:
            bag = ['market'] * (self.teams - 1)
        rng.shuffle(bag)
        opponent_slots = [s for s in range(1, self.teams + 1) if s != self.slot]
        return dict(zip(opponent_slots, bag))

    def _seed_opponent_counts(self, draft_history: list[dict] | None) -> np.ndarray:
        """Seed simulated opponent roster counts from real picks already made.

        Each real selection is tied to its snake-draft owner slot. This means
        future simulated behavior reflects that specific team's existing roster:
        e.g. a team that already drafted a QB is less likely to take another.
        """
        counts = np.zeros((self.teams + 1, len(self.POSITIONS)), dtype=np.int16)
        for pick in draft_history or []:
            try:
                owner_slot = int(pick['owner_slot'])
                pid = str(pick['player_id'])
            except (KeyError, TypeError, ValueError):
                continue
            if owner_slot == self.slot:
                continue
            idx = self.id_to_idx.get(pid)
            if idx is None:
                continue
            code = self.pos_code[idx]
            if code >= 0 and 1 <= owner_slot <= self.teams:
                counts[owner_slot, code] += 1
        return counts

    def _roster_fit_vector(self, roster_counts: np.ndarray) -> np.ndarray:
        fit = np.full(self.n, 0.45, dtype=float)
        rcfg = self.engine.roster_cfg
        starters = {p: int(rcfg.get(p, 0)) for p in self.POSITIONS}
        for p, code in self.pos_to_code.items():
            mask = self.pos_code == code
            count = int(roster_counts[code])
            if count < starters.get(p, 0):
                fit[mask] = 1.0
                continue
            if p in set(rcfg.get('flex_eligible', [])):
                flex_codes = [self.pos_to_code[x] for x in rcfg.get('flex_eligible', []) if x in self.pos_to_code]
                base_filled = sum((min(int(roster_counts[c]), starters[self.POSITIONS[c]]) for c in flex_codes))
                flex_excess = sum((int(roster_counts[c]) for c in flex_codes)) - base_filled
                if flex_excess < int(rcfg.get('FLEX', 0)):
                    fit[mask] = 0.85
            cap = self.bench_caps.get(p)
            if cap is not None and count >= int(cap):
                fit[mask] = 0.05
        return fit

    def _scarcity_pct(self, available: np.ndarray) -> np.ndarray:
        scarcity = np.zeros(self.n, dtype=float)
        for code in range(len(self.POSITIONS)):
            idx = np.where(available & (self.pos_code == code))[0]
            if len(idx) == 0:
                continue
            ordered = idx[np.argsort(self.proj[idx])[::-1]]
            vals = self.proj[ordered]
            future_index = np.minimum(np.arange(len(vals)) + 4, len(vals) - 1)
            scarcity[ordered] = np.maximum(0.0, vals - vals[future_index])
        if not available.any():
            return scarcity
        vals = scarcity[available]
        order = np.argsort(vals, kind='mergesort')
        ranks = np.empty(len(vals), dtype=float)
        ranks[order] = np.arange(1, len(vals) + 1, dtype=float)
        out = np.zeros(self.n, dtype=float)
        out[np.where(available)[0]] = ranks / len(vals)
        return out

    def _urgency(self, next_pick: int) -> np.ndarray:
        sd = np.maximum(6.0, self.adp * 0.15)
        z = (float(next_pick) - self.adp) / sd
        return np.array([_norm_cdf(x) for x in z], dtype=float)

    def _next_my_pick(self, current_pick: int) -> int:
        for p in sorted(self.my_picks):
            if p > current_pick:
                return p
        return current_pick + self.teams * 2

    def patbot_pick(self, available: np.ndarray, roster_counts: np.ndarray, pick: int) -> int:
        next_pick = self._next_my_pick(pick)
        urgency = self._urgency(next_pick)
        roster_fit = self._roster_fit_vector(roster_counts)
        scarcity_pct = self._scarcity_pct(available)
        score = 100.0 * (self.vorp_pct * self.w_vorp + self.proj_pct * self.w_proj + urgency * self.w_urgency + scarcity_pct * self.w_scarcity + roster_fit * self.w_roster + self.expert_pct * self.expert_weight)
        score -= self.injury * self.injury_penalty
        round_no = (pick - 1) // self.teams + 1
        if round_no < self.min_round_k:
            score[self.pos == 'K'] -= 35.0
        if round_no < self.min_round_def:
            score[self.pos == 'DEF'] -= 35.0
        score = np.where(available, score, -1000000000.0)
        return int(np.argmax(score))

    def _base_roster_need_penalty(self, roster_counts: np.ndarray, round_no: int) -> np.ndarray:
        penalty = np.zeros(self.n, dtype=float)
        for p in ('K', 'DEF'):
            code = self.pos_to_code[p]
            mask = self.pos_code == code
            if round_no < 12:
                penalty[mask] += 100.0
            if roster_counts[code] >= 1:
                penalty[mask] += 150.0
        code = self.pos_to_code['QB']
        mask = self.pos_code == code
        if roster_counts[code] >= 2:
            penalty[mask] += 100.0
        elif roster_counts[code] >= 1 and round_no < 9:
            penalty[mask] += 18.0
        elif roster_counts[code] == 0 and round_no >= 8:
            penalty[mask] -= 8.0
        code = self.pos_to_code['TE']
        mask = self.pos_code == code
        if roster_counts[code] >= 2:
            penalty[mask] += 80.0
        elif roster_counts[code] >= 1 and round_no < 9:
            penalty[mask] += 15.0
        for p, need, cap in (('RB', 2, 6), ('WR', 3, 7)):
            code = self.pos_to_code[p]
            mask = self.pos_code == code
            if roster_counts[code] < need:
                penalty[mask] -= 5.0
            if roster_counts[code] >= cap:
                penalty[mask] += 60.0
        return penalty

    def opponent_pick(self, available: np.ndarray, market_latent: np.ndarray, custom_latent: np.ndarray, roster_counts: np.ndarray, round_no: int, archetype: str) -> int:
        acfg = self.archetype_cfg.get(archetype, {})
        market_w = float(acfg.get('market_weight', 0.8))
        custom_w = float(acfg.get('custom_weight', 0.2))
        need_strength = float(acfg.get('roster_need_strength', 1.0))
        score = market_w * market_latent + custom_w * custom_latent
        score += self._base_roster_need_penalty(roster_counts, round_no) * need_strength
        score = np.where(available, score, 1000000000.0)
        return int(np.argmin(score))

    def evaluate_roster(self, mine: list[int]) -> dict:
        """Evaluate the actual starting lineup plus discounted bench.

        This prevents raw summed VORP from treating all drafted players as
        equally valuable even if roster construction leaves major starter
        groups unfilled.
        """
        rcfg = self.engine.roster_cfg
        evalcfg = self.roster_eval_cfg
        by_pos = {}
        for p in self.POSITIONS:
            idxs = [i for i in mine if self.pos[i] == p]
            idxs.sort(key=lambda i: self.proj[i], reverse=True)
            by_pos[p] = idxs
        starters = []
        used = set()
        missing = Counter()
        for p in ('QB', 'RB', 'WR', 'TE'):
            need = int(rcfg.get(p, 0))
            chosen = by_pos[p][:need]
            starters.extend(chosen)
            used.update(chosen)
            if len(chosen) < need:
                missing[p] = need - len(chosen)
        flex_need = int(rcfg.get('FLEX', 0))
        flex_pool = [i for i in mine if i not in used and self.pos[i] in set(rcfg.get('flex_eligible', []))]
        flex_pool.sort(key=lambda i: self.proj[i], reverse=True)
        flex_chosen = flex_pool[:flex_need]
        starters.extend(flex_chosen)
        used.update(flex_chosen)
        if len(flex_chosen) < flex_need:
            missing['FLEX'] = flex_need - len(flex_chosen)
        starter_vorp = float(np.sum(self.vorp[starters])) if starters else 0.0
        bench = [i for i in mine if i not in used and self.pos[i] not in {'K', 'DEF'}]
        bench_vorp = float(np.sum(np.maximum(self.vorp[bench], 0.0))) if bench else 0.0
        bench_discount = float(evalcfg.get('bench_vorp_discount', 0.2))
        score = starter_vorp + bench_discount * bench_vorp
        missing_penalty_cfg = evalcfg.get('missing_starter_penalty', {})
        for p, count in missing.items():
            score -= float(missing_penalty_cfg.get(p, 0.0)) * int(count)
        counts = Counter((self.pos[i] for i in mine))
        empty_cfg = evalcfg.get('empty_group_penalty', {})
        for p, penalty in empty_cfg.items():
            if counts[p] == 0:
                score -= float(penalty)
        bonus_cfg = evalcfg.get('construction_bonus', {})
        if counts['QB'] >= 1:
            score += float(bonus_cfg.get('has_qb', 0))
        if counts['TE'] >= 1:
            score += float(bonus_cfg.get('has_te', 0))
        if counts['RB'] >= 2:
            score += float(bonus_cfg.get('has_two_rb', 0))
        if counts['WR'] >= 3:
            score += float(bonus_cfg.get('has_three_wr', 0))
        flex_eligible_count = counts['RB'] + counts['WR'] + counts['TE']
        base_flex_need = int(rcfg.get('RB', 0)) + int(rcfg.get('WR', 0)) + int(rcfg.get('TE', 0)) + int(rcfg.get('FLEX', 0))
        if flex_eligible_count >= base_flex_need + 1:
            score += float(bonus_cfg.get('has_flex_depth', 0))
        return {'lineup_score': float(score), 'starter_vorp': starter_vorp, 'bench_vorp': bench_vorp, 'missing_starters': dict(missing)}

    def simulate_candidate(self, current_pick: int, drafted_ids: set[str], my_roster_ids: list[str], candidate_id: str, runs: int, through_round: int, seed: int, draft_history: list[dict] | None=None) -> dict:
        drafted_idx = {self.id_to_idx[str(x)] for x in drafted_ids if str(x) in self.id_to_idx}
        my_idx = [self.id_to_idx[str(x)] for x in my_roster_ids if str(x) in self.id_to_idx]
        candidate_idx = self.id_to_idx[str(candidate_id)]
        rng = np.random.default_rng(seed)
        last_pick = self.teams * int(through_round)
        lineup_scores = np.empty(runs, dtype=float)
        starter_vorps = np.empty(runs, dtype=float)
        total_projections = np.empty(runs, dtype=float)
        second_names = Counter()
        third_names = Counter()
        archetype_pick_counts = Counter()
        base_available = np.ones(self.n, dtype=bool)
        if drafted_idx:
            base_available[list(drafted_idx)] = False
        latent_sd = np.maximum(self.sd_floor, self.adp * self.sd_pct)
        for run in range(runs):
            available = base_available.copy()
            opp_counts = self._seed_opponent_counts(draft_history)
            archetypes = self._archetype_assignments(rng)
            my_counts = np.zeros(len(self.POSITIONS), dtype=np.int16)
            mine = list(my_idx)
            for idx in mine:
                code = self.pos_code[idx]
                if code >= 0:
                    my_counts[code] += 1
            market_latent = np.maximum(1.0, rng.normal(self.adp, latent_sd))
            custom_noise_base = rng.normal(0.0, np.maximum(3.0, self.custom_rank * 0.06))
            followup_pick_no = 0
            for pick in range(int(current_pick), last_pick + 1):
                if not available.any():
                    break
                if pick in self.my_picks:
                    if pick == int(current_pick):
                        idx = candidate_idx
                        if not available[idx]:
                            break
                    else:
                        idx = self.patbot_pick(available, my_counts, pick)
                    available[idx] = False
                    mine.append(idx)
                    code = self.pos_code[idx]
                    if code >= 0:
                        my_counts[code] += 1
                    followup_pick_no += 1
                    if followup_pick_no == 2:
                        second_names[self.names[idx]] += 1
                    elif followup_pick_no == 3:
                        third_names[self.names[idx]] += 1
                else:
                    team_slot = _team_slot_for_pick(pick, self.teams)
                    round_no = (pick - 1) // self.teams + 1
                    archetype = archetypes.get(team_slot, 'market')
                    acfg = self.archetype_cfg.get(archetype, {})
                    randomness = float(acfg.get('randomness', 1.0))
                    custom_latent = np.maximum(1.0, self.custom_rank + custom_noise_base * randomness)
                    idx = self.opponent_pick(available, market_latent, custom_latent, opp_counts[team_slot], round_no, archetype)
                    available[idx] = False
                    code = self.pos_code[idx]
                    if code >= 0:
                        opp_counts[team_slot, code] += 1
                    archetype_pick_counts[archetype] += 1
            eval_result = self.evaluate_roster(mine)
            lineup_scores[run] = eval_result['lineup_score']
            starter_vorps[run] = eval_result['starter_vorp']
            total_projections[run] = float(np.sum(self.proj[mine]))

        def top_counter(counter: Counter, n=5):
            total = sum(counter.values()) or 1
            return [{'player': name, 'pct': round(100.0 * count / total, 1)} for name, count in counter.most_common(n)]
        return {'candidate': self.names[candidate_idx], 'candidate_id': str(candidate_id), 'runs': runs, 'avg_lineup_score': round(float(np.mean(lineup_scores)), 2), 'p25_lineup_score': round(float(np.percentile(lineup_scores, 25)), 2), 'p75_lineup_score': round(float(np.percentile(lineup_scores, 75)), 2), 'avg_starter_vorp': round(float(np.mean(starter_vorps)), 2), 'avg_roster_projected_points': round(float(np.mean(total_projections)), 2), 'most_common_second_pick': top_counter(second_names), 'most_common_third_pick': top_counter(third_names)}

def simulate_candidate(engine, current_pick: int, drafted_ids: set[str], my_roster_ids: list[str], candidate_id: str, runs: int=300, through_round: int=8, seed: int=42, draft_history: list[dict] | None=None) -> dict:
    sim = FastDraftSimulator(engine)
    return sim.simulate_candidate(current_pick=current_pick, drafted_ids=drafted_ids, my_roster_ids=my_roster_ids, candidate_id=candidate_id, runs=int(runs), through_round=int(through_round), seed=int(seed), draft_history=draft_history)

def compare_candidates(engine, current_pick: int, drafted_ids: set[str], my_roster_ids: list[str], candidate_ids: list[str], runs: int=300, through_round: int=8, draft_history: list[dict] | None=None):
    sim = FastDraftSimulator(engine)
    results = []
    for i, pid in enumerate(candidate_ids):
        results.append(sim.simulate_candidate(current_pick=current_pick, drafted_ids=drafted_ids, my_roster_ids=my_roster_ids, candidate_id=str(pid), runs=int(runs), through_round=int(through_round), seed=20260818 + i * 97, draft_history=draft_history))
    summary = pd.DataFrame([{'Candidate': r['candidate'], 'Avg Lineup Score': r['avg_lineup_score'], '25th %ile': r['p25_lineup_score'], '75th %ile': r['p75_lineup_score'], 'Avg Starter VORP': r['avg_starter_vorp'], 'Avg Drafted Proj': r['avg_roster_projected_points'], 'Runs': r['runs']} for r in results]).sort_values('Avg Lineup Score', ascending=False)
    return (summary.reset_index(drop=True), results)
