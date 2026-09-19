"""Deterministic weekly optimizer. No network, draft imports, or Yahoo actions.

Scores are decision utilities, not revised fantasy-point forecasts. No fitted
distribution or invented floor/ceiling. See MODEL for all adjustable weights.
"""
import hashlib
import itertools
import json
import math
import os
import re
import uuid
from pathlib import Path

from gm_snapshot import load_snapshot, utc_now
from gm_views import payload

BASE = Path(__file__).resolve().parent
SLOTS = ('QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'K', 'DST')
MODEL = dict(version='0.2', favorite_probability=.65, underdog_probability=.35,
             close_points=1.0, max_lineup_sacrifice=1.0, expert_weight=.15,
             ecr_weight=.10, injury_weight=.25, strategy_weight=.35,
             workload_scale=25, low_confidence_gap=.25)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def identity(p):
    for key in ('fp_id', 'fpId', 'fpid', 'player_id'):
        if p.get(key) is not None:
            return str(p[key])
    return None


def name(p):
    return p.get('player_name') or p.get('full') or p.get('name') or ''


def normalized(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())


def join(player, rows):
    """ID first, unique conservative name fallback; conflicting IDs never join."""
    pid = identity(player)
    hits = [r for r in rows if pid is not None and identity(r) == pid]
    if not hits:
        hits = [r for r in rows if name(player) and normalized(name(r)) == normalized(name(player))
                and (pid is None or identity(r) is None)]
    return hits[0] if len(hits) == 1 else {}


def slot(value):
    return {'WR/RB/TE': 'FLEX', 'WR+RB+TE': 'FLEX', 'DEF': 'DST', 'D/ST': 'DST'}.get(value, value)


def score_projection(stats, position, rules, fallback):
    """Reconstruct only from explicit stats. Missing category => whole-player fallback.

    Threshold bonuses require event-count/probability projections, never a test
    against mean yards. Fumbles are not treated as fumbles LOST. K distance/miss
    buckets and DST points-allowed probabilities must be explicitly provided.
    """
    stats = normalize_stats(stats)
    s = rules['scoring']
    components, missing = {}, []

    def add(label, stat, weight):
        if not weight:
            return
        if numeric(stats.get(stat)):
            components[label] = stats[stat] * weight
        else:
            missing.append(stat)

    if position == 'K':
        k = rules['kicker_scoring']
        for bucket, weight in k['field_goal_made'].items():
            add('FG ' + bucket, 'fg_' + bucket, weight)
        for bucket, weight in k['field_goal_missed'].items():
            add('FG missed ' + bucket, 'fg_missed_' + bucket, weight)
        add('PAT', 'xpt', k['extra_point_made'])
        add('PAT missed', 'xpt_missed', k['extra_point_missed'])
    elif position == 'DST':
        d = rules['defense_scoring']
        for key in ('sack', 'interception', 'fumble_recovery', 'touchdown', 'safety', 'blocked_kick', 'extra_point_returned'):
            add(key, key, d[key])
        for bucket, weight in d['points_allowed'].items():
            add('Points allowed ' + str(bucket), 'points_allowed_probability_' + str(bucket), weight)
    else:
        groups = [('rush', 'rush_yards_per_point', 'rush_td', 'rush_yard_bonuses'),
                  ('rec', 'rec_yards_per_point', 'rec_td', 'rec_yard_bonuses')]
        if position == 'QB':
            groups.append(('pass', 'pass_yards_per_point', 'pass_td', 'pass_yard_bonuses'))
            add('Completions', 'pass_cmp', s['pass_completion'])
            add('Interceptions', 'pass_ints', s['interception'])
        for prefix, yards, td, bonuses in groups:
            add(prefix + ' yards', prefix + '_yds', 1 / s[yards])
            add(prefix + ' TD', prefix + '_tds', s[td])
            for bonus in s.get(bonuses, []):
                add(prefix + ' bonus ' + str(bonus['threshold']),
                    prefix + '_yds_' + str(bonus['threshold']), bonus['points'])
        add('Receptions', 'rec', s['reception'])
        for label, stat, rule in [('Fumbles lost', 'fumbles_lost', 'fumble_lost'),
                                  ('Return TD', 'ret_tds', 'return_td'),
                                  ('Two point', '2pt_tds', 'two_point_conversion'),
                                  ('Offensive fumble TD', 'offensive_fumble_return_td', 'offensive_fumble_return_td')]:
            add(label, stat, s[rule])
    return dict(points=fallback if missing else sum(components.values()),
                method='FantasyPros whole-player fallback' if missing else 'Yahoo custom scoring',
                calculated_components=components, unavailable_components=missing)


def normalize_stats(stats):
    """Only aliases confirmed in the captured FP component payloads."""
    stats = dict(stats)
    for source, target in {'rec_rec': 'rec', 'def_sack': 'sack', 'def_int': 'interception',
                           'def_fr': 'fumble_recovery', 'def_safety': 'safety',
                           'def_td': 'touchdown'}.items():
        if target not in stats and source in stats:
            stats[target] = stats[source]
    return stats


def build_players(snapshot, rules):
    lineup = payload(snapshot, 'lineup')
    raw = lineup.get('raw_payload', {}).get('matchup', {}).get('team1', {})
    raw_rows = sum([raw.get(g, []) or [] for g in ('starters', 'bench', 'ir')], [])
    advice = payload(snapshot, 'start_sit')
    experts = {}
    for group in ('players_to_consider', 'players_to_sit', 'players_to_start', 'recommended_lineup'):
        for p in advice.get(group, []) or []:
            experts[identity(p) or normalized(name(p))] = p
    injuries = payload(snapshot, 'injuries').get('practice_details', []) or []
    news = payload(snapshot, 'injuries').get('player_news', []) or []
    players, warnings, counts = [], [], {}
    for group in ('starters', 'bench', 'ir'):
        for p in lineup.get(group, []) or []:
            r = {**join(p, raw_rows), **{k: v for k, v in p.items() if v is not None}}
            pos = slot(r.get('player_pos') or r.get('real_position'))
            current_slot = slot(r.get('slot') or r.get('position'))
            current_index = None
            if group == 'starters' and current_slot in SLOTS:
                indices = [i for i, s in enumerate(SLOTS) if s == current_slot]
                n = counts.get(current_slot, 0)
                if n < len(indices):
                    current_index = indices[n]
                counts[current_slot] = n + 1
            projection = payload(snapshot, 'projection_weekly_' + str(pos))
            # An unspecified matchup week does not establish compatibility.
            compatible = (lineup.get('week') is not None and
                          str(lineup['week']) == str(projection.get('week')))
            projected = join(r, projection.get('projections', []) or []) if compatible else {}
            source_stats = projected.get('stats', {}) or {}
            stats = normalize_stats(source_stats)
            fp_points = r.get('original_proj')
            if not numeric(fp_points):
                point_key = {'PPR': 'points_ppr', 'HALF': 'points_half', 'STD': 'points'}.get(payload(snapshot, 'settings').get('scoring'))
                fp_points = stats.get(point_key)
            scoring = score_projection(stats, pos, rules, fp_points)
            if not compatible:
                scoring['unavailable_components'].append('Weekly component projection not joined: matchup week absent or different')
            injury = join(r, injuries).get('status') or r.get('injuryStatus') or r.get('injury_status')
            eligibility = r.get('eligibility') or [pos]
            if isinstance(eligibility, str):
                eligibility = re.split(r'[/,+ ]+', eligibility)
            eligible = sorted({slot(e) for e in eligibility if isinstance(e, str)})
            pid = identity(r) or normalized(name(r))
            expert = experts.get(pid, {})
            locked = r.get('locked') is True or r.get('isPreGame') is False or r.get('gameStatus') in ('In Progress', 'Final')
            rank = r.get('ecr')
            # Raw matchup positional ECR is aligned with its player projections.
            role = r.get('depth_chart_role') or r.get('depthChartRole')
            players.append(dict(id=pid, name=name(r), position=pos, eligibility=eligible,
                team=r.get('player_team') or r.get('real_team'), current_index=current_index,
                locked=locked, lock_known=r.get('locked') is not None or r.get('isPreGame') is not None,
                excluded=group == 'ir' or str(injury).upper() in ('O', 'OUT', 'IR', 'PUP', 'SUSP', 'SUSPENDED'),
                points=scoring['points'], scoring=scoring, fp_points=fp_points, ecr=rank,
                expert_percent=expert.get('percent_of_experts'), injury=injury, role=role,
                news=join(r, news).get('news'),
                opponent=r.get('opponent'), sos=r.get('sos'), stats=stats, source_stats=source_stats,
                source_sections=['lineup', 'settings', 'start_sit', 'injuries'] +
                    (['projection_weekly_' + str(pos)] if projected else [])))
    if len({p['id'] for p in players}) != len(players):
        raise ValueError('Duplicate roster player identity; cannot safely optimize.')
    if any(not p['lock_known'] for p in players):
        warnings.append('Some player locks are unavailable; confirm eligibility and locks in Yahoo.')
    if any(p['scoring']['unavailable_components'] for p in players):
        warnings.append('Custom scoring is incomplete: fallback FantasyPros points are not verified Yahoo-scored points. See per-player scoring audit.')
    if lineup.get('week') is None:
        warnings.append('Matchup week is unspecified; separate weekly projections and rankings were not joined.')
    warnings.append('No calibrated floor, ceiling, or revised win probability is available. SOS scale is not documented and receives no numerical weight.')
    warnings.append('GM-local scoring rules copied from config/league.yaml take precedence over FantasyPros settings. Both rule sets are recorded in the audit; fallback totals are not corrected for scoring differences.')
    return players, warnings


def eligible(player, target):
    return bool(set(player['eligibility']) & {'RB', 'WR', 'TE'}) if target == 'FLEX' else target in player['eligibility']


def legal_lineups(players):
    """Every legal assignment, collapsing only permutations of identical slots.

    FLEX placements remain distinct, including locked slot constraints. A locked
    bench player cannot enter; a locked starter must stay in their exact slot.
    """
    fixed = {p['current_index']: p for p in players if p['locked'] and p['current_index'] is not None}
    choices = [[p for p in players if eligible(p, target) and
                (p is fixed[i] if i in fixed else not p['locked']) and
                (not p['excluded'] or p is fixed.get(i))] for i, target in enumerate(SLOTS)]

    def visit(chosen, used):
        i = len(chosen)
        if i == len(SLOTS):
            yield tuple(chosen)
            return
        for p in choices[i]:
            if p['id'] in used:
                continue
            # Canonical order only across unlocked identical slots.
            prior = [j for j in range(i) if SLOTS[j] == SLOTS[i] and j not in fixed]
            if i not in fixed and prior and chosen[prior[-1]]['id'] > p['id']:
                continue
            yield from visit(chosen + [p], used | {p['id']})
    yield from visit([], set())


def strategy(probability, model=MODEL):
    if numeric(probability) and 0 <= probability <= 1:
        if probability >= model['favorite_probability']:
            return 'Floor-Leaning'
        if probability <= model['underdog_probability']:
            return 'Ceiling-Leaning'
    return 'Neutral'


def adjustment(player, mode, model=MODEL):
    """Bounded qualitative support, not fantasy-point floor/ceiling estimates.

    Workload proxy: projected carries + receptions / 25, capped at 1.
    Variance proxy: TD share of projected points; never injury-as-upside.
    Positional ECR contributes at most .10, experts .15. SOS and free-text role
    remain context only because their scale/semantics are not established.
    """
    parts = {}
    pct = player.get('expert_percent')
    if numeric(pct) and 0 <= pct <= 100:
        parts['expert_support'] = model['expert_weight'] * pct / 100
    rank = re.fullmatch(r'(QB|RB|WR|TE|K|DST)(\d+)', str(player.get('ecr')))
    if rank and int(rank[2]) > 0:
        parts['positional_ecr'] = model['ecr_weight'] / int(rank[2])
    if player.get('injury'):
        parts['injury_uncertainty'] = -model['injury_weight']
    stats = player.get('stats', {})
    if mode == 'Floor-Leaning' and all(numeric(stats.get(k)) for k in ('rush_att', 'rec')):
        parts['workload_reliability_proxy'] = model['strategy_weight'] * min(1, max(0, stats['rush_att'] + stats['rec']) / model['workload_scale'])
    if mode == 'Ceiling-Leaning' and all(numeric(stats.get(k)) for k in ('rush_tds', 'rec_tds')) and numeric(player.get('points')) and player['points'] > 0:
        parts['touchdown_variance_proxy'] = model['strategy_weight'] * min(1, max(0, 6 * (stats['rush_tds'] + stats['rec_tds']) / player['points']))
    return parts


def total(lineup):
    return sum(p['points'] for p in lineup) if lineup and all(numeric(p['points']) for p in lineup) else None


def optimize(players, mode, model=MODEL):
    alternatives = list(legal_lineups(players))
    scored = [a for a in alternatives if total(a) is not None]
    if not scored:
        return None, None, alternatives
    current_ids = {p['id'] for p in players if p['current_index'] is not None}
    def stable(a):
        return (len(current_ids & {p['id'] for p in a}), tuple(p['id'] for p in a))
    best = max(scored, key=lambda a: (total(a), stable(a)))
    best_ids = {p['id'] for p in best}
    def close(a):
        if total(best) - total(a) > model['max_lineup_sacrifice'] + 1e-9:
            return False
        incoming = [p for p in a if p['id'] not in best_ids]
        outgoing = [p for p in best if p['id'] not in {q['id'] for q in a}]
        # Require a legal assignment of paired close swaps; prevents sacrificing
        # a materially better player behind offsetting improvements elsewhere.
        return any(all(abs(p['points'] - q['points']) <= model['close_points'] and
                       bool(set(p['eligibility']) & set(q['eligibility']))
                       for p, q in zip(incoming, perm)) for perm in itertools.permutations(outgoing))
    recommended = max((a for a in scored if close(a)),
                      key=lambda a: (total(a) + sum(sum(adjustment(p, mode, model).values()) for p in a), stable(a)))
    return best, recommended, alternatives


def present(lineup, current):
    if lineup is None:
        return []
    # Keep current slot placements when legal, to avoid cosmetic changes.
    result = list(lineup)
    for i, p in enumerate(current):
        if p is None or p not in result or result[i]['locked']:
            continue
        j = result.index(p)
        if not p['locked'] and eligible(result[i], SLOTS[j]) and eligible(p, SLOTS[i]):
            result[i], result[j] = result[j], result[i]
    return [dict(slot=SLOTS[i], id=p['id'], name=p['name'], points=p['points'],
                 changed=p['id'] not in {q['id'] for q in current if q}) for i, p in enumerate(result)]


def explain(p, q, mode, model):
    if q is None or not numeric(q['points']):
        return 'Fills a starting slot with an eligible player who has an available projection.'
    gap = p['points'] - q['points']
    pieces = [f"Projection: {p['points']:.2f} vs {q['points']:.2f} ({gap:+.2f})."]
    if abs(gap) <= model['close_points']:
        pieces.append('This is a close decision.')
    p_adj, q_adj = adjustment(p, mode, model), adjustment(q, mode, model)
    labels = {'expert_support': 'expert start support', 'positional_ecr': 'positional ECR',
              'injury_uncertainty': 'less reported injury uncertainty',
              'workload_reliability_proxy': 'projected workload reliability proxy',
              'touchdown_variance_proxy': 'projected touchdown variance proxy'}
    supports = [label for key, label in labels.items() if p_adj.get(key, 0) > q_adj.get(key, 0)]
    if supports:
        pieces.append('Additional support: ' + ', '.join(supports) + '.')
    pieces.append(f'{mode} strategy; all adjustments stay within the close-decision limits.')
    if p['scoring']['unavailable_components'] or q['scoring']['unavailable_components']:
        pieces.append('Points use an incomplete-scoring fallback; treat a small edge as tentative.')
    return ' '.join(pieces)


def analyze(snapshot, rules=None, model=None, timestamp=None):
    rules = rules or json.loads((BASE / 'scoring_rules.json').read_text())
    model = dict(MODEL if model is None else model)
    players, warnings = build_players(snapshot, rules)
    probability = payload(snapshot, 'matchup').get('win_probability')
    mode = strategy(probability, model)
    best, recommended, alternatives = optimize(players, mode, model)
    current = [next((p for p in players if p['current_index'] == i), None) for i in range(len(SLOTS))]
    if recommended is None:
        warnings.append('No fully projected legal lineup is available; no actionable recommendation produced.')
    if any(not numeric(p['points']) and not p['excluded'] for p in players):
        warnings.append('Some legal alternatives lack projections and cannot be ranked; recommendation is provisional.')
    rec_rows = present(recommended, current)
    current_ids = {p['id'] for p in current if p}
    rec_ids = {p['id'] for p in recommended or []}
    fp = payload(snapshot, 'start_sit').get('recommended_lineup')
    fp_ids = {identity(p) for p in fp or []}
    changes = []
    outgoing = [p for p in current if p and p['id'] not in rec_ids] if recommended else []
    incoming = [p for p in recommended or [] if p['id'] not in current_ids]
    # Match replacements by actual final slot first; then common eligibility.
    for p in incoming:
        new_index = next(i for i, row in enumerate(rec_rows) if row['id'] == p['id'])
        q = next((q for q in outgoing if q['current_index'] == new_index), None)
        q = q or next((q for q in outgoing if set(q['eligibility']) & set(p['eligibility'])), None)
        q = q or (outgoing[0] if outgoing else None)
        if q:
            outgoing.remove(q)
        gap = p['points'] - q['points'] if q and numeric(q['points']) else None
        confidence = 'High' if gap is not None and gap > model['close_points'] else 'Medium'
        if p['scoring']['unavailable_components'] or (q and q['scoring']['unavailable_components']) or gap is None or abs(gap) < model['low_confidence_gap']:
            confidence = 'Low'
        changes.append(dict(action=f"START {p['name']} over {q['name'] if q else '(empty slot)'}",
            gm_recommendation=p['name'], sit=q['name'] if q else None,
            projected_point_difference=gap, confidence=confidence, explanation=explain(p, q, mode, model),
            fantasypros_recommendation=('Agrees' if p['id'] in fp_ids and (q is None or q['id'] not in fp_ids) else 'Disagrees') if fp is not None else 'Unavailable',
            inputs={'start': p, 'sit': q, 'strategy': mode,
                    'start_adjustments': adjustment(p, mode, model),
                    'sit_adjustments': adjustment(q, mode, model) if q else {}}))
    # Retained starters with a directly legal close bench alternative are real decisions too.
    decisions = list(changes)
    if recommended:
        for p in recommended:
            if p['id'] not in current_ids or p['locked']:
                continue
            i = list(recommended).index(p)
            rivals = [q for q in players if q['id'] not in rec_ids and not q['locked'] and not q['excluded']
                      and eligible(q, SLOTS[i]) and numeric(q['points']) and abs(p['points'] - q['points']) <= model['close_points']]
            if rivals:
                q = max(rivals, key=lambda q: q['points'])
                decisions.append(dict(action=f"KEEP {p['name']} over {q['name']}", gm_recommendation=p['name'],
                    sit=q['name'], confidence='Low', projected_point_difference=p['points']-q['points'],
                    explanation=explain(p, q, mode, model),
                    fantasypros_recommendation=('Agrees' if p['id'] in fp_ids and q['id'] not in fp_ids else 'Disagrees') if fp is not None else 'Unavailable',
                    inputs={'start': p, 'sit': q, 'strategy': mode,
                            'start_adjustments': adjustment(p, mode, model), 'sit_adjustments': adjustment(q, mode, model)}))
    return dict(analysis_timestamp=timestamp or utc_now(), snapshot_id=snapshot.get('snapshot_id'),
        snapshot_sha256=hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
        engine_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        snapshot_scoring_settings=payload(snapshot, 'settings').get('raw_settings', {}),
        model=model, scoring_rules=rules, strategy=mode, win_probability=probability,
        win_probability_source='FantasyPros current lineup; not recalculated for GM lineup',
        current_lineup=[dict(slot=SLOTS[i], id=p['id'], name=p['name'], points=p['points']) for i,p in enumerate(current) if p],
        current_projection=total(current) if all(current) else None,
        recommended_lineup=rec_rows, recommended_projection=total(recommended),
        highest_projected_lineup=present(best, current), alternative_lineup_projection=total(best),
        alternatives=[dict(player_ids=[p['id'] for p in a], projection=total(a)) for a in alternatives],
        lineup_changes=changes, decisions=decisions, fantasypros_recommendation=fp,
        players=players, warnings=warnings,
        unavailable_inputs=['Calibrated floor/ceiling', 'Calibrated GM win probability', 'Documented SOS scale'] +
            (['Structured depth-chart role'] if not any(p['role'] for p in players) else []))


def save_analysis(analysis, folder=None):
    folder = Path(folder) if folder else BASE / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    digest = hashlib.sha256(text.encode()).hexdigest()
    history = folder / 'analyses'
    history.mkdir(exist_ok=True)
    destination = history / (digest + '.json')
    try:
        with destination.open('x', encoding='utf-8') as stream:
            stream.write(text)
    except FileExistsError:
        pass
    temporary = folder / ('.analysis-' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, folder / 'latest_analysis.json')
    finally:
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == '__main__':
    result = analyze(load_snapshot())
    print(save_analysis(result))
    print(result['strategy'], result['recommended_projection'])
    for change in result['lineup_changes']:
        print(change['action'], '(' + change['confidence'] + ' confidence)')
