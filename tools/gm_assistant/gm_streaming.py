"""GM v0.4 streaming: snapshot-only decisions, explicit weeks, manual transactions.

Outlook scores are heuristic utilities, never future fantasy-point projections.
Only single-week, documented FantasyPros SOS ratings affect those utilities.
"""
import hashlib
import copy
import json
import os
import re
import uuid
from pathlib import Path

from gm_lineup import join, name, normalized, numeric, score_projection
from gm_schema import BASE
from gm_snapshot import decode, load_snapshot, stamp, utc_now
from gm_views import payload
from gm_waivers import ownership, rank
from gm_week import establish_week, kickoff_for

CONFIG = json.loads((Path(__file__).parent/'streaming_config.json').read_text(encoding='utf-8'))
RULES = json.loads((Path(__file__).parent/'scoring_rules.json').read_text(encoding='utf-8'))
POSITIONS = ('DST', 'K')


def fresh(timestamp, now, started=None):
    try:
        return (0 <= (stamp(now)-stamp(timestamp)).total_seconds() <= 86400
                and (started is None or stamp(timestamp) >= stamp(started)))
    except (ValueError, TypeError, AttributeError):
        return False


def read_optional(filename):
    path = BASE/'data'/filename
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def schedule_sources(snapshot, sidecar, now):
    sources = []
    for pos in POSITIONS:
        section = snapshot['sections'].get('schedule_'+pos, {})
        if section.get('status') == 'ok':
            sources.append(dict(id='schedule_'+pos, data=payload(snapshot,'schedule_'+pos),
                                source_ids=section.get('source_ids', [])))
    if sidecar.get('snapshot_id') == snapshot.get('snapshot_id'):
        for i, call in enumerate(sidecar.get('calls', [])):
            if call.get('kind') != 'schedule' or not fresh(call.get('fetched_at'), now, snapshot['refresh_started_at']):
                continue
            try:
                data = decode(call['raw'])
                if isinstance(data, dict):
                    sources.append(dict(id='streaming_sources:'+str(i), data=data,
                                        arguments=call.get('arguments'), fetched_at=call['fetched_at']))
            except (ValueError, KeyError, TypeError):
                continue
    return sources


def schedule_for(team, pos, week, sources):
    """Never distribute a multiweek SOS average across individual games."""
    matches, ratings, refs = set(), set(), []
    for source in sources:
        d = source['data']
        for key in ('team schedule', 'easiest schedules'):
            group = d.get(key, {})
            documented = 'high star rating and low rank is good' in group.get('IMPORTANT INFO','').lower()
            for row in group.get('schedules', []):
                if row.get('team') != team:
                    continue
                detail = row.get('Weekly Matchups', {}).get('Week '+str(week), {})
                for item in row.get('Matchups', []):
                    detail = item.get('Week '+str(week), detail)
                if isinstance(detail, str):
                    detail = dict(matchup=detail)
                if detail.get('matchup'):
                    # Normalize home/away formatting; opponent identity is what this model uses.
                    matchup = detail['matchup']
                    tokens = re.findall(r'\b[A-Z]{2,3}\b', matchup)
                    opponent = 'BYE' if 'BYE' in matchup.upper() else next((t for t in tokens if t!=team), None)
                    if opponent:
                        matches.add(opponent)
                rating = detail.get('strength_of_schedule_ratings', {}) or {}
                window = d.get('sos_window', {})
                if not rating and window.get('week_start')==window.get('week_end')==week:
                    rating = row.get('Strength of Schedule', {})
                if documented:
                    stars = rating.get(pos.lower()+'_stars_out_of_5', rating.get(pos.lower()+'_stars'))
                    if numeric(stars) and 1 <= stars <= 5:
                        # FP exposes one decimal in positional windows and two in
                        # all-position detail. Compare at their common precision.
                        ratings.add(round(float(stars),1))
                if detail or rating:
                    refs.append(dict(source=source['id'], detail=detail, rating=rating))
    conflict = len(matches)>1 or len(ratings)>1
    return dict(week=week, opponent=next(iter(matches)) if len(matches)==1 else None,
                bye=matches=={'BYE'}, stars=next(iter(ratings)) if len(ratings)==1 and not conflict else None,
                conflict=conflict, sources=refs)


def weighted(scores, weights):
    return round(sum(s*w for s,w in zip(scores,weights)),3) if all(numeric(s) for s in scores) else None


def candidate_pool(snapshot, evidence, now):
    result = []
    owned = {normalized(n) for t in payload(snapshot,'all_rosters').get('teams', [])
             for names in t.get('roster', {}).values() for n in names}
    mine = payload(snapshot,'my_roster').get('roster', {})
    for pos in POSITIONS:
        projections = payload(snapshot,'projection_weekly_'+pos).get('projections', [])
        ecr = payload(snapshot,'ranking_WEEKLY_'+pos).get('rankings', [])
        waiver = payload(snapshot,'waiver_'+pos)
        rows = [(r,'Waiver Finder') for r in waiver.get('top_starts_this_week',{}).get(pos, [])]
        rows += [(r.get('Player to Add', r),'Waiver Finder') for r in waiver.get('recommended_adds', [])]
        rows += [(r,'Weekly projection expansion') for r in projections]
        rows += [(r,'Weekly ECR expansion') for r in ecr]
        rows += [(dict(name=n),'Current roster') for n in mine.get(pos, [])]
        unique = {}
        for r, source in rows:
            if not name(r):
                continue
            key = normalized(name(r))
            unique.setdefault(key, dict(name=name(r), position=pos, candidate_sources=[]))['candidate_sources'].append(source)
        for key, p in unique.items():
            p['current'] = key in {normalized(n) for n in mine.get(pos, [])}
            projection = join(p, projections)
            ranking = join(p, ecr)
            p.update(team=projection.get('team') or ranking.get('team'), projection_source=projection,
                     ecr_source=ranking, availability=ownership(snapshot,evidence,p['name'],now))
            if p['current'] or key in owned:
                p['availability'] = dict(confirmed=False,status='Current roster' if p['current'] else 'Owned / unavailable')
            result.append(p)
    return result


def score_candidate(p, snapshot, week, sources, config, rules, season=None):
    p = dict(p)
    pos = p['position']
    data = payload(snapshot,'projection_weekly_'+pos)
    aligned = (week is not None and str(data.get('week'))==str(week)
               and (season is None or str(data.get('season'))==str(season)))
    p['projection_context'] = dict(source='projection_weekly_'+pos,week=data.get('week'),
                                   season=data.get('season'),scoring=data.get('scoring'),aligned=aligned)
    stats = p['projection_source'].get('stats', {}) if aligned else {}
    scoring = score_projection(stats,pos,rules,stats.get('points'))
    p['scoring'] = scoring
    p['projection'] = scoring['points'] if numeric(scoring['points']) else None
    p['ecr'] = rank(p['ecr_source'].get('pos_rank') or p['ecr_source'].get('rank_ecr'))
    # Many ECR responses omit week. Keep these visible, but do not silently align them.
    ecr_week = payload(snapshot,'ranking_WEEKLY_'+pos).get('week')
    p['ecr_aligned'] = week is not None and str(ecr_week)==str(week)
    warnings = []
    if not aligned:
        warnings.append('Projection week/season unavailable or mismatched; excluded.')
    if not p['ecr_aligned']:
        warnings.append('Weekly ECR has no matching explicit week; context only.')
    if scoring['unavailable_components']:
        warnings.append('Custom scoring incomplete: using whole-player FantasyPros fallback; no invented components.')
    weeks = []
    for offset in range(4):
        row = schedule_for(p['team'],pos,week+offset,sources) if week else dict(week=None,opponent=None,bye=False,stars=None,conflict=False,sources=[])
        components = {}
        if row['bye']:
            score = 0.0
            components['bye'] = 0.0
        elif offset==0:
            score = p['projection']
            if numeric(score):
                components['projection'] = score
                if numeric(row['stars']):
                    components['matchup'] = config['dst_matchup_weight' if pos=='DST' else 'k_matchup_weight']*(row['stars']-3)
                if p['ecr_aligned'] and numeric(p['ecr']):
                    components['ecr'] = config['ecr_weight']*max(-1,min(1,(16.5-p['ecr'])/15.5))
                score = sum(components.values())
        else:
            score = None
            if numeric(row['stars']) and row['opponent']:
                components['future_schedule_baseline'] = config['future_dst_base' if pos=='DST' else 'future_k_base']
                components['matchup'] = config['future_dst_matchup_weight' if pos=='DST' else 'future_k_matchup_weight']*(row['stars']-3)
                score = sum(components.values())
        if row['conflict']:
            score = None
            warnings.append('Conflicting schedule inputs; affected week excluded.')
        if row['opponent'] is None or row['stars'] is None:
            warnings.append(f'Week +{offset}: opponent or weekly SOS unavailable.')
        row.update(score=round(score,3) if numeric(score) else None,components=components)
        weeks.append(row)
    p.update(weeks=weeks, composite=weighted([r['score'] for r in weeks[:3]],config['weights']),
             warnings=list(dict.fromkeys(warnings)), gm_action='Current option' if p['current'] else 'Reference only')
    return p


def decide(pos, rows, roster, config, enabled=True, reserved_drops=()):
    currents = [p for p in rows if p['current']]
    current = max(currents,key=lambda p:p['weeks'][0]['score'] if numeric(p['weeks'][0]['score']) else -999,default=None)
    call = dict(position=pos,current=current['name'] if current else None, recommendation='NO ACTION',
                add=None,drop=None,this_week_edge=None,score_edge=None,composite_edge=None,
                transaction_required=False,confidence='Low',reason='Current option or comparable data unavailable.',
                three_week_outlook=current['composite'] if current else None)
    targets = []
    if not enabled or not current or not numeric(current['weeks'][0]['score']):
        return call, targets
    current_roster = next((r for r in roster if normalized(r['name'])==normalized(current['name'])), {})
    if current_roster.get('locked') or current.get('locked'):
        call['reason'] = 'Current option is locked; no transaction recommended.'
        return call, targets
    call.update(recommendation='HOLD',reason='No confirmed available option clears the churn thresholds.')
    available = [p for p in rows if not p['current'] and p['availability'].get('confirmed') is True
                 and numeric(p['weeks'][0]['score']) and not p['weeks'][0]['bye'] and p['weeks'][0].get('opponent')
                 and not p.get('injury') and not p.get('locked')]
    swaps = []
    for p in available:
        pp = p['projection']
        cp = 0 if current['weeks'][0]['bye'] else current['projection']
        if not numeric(pp) or not numeric(cp):
            continue
        edge = pp-cp
        score_edge = p['weeks'][0]['score']-current['weeks'][0]['score']
        composite_edge = p['composite']-current['composite'] if numeric(p['composite']) and numeric(current['composite']) else None
        matchup_edge = (p['weeks'][0]['stars']-current['weeks'][0]['stars']
                        if numeric(p['weeks'][0]['stars']) and numeric(current['weeks'][0]['stars']) else 0)
        worth = (edge>=config['projection_edge'] or
                 (pos=='DST' and matchup_edge>=1 and edge>=-config['max_current_sacrifice']))
        future_ok = composite_edge is None or composite_edge>=config['composite_edge'] or edge>=config['material_projection_edge']
        if score_edge>=config['current_score_edge'] and worth and future_ok:
            swaps.append((p,edge,score_edge,composite_edge))
    if swaps:
        p,edge,score_edge,composite_edge = max(swaps,key=lambda x:(x[0]['composite'] if numeric(x[0]['composite']) else x[0]['weeks'][0]['score'],x[2]))
        call.update(recommendation='STREAM',add=p['name'],drop=current['name'],this_week_edge=round(edge,3),
                    score_edge=round(score_edge,3),composite_edge=round(composite_edge,3) if numeric(composite_edge) else None,
                    three_week_outlook=p['composite'],transaction_required=True,
                    confidence='Medium' if numeric(p['composite']) and numeric(current['composite']) else 'Low',
                    reason=f'This week: {edge:+.2f} projected points; {score_edge:+.2f} GM score. '
                    f'Matchup stars: {p["weeks"][0]["stars"]} vs {current["weeks"][0]["stars"]}. '
                    f'Next-week stars: {p["weeks"][1]["stars"]} vs {current["weeks"][1]["stars"]}. '
                    'Current edge clears churn threshold; large projection edges can outweigh a weaker future schedule.')
    if pos=='DST':
        for p in available:
            future = p['weeks'][1:3]
            baseline = current['weeks'][1:3]
            if not all(numeric(w['stars']) and numeric(w['score']) and not w['bye'] for w in future):
                continue
            if not all(numeric(w['score']) for w in baseline):
                continue
            future_edge = weighted([a['score']-b['score'] for a,b in zip(future,baseline)],[.625,.375])
            if (future[0]['stars']>=config['stash_next_stars'] and
                weighted([w['stars'] for w in future],[.625,.375])>=config['stash_future_stars'] and
                future_edge>=config['stash_future_edge']):
                targets.append(dict(name=p['name'],availability=p['availability'],label='Potential early stash',
                                    next_opponent=future[0]['opponent'],following_opponent=future[1]['opponent'],
                                    future_edge=future_edge,reason='Favorable next two matchups; bench cost must also clear the stash gate.'))
        # Never sacrifice meaningful depth/upside. Reuse the existing ordered, classified roster.
        drops = [r for r in roster if r['position'] not in POSITIONS and
                 r.get('classification') in ('Immediate drop','Replaceable') and
                 not r.get('gm_recommended_starter') and r.get('current_index') is None and
                 r.get('weekly_comparable') is not False and
                 not r.get('locked') and numeric(r.get('utility')) and r['name'] not in reserved_drops]
        if targets and call['recommendation']=='HOLD' and not current['weeks'][0]['bye'] and len(currents)<config['max_defenses'] and drops:
            target = max(targets,key=lambda t:t['future_edge'])
            drop = drops[0]
            net = .4*target['future_edge']-config['bench_cost_weight']*max(0,drop['utility'])
            target.update(bench_drop=drop['name'],bench_cost=drop['utility'],net_advance_value=round(net,3))
            if net>=config['stash_net_edge']:
                call.update(recommendation='STASH FOR NEXT WEEK',add=target['name'],drop=drop['name'],
                            transaction_required=True,confidence='Low',
                            reason=f'Keep {current["name"]} this week. Future schedule edge {target["future_edge"]:.2f}; '
                            f'net advance value {net:.2f} after bench cost for {drop["name"]} ({drop["classification"]}).')
    return call, targets


def analyze_streaming(snapshot, waiver_analysis=None, evidence=None, sources=None, config=None, rules=None, now=None, calendar=None):
    now = now or utc_now()
    config = dict(CONFIG if config is None else config)
    if len(config['weights'])!=3 or not all(numeric(w) and w>=0 for w in config['weights']) or abs(sum(config['weights'])-1)>1e-8:
        raise ValueError('Three nonnegative composite weights must sum to one.')
    rules = rules or RULES
    evidence = read_optional('streaming_ownership.json') if evidence is None else evidence
    sources = read_optional('streaming_sources.json') if sources is None else sources
    schedules = schedule_sources(snapshot,sources,now)
    week_info = establish_week(schedules, now=now, calendar=calendar)
    week_info['projection_weeks'] = {pos:payload(snapshot,'projection_weekly_'+pos).get('week') for pos in POSITIONS}
    week = week_info['week']
    if waiver_analysis is None:
        from gm_waivers import analyze_waivers
        waiver_analysis = analyze_waivers(snapshot)
    roster = waiver_analysis['drop_hierarchy']
    expected = {normalized(n) for names in payload(snapshot,'my_roster').get('roster',{}).values() for n in names}
    complete = bool(expected) and expected=={normalized(r['name']) for r in roster}
    enabled = bool(week) and complete and fresh(snapshot['refreshed_at'],now)
    pool = candidate_pool(snapshot,evidence,now)
    pool = [score_candidate(p,snapshot,week,schedules,config,rules,season=week_info['season']) for p in pool]
    for p in pool:
        injury = join(p,payload(snapshot,'injuries').get('practice_details',[]))
        p['injury'] = injury.get('status')
        own = next((r for r in roster if normalized(r['name'])==normalized(p['name'])),{})
        p['kickoff'] = kickoff_for(week_info['date_evidence'],p['team'],week)
        p['locked'] = own.get('locked',False) or bool(p['kickoff'] and stamp(p['kickoff']) <= stamp(now))
    calls, upcoming = [], []
    reserved = {r['drop'] for r in waiver_analysis.get('claim_plan',[]) if r.get('drop')}
    for pos in POSITIONS:
        rows = [p for p in pool if p['position']==pos]
        call, targets = decide(pos,rows,roster,config,enabled,reserved)
        if not enabled:
            call['reason'] = 'Analysis unavailable: explicit current week, fresh snapshot, and complete roster are required.'
        calls.append(call)
        upcoming += targets
        for p in rows:
            if p['name']==call['add']:
                p['gm_action'] = call['recommendation']
            elif p['current']:
                p['gm_action'] = ('DROP for stream' if p['name']==call['drop'] else
                                  'NO ACTION' if call['recommendation']=='NO ACTION' else 'HOLD')
            elif p['availability']['confirmed']:
                p['gm_action'] = 'No transaction'
    pool.sort(key=lambda p:(p['position'],-(p['composite'] if numeric(p['composite']) else -999),
                            -(p['weeks'][0]['score'] if numeric(p['weeks'][0]['score']) else -999),p['name']))
    return dict(version='0.4',generated_at=now,snapshot_id=snapshot['snapshot_id'],current_week=week_info,
                enabled=enabled,config=config,scoring_rules=rules,candidate_pool=pool,calls=calls,upcoming_targets=upcoming,
                source_schedule_inputs=schedules,ownership_evidence=evidence,
                input_sha256=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest(),
                warnings=['Future outlooks and composites are heuristic scores, not point forecasts.',
                          'Week +3 is displayed for planning; only current/+1/+2 enter the composite.',
                          'No betting lines, implied totals, invented distance distributions or opponent statistics.',
                          'Explicit NFL game dates determine the playing slate; provider and projection weeks remain separate.',
                          'Unknown ECR week is context only. Missing SOS is never treated as an easy matchup.',
                          'Yahoo actions and FAAB submission remain manual. Recheck player locks before acting.'])


def save_streaming_analysis(analysis, folder=None):
    folder = Path(folder) if folder else BASE/'data'
    folder.mkdir(parents=True,exist_ok=True)
    temporary = folder/('.streaming-'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(analysis,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(temporary,folder/'latest_streaming_analysis.json')
    finally:
        temporary.unlink(missing_ok=True)


def route_waivers(waivers, streaming):
    """Presentation/audit handoff only; the validated generic waiver engine is unchanged."""
    result = copy.deepcopy(waivers)
    result['streaming_calls'] = streaming['calls']
    result['streaming_snapshot_id'] = streaming['snapshot_id']
    for row in result['board']:
        if row['position'] in POSITIONS:
            row.update(recommendation='SEE STREAMING',drop=None,opportunity_cost=None,
                       strength='See Streaming',faab=dict(conservative=0,recommended=0,aggressive_ceiling=0))
    stream_names = {p['name'] for p in streaming['candidate_pool']}
    result['claim_plan'] = [r for r in result['claim_plan'] if r['add'] not in stream_names]
    result['overall'] = 'CLAIM' if any(r['recommendation']=='CLAIM' for r in result['board']) else 'NO CLAIM'
    return result


if __name__=='__main__':
    result = analyze_streaming(load_snapshot())
    save_streaming_analysis(result)
    print(json.dumps(dict(current_week=result['current_week'],calls=result['calls'],upcoming=result['upcoming_targets']),indent=2))
