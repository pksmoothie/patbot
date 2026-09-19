"""GM v0.3.1: auditable roster opportunity costs and bounded heuristic FAAB bids.

Pure analysis of validated snapshots and separately preserved ownership evidence.
No Yahoo actions or network access. Utilities are not point forecasts.
"""
import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path

from gm_faab import load_history, owner_priors, pressure, THRESHOLDS
from gm_lineup import analyze as analyze_lineup, join, name, normalized, numeric
from gm_schema import BASE, LEAGUE_KEY, POSITIONS
from gm_snapshot import decode, load_snapshot, stamp, utc_now
from gm_views import payload

STARTERS = dict(QB=1, RB=2, WR=3, TE=1, DST=1, K=1)
CUTOFF = dict(QB=12, RB=24, WR=36, TE=12, DST=12, K=12)
TIERS = {'League-changing / potential long-term starter':40, 'Strong roster upgrade':15,
         'Useful depth/upside':3, 'Speculative stash':1, 'Streamer':1}
DROP_ORDER = {'Immediate drop':0, 'Replaceable':1, 'Upside stash':2, 'Useful depth':3, 'Core hold':4}


def rank(value):
    if numeric(value):
        return float(value) if value > 0 else None
    match = re.fullmatch(r'(?:QB|RB|WR|TE|DST|K)?(\d+(?:\.\d+)?)', str(value))
    return float(match[1]) if match and float(match[1])>0 else None


def ownership(snapshot, evidence, player, now=None):
    """Exact league-name bridge only when the preserved key/name mapping is unique."""
    unknown = dict(confirmed=False, status='Unconfirmed', evidence=None)
    if not evidence or evidence.get('league_key') != snapshot['league_key']:
        return unknown
    try:
        leagues = decode(evidence['league_lookup']).get('leagues', [])
        matches = [l for l in leagues if l.get('key')==snapshot['league_key']]
        if len(matches)!=1 or sum(l.get('name')==matches[0]['name'] for l in leagues)!=1:
            return unknown
        calls = [c for c in evidence['calls'] if normalized(c['player_name'])==normalized(player)]
        if not calls:
            return unknown
        call = max(calls, key=lambda c: stamp(c['fetched_at']))
        age = (stamp(now or utc_now())-stamp(call['fetched_at'])).total_seconds()/3600
        if not 0 <= age <= 24 or stamp(call['fetched_at']) < stamp(snapshot['refresh_started_at']):
            return dict(unknown, status='Stale ownership confirmation')
        raw = decode(call['raw'])
        player_rows = [rows for key, rows in raw.items()
                       if normalized(re.sub(r' \([^()]*\)$', '', key))==normalized(player)]
        if len(player_rows)!=1:
            return unknown
        rows = [r for r in player_rows[0] if r.get('league_name')==matches[0]['name']]
        if len(rows)!=1:
            return unknown
        available = rows[0].get('ownership_info')=='Not owned - free agent'
        return dict(confirmed=available, status='Confirmed free agent' if available else 'Owned / unavailable',
                    evidence=call, league_mapping=matches[0])
    except (ValueError, KeyError, TypeError):
        return unknown


def enrich(snapshot, p, context=None):
    p = dict(p)
    pos = p['position']
    probe = {'name':p['name'], **({'fp_id':p['id']} if p.get('id') else {})}
    ros = join(probe, payload(snapshot, 'ranking_ROS_'+pos).get('rankings', []))
    p['ros_rank'] = rank(ros.get('pos_rank') or ros.get('rank_ecr'))
    p['ros_source'] = 'ranking_ROS_'+pos if ros else None
    proj = payload(snapshot, 'projection_weekly_'+pos)
    weekly = join(probe, proj.get('projections', []))
    key = {'PPR':'points_ppr','HALF':'points_half','STD':'points'}.get(payload(snapshot,'settings').get('scoring'))
    p['weekly_reference'] = dict(points=weekly.get('stats',{}).get(key), week=proj.get('week'),
                                 source='projection_weekly_'+pos, scoring=payload(snapshot,'settings').get('scoring'))
    compatible = payload(snapshot,'lineup').get('week') is not None and str(payload(snapshot,'lineup')['week'])==str(proj.get('week'))
    p['weekly_comparable'] = compatible
    if 'points' not in p:
        p['points'] = p['weekly_reference']['points'] if compatible else None
    injury = join(probe, payload(snapshot,'injuries').get('practice_details',[]))
    p['injury'] = injury.get('status') or p.get('injury')
    p['role'] = p.get('role') or weekly.get('depth_chart_role')
    p['workload_path'] = p.get('workload_path') or weekly.get('workload_path')
    for call in (context or {}).get('calls',[]):
        try:
            age = (stamp(utc_now())-stamp(call['fetched_at'])).total_seconds()/3600
            if not 0<=age<=24 or stamp(call['fetched_at'])<stamp(snapshot['refresh_started_at']):
                continue
            raw = decode(call['raw'])
            rows = [dict(name=r.get('playerName'),fp_id=r.get('playerId'),**r)
                    for r in raw.get(pos,[]) if isinstance(r,dict)]
            detail = join(probe,rows)
            if detail and numeric(detail.get('depth')):
                p['role'] = pos+str(detail['depth'])
                p['depth_source'] = call
                ahead = [r['name'] for r in rows if r.get('depth',999)<detail['depth']]
                p['workload_path'] = ('Conditional: would need more work than '+', '.join(ahead)+'. No workload change confirmed.'
                                      if ahead else 'First on supplied positional depth chart; future volume is unconfirmed.')
                p['workload_path_confirmed'] = False
        except (ValueError,KeyError,TypeError):
            continue
    p['schedule'] = payload(snapshot, 'schedule_'+pos)
    p['sos_adjustment'] = 0  # Source scale is undocumented; never invent favorable direction.
    return p


def value(p):
    """Position-normalized ROS utility. Missing ROS never becomes a bad rank."""
    r = p.get('ros_rank')
    ros = max(0,18-9*r/CUTOFF[p['position']]) if numeric(r) else None
    weekly = p.get('points') if p.get('weekly_comparable') else None
    if ros is None and not numeric(weekly):
        return None
    result = ros if ros is not None else weekly
    if ros is not None and numeric(weekly):
        result = .75*ros+.25*weekly
    if p.get('workload_path') and p.get('workload_path_confirmed',True):
        result += .5
    return round(result,3)


def drop_hierarchy(roster):
    result = []
    for p in roster:
        p = dict(p, utility=value(p))
        u = p['utility']
        positional_order = sorted([r for r in roster if r['position']==p['position']],
            key=lambda r:value(r) if value(r) is not None else -1,reverse=True)
        depth_rank = next(i for i,r in enumerate(positional_order,1) if r['name']==p['name'])
        if p.get('gm_recommended_starter') or p.get('current_index') is not None or (numeric(p.get('ros_rank')) and p['ros_rank']<=CUTOFF[p['position']]):
            label = 'Core hold'
        elif u is None:
            label = 'Useful depth'  # Missing evidence never makes a player an automatic drop.
        elif p.get('role') == 'released' and not p.get('workload_path'):
            label = 'Immediate drop'
        elif (p.get('workload_path') and (p.get('workload_path_confirmed',True) or p.get('role') in ('RB2','RB3'))) or p.get('injury'):
            label = 'Upside stash'
        elif u>=3 and depth_rank<=STARTERS[p['position']]+1:
            label = 'Useful depth'
        elif u < 6:
            label = 'Replaceable'
        else:
            label = 'Useful depth'
        p['classification'] = label
        p['drop_reason'] = f'{label}; ROS positional rank {p.get("ros_rank")}; current projection {p.get("points")}. '
        if p.get('gm_recommended_starter'):
            p['drop_reason'] += 'GM-recommended starter; protected from waiver drops. '
        if p.get('injury'):
            p['drop_reason'] += 'Injury alone does not establish a permanent loss of value. '
        p['drop_reason'] += 'Workload path: '+str(p.get('workload_path') or 'not supplied')
        if p.get('owner_expendable'):
            p['drop_reason'] += ' Explicit owner preference: first expendable drop; classification and valuation unchanged.'
        if p.get('retain_on_ir'):
            p['drop_reason'] += ' Retain on confirmed IR space; excluded from claim drops.'
        result.append(p)
    return sorted(result,key=lambda p:(0 if p.get('owner_expendable') and
        p['classification']!='Core hold' and not p.get('locked') else 1,
        DROP_ORDER[p['classification']],p['utility'] if p['utility'] is not None else 999,p['name']))


def starter_value(roster):
    """Best positional utility plus one RB/WR/TE FLEX; no cross-position raw rank comparison."""
    used, total = [], 0
    for pos, count in STARTERS.items():
        rows = sorted([p for p in roster if p['position']==pos],key=lambda p:value(p) if value(p) is not None else -1,reverse=True)
        for p in rows[:count]:
            used.append(p['name'])
            total += value(p) or 0
    flex = [value(p) or 0 for p in roster if p['position'] in ('RB','WR','TE') and p['name'] not in used]
    return total + max(flex,default=0)


def scarcity(candidate, roster, pool):
    pos = candidate['position']
    depth = sum(p['position']==pos and not p.get('injury') for p in roster)
    alternatives = sum(p['position']==pos and p.get('availability',{}).get('confirmed') and
                       value(p) is not None and value(p)>= (value(candidate) or 0)-1 for p in pool)
    # Candidate-set scarcity, never a claim about the full free-agent pool.
    return .10 if depth<=STARTERS[pos] and alternatives<=2 else .05 if depth<=STARTERS[pos] else 0


def bid_range(tier, gain, starter_gain, scarce, fp_bid, prior_fraction, remaining=None, role=None, ros_rank=None, position='WR', urgency=False):
    base = TIERS[tier]
    components = dict(roster_improvement=min(.25,max(0,gain)/20),
        regular_starter=min(.20,max(0,starter_gain)/15), scarcity=min(.10,max(0,scarce)),
        role=.10 if role in ('starter','lead back','WR1','RB1','TE1') else 0,
        ros_value=.10 if numeric(ros_rank) and ros_rank<=CUTOFF[position] else 0,
        urgency=.10 if urgency else 0,
        likely_market_interest=.10 if numeric(ros_rank) and ros_rank<=CUTOFF[position] else
            .05 if numeric(fp_bid) and fp_bid>=10 else 0)
    player_value = base*(1+sum(components.values()))
    fp_adjustment = max(-.20,min(.20,(fp_bid-player_value)/max(1,player_value))) if numeric(fp_bid) else 0
    before_history = player_value*(1+fp_adjustment)
    historical = before_history*max(-.05,min(.05,prior_fraction))
    central = before_history+historical
    cap = math.floor(remaining) if numeric(remaining) and 0<=remaining<=100 else 100
    if tier=='Streamer':
        cap = min(cap,5)
    amounts = [min(cap,max(0,math.floor(central*.65))),min(cap,max(0,math.floor(central+.5))),min(cap,max(0,math.ceil(central*1.4)))]
    return dict(conservative=amounts[0],recommended=amounts[1],aggressive_ceiling=amounts[2],
        audit=dict(base=base,adjustments=components,player_value=player_value,fp_fraction=fp_adjustment,
                   before_history=before_history,historical_dollars=historical,remaining_budget=remaining,
                   budget_cap=cap,budget_assumption='Current budget supplied' if remaining is not None else 'Unknown; nominal $100 maximum, verify before submitting'))


def evaluate(candidate, roster, pool, market, remaining=None):
    c = candidate
    warnings = []
    # A weekly-only candidate must be compared on a common weekly-only basis.
    if c.get('ros_rank') is None and c.get('weekly_comparable'):
        if any(not p.get('weekly_comparable') or not numeric(p.get('points')) for p in roster):
            c = dict(c,weekly_comparable=False)
        else:
            roster = [dict(p,ros_rank=None) for p in roster]
            warnings.append('ROS unavailable for candidate: all opportunity costs use aligned weekly points only.')
    if c.get('ros_rank') is None:
        warnings.append('ROS data missing; no ROS value inferred.')
    if not c.get('weekly_comparable'):
        warnings.append('Weekly reference is not aligned to a confirmed matchup week; excluded from comparisons.')
    if not c.get('role') or not c.get('workload_path'):
        warnings.append('Depth-chart role/workload path unconfirmed; no assumed breakout premium.')
    result = dict(add=c['name'],drop=None,position=c['position'],availability=c['availability'],
        candidate_source=c.get('candidate_source'),fantasypros_recommendation=c.get('fantasypros_recommendation'),
        weekly_value=c.get('weekly_reference'),ros_value=c.get('ros_rank'),inputs=c,
        recommendation='NO CLAIM',strength='Pass',tier='Speculative stash',confidence='Low',warnings=warnings,
        faab=dict(conservative=0,recommended=0,aggressive_ceiling=0),roster_need='No demonstrated upgrade',
        opportunity_cost=None,utility_gain=0,starter_gain=0)
    if not c['availability']['confirmed'] or value(c) is None or c.get('injury') in ('O','OUT','IR','PUP','SUSP'):
        warnings.append('Unavailable, injured, or insufficient comparable valuation evidence.')
        return result
    if c['position'] in ('K','DST') and not c.get('weekly_comparable'):
        warnings.append('Streaming requires aligned weekly evidence; ROS rank alone cannot justify a claim.')
        return result
    baseline = starter_value(roster)
    options = []
    for d in drop_hierarchy(roster):
        if d.get('gm_recommended_starter') or d.get('locked') or d.get('utility') is None or d.get('retain_on_ir'):
            continue
        if c.get('ros_rank') is not None and d.get('ros_rank') is None:
            continue  # Do not compare a ROS/weekly blend to a weekly-only drop.
        if c['position'] in ('K','DST') and d['position'] != c['position']:
            continue
        # Preserve positional starters and scarce depth. Core replacements require same position.
        if d['classification']=='Core hold' and d['position']!=c['position']:
            continue
        after = [p for p in roster if p['name']!=d['name']]+[c]
        if any(sum(p['position']==pos for p in after)<min(count,sum(p['position']==pos for p in roster)) for pos,count in STARTERS.items()):
            continue
        gain = value(c)-d['utility']
        starting = starter_value(after)-baseline
        same = c['position']==d['position']
        owner_release = (d.get('owner_expendable') and d['classification']!='Core hold'
                         and c['position'] in d.get('owner_drop_positions', ('RB','WR')))
        if gain>=.6 and starting>=-.01 and (same or starting>=.6 or owner_release):
            options.append((gain+2*starting,d,gain,starting))
    if not options:
        return result
    _,drop,gain,starting = max(options,key=lambda o:(bool(o[1].get('owner_expendable')),
                                                   o[0]))
    scarce = scarcity(c,roster,pool)
    regular = numeric(c.get('ros_rank')) and c['ros_rank']<=CUTOFF[c['position']]
    tier = ('Streamer' if c['position'] in ('K','DST') else
            'League-changing / potential long-term starter' if regular and starting>=5 else
            'Strong roster upgrade' if starting>=2 else 'Useful depth/upside' if gain>=1 else 'Speculative stash')
    bids = bid_range(tier,gain,starting,scarce,c.get('fp_bid'),market['fraction'],remaining,
                     c.get('role'),c.get('ros_rank'),c['position'],scarce>0 and starting>0)
    result.update(drop=drop['name'],recommendation='CLAIM',strength='Strong' if starting>=2 else 'Optional',tier=tier,
        confidence='Medium' if c.get('weekly_comparable') and c.get('ros_rank') else 'Low',
        faab=bids,roster_need=f'{c["position"]} '+('starter/FLEX improvement' if starting>=.6 else 'depth improvement; not a current starter upgrade'),
        opportunity_cost=dict(player=drop['name'],classification=drop['classification'],utility=drop['utility'],
                              ros_rank=drop.get('ros_rank'),injury=drop.get('injury'),reason=drop['drop_reason']),
        utility_gain=round(gain,3),starter_gain=round(starting,3),scarcity=scarce)
    return result


def claim_plan(board, remaining=None):
    """Shared-drop claims are mutually exclusive fallback groups; cap all-success spend."""
    plan, groups, reserved = [], {}, 0
    for row in board:
        if row['recommendation']!='CLAIM':
            continue
        group = groups.get(row['drop'])
        bid = row['faab']['recommended']
        if group is None:
            budget = max(0,remaining-reserved) if numeric(remaining) else None
            if budget is not None and bid>budget:
                continue
            group = {'orders':[], 'reserved':bid}
            groups[row['drop']] = group
            reserved += bid
        elif bid>group['reserved']:
            if numeric(remaining) and reserved+bid-group['reserved']>remaining:
                continue
            reserved += bid-group['reserved']
            group['reserved'] = bid
        order = len(plan)+1
        plan.append(dict(order=order,add=row['add'],drop=row['drop'],bid=bid,
            requires_failed_claims=list(group['orders']),
            instruction='Submit as fallback: earlier successful claim using this drop invalidates this claim.' if group['orders'] else 'Primary claim for this drop.',
            budget_condition='Verify current FAAB before submitting' if remaining is None else 'Within worst-case successful-claim budget'))
        group['orders'].append(order)
    return plan


def collect_candidates(snapshot, evidence, context=None):
    candidates = {}
    for pos in POSITIONS:
        data = payload(snapshot,'waiver_'+pos)
        rows = [(r,'FantasyPros weekly candidate',None) for r in data.get('top_starts_this_week',{}).get(pos,[])]
        rows += [(r.get('Player to Add',r),'FantasyPros recommended add',r) for r in data.get('recommended_adds',[])]
        # Expansion only includes players for whom ownership was actually queried.
        queried = {normalized(c['player_name']) for c in (evidence or {}).get('calls',[])}
        rows += [(r,'ROS ranking expansion',None) for r in payload(snapshot,'ranking_ROS_'+pos).get('rankings',[])
                 if normalized(name(r)) in queried and normalized(name(r)) not in candidates]
        for r,source,advice in rows:
            key = normalized(name(r))
            if not key:
                continue
            prior = candidates.get(key,{})
            sources = prior.get('candidate_source',[])
            if source not in sources:
                sources.append(source)
            fp = advice or prior.get('fantasypros_recommendation')
            dollars = re.fullmatch(r'\$(\d+)',str((fp or {}).get('FAB Bid','')))
            p = dict(name=name(r),position=pos,candidate_source=sources,fantasypros_recommendation=fp,
                     fp_bid=int(dollars[1]) if dollars else None,availability=ownership(snapshot,evidence,name(r)))
            candidates[key] = enrich(snapshot,p,context)
    return list(candidates.values())


def apply_drop_preference(snapshot, context, roster, now=None):
    """Explicit owner release preference changes drop order, never player ranks."""
    preference = (context or {}).get('drop_preference', {})
    try:
        age = (stamp(now or utc_now())-stamp(preference['confirmed_at'])).total_seconds()
        if (preference.get('source')!='user' or preference.get('league_key')!=snapshot['league_key']
                or preference.get('snapshot_id')!=snapshot['snapshot_id']
                or not 0<=age<=86400
                or stamp(preference['confirmed_at'])<stamp(snapshot['refresh_started_at'])):
            return roster
        result = []
        for p in roster:
            p = dict(p)
            if normalized(p['name'])==normalized(preference['preferred_drop']):
                p['retain_on_ir'] = preference.get('ir_move_confirmed') is True
                p['owner_expendable'] = (not p['retain_on_ir'] and not p.get('gm_recommended_starter')
                    and p.get('current_index') is None and not p.get('locked'))
                p['owner_drop_positions'] = preference.get('positions', ['RB','WR'])
                p['owner_drop_preference'] = preference
            result.append(p)
        return result
    except (KeyError,TypeError,ValueError,AttributeError):
        return roster


def confirmed_budget(snapshot, context, now=None):
    """Keep user-confirmed balances separate from unmodified provider evidence."""
    confirmation = (context or {}).get('faab_confirmation', {})
    try:
        if (confirmation.get('source') != 'user'
                or confirmation.get('league_key') != snapshot['league_key']
                or confirmation.get('snapshot_id') != snapshot['snapshot_id']):
            return None
        confirmed_at = stamp(confirmation['confirmed_at'])
        age = (stamp(now or utc_now()) - confirmed_at).total_seconds()
        if not 0 <= age <= 86400 or confirmed_at < stamp(snapshot['refresh_started_at']):
            return None
        balances = confirmation['team_balances']
        amount = balances[str(snapshot['team_id'])]
        if not numeric(amount) or not 0 <= amount <= 100:
            return None
        return dict(confirmation, remaining=amount)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def analyze_waivers(snapshot, evidence=None, history=None, context=None):
    history = history if history is not None else load_history()
    if evidence is None:
        path = BASE/'data'/'waiver_ownership.json'
        evidence = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    priors = owner_priors(history)
    if context is None:
        path = BASE/'data'/'waiver_context.json'
        context = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    market = pressure(priors)
    lineup = analyze_lineup(snapshot)
    warnings = list(lineup['warnings'])
    recommended_ids = {p['id'] for p in lineup['recommended_lineup']}
    roster = [enrich(snapshot,dict(p,gm_recommended_starter=p['id'] in recommended_ids),context)
              for p in lineup['players']]
    roster = apply_drop_preference(snapshot,context,roster)
    expected = {normalized(n) for names in payload(snapshot,'my_roster').get('roster',{}).values() for n in names}
    actual = {normalized(p['name']) for p in roster}
    complete = expected==actual and bool(actual)
    snapshot_age = (stamp(utc_now())-stamp(snapshot['refreshed_at'])).total_seconds()/3600
    if snapshot_age>24 or snapshot_age<0:
        complete = False
        warnings.append('Snapshot is stale or future-dated; refresh before generating actionable claims.')
    pool = collect_candidates(snapshot,evidence,context)
    # Positive ownership confirmation must not conflict with the validated roster capture.
    owned = {normalized(n) for t in payload(snapshot,'all_rosters').get('teams',[])
             for names in t.get('roster',{}).values() for n in names}
    for c in pool:
        if normalized(c['name']) in owned:
            c['availability'] = dict(c['availability'],confirmed=False,status='Roster/ownership conflict; refresh')
    remaining = payload(snapshot,'my_roster').get('faab_remaining')
    budget_confirmation = None
    if not numeric(remaining) or not 0<=remaining<=100:
        budget_confirmation = confirmed_budget(snapshot, context)
        remaining = budget_confirmation['remaining'] if budget_confirmation else None
    if remaining is None:
        warnings.append('Current remaining FAAB unavailable. Bid ranges assume at most $100; verify budget before submitting.')
    elif budget_confirmation:
        market['caveat'] = market['caveat'].replace('Competitors and current budgets are unconfirmed.',
            'Current balances use the recorded user confirmation; competitors participating in these claims are unconfirmed.')
    if expected!=actual or not actual:
        warnings.append('Lineup players do not exactly cover the actual roster. Claims disabled pending a complete roster capture.')
    board = [evaluate(c,roster,pool,market,remaining) for c in pool]
    if not complete:
        for row in board:
            row.update(recommendation='NO CLAIM',drop=None,strength='Pass',opportunity_cost=None,
                       utility_gain=0,starter_gain=0,faab=dict(conservative=0,recommended=0,aggressive_ceiling=0))
            row['warnings'].append('Claims disabled: roster inputs incomplete or snapshot stale.')
    board.sort(key=lambda r:(r['recommendation']=='CLAIM',r['starter_gain'],r['utility_gain'],r['add']),reverse=True)
    for i,row in enumerate(board,1):
        row['priority']=i
    warnings += ['Waiver Finder and expanded rankings are a partial candidate pool.',
                 'Bids are transparent heuristics, not winning-bid forecasts. Owner history adjusts unrounded bids by at most 5%.',
                 'Depth-chart/workload evidence is missing unless explicitly populated; SOS is context only.']
    return dict(version='0.3.1',generated_at=utc_now(),snapshot_id=snapshot['snapshot_id'],
        snapshot_refreshed_at=snapshot['refreshed_at'],league_key=snapshot['league_key'],
        candidate_coverage={p:sum(c['position']==p for c in pool) for p in POSITIONS},
        lineup_recommendation=lineup['recommended_lineup'],
        roster_inputs=roster,drop_hierarchy=drop_hierarchy(roster),
        roster_needs=[dict(position=p,rostered=sum(r['position']==p for r in roster),
            required_starters=n,healthy_depth=sum(r['position']==p and not r.get('injury') for r in roster)-n,
            note='RB/WR/TE also compete for one FLEX slot' if p in ('RB','WR','TE') else 'One starting slot') for p,n in STARTERS.items()],
        historical_owner_prior=dict(owners=priors,thresholds=THRESHOLDS,pressure=market,
            source='config/faab_history.json',sha256=hashlib.sha256(json.dumps(history,sort_keys=True).encode()).hexdigest()),
        remaining_faab=remaining,budget_confirmation=budget_confirmation,
        board=board,claim_plan=claim_plan(board,remaining),
        overall='CLAIM' if any(r['recommendation']=='CLAIM' for r in board) else 'NO CLAIM',warnings=warnings)


def save_waiver_analysis(analysis, folder=None):
    folder = Path(folder) if folder else BASE/'data'
    folder.mkdir(parents=True,exist_ok=True)
    temporary = folder/('.waivers-'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(analysis,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(temporary,folder/'latest_waiver_analysis.json')
    finally:
        temporary.unlink(missing_ok=True)


if __name__=='__main__':
    analysis = analyze_waivers(load_snapshot())
    save_waiver_analysis(analysis)
    print(json.dumps(dict(overall=analysis['overall'],claims=analysis['claim_plan']),indent=2))
