"""GM v0.5: both-roster trade impact using the unchanged v0.2 optimizer.

Read-only FantasyPros evidence. No Yahoo actions, trade-history assumptions,
draft imports, or numerical championship-probability claims.
"""
import copy
import hashlib
import itertools
import json
import os
import uuid
from pathlib import Path

from gm_faab import canonical_owner, load_history
from gm_lineup import (SLOTS, analyze as analyze_lineup, identity, join, name,
                       normalized, numeric, optimize, present, score_projection, total)
from gm_schema import BASE, TEAM_ID
from gm_snapshot import decode, load_snapshot, utc_now
from gm_streaming import analyze_streaming, fresh, read_optional
from gm_week import kickoff_for
from gm_snapshot import stamp
from gm_views import payload
from gm_waivers import CUTOFF, STARTERS, analyze_waivers, rank

CONFIG = json.loads((Path(__file__).parent/'trade_config.json').read_text(encoding='utf-8'))
RULES = json.loads((Path(__file__).parent/'scoring_rules.json').read_text(encoding='utf-8'))
SKILL = ('QB','RB','WR','TE')
TIERS = {'Strong target':0,'Worth offering':1,'Research only':2,'Fair but unnecessary':3,'Only if discounted':4,'Reject':5}


def is_protected(player):
    return bool(player.get('protected') or player.get('gm_recommended_starter'))


def owner_identity(team, history, mappings=None):
    tid = str(team['team_id'])
    if tid==TEAM_ID and team.get('is_user_team'):
        return dict(owner='Pat',identity_source='Validated user team ID')
    entry = (mappings or {}).get('teams',{}).get(tid)
    if entry and entry.get('team_name')==team['team_name'] and entry.get('source'):
        try:
            return dict(owner=canonical_owner(history,entry['owner']),identity_source=entry['source'])
        except ValueError:
            pass
    try:
        return dict(owner=canonical_owner(history,team['team_name']),identity_source='Exact unique saved historical alias; no behavior inference')
    except ValueError:
        return dict(owner='Unconfirmed owner (team '+tid+')',identity_source='No reliable owner mapping')


def asset_value(p, config=CONFIG):
    """Explicit position-normalized convex ROS curve; not a market-price forecast."""
    r = p.get('ros_rank')
    if p['position'] not in SKILL or not numeric(r) or r<=0:
        return None
    return round(100*config['scarcity'][p['position']]/(1+(r/(CUTOFF[p['position']]/2))**1.4),3)


def construct_rosters(snapshot, lineup, waivers, week, history, mappings, config, season=None, calendar=None, now=None):
    original = {normalized(p['name']):p for p in lineup['players']}
    protected = {normalized(p['name']):p for p in waivers['drop_hierarchy']}
    starters = {normalized(p['name']) for p in lineup['recommended_lineup']}
    teams = []
    for team in payload(snapshot,'all_rosters').get('teams',[]):
        players = []
        mine = str(team['team_id'])==TEAM_ID
        for pos, names in team.get('roster',{}).items():
            for label in names:
                probe = dict(name=label)
                projection = payload(snapshot,'projection_weekly_'+pos)
                row = join(probe,projection.get('projections',[]))
                ros = join(probe,payload(snapshot,'ranking_ROS_'+pos).get('rankings',[]))
                old = original.get(normalized(label),{}) if mine else {}
                aligned = (week is not None and str(projection.get('week'))==str(week)
                           and season is not None and str(projection.get('season'))==str(season))
                stats = row.get('stats',{}) if aligned else {}
                key = {'PPR':'points_ppr','HALF':'points_half','STD':'points'}.get(payload(snapshot,'settings').get('scoring'))
                scoring = score_projection(stats,pos,RULES,stats.get(key))
                injury = join(probe,payload(snapshot,'injuries').get('practice_details',[])).get('status') or old.get('injury')
                p = dict(id=identity(row) or identity(ros) or normalized(label),name=label,position=pos,
                         eligibility=old.get('eligibility',[pos]),current_index=old.get('current_index'),
                         locked=old.get('locked',False),excluded=old.get('excluded',False) or
                         str(injury).upper() in ('O','OUT','IR','PUP','SUSP','SUSPENDED'),injury=injury,
                         points=scoring['points'],stats=stats,scoring=scoring,expert_percent=old.get('expert_percent'),
                         ecr=None,team=row.get('team') or ros.get('team'),ros_rank=rank(ros.get('pos_rank') or ros.get('rank_ecr')),
                         role=old.get('role') or row.get('depth_chart_role'),bye=row.get('bye_week',ros.get('bye_week')),
                         projection_source=dict(section='projection_weekly_'+pos,week=projection.get('week'),season=projection.get('season'),row=row,aligned=aligned),
                         ros_source=ros,waiver_classification=protected.get(normalized(label),{}).get('classification'),
                         ros_projection_available=bool(join(probe,payload(snapshot,'projection_ros_'+pos).get('projections',[]))),
                         protected=mine and (normalized(label) in starters or
                             protected.get(normalized(label),{}).get('classification')=='Core hold' or
                             protected.get(normalized(label),{}).get('gm_recommended_starter',False)),
                         gm_recommended_starter=mine and (normalized(label) in starters or
                             protected.get(normalized(label),{}).get('gm_recommended_starter',False)))
                p['trade_classification'] = ('Core / protected starter' if p['protected'] else
                                             p['waiver_classification'] or 'Unclassified asset')
                p['asset_value'] = asset_value(p,config)
                kickoff = kickoff_for(calendar or {},p['team'],week)
                if kickoff and now and stamp(kickoff) <= stamp(now):
                    # Opponent slot assignments are unknown; do not trade played assets.
                    p['trade_locked'] = True
                else:
                    p['trade_locked'] = p['locked']
                players.append(p)
        teams.append(dict(team_id=str(team['team_id']),team_name=team['team_name'],
                          **owner_identity(team,history,mappings),players=players))
    return teams


def quick_lineup(players, field='points'):
    """Cheap single-position screening bound; final decisions use optimize()."""
    used, selected = set(), []
    for pos, count in STARTERS.items():
        options = sorted([p for p in players if p['position']==pos and not p['excluded'] and numeric(p.get(field))],key=lambda p:(-p[field],p['id']))
        if len(options)<count:
            return []
        for p in options[:count]:
            used.add(p['id'])
            selected.append(p)
    flex = sorted([p for p in players if p['id'] not in used and p['position'] in ('RB','WR','TE') and not p['excluded'] and numeric(p.get(field))],key=lambda p:(-p[field],p['id']))
    return selected+flex[:1] if flex else []


def quick_total(players, field='points'):
    selected = quick_lineup(players,field)
    return sum(p[field] for p in selected) if selected else None


def optimize_roster(players, mode='Neutral'):
    _, recommended, _ = optimize(players,mode)
    current = [next((p for p in players if p['current_index']==i),None) for i in range(len(SLOTS))]
    return dict(projection=total(recommended),lineup=present(recommended,current))


def roster_values(players):
    """ROS starter utility and bench insurance remain separate from weekly points."""
    converted = [dict(p,points=(18-9*p['ros_rank']/CUTOFF[p['position']]) if numeric(p['ros_rank']) else None,
                      excluded=False,locked=False,current_index=None) for p in players]
    # K/DST are fixed placeholders for ROS skill-position comparison, never trade assets.
    for p in converted:
        if p['position'] not in SKILL:
            p['points']=0
        elif numeric(p['points']):
            p['points']=max(0,p['points'])
    selected = quick_lineup(converted)
    ids = {p['id'] for p in selected}
    missing = [p['name'] for p in players if p['position'] in SKILL and
               (not numeric(p.get('ros_rank')) or not numeric(p.get('asset_value')))]
    ros = sum(p['points'] for p in selected) if selected and not missing else None
    active = sorted([p for p in players if p['position'] in SKILL and numeric(p['asset_value'])],key=lambda p:p['asset_value'],reverse=True)
    return dict(starter_ros=ros,starter_ids=sorted(ids),
                bench_assets=sum(p['asset_value'] for p in active if p['id'] not in ids) if not missing and selected else None,
                asset_total=sum(p['asset_value'] for p in active) if not missing else None,missing_ros=missing)


def profile(team, baseline):
    players = team['players']
    starts = {p['id'] for p in baseline['lineup']}
    positions = []
    for pos in SKILL:
        rows = [p for p in players if p['position']==pos]
        startable = [p for p in rows if numeric(p['ros_rank']) and p['ros_rank']<=CUTOFF[pos] and not p['excluded']]
        count = STARTERS[pos]
        positions.append(dict(position=pos,rostered=len(rows),ros_startable=len(startable),
            surplus=max(0,len(startable)-count),need=max(0,count-len(startable)),
            projected_starters=[p['name'] for p in rows if p['id'] in starts],
            excess_depth=[p['name'] for p in rows if p['id'] not in starts],
            missing_ros=[p['name'] for p in rows if p['ros_rank'] is None]))
    return dict(owner=team['owner'],team_id=team['team_id'],team_name=team['team_name'],
                identity_source=team['identity_source'],positions=positions,
                strengths=[p['position'] for p in positions if p['ros_startable']>=STARTERS[p['position']]],
                weaknesses=[p['position'] for p in positions if p['need']],
                surplus=[p['position'] for p in positions if p['surplus']],
                starting_bottlenecks=[p['name'] for p in players if p['id'] in starts and p['position'] in SKILL and
                                     numeric(p['ros_rank']) and p['ros_rank']>CUTOFF[p['position']]],
                modeled_lineup=baseline)


def trade_key(owner_id, give, receive):
    return str(owner_id)+'|'+','.join(sorted(normalized(n) for n in give))+'|'+','.join(sorted(normalized(n) for n in receive))


def trusted_calls(snapshot, evidence, now):
    if evidence.get('snapshot_id')!=snapshot['snapshot_id'] or evidence.get('league_key')!=snapshot['league_key']:
        return []
    result=[]
    for call in evidence.get('calls',[]):
        if call.get('arguments',{}).get('league_key')!=snapshot['league_key'] or not fresh(call.get('fetched_at'),now,snapshot['refresh_started_at']):
            continue
        try:
            data=decode(call['raw'])
            if isinstance(data,dict) and not data.get('status'):
                result.append(dict(call,data=data))
        except (ValueError,TypeError,KeyError):
            continue
    return result


def verification(calls, give, receive, partner, user_team_name=None):
    for call in reversed(calls):
        args=call['arguments']
        if call.get('kind')!='analyzer':
            continue
        sent={normalized(n.strip()) for n in args.get('players_trade_away','').split(',')}
        got={normalized(n.strip()) for n in args.get('players_trade_for','').split(',')}
        d=call['data']
        if sent!={normalized(p['name']) for p in give} or got!={normalized(p['name']) for p in receive}:
            continue
        # Match the returned teams AND exact player IDs; never trust request args alone.
        if d.get('team2')!=partner['team_name']:
            continue
        if user_team_name is not None and d.get('team1')!=user_team_name:
            continue
        if {str(p.get('id')) for p in d.get('team1GetsDetails',[])}!={p['id'] for p in receive}:
            continue
        if {str(p.get('id')) for p in d.get('team2GetsDetails',[])}!={p['id'] for p in give}:
            continue
        ros=d.get('rest_of_season_power_ranking_increase')
        if not isinstance(ros,str):
            continue
        try:
            gain=float(ros.rstrip('%'))
        except ValueError:
            continue
        if not numeric(gain):
            continue
        other_ros=d.get('rest_of_season_power_ranking_increase_team2')
        if other_ros is None and d.get('default_power_rank_change_horizon')=='rest_of_season':
            other_ros=d.get('default_team_2_power_rank_change')
        try:
            other_gain=float(str(other_ros).rstrip('%'))
            other_gain=other_gain if numeric(other_gain) else None
        except ValueError:
            other_gain=None
        return dict(verified=True,verdict='Live analyzer ROS '+ros,ros_gain=gain,
                    counterparty_ros_gain=other_gain,source=call)
    return dict(verified=False,verdict='Not verified - analyzer required',ros_gain=None,source=None)


def package_levels(rows, config=CONFIG):
    approved = [r for r in rows if r['approved'] and r.get('plausible') is True
                and r['verification']['verified'] and r['tier'] in ('Strong target','Worth offering')
                and numeric(r['asset_ratio']) and config['minimum_asset_ratio']<=r['asset_ratio']<=config['maximum_asset_ratio']]
    if not approved:
        return {}
    # Net cost of the principal target deducts the other assets received.
    approved.sort(key=lambda r:(r['effective_target_cost'],-r['roster_equity_delta'],r['id']))
    fair = min(approved,key=lambda r:(abs(r['asset_ratio']-1),-r['roster_equity_delta'],r['id']))
    return dict(opening=approved[0],fair=fair,walk_away=approved[-1])


def grade_trade(row, config=CONFIG):
    """ROS-first decision gates shared by actual offers and behavioral tests."""
    row.update(approved=False,plausible=False,confidence='Low',local_tier='Not eligible')
    def stop(tier,reason):
        row.update(tier=tier,reason=reason)
        return row
    if not row['ros_assessment']['complete']:
        return stop('Research only','Incomplete ROS data; long-term value is unknown, not replaced by weekly projections.')
    ratio=row['asset_ratio']
    if not numeric(ratio) or ratio<config['minimum_asset_ratio']:
        return stop('Reject','Lopsided opening price: insufficient ROS assets for the counterparty. A low-cost opening is not exempt.')
    if ratio>config['maximum_asset_ratio']:
        return stop('Only if discounted','Price exceeds the maximum plausible ROS asset exchange.')
    if row['ros_lineup_delta']<0 or row['roster_equity_delta']<0:
        return stop('Only if discounted','ROS roster value declines; a weekly projection gain cannot offset that loss.')
    if (row['counterparty_ros_delta'] < -config['counterparty_ros_loss_limit'] or
        row['counterparty_equity_delta']<0 or
        row['counterparty_delta'] < -config['counterparty_weekly_loss_limit']):
        return stop('Reject','Counterparty roster fit fails: starting lineup or ROS depth loss is not adequately replaced.')
    if not (row['counterparty_ros_delta']>=config['counterparty_minimum_ros_gain'] or
            (row['counterparty_delta']>=config['counterparty_minimum_weekly_gain'] and row['counterparty_ros_delta']>=0)):
        return stop('Reject','No concrete counterparty starting-lineup or ROS roster benefit; asset arithmetic alone is insufficient.')
    if row['protected_sent'] and (row['ros_lineup_delta']<config['protected_ros_gain'] or
                                  row['roster_equity_delta']<config['protected_equity_gain']):
        return stop('Only if discounted','Protected GM starter/core asset requires a material ROS upgrade; ordinary market value is insufficient.')
    if (row['ros_lineup_delta']<config['minimum_ros_gain'] or
            row['roster_equity_delta']<config['minimum_equity_gain']):
        return stop('Fair but unnecessary','No material ROS roster improvement. A tiny weekly gain does not justify the trade.')
    if row['lineup_delta'] < -config['maximum_weekly_sacrifice']:
        return stop('Only if discounted','ROS benefit comes with too much current-week starting-lineup loss.')
    if row.get('acquired_inactive'):
        return stop('Research only','Injury/return timing unresolved; ROS proxy is incomplete for this acquisition.')
    row.update(plausible=True,local_tier='Plausible ROS upgrade')
    verified=row['verification']
    if verified['verified'] and verified['ros_gain']<0:
        return stop('Only if discounted','FantasyPros reports negative ROS impact; weekly upside cannot override that verdict.')
    if verified['verified'] and numeric(verified.get('counterparty_ros_gain')) and verified['counterparty_ros_gain'] < -config['analyzer_counterparty_loss_limit']:
        row['plausible']=False
        return stop('Reject','FantasyPros counterparty ROS loss contradicts a plausible offer; improve the return to that roster.')
    if not verified['verified'] or not numeric(verified.get('counterparty_ros_gain')):
        return stop('Research only','Not send-ready: exact fresh FantasyPros verification of both ROS impacts is required.')
    if row['owner'].startswith('Unconfirmed owner'):
        return stop('Research only','Not send-ready: owner identity remains unresolved; no owner was guessed.')
    row.update(approved=True,confidence='Low' if row['ros_assessment'].get('context_incomplete') else 'Medium',tier='Strong target' if row['ros_lineup_delta']>=2 and row['roster_equity_delta']>=1 else 'Worth offering',
               reason='Material ROS roster improvement, plausible asset price and concrete counterparty fit; exact FantasyPros verification passed.')
    return row


def evaluate_trade(mine, partner, give_names, receive_names, baselines, config=CONFIG, calls=(), capacity=15, mode='Neutral', cache=None):
    give=[p for p in mine['players'] if p['name'] in give_names]
    receive=[p for p in partner['players'] if p['name'] in receive_names]
    result=dict(id=trade_key(partner['team_id'],give_names,receive_names),owner=partner['owner'],team_id=partner['team_id'],
                team_name=partner['team_name'],give=give_names,receive=receive_names,target=', '.join(receive_names),
                tier='Reject',local_tier='Reject',confidence='Low',lineup_delta=None,asset_give=None,asset_receive=None,
                asset_ratio=None,approved=False,plausible=False,warnings=[],reason='Invalid trade or incomplete comparable inputs.',
                ros_assessment=dict(complete=False,summary='ROS roster assessment unavailable.'),
                verification=dict(verified=False,verdict='Not verified - analyzer required'),before=baselines[mine['team_id']],
                after=None,counterparty_before=baselines[partner['team_id']],counterparty_after=None)
    if (len(give)!=len(give_names) or len(receive)!=len(receive_names) or not give or not receive or
        len(set(give_names))!=len(give_names) or len(set(receive_names))!=len(receive_names) or
        any(p['position'] not in SKILL or p['locked'] or p.get('trade_locked') for p in give+receive)):
        return result
    after=[p for p in mine['players'] if p['name'] not in give_names]+[dict(p,current_index=None,locked=False) for p in receive]
    theirs=[p for p in partner['players'] if p['name'] not in receive_names]+[dict(p,current_index=None,locked=False) for p in give]
    if len({p['id'] for p in after})!=len(after) or len({p['id'] for p in theirs})!=len(theirs):
        return result
    # Preserve existing captured excess/IR ambiguity; never create extra active slots.
    if len(after)>max(capacity,len(mine['players'])) or len(theirs)>max(capacity,len(partner['players'])):
        result['reason']='Receiving roster exceeds captured capacity; separate drop/IR resolution required. No implicit player drop.'
        return result
    if len(mine['players'])>capacity or len(partner['players'])>capacity:
        result['warnings'].append('Captured roster exceeds active-slot count; verify existing IR allocation. Trade does not increase that excess.')
    def run(team,players):
        key=(team['team_id'],tuple(sorted(p['id'] for p in players)),mode if team is mine else 'Neutral')
        if cache is not None and key in cache:
            return cache[key]
        value=optimize_roster(players,mode if team is mine else 'Neutral')
        if cache is not None:
            cache[key]=value
        return value
    post,other=run(mine,after),run(partner,theirs)
    result.update(after=post,counterparty_after=other)
    before=baselines[mine['team_id']]
    their_before=baselines[partner['team_id']]
    if any(not numeric(x['projection']) for x in (before,post,other,their_before)):
        values=[roster_values(p) for p in (mine['players'],after,partner['players'],theirs)]
        result['ros_assessment']=dict(complete=all(numeric(v['starter_ros']) for v in values),
            before=values[0],after=values[1],counterparty_before=values[2],counterparty_after=values[3],
            summary='ROS inputs retained separately; current-week legal lineup comparison is unavailable.')
        result['reason']='No fully projected legal lineup after trade; unavailable inputs are not zero.'
        return result
    delta=round(post['projection']-before['projection'],3)
    other_delta=round(other['projection']-their_before['projection'],3)
    start_ids={p['id'] for p in post['lineup']}
    starts=[p['name'] for p in receive if p['id'] in start_ids]
    before_ids={p['id'] for p in before['lineup']}
    slot_changes=[dict(slot=p['slot'],incoming=p['name']) for p in post['lineup'] if p['id'] not in before_ids]
    depth={pos:sum(p['position']==pos for p in after)-sum(p['position']==pos for p in mine['players']) for pos in SKILL}
    rv0,rv1,ov0,ov1=map(roster_values,(mine['players'],after,partner['players'],theirs))
    rosdelta=rv1['starter_ros']-rv0['starter_ros'] if numeric(rv0['starter_ros']) and numeric(rv1['starter_ros']) else None
    other_ros=ov1['starter_ros']-ov0['starter_ros'] if numeric(ov0['starter_ros']) and numeric(ov1['starter_ros']) else None
    bench_before=[p for p in mine['players'] if p['id'] not in before_ids and p['position'] in SKILL]
    bench_after=[p for p in after if p['id'] not in start_ids and p['position'] in SKILL]
    ros_complete=all(numeric(v['starter_ros']) and numeric(v['bench_assets']) and not v['missing_ros'] for v in (rv0,rv1,ov0,ov1))
    bench_loss=rv0['bench_assets']-rv1['bench_assets'] if ros_complete else None
    result.update(lineup_delta=delta,counterparty_delta=other_delta,acquired_starters=starts,slot_changes=slot_changes,
                  depth_change=depth,ros_lineup_delta=rosdelta,counterparty_ros_delta=other_ros,
                  bench_before=[p['name'] for p in bench_before],bench_after=[p['name'] for p in bench_after],
                  bench_asset_loss=round(bench_loss,3) if numeric(bench_loss) else None,
                  bye_context=[dict(player=p['name'],bye=p['bye']) for p in give+receive])
    result['verification']=verification(calls,give,receive,partner,mine['team_name'])
    other_bench_gain=ov1['bench_assets']-ov0['bench_assets'] if ros_complete else None
    result['counterparty_bench_asset_gain']=round(other_bench_gain,3) if ros_complete else None
    result['roster_equity_delta']=round(rosdelta-config['bench_weight']*bench_loss,3) if ros_complete else None
    result['counterparty_equity_delta']=round(other_ros+config['bench_weight']*other_bench_gain,3) if ros_complete else None
    result['ros_assessment']=dict(complete=ros_complete,basis='ROS positional ranking proxies; no weekly projections in ROS value',
        before=rv0,after=rv1,counterparty_before=ov0,counterparty_after=ov1,
        missing_players=sorted(set(n for v in (rv0,rv1,ov0,ov1) for n in v['missing_ros'])),
        summary=(f'ROS starter utility {rosdelta:+.2f}; depth-adjusted ROS {result["roster_equity_delta"]:+.2f}; '
                 f'counterparty ROS {other_ros:+.2f}, depth-adjusted {result["counterparty_equity_delta"]:+.2f}. Ranking proxies only.'
                 if ros_complete else 'Incomplete ROS rankings; long-term value unknown. Weekly projections are not substituted.'))
    missing_context=[dict(player=p['name'],missing=[label for label,value in (
        ('ROS projection',p.get('ros_projection_available')),('role',p.get('role')),('bye',p.get('bye'))) if not value])
        for p in give+receive]
    result['ros_assessment']['missing_context']=[p for p in missing_context if p['missing']]
    result['ros_assessment']['context_incomplete']=bool(result['ros_assessment']['missing_context'])
    if result['ros_assessment']['context_incomplete']:
        result['ros_assessment']['summary']+=' ROS projections/role/bye context incomplete; confidence reduced.'
    if not ros_complete:
        return grade_trade(result,config)
    av,bv=sum(p['asset_value'] for p in give),sum(p['asset_value'] for p in receive)
    ratio=av/bv
    result.update(asset_give=round(av,3),asset_receive=round(bv,3),asset_ratio=round(ratio,4),
                  protected_sent=[p['name'] for p in give if is_protected(p)])
    their_start_ids={p['id'] for p in other['lineup']}
    their_new=[p['name'] for p in give if p['id'] in their_start_ids]
    result['counterparty_fit']=f'Roster fit: incoming {", ".join(their_new) or "bench insurance"}; modeled weekly change {other_delta:+.2f}, ROS starter utility {other_ros:+.2f}. No willingness or trade-history inference.'
    result['roster_fit']=f'Acquired starters: {", ".join(starts) or "none"}; weekly lineup {delta:+.2f}, ROS starter utility {rosdelta:+.2f}; bench asset loss {bench_loss:+.2f}.'
    principal=max(receive,key=lambda p:(p['asset_value'],p['id']))
    result.update(principal_target=principal['name'],
                  effective_target_cost=round(av-(bv-principal['asset_value']),3),
                  acquired_inactive=any(p['excluded'] for p in receive))
    result['warnings'] += ['ROS ranking proxies omit unknown roles, injury return timing and some bye coverage; confidence is at most Medium.',
                           'Current-week lineup delta is a separate short-term comparison, not a replacement for ROS value.']
    return grade_trade(result,config)


def generate_candidates(mine, opponents, profiles, config, calls):
    skill=[p for p in mine['players'] if p['position'] in SKILL and not p['locked'] and not p.get('trade_locked') and numeric(p['asset_value'])]
    # Good depth is a trade asset, including protected starters only at an upgrade price.
    chips=sorted([p for p in skill if not is_protected(p)],key=lambda p:(-p['asset_value'],p['id']))[:config['max_offer_chips']]
    offers=[(p,) for p in skill]+list(itertools.combinations(chips,2))
    proposals={}
    before=quick_total(mine['players'])
    ros_before=roster_values(mine['players'])['starter_ros']
    for opponent in opponents:
        other_before=quick_total(opponent['players'])
        other_ros_before=roster_values(opponent['players'])['starter_ros']
        targets=sorted([p for p in opponent['players'] if p['position'] in SKILL and not p['excluded'] and not p.get('trade_locked') and numeric(p['asset_value'])],
                       key=lambda p:(-p['asset_value'],p['id']))[:config['max_targets_per_owner']]
        for target in targets:
            returns=[(target,)]
            # Two-for-two balances a position-for-position upgrade or different roster needs.
            returns += [(target,p) for p in opponent['players'] if p['position'] in SKILL and p['id']!=target['id'] and
                        numeric(p['asset_value']) and p['asset_value']<=target['asset_value'] and not p['excluded'] and not p.get('trade_locked')]
            for give in offers:
                for receive in returns:
                    if len(give)==1 and len(receive)==2:
                        continue
                    av,bv=sum(p['asset_value'] for p in give),sum(p['asset_value'] for p in receive)
                    if not config['minimum_asset_ratio']<=av/bv<=config['maximum_asset_ratio']:
                        continue
                    after=[p for p in mine['players'] if p not in give]+list(receive)
                    bound=quick_total(after)
                    if before is None or bound is None or bound-before < -config['maximum_weekly_sacrifice']:
                        continue
                    theirs=[p for p in opponent['players'] if p not in receive]+list(give)
                    other_bound=quick_total(theirs)
                    if other_before is None or other_bound is None or other_bound-other_before < -config['counterparty_weekly_loss_limit']-1:
                        continue
                    ros_after=roster_values(after)['starter_ros']
                    other_ros_after=roster_values(theirs)['starter_ros']
                    if (numeric(ros_before) and numeric(ros_after) and ros_after<ros_before) or (numeric(other_ros_before) and numeric(other_ros_after) and other_ros_after-other_ros_before < -config['counterparty_ros_loss_limit']):
                        continue
                    key=trade_key(opponent['team_id'],[p['name'] for p in give],[p['name'] for p in receive])
                    origin=dict(target=target['name'],screened_weekly_delta=round(bound-before,3),
                                screened_ros_delta=round(ros_after-ros_before,3) if numeric(ros_after) and numeric(ros_before) else None)
                    if key in proposals:
                        proposals[key]['generation_paths'].append(origin)
                    else:
                        proposals[key]=dict(team_id=opponent['team_id'],give=[p['name'] for p in give],receive=[p['name'] for p in receive],
                            source='Independent ROS roster-fit search',generation_paths=[origin],
                            explanation='All eligible singles and pairs of unprotected ROS chips are tested against bounded owner targets. The same outgoing package may fit multiple targets; it is not a fixed offer template.')
    for call in calls:
        if call.get('kind')=='analyzer':
            # Reevaluate previously checked exact packages even if new gates screen
            # them out, preserving an explainable rejection audit.
            opponent=next((t for t in opponents if t['team_name']==call['data'].get('team2')),None)
            if opponent:
                give=[n.strip() for n in call['arguments'].get('players_trade_away','').split(',') if n.strip()]
                receive=[n.strip() for n in call['arguments'].get('players_trade_for','').split(',') if n.strip()]
                key=trade_key(opponent['team_id'],give,receive)
                proposals.setdefault(key,dict(team_id=opponent['team_id'],give=give,receive=receive,
                    source='Previously verified package reevaluation',explanation='Exact analyzer package retained to explain survival or rejection under current gates.'))
            continue
        if call.get('kind')!='finder': continue
        for move in call['data'].get('moves',[]):
            give=[p['playerName'] for p in move.get('players_trade_away',[])]
            receive=[p['playerName'] for p in move.get('players_trade_for',[])]
            tid=str(move.get('theirTeamId'))
            key=trade_key(tid,give,receive)
            proposals.setdefault(key,dict(team_id=tid,give=give,receive=receive,source='FantasyPros Trade Finder idea; cached scores diagnostic only',finder=move))
    return list(proposals.values())


def analyze_trades(snapshot, waiver_analysis=None, streaming_analysis=None, evidence=None, mappings=None, config=None, now=None):
    config=dict(CONFIG if config is None else config)
    now=now or utc_now()
    lineup=analyze_lineup(snapshot)
    waivers=waiver_analysis if waiver_analysis is not None else analyze_waivers(snapshot)
    streaming=streaming_analysis if streaming_analysis is not None else analyze_streaming(snapshot,waiver_analysis=waivers,now=now)
    evidence=read_optional('trade_sources.json') if evidence is None else evidence
    mappings=read_optional('owner_mappings.json') if mappings is None else mappings
    if mappings.get('league_key')!=snapshot['league_key']:
        mappings={}
    week=streaming['current_week']['week']
    teams=construct_rosters(snapshot,lineup,waivers,week,load_history(),mappings,config,
                            season=streaming['current_week'].get('season'),
                            calendar=streaming['current_week'].get('date_evidence'),now=now)
    known_byes={p['team']:w['week'] for p in streaming.get('candidate_pool',[]) for w in p['weeks'] if w['bye']}
    for team in teams:
        for p in team['players']:
            if p['bye'] is None and p['team'] in known_byes:
                p['bye']=known_byes[p['team']]
                p['bye_source']='Explicit streaming schedule BYE for same NFL team'
    mine=next((t for t in teams if t['team_id']==TEAM_ID),None)
    baselines={t['team_id']:optimize_roster(t['players'],lineup['strategy'] if t is mine else 'Neutral') for t in teams}
    profiles=[profile(t,baselines[t['team_id']]) for t in teams]
    calls=trusted_calls(snapshot,evidence,now)
    complete=(len(teams)==12 and len({t['team_id'] for t in teams})==12 and mine is not None and
              {normalized(p['name']) for p in mine['players']}=={normalized(p['name']) for p in lineup['players']})
    all_names=[normalized(p['name']) for t in teams for p in t['players']]
    enabled=(complete and week is not None and fresh(snapshot['refreshed_at'],now)
             and all(numeric(b['projection']) for b in baselines.values())
             and len(all_names)==len(set(all_names)))
    rows=[]
    chips=[]
    capacity=sum(r['count'] for r in payload(snapshot,'settings').get('roster_positions',[]) if r['type'] not in ('IR','IR+')) or 15
    if mine and complete:
        cache={}
        opponents=[t for t in teams if t is not mine]
        proposals=generate_candidates(mine,opponents,profiles,config,calls)
        lookup={t['team_id']:t for t in opponents}
        for idea in proposals:
            if idea['team_id'] not in lookup:
                continue
            row=evaluate_trade(mine,lookup[idea['team_id']],idea['give'],idea['receive'],baselines,config,calls,capacity,lineup['strategy'],cache)
            row['candidate_source']=idea
            if not enabled:
                row.update(approved=False,confidence='Low')
                if row['tier'] in ('Strong target','Worth offering'):
                    row.update(tier='Research only',reason='Not send-ready: current roster/projection inputs are incomplete or stale.')
                row['warnings'].append('Stale/incomplete roster or unavailable week: recommendations disabled.')
            rows.append(row)
        for p in mine['players']:
            if p['position'] not in SKILL or p['asset_value'] is None or is_protected(p) or p['excluded'] or p.get('trade_locked'):
                continue
            removed=optimize_roster([r for r in mine['players'] if r['id']!=p['id']],lineup['strategy'])
            loss=baselines[TEAM_ID]['projection']-removed['projection'] if numeric(removed['projection']) and numeric(baselines[TEAM_ID]['projection']) else None
            ros_before=roster_values(mine['players'])
            ros_removed=roster_values([r for r in mine['players'] if r['id']!=p['id']])
            ros_cost=ros_before['starter_ros']-ros_removed['starter_ros'] if numeric(ros_before['starter_ros']) and numeric(ros_removed['starter_ros']) else None
            chips.append(dict(player=p['name'],position=p['position'],classification=p['trade_classification'],
                              asset_value=p['asset_value'],weekly_removal_cost=round(loss,3) if numeric(loss) else None,
                              ros_starter_removal_cost=round(ros_cost,3) if numeric(ros_cost) else None,
                              protected=False,waiver_classification=p['waiver_classification'],
                              chip_score=round(p['asset_value']/(1+max(0,ros_cost)),3) if numeric(ros_cost) else 0,
                              reason='Unprotected depth; trade only at a plausible ROS price. Weekly removal cost is separate.',
                              confidence='Low' if ros_cost is None else 'Medium'))
    rows.sort(key=lambda r:(not r['approved'],TIERS[r['tier']],-(r.get('roster_equity_delta') or 0),-(r.get('ros_lineup_delta') or 0),r['id']))
    groups={}
    for row in rows:
        if row['approved']:
            partner=next(t for t in teams if t['team_id']==row['team_id'])
            target=max((p for p in partner['players'] if p['name'] in row['receive']),key=lambda p:p['asset_value'])['name']
            groups.setdefault((row['team_id'],target),[]).append(row)
    targets=[]
    for (tid,target),offers in groups.items():
        levels=package_levels(offers,config)
        if not levels: continue
        fair=levels['fair']
        targets.append(dict(target=target,owner=fair['owner'],team_id=tid,packages=levels,proposal=fair,
                            approved=True,package_explanation=(
                                'Only one verified plausible package exists; all three levels intentionally coincide. No higher price is authorized.'
                                if len({p['id'] for p in levels.values()})==1 else
                                'Opening minimizes net ROS target cost; fair is closest to balanced assets; walk-away is the highest verified plausible net cost. Repeated levels mean no distinct tested package justifies another price.')))
    targets.sort(key=lambda t:(TIERS[t['proposal']['tier']],-t['proposal']['roster_equity_delta'],t['target']))
    targets=targets[:config['max_priority_targets']]
    for i,t in enumerate(targets,1):
        t['priority']=i
    research=[]
    seen=set()
    for row in rows:
        key=(row['team_id'],row.get('principal_target',row['target']))
        if row['tier']=='Research only' and key not in seen:
            research.append(row)
            seen.add(key)
    reuse={}
    for row in rows:
        key=', '.join(sorted(row['give']))
        reuse.setdefault(key,dict(give=sorted(row['give']),packages=0,approved=0,targets=set()))
        reuse[key]['packages']+=1
        reuse[key]['approved']+=int(row['approved'])
        reuse[key]['targets'].add(row['team_id']+': '+row.get('principal_target',row['target']))
    reuse=[dict(v,targets=sorted(v['targets'])) for v in reuse.values()]
    reuse.sort(key=lambda r:(-r['packages'],r['give']))
    return dict(version='0.5',generated_at=now,snapshot_id=snapshot['snapshot_id'],league_key=snapshot['league_key'],
                enabled=enabled,config=config,all_roster_inputs=teams,league_profiles=profiles,
                validated_lineup_reference=lineup['recommended_lineup'],validated_lineup_projection=lineup['recommended_projection'],
                comparison_week=week,comparison_baseline=baselines.get(TEAM_ID),streaming_calls=streaming['calls'],
                candidates_considered=rows,priority_targets=targets,verified_board=[r for r in rows if r['approved']],
                research_candidates=research[:config['max_priority_targets']],
                generation_audit=dict(method='Deterministic bounded ROS search; exact packages deduplicated. Repeated outgoing assets indicate separate evaluated targets, not independent endorsements.',
                                      package_reuse=reuse,limits={k:config[k] for k in ('max_offer_chips','max_targets_per_owner')},
                                      considered=len(rows)),
                trade_chips=sorted(chips,key=lambda c:(-c['chip_score'],c['player'])),
                core_players=[dict(player=p['name'],position=p['position'],reason='Protected by validated waiver/GM lineup; no normal-market sale.')
                              for p in (mine or {}).get('players',[]) if is_protected(p) and p['position'] in SKILL],
                source_evidence=evidence,input_sha256=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest(),
                warnings=['All 12 rosters required; opponent starting lineups are modeled, not captured Yahoo slot assignments.',
                          'Weekly comparison reruns the unchanged optimizer on common explicitly dated projections; the undated validated lineup remains a separate reference.',
                          'ROS values are heuristic asset proxies, not market prices or championship probabilities.',
                          'Unknown owner mappings remain unconfirmed; no inferred trade tendencies or acceptance probabilities.',
                          'K/DST are excluded from trade packages; pending streaming actions are not assumed executed.',
                          'Offers are alternatives, not a cumulative plan. Recalculate after any roster change.',
                          'No implicit free-agent replacement after consolidation. Missing projections/ROS/roles/byes remain limitations.'])


def save_trade_analysis(analysis, folder=None):
    folder=Path(folder) if folder else BASE/'data'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/('.trades-'+uuid.uuid4().hex+'.tmp')
    try:
        path.write_text(json.dumps(analysis,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(path,folder/'latest_trade_analysis.json')
    finally:
        path.unlink(missing_ok=True)


if __name__=='__main__':
    result=analyze_trades(load_snapshot())
    save_trade_analysis(result)
    print(json.dumps(dict(candidates=len(result['candidates_considered']),targets=[dict(target=t['target'],owner=t['owner'],
        offers={k:dict(give=v['give'],receive=v['receive'],delta=v['lineup_delta'],tier=v['tier']) for k,v in t['packages'].items()})
        for t in result['priority_targets']]),indent=2))
