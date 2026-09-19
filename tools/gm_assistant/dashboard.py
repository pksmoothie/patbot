"""Local snapshot-only GM dashboard. No refresh, Yahoo client, or execution actions."""
import sys
from pathlib import Path

# Streamlit executes this file directly; resolve only this independent tool's modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
from gm_schema import BASE, LEAGUE_KEY, POSITIONS
from gm_snapshot import load_snapshot, stamp, utc_now
from gm_lineup import analyze, save_analysis
from gm_waivers import analyze_waivers, save_waiver_analysis
from gm_streaming import analyze_streaming, save_streaming_analysis, route_waivers
from gm_trades import analyze_trades, save_trade_analysis
from gm_views import (payload, lineup_rows, roster_rows, matchup_metrics, obvious_warnings,
                      candidates, ranking_rows, projection_rows, schedule_rows, source_rows, injury_rows, opponent_rows)


def table(rows, empty='No rows returned.'):
    if rows:
        st.dataframe(rows, width='stretch', hide_index=True)
    else:
        st.info(empty)


def section_notice(snapshot, name):
    section = snapshot['sections'][name]
    if section['status'] not in ('ok', 'empty'):
        st.warning(f'{name}: {section["status"]}. ' + ' '.join(section['warnings']))
        return False
    if section['status'] == 'empty':
        st.info(f'{name}: FantasyPros explicitly returned an empty result.')
    return True


def number(value, probability=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 'Unavailable'
    if probability:
        return f'{value:.1%}' if 0 <= value <= 1 else 'Invalid source probability'
    return f'{value:.2f}'


def lineup_table(rows):
    table([{'Slot': r['slot'], 'Player': ('★ ' if r.get('changed') else '') + r['name'],
            'GM projected points*': r['points']} for r in rows])


def this_week(snapshot):
    metrics = matchup_metrics(snapshot)
    analysis = analyze(snapshot)
    try:
        save_analysis(analysis, BASE / 'data')
    except OSError as exc:
        st.warning(f'Analysis could not be saved: {exc}')
    st.subheader('GM Final Call')
    st.caption(f'Opponent: {metrics["opponent"] or "Unavailable"} · Matchup week: {metrics["week"] or "not supplied"}')
    a, b, c, d = st.columns(4)
    a.metric('GM projected score*', number(analysis['recommended_projection']))
    b.metric('Opponent projection (FantasyPros)', number(metrics['opponent_projection']))
    c.metric('Win probability (FantasyPros current lineup)', number(analysis['win_probability'], True))
    d.metric('Lineup changes', len(analysis['lineup_changes']))
    st.write(f'Current strategy: **{analysis["strategy"]}**')
    st.caption('*Uses FantasyPros player-point fallback where custom Yahoo scoring cannot be reconstructed. Win probability is source context, not a new GM forecast.')
    for change in analysis['lineup_changes']:
        st.write(f'**{change["action"]}** — {change["confidence"]} confidence')
    if not analysis['lineup_changes'] and analysis['recommended_lineup']:
        st.write('Keep the current starters.')
    limitations = list(dict.fromkeys(obvious_warnings(snapshot) + analysis['warnings']))
    if limitations:
        st.warning('Scoring and data limitations apply; review the details before changing your lineup.')
        with st.expander('Scoring and data limitation details'):
            for warning in limitations:
                st.write(warning)
    st.subheader('Recommended lineup')
    lineup_table(analysis['recommended_lineup'])
    st.caption('★ New starter compared with the current lineup.')
    st.subheader('Decisions')
    if not analysis['decisions']:
        st.caption('No close start/sit decisions with usable inputs.')
    for decision in analysis['decisions']:
        st.markdown(f'**{decision["action"]}** · {decision["confidence"]} confidence')
        rows = []
        for label in ('start', 'sit'):
            p = decision['inputs'][label]
            if p:
                rows.append({'Player': p['name'], 'Our projected points*': p['points'],
                    'FP positional ECR': p['ecr'] or '—', 'FP expert start %': p['expert_percent'],
                    'Opponent': p['opponent'] or '—', 'FP SOS (source scale)': p['sos'],
                    'Injury': p['injury'] or '—', 'Depth-chart role': p['role'] or '—'})
        table(rows)
        st.write(decision['explanation'])
        for label in ('start', 'sit'):
            p = decision['inputs'][label]
            if p and p.get('news'):
                st.caption(f'FantasyPros news for {p["name"]}: {p["news"]}')
        with st.expander('Decision inputs and scoring limitations'):
            st.json(decision['inputs'])
    st.subheader('FantasyPros comparison')
    table([{'Decision': d['action'], 'FantasyPros': d['fantasypros_recommendation']}
           for d in analysis['decisions']], 'No comparable start/sit decisions.')
    a, b = st.columns(2)
    with a:
        st.write('Current Yahoo/FantasyPros lineup')
        lineup_table(analysis['current_lineup'])
        st.caption('Current total: ' + number(analysis['current_projection']))
    with b:
        st.write('Highest projected-points lineup')
        lineup_table(analysis['highest_projected_lineup'])
        st.caption('Highest total: ' + number(analysis['alternative_lineup_projection']))
    with st.expander('FantasyPros recommended lineup'):
        table(analysis['fantasypros_recommendation'], 'FantasyPros recommendation unavailable.')
    with st.expander('Opponent roster'):
        table(opponent_rows(snapshot))
    with st.expander('Raw source data / diagnostics'):
        st.caption(f'{len(analysis["alternatives"])} legal alternatives enumerated. Audit: data/latest_analysis.json and data/analyses/.')
        st.json(analysis)
        st.json(payload(snapshot, 'start_sit'))


def waiver_board(snapshot, analysis=None, streaming=None):
    analysis = analysis if analysis is not None else analyze_waivers(snapshot)
    if streaming is not None:
        analysis = route_waivers(analysis, streaming)
    try:
        save_waiver_analysis(analysis, BASE / 'data')
    except OSError as exc:
        st.warning(f'Waiver analysis could not be saved: {exc}')
    st.subheader('GM Waiver Board')
    st.caption('Availability is confirmed as of the recorded ownership check. Verify current FAAB before manual Yahoo submission.')
    rows = [r for r in analysis['board'] if r['recommendation']!='SEE STREAMING']
    if streaming is not None:
        for call in streaming['calls']:
            st.write(f'{call["position"]}: See Streaming - {call["recommendation"]}; {call["add"] or call["current"] or "data unavailable"}')
    if analysis['overall'] == 'NO CLAIM':
        st.info('NO CLAIM — no confirmed candidate demonstrates a worthwhile roster upgrade.')
    table([{'Priority':r['priority'], 'Add':r['add'], 'Drop':r['drop'] or 'NO CLAIM',
        'Position':r['position'], 'GM tier':r['tier'] if r['recommendation']=='CLAIM' else 'NO CLAIM',
        'Conservative bid':r['faab']['conservative'], 'Recommended bid':r['faab']['recommended'],
        'Aggressive ceiling':r['faab']['aggressive_ceiling'], 'Confidence':r['confidence'],
        'Availability':r['availability']['status']} for r in rows])
    for row in rows:
        with st.expander(f'{row["add"]} — {row["strength"]}: {row["roster_need"]}'):
            st.write(f'Add: {row["add"]} / Drop: {row["drop"] or "NO CLAIM"}')
            st.write('FantasyPros: ' + ('Recommended add; suggested bid '+str(row['inputs'].get('fp_bid')) if row['fantasypros_recommendation'] else 'Candidate/reference only; no explicit add recommendation'))
            st.write(f'Weekly reference: {row["weekly_value"]}. ROS positional rank: {row["ros_value"] or "Unavailable"}.')
            st.write(f'Depth-chart role: {row["inputs"].get("role") or "Unavailable"}. Workload path: {row["inputs"].get("workload_path") or "Unconfirmed"}.')
            if row['availability'].get('evidence'):
                st.caption('Ownership checked: '+row['availability']['evidence']['fetched_at'])
            if row['opportunity_cost']:
                st.write('Drop opportunity cost: '+row['opportunity_cost']['reason'])
            st.write(f'GM utility gain: {row["utility_gain"]}; starter/FLEX utility gain: {row["starter_gain"]}. Utilities are not projected points.')
            st.write(' '.join(row['warnings']))
            if row['faab'].get('audit'):
                st.write('Bid calculation', row['faab']['audit'])
    st.subheader('Best candidates by position')
    for pos in POSITIONS:
        if streaming is not None and pos in ('DST','K'):
            st.write(f'{pos}: See Streaming')
            continue
        best = next((r for r in rows if r['position']==pos and r['recommendation']=='CLAIM' and r['availability'].get('confirmed') is True),None)
        if best:
            st.write(f'{pos}: {best["add"]} / drop {best["drop"]} — ${best["faab"]["recommended"]}')
        else:
            st.write(f'{pos}: NO CLAIM — no confirmed available candidate improves the roster')
    st.subheader("Pat's drop hierarchy")
    table([{'Order':i,'Player':p['name'],'Classification':p['classification'],'ROS rank':p.get('ros_rank'),
            'Injury':p.get('injury'),'Reason':p['drop_reason']} for i,p in enumerate((p for p in analysis['drop_hierarchy'] if p['classification']!='Core hold'),1)])
    with st.expander('Protected core holds'):
        table([{'Player':p['name'],'Classification':p['classification'],'Reason':p['drop_reason']}
               for p in analysis['drop_hierarchy'] if p['classification']=='Core hold'])
    st.subheader('Roster needs')
    table(analysis['roster_needs'])
    st.subheader('Manual Yahoo claim plan')
    table(analysis['claim_plan'],'NO CLAIM')
    st.caption('Shared-drop claims are ordered fallbacks. A successful earlier claim invalidates later claims using that player. Re-evaluate after each run; verify FAAB and player locks.')
    st.subheader('League FAAB behavior by owner')
    prior = analysis['historical_owner_prior']
    table(prior['owners'])
    market = prior['pressure']
    st.write('Historical pressure: '+market['label'])
    st.write('Historically aggressive possible competitors: '+', '.join(market['possible_aggressive_competitors']))
    st.write('Historical budget preservers: '+', '.join(market['historical_budget_preservers']))
    st.caption(market['caveat'])
    with st.expander('Methodology and limitations'):
        st.write('Spend: aggressive >= $75 average, conservative <= $35, moderate between; requires 3 seasons. Activity: high >=25 moves, low <15, otherwise moderate. Consistency: spend range <=$20 consistent, >$50 variable, otherwise mixed.')
        st.write('History adjusts unrounded bids by at most 5%; activity does not affect bidding. League-changing targets may justify substantial early spending. No fixed late-season reserve.')
        for warning in analysis['warnings']:
            st.write(warning)
    with st.expander('Raw FantasyPros waiver data / diagnostics'):
        st.json({p:payload(snapshot,'waiver_'+p) for p in POSITIONS})
        st.json(analysis)


def streaming_board(analysis):
    try:
        save_streaming_analysis(analysis, BASE / 'data')
    except OSError as exc:
        st.warning(f'Streaming audit could not be saved: {exc}')
    st.subheader('GM Streaming Call')
    week = analysis['current_week']['week']
    info = analysis['current_week']
    st.caption(f'NFL schedule week: {week or "Unavailable"} ({info.get("phase", "Unavailable")}). '
               f'Playing week: {info.get("playing_week") or "None"}; next scheduled week: {info.get("next_week") or "Unavailable"}. '
               f'Projection weeks: {info.get("projection_weeks", {})}.')
    if info.get('provider_conflict'):
        st.warning('Provider week conflicts with explicit NFL game dates. Streaming uses the dated NFL schedule.')
    if not analysis['enabled']:
        st.warning('Streaming analysis unavailable: refresh explicit week and complete roster inputs.')
    table([{'Position':r['position'],'Current':r['current'],'Recommendation':r['recommendation'],
            'Add':r['add'],'Drop':r['drop'],'This Week Edge':r['this_week_edge'],
            '3-Week Outlook':r['three_week_outlook'],'Confidence':r['confidence']} for r in analysis['calls']])
    st.caption('This Week Edge is projected points. Outlooks/composites are GM utilities, not future point forecasts. Yahoo moves remain manual.')
    for call in analysis['calls']:
        st.write(f'{call["position"]}: {call["recommendation"]} - {call["reason"]}')
    for pos, title in [('DST','D/ST Rankings'),('K','Kicker Rankings')]:
        st.subheader(title)
        rows = [p for p in analysis['candidate_pool'] if p['position']==pos and
                (p['current'] or p['availability'].get('confirmed') is True)]
        table([{'Rank':i,'Defense' if pos=='DST' else 'Kicker':p['name'],
                'Availability':'Rostered' if p['current'] else p['availability']['status'],'Current opponent':p['weeks'][0]['opponent'],
                'Current-week projection':p['projection'],'Weekly ECR (context if week unknown)':p['ecr'],
                'Matchup stars (5 easiest)':p['weeks'][0]['stars'],
                'Current GM score':p['weeks'][0]['score'],
                'Next-week opponent':p['weeks'][1]['opponent'],'Week +1 outlook':p['weeks'][1]['score'],
                'Week+2 opponent':p['weeks'][2]['opponent'],'Week +2 outlook':p['weeks'][2]['score'],
                'Week+3 opponent':p['weeks'][3]['opponent'],'Week +3 outlook':p['weeks'][3]['score'],
                '3-week composite':p['composite'],'GM action':p['gm_action']} for i,p in enumerate(rows,1)])
        if pos=='K':
            st.caption('League FG scoring: 3/3/3/4/5 by distance; missed FG under 50 yards -1, missed PAT -1. Missing distance/miss projections require the FantasyPros whole-player fallback.')
    st.subheader('Upcoming Targets')
    table([{'Defense':r['name'],'Flag':r['label'],'Next opponent':r['next_opponent'],
            'Following opponent':r['following_opponent'],'Future outlook edge':r['future_edge'],
            'Reason':r['reason']} for r in analysis['upcoming_targets']], 'No confirmed defense clears the early-stash thresholds.')
    with st.expander('Streaming source data / diagnostics'):
        for warning in analysis['warnings']:
            st.write(warning)
        st.json(analysis)


@st.cache_data(show_spinner=False)
def cached_trade_analysis(snapshot, waivers, streaming, evidence, mappings, freshness_hour):
    return analyze_trades(snapshot,waiver_analysis=waivers,streaming_analysis=streaming,
                          evidence=evidence,mappings=mappings)


def trade_board(snapshot, waivers, streaming):
    from gm_streaming import read_optional
    analysis = cached_trade_analysis(snapshot,waivers,streaming,read_optional('trade_sources.json'),
                                     read_optional('owner_mappings.json'),utc_now()[:13])
    try:
        save_trade_analysis(analysis, BASE / 'data')
    except OSError as exc:
        st.warning(f'Trade audit could not be saved: {exc}')
    st.subheader('Verified Trade Board')
    st.caption(f'Common projection basis: FantasyPros Week {analysis["comparison_week"] or "unavailable"}. '
               f'Validated lineup reference: {number(analysis["validated_lineup_projection"])}. '
               f'Comparable before-trade lineup: {number((analysis["comparison_baseline"] or {}).get("projection"))}.')
    if not analysis['enabled']:
        st.warning('Trade recommendations unavailable: all 12 complete roster inputs, a current week and fresh snapshot are required.')
    st.caption('Owner identity and roster fit only; no inferred trade behavior. Unverified packages are research ideas, not approved offers.')
    targets = analysis['priority_targets']
    table([{'Target':r.get('principal_target',r['target']),'Owner':r['owner'],
            'Pat gives':', '.join(r['give']),'Pat receives':', '.join(r['receive']),
            'Current-week lineup delta':r['lineup_delta'],'ROS roster assessment':r['ros_assessment']['summary'],
            'FantasyPros verification':r['verification']['verdict'],
            'GM tier':r['tier'],'Confidence':r['confidence']} for r in analysis['verified_board']],
          'No send-ready offer passes ROS, price, counterparty-fit and exact verification gates.')
    for t in targets:
        r=t['proposal']
        with st.expander(f'{t["target"]} - {t["owner"]}'):
            st.write('Why Pat benefits: '+r.get('roster_fit',r['reason']))
            st.write('Why the counterparty may consider it: '+r.get('counterparty_fit','Fit unavailable'))
            st.caption('Current team alias: '+r['team_name'])
            st.write(r['reason'])
            left,right=st.columns(2)
            with left:
                st.write('Before: common-week GM lineup')
                lineup_table(r['before']['lineup'])
            with right:
                st.write('After: hypothetical GM lineup')
                lineup_table(r['after']['lineup'])
            table([{'Asset value given':r['asset_give'],'Asset value received':r['asset_receive'],
                    'ROS starter utility delta':r.get('ros_lineup_delta'),'Bench asset loss':r.get('bench_asset_loss')}])
            st.write('Depth change',r.get('depth_change'))
            st.write('Bench before',r.get('bench_before'))
            st.write('Bench after',r.get('bench_after'))
            st.write(t['package_explanation'])
            table([{'Offer':label.replace('_',' ').title(),'Give':', '.join(p['give']),
                    'Receive':', '.join(p['receive']),'Net ROS target cost':p['effective_target_cost'],
                    'Lineup delta':p['lineup_delta'],'Approval':'GM approved' if p['approved'] else 'Awaiting verification / not approved',
                    'FantasyPros verdict':p['verification']['verdict']} for label,p in t['packages'].items()])
            for warning in r['warnings']:
                st.caption(warning)
    st.subheader('Research Candidates')
    st.caption('Not send-ready. These packages are research candidates, not offers to submit.')
    table([{'Target':r.get('principal_target',r['target']),'Owner':r['owner'],
            'Pat gives':', '.join(r['give']),'Pat receives':', '.join(r['receive']),
            'Current-week lineup delta':r['lineup_delta'],'ROS roster assessment':r['ros_assessment']['summary'],
            'FantasyPros verification':r['verification']['verdict'],'GM tier':'Research only',
            'Confidence':r['confidence'],'Why not send-ready':r['reason']}
           for r in analysis['research_candidates']], 'No additional research candidate clears the local screening gates.')
    st.subheader('Trade Chips')
    table(analysis['trade_chips'])
    st.subheader('Core / untouchable at normal market value')
    table(analysis['core_players'])
    st.subheader('League Positional Needs')
    table([{'Owner':p['owner'],'Current team alias':p['team_name'],
            'Strengths':', '.join(p['strengths']) or 'Unconfirmed',
            'Roster fit suggests need':', '.join(p['weaknesses']) or 'No clear positional shortfall',
            'Potential surplus':', '.join(p['surplus']) or 'None established'} for p in analysis['league_profiles']])
    with st.expander('Raw FantasyPros trade data / diagnostics'):
        st.write(analysis['generation_audit']['method'])
        st.json(analysis['generation_audit'])
        for warning in analysis['warnings']:
            st.write(warning)
        st.json(analysis)


def main():
    st.set_page_config(page_title='Fantasy Football GM v0.5', layout='wide')
    st.title('Fantasy Football GM')
    st.caption('v0.5 · Weekly lineup and waiver analysis · Team 6 · Yahoo actions are manual')
    st.sidebar.subheader('Snapshot controls')
    st.sidebar.button('Reload local snapshot')  # Rerun only; no network or MCP invocation.
    st.sidebar.code(LEAGUE_KEY, language=None)
    st.sidebar.caption('Refresh separately with REFRESH_GM.bat, then reload here.')
    path = BASE / 'data' / 'latest_snapshot.json'
    try:
        snapshot = load_snapshot(path)
    except FileNotFoundError:
        st.info('No snapshot yet. Run REFRESH_GM.bat or use the one-prompt Codex workflow in README.md.')
        for tab in st.tabs(['This Week', 'My Roster', 'Waivers', 'Streaming', 'Trades', 'League']):
            with tab:
                st.caption('Awaiting a validated FantasyPros refresh.')
        return
    except (ValueError, OSError, TypeError, KeyError) as exc:
        st.error(f'Snapshot rejected: {exc}. Refresh through Codex; unvalidated data will not be displayed.')
        return
    collected = [stamp(source['fetched_at']) for source in snapshot['sources']]
    oldest, newest = min(collected), max(collected)
    age = (stamp(utc_now()) - oldest).total_seconds() / 3600
    st.caption(f'Data collected {oldest.isoformat()} to {newest.isoformat()} · oldest source {age:.1f} hours ago')
    st.caption(f'Snapshot published {snapshot["refreshed_at"]} · {snapshot["refresh_status"]}')
    if age > 24:
        st.warning('This snapshot is over 24 hours old. Refresh before acting in Yahoo.')
    if snapshot['refresh_status'] == 'partial':
        st.caption('Some sections are unavailable. See League → source coverage for details.')
    week, roster, waivers, streaming, trades, league = st.tabs(
        ['This Week', 'My Roster', 'Waivers', 'Streaming', 'Trades', 'League'])
    with week:
        this_week(snapshot)
    with roster:
        st.subheader('Full roster')
        details = lineup_rows(snapshot) + lineup_rows(snapshot, 'bench') + lineup_rows(snapshot, 'ir')
        table(roster_rows(payload(snapshot, 'my_roster').get('roster'), details))
        st.subheader('Current bench')
        table(lineup_rows(snapshot, 'bench'), 'Bench data unavailable or empty; consult raw lineup source.')
        st.caption('Unreported slots and statuses stay blank. IR is not inferred from an injury designation.')
        st.subheader('Injuries and recent news')
        if section_notice(snapshot, 'injuries'):
            table(injury_rows(snapshot), 'No structured injury statuses returned; this is not a declaration that every player is healthy.')
            for item in payload(snapshot, 'injuries').get('player_news', []):
                st.write(item.get('player_name', 'Player not supplied'))
                st.write(item.get('news', 'News text unavailable'))
            with st.expander('Raw injury/news response'):
                st.json(payload(snapshot, 'injuries'))
        st.subheader('Weekly / ROS reference data')
        pos = st.selectbox('Position', POSITIONS, key='roster_pos')
        timeline = st.radio('Timeframe', ['WEEKLY', 'ROS'], horizontal=True)
        st.caption('Global FantasyPros reference data, not league availability. Up to 100 rows requested per position; league-specific scoring can differ.')
        if section_notice(snapshot, f'ranking_{timeline}_{pos}'):
            table(ranking_rows(snapshot, timeline, pos))
        if section_notice(snapshot, f'projection_{timeline.lower()}_{pos}'):
            table(projection_rows(snapshot, timeline.lower(), pos))
    waiver_analysis = analyze_waivers(snapshot)
    streaming_analysis = analyze_streaming(snapshot, waiver_analysis=waiver_analysis)
    with waivers:
        waiver_board(snapshot, waiver_analysis, streaming_analysis)
    with streaming:
        streaming_board(streaming_analysis)
    with trades:
        trade_board(snapshot, waiver_analysis, streaming_analysis)
    with league:
        settings = payload(snapshot, 'settings')
        st.subheader('League settings')
        st.write(f'Teams: {settings.get("n_teams", "unavailable")} · Scoring: {settings.get("scoring", "unavailable")}')
        with st.expander('Full scoring and roster settings'):
            st.json(settings)
        st.subheader('Power rankings / playoff odds')
        if section_notice(snapshot, 'league_analysis'):
            analysis = payload(snapshot, 'league_analysis')
            table(analysis.get('league_ranks', []), 'League power rankings not supplied.')
            if 'playoff_odds' in analysis:
                st.write('Playoff odds (FantasyPros)')
                st.json(analysis['playoff_odds'])
            else:
                st.caption('A playoff_odds field was not supplied. Any other provider fields remain in the full analysis below.')
            with st.expander('Full league analysis, including odds when supplied'):
                st.json(analysis)
        st.subheader('All league rosters')
        teams = payload(snapshot, 'all_rosters').get('teams', [])
        for team in teams:
            with st.expander(f'{team.get("team_name")} · team {team.get("team_id")}'):
                table(roster_rows(team.get('roster')))
        st.subheader('Source coverage')
        table(source_rows(snapshot))
        with st.expander('All refresh warnings'):
            for warning in snapshot['warnings']:
                st.write(warning)
        with st.expander('Auditable raw FantasyPros results'):
            st.json(snapshot['sources'])


if __name__ == '__main__':
    main()
