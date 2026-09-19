"""Offline streaming behavior and integration contracts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gm_schema
from gm_snapshot import build_snapshot, persist, utc_now
from gm_streaming import (CONFIG, RULES, analyze_streaming, candidate_pool, decide,
                         establish_week, route_waivers, save_streaming_analysis,
                         schedule_for, schedule_sources, score_candidate, weighted)
from gm_waivers import analyze_waivers
from test_gm import fixture_bundle


def option(label, pos='DST', projection=7, stars=(3,3,3,3), current=False, confirmed=True):
    weeks = [dict(week=2+i,opponent='OPP',bye=False,stars=star,
                  score=projection+(star-3) if i==0 else 6+1.5*(star-3)) for i,star in enumerate(stars)]
    return dict(name=label,position=pos,current=current,projection=projection,weeks=weeks,
                composite=weighted([w['score'] for w in weeks[:3]],CONFIG['weights']),
                availability=dict(confirmed=confirmed,status='Current roster' if current else 'Confirmed free agent'))


def schedule(week=2, stars=4, matchup='PIT at NE', state='in_season'):
    return dict(id='explicit',data=dict(current_week=week,season_state=state,
        sos_window=dict(week_start=week,week_end=week),
        **{'team schedule':{'IMPORTANT INFO':'high star rating and low rank is good',
            'schedules':[{'team':'PIT','Weekly Matchups':{'Week '+str(week):{
                'matchup':matchup,'strength_of_schedule_ratings':{'dst_stars_out_of_5':stars}}} }]}}))


class WeekAndScoringTests(unittest.TestCase):
    def test_explicit_current_week(self):
        from test_week import calendar_at
        result = establish_week([schedule()],now='2026-09-10T00:30:00Z',calendar=calendar_at())
        self.assertEqual(result['week'],1)
        self.assertEqual(result['sources'][0]['source'],'explicit')

    def test_missing_conflicting_and_offseason_week(self):
        for sources in ([],[schedule(2),schedule(3)],[schedule(0,state='offseason')]):
            self.assertIsNone(establish_week(sources,calendar={})['week'])

    def test_three_week_weights_and_missing(self):
        self.assertEqual(weighted([10,6,2],[.6,.25,.15]),7.8)
        self.assertIsNone(weighted([10,None,2],CONFIG['weights']))

    def test_opponent_and_weekly_sos(self):
        row = schedule_for('PIT','DST',2,[schedule()])
        self.assertEqual(row['opponent'],'NE')
        self.assertEqual(row['stars'],4)
        self.assertIsNone(schedule_for('PIT','DST',3,[schedule()])['stars'])

    def test_multiweek_sos_cannot_become_weekly_rating(self):
        s = schedule()
        s['data']['sos_window']['week_end']=5
        row = s['data']['team schedule']['schedules'][0]
        row.pop('Weekly Matchups')
        row['Strength of Schedule']={'dst_stars_out_of_5':5}
        self.assertIsNone(schedule_for('PIT','DST',2,[s])['stars'])

    def test_unknown_sos_scale_and_conflicting_schedule(self):
        s = schedule()
        s['data']['team schedule']['IMPORTANT INFO']='Undocumented scale'
        self.assertIsNone(schedule_for('PIT','DST',2,[s])['stars'])
        row = schedule_for('PIT','DST',2,[schedule(),schedule(matchup='PIT at CLE')])
        self.assertTrue(row['conflict'])

    def test_bye_including_string_response(self):
        s = schedule(matchup='BYE')
        self.assertTrue(schedule_for('PIT','DST',2,[s])['bye'])
        s['data']['team schedule']['schedules'][0]['Weekly Matchups']['Week 2']='BYE'
        self.assertTrue(schedule_for('PIT','DST',2,[s])['bye'])

    def test_source_rounding_is_not_a_conflict(self):
        row = schedule_for('PIT','DST',2,[schedule(stars=1.2),schedule(stars=1.18)])
        self.assertFalse(row['conflict'])
        self.assertEqual(row['stars'],1.2)


    def score_kicker(self, stats, week=2):
        s = {'sections':{'projection_weekly_K':{'status':'ok','data':{'week':week}},
                         'ranking_WEEKLY_K':{'status':'ok','data':{}}}}
        p = dict(name='K',position='K',team='PIT',current=True,
                 projection_source={'stats':stats},ecr_source={})
        return score_candidate(p,s,2,[],CONFIG,RULES)

    def test_league_distance_scoring_and_miss_penalties(self):
        stats = {**{'fg_'+b:0 for b in RULES['kicker_scoring']['field_goal_made']},
                 **{'fg_missed_'+b:0 for b in RULES['kicker_scoring']['field_goal_missed']},
                 'xpt':2,'xpt_missed':1,'fg_50_plus':1,'fg_40_49':1,'fg_missed_20_29':1,'points':999}
        p = self.score_kicker(stats)
        self.assertEqual(p['projection'],9)
        self.assertEqual(p['scoring']['method'],'Yahoo custom scoring')

    def test_kicker_fallback_never_invents_distance_or_misses(self):
        p = self.score_kicker(dict(points=8,fg=2,fga=2.2,xpt=2))
        self.assertEqual(p['projection'],8)
        self.assertEqual(p['scoring']['calculated_components'],{'PAT':2})
        self.assertIn('fg_50_plus',p['scoring']['unavailable_components'])
        self.assertIsNone(self.score_kicker(dict(fg=2))['projection'])

    def test_missing_projection_ecr_sos_and_mismatched_week(self):
        for p in (self.score_kicker({}),self.score_kicker({'points':10},week=3)):
            self.assertIsNone(p['projection'])
            self.assertIsNone(p['ecr'])
            self.assertIsNone(p['composite'])


class DecisionTests(unittest.TestCase):
    def call(self, candidate, current=None, roster=None, enabled=True):
        current = current or option('Current',pos=candidate['position'],current=True)
        return decide(candidate['position'],[current,candidate],roster or [],CONFIG,enabled)

    def test_strong_current_stream(self):
        call,_ = self.call(option('Upgrade',projection=10))
        self.assertEqual(call['recommendation'],'STREAM')
        self.assertEqual(call['drop'],'Current')
        self.assertEqual(call['this_week_edge'],3)

    def test_tiny_point_edge_holds(self):
        for edge in (.1,.2):
            call,_ = self.call(option('Tiny',projection=7+edge))
            self.assertEqual(call['recommendation'],'HOLD')
            self.assertFalse(call['transaction_required'])

    def test_matchup_quality_can_change_dst_decision(self):
        call,_ = self.call(option('Weak opponent',projection=7,stars=(5,5,5,5)))
        self.assertEqual(call['recommendation'],'STREAM')
        self.assertEqual(call['this_week_edge'],0)

    def test_future_cannot_override_bad_current_week(self):
        call,_ = self.call(option('Future',projection=3,stars=(3,5,5,5)))
        self.assertNotEqual(call['recommendation'],'STREAM')

    def test_confirmed_only(self):
        p = option('Unconfirmed',projection=15,confirmed=False)
        call,targets = self.call(p)
        self.assertEqual(call['recommendation'],'HOLD')
        self.assertEqual(targets,[])

    def test_early_stash_flag_and_bench_cost(self):
        p = option('Future',stars=(3,5,5,5))
        call,targets = self.call(p)
        self.assertEqual(call['recommendation'],'HOLD')
        self.assertEqual(targets[0]['label'],'Potential early stash')
        cheap = dict(name='Expendable',position='WR',classification='Replaceable',utility=0,current_index=None)
        call,_ = self.call(p,roster=[cheap])
        self.assertEqual(call['recommendation'],'STASH FOR NEXT WEEK')
        self.assertEqual(call['drop'],'Expendable')
        for update in ({'classification':'Upside stash'}, {'classification':'Useful depth'},
                       {'gm_recommended_starter':True},{'locked':True},{'utility':10},
                       {'weekly_comparable':False,'utility':0,'points':6.07}):
            call,_ = self.call(p,roster=[dict(cheap,**update)])
            self.assertEqual(call['recommendation'],'HOLD')

    def test_no_third_defense_or_conflicting_waiver_drop(self):
        p = option('Future',stars=(3,5,5,5))
        cheap = dict(name='Expendable',position='WR',classification='Replaceable',utility=0,current_index=None)
        for rows,reserved in (([option('A',current=True),option('B',current=True),p],()),
                              ([option('A',current=True),p],('Expendable',))):
            call,_ = decide('DST',rows,[cheap],CONFIG,reserved_drops=reserved)
            self.assertNotEqual(call['recommendation'],'STASH FOR NEXT WEEK')

    def test_never_carry_two_kickers(self):
        for projection in (7,10):
            call,targets = self.call(option('K upgrade',pos='K',projection=projection,stars=(3,5,5,5)))
            self.assertNotEqual(call['recommendation'],'STASH FOR NEXT WEEK')
            self.assertEqual(targets,[])
            if call['transaction_required']:
                self.assertEqual(call['drop'],'Current')

    def test_bye_option_cannot_be_added_and_current_bye_replaced(self):
        p = option('Bye',projection=15)
        p['weeks'][0].update(bye=True,score=0)
        self.assertEqual(self.call(p)[0]['recommendation'],'HOLD')
        current = option('Current',current=True)
        current['weeks'][0].update(bye=True,score=0)
        self.assertEqual(self.call(option('Active',projection=8),current=current)[0]['recommendation'],'STREAM')

    def test_missing_week_and_locked_current_disable_transactions(self):
        self.assertEqual(self.call(option('Great',projection=20),enabled=False)[0]['recommendation'],'NO ACTION')
        call,_ = self.call(option('Great',projection=20),roster=[dict(name='Current',locked=True)])
        self.assertFalse(call['transaction_required'])

    def test_next_week_bye_can_motivate_advance_target(self):
        current = option('Current',current=True)
        current['weeks'][1].update(bye=True,score=0,stars=None)
        _,targets = self.call(option('Future',stars=(3,5,5,5)),current=current)
        self.assertEqual(targets[0]['name'],'Future')


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = build_snapshot(fixture_bundle())

    def test_current_rostered_included_and_ownership_conflict(self):
        self.snapshot['sections']['my_roster']['data']['roster']['K']=['Own kicker']
        pool = candidate_pool(self.snapshot,{},utc_now())
        p = next(p for p in pool if p['name']=='Own kicker')
        self.assertTrue(p['current'])
        self.assertFalse(p['availability']['confirmed'])
        with patch('gm_streaming.ownership',return_value={'confirmed':True,'status':'Confirmed free agent'}):
            pool = candidate_pool(self.snapshot,{},utc_now())
        self.assertFalse(next(p for p in pool if p['name']=='Own kicker')['availability']['confirmed'])

    def test_missing_week_audit_and_save(self):
        for pos in ('DST','K'):
            self.snapshot['sections']['schedule_'+pos]['data']={}
        self.snapshot['sections']['matchup']['data']['week']=9
        result = analyze_streaming(self.snapshot,evidence={},sources={},calendar={})
        self.assertIsNone(result['current_week']['week'])
        self.assertFalse(any(c['transaction_required'] for c in result['calls']))
        with tempfile.TemporaryDirectory() as folder:
            save_streaming_analysis(result,folder)
            self.assertEqual(json.loads((Path(folder)/'latest_streaming_analysis.json').read_text())['version'],'0.4')

    def test_sidecar_wrong_snapshot_or_stale_rejected(self):
        call = dict(kind='schedule',fetched_at=utc_now(),raw=schedule()['data'])
        for sidecar in ({'snapshot_id':'wrong','calls':[call]},
                        {'snapshot_id':self.snapshot['snapshot_id'],'calls':[dict(call,fetched_at='2020-01-01T00:00:00Z')]}):
            sources = schedule_sources(self.snapshot,sidecar,utc_now())
            self.assertFalse(any(s['id'].startswith('streaming_sources:') for s in sources))

    def test_opponent_rostered_candidate_cannot_be_confirmed(self):
        self.snapshot['sections']['projection_weekly_DST']['status'] = 'ok'
        self.snapshot['sections']['projection_weekly_DST']['data'] = {
            'projections':[dict(name='Owned defense',team='PIT',stats={'points':20})]}
        self.snapshot['sections']['all_rosters']['data']['teams'].append({'roster':{'DST':['Owned defense']}})
        with patch('gm_streaming.ownership',return_value={'confirmed':True,'status':'Confirmed free agent'}):
            pool = candidate_pool(self.snapshot,{},utc_now())
        p = next(p for p in pool if p['name']=='Owned defense')
        self.assertFalse(p['availability']['confirmed'])

    def test_stale_snapshot_disables_streaming(self):
        self.snapshot['refreshed_at']='2020-01-01T00:00:00Z'
        result = analyze_streaming(self.snapshot,evidence={},sources={})
        self.assertFalse(result['enabled'])
        self.assertTrue(all(c['recommendation']=='NO ACTION' for c in result['calls']))

    def test_waiver_streaming_consistency_without_mutating_engine(self):
        original = dict(board=[dict(position=p,recommendation='CLAIM',add=p) for p in ('DST','K','WR')],
                        claim_plan=[dict(add=p,drop='Old '+p) for p in ('DST','K','WR')],overall='CLAIM')
        saved = copy.deepcopy(original)
        streaming = dict(calls=[dict(position='DST',recommendation='STREAM',add='DST')],
                         snapshot_id='test',candidate_pool=[dict(name='DST'),dict(name='K')])
        result = route_waivers(original,streaming)
        self.assertEqual(original,saved)
        self.assertEqual([r['recommendation'] for r in result['board']],['SEE STREAMING','SEE STREAMING','CLAIM'])
        self.assertEqual(result['claim_plan'],[dict(add='WR',drop='Old WR')])
        self.assertEqual(result['streaming_calls'],streaming['calls'])

    def test_dashboard_streaming_rendering(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            persist(self.snapshot,folder/'data')
            with patch.object(gm_schema,'BASE',folder):
                app = AppTest.from_file(str(Path(__file__).parent/'dashboard.py')).run(timeout=45)
            self.assertEqual(len(app.exception),0)
            for title in ('GM Streaming Call','D/ST Rankings','Kicker Rankings','Upcoming Targets'):
                self.assertIn(title,[s.value for s in app.subheader])
            self.assertIn('Streaming source data / diagnostics',[e.label for e in app.expander])
            self.assertTrue((folder/'data'/'latest_streaming_analysis.json').exists())
            self.assertTrue(any('See Streaming' in m.value for m in app.markdown))


if __name__=='__main__':
    unittest.main()
