"""Trade roster-impact, evidence, protection and dashboard regressions."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gm_schema
from gm_faab import load_history
from gm_snapshot import build_snapshot, persist, utc_now
from gm_trades import (CONFIG, analyze_trades, asset_value, evaluate_trade, optimize_roster,
                       owner_identity, package_levels, profile, roster_values,
                       save_trade_analysis, trusted_calls, verification, grade_trade,
                       construct_rosters, is_protected, generate_candidates)
from gm_lineup import SLOTS
from test_gm import fixture_bundle


def player(label,pos='WR',points=10,ros=30,**extra):
    p=dict(id=label,name=label,position=pos,points=points,ros_rank=ros,eligibility=[pos],
           current_index=None,locked=False,excluded=False,stats={},injury=None,bye=None,
           protected=False,gm_recommended_starter=False,scoring={},waiver_classification=None)
    p.update(extra)
    p['asset_value']=asset_value(p)
    return p


def roster(prefix):
    return [player(prefix+str(i),'RB' if pos=='FLEX' else pos,current_index=i) for i,pos in enumerate(SLOTS)]


class TradeImpactTests(unittest.TestCase):
    def setUp(self):
        self.mine=dict(team_id='6',owner='Pat',team_name='Pat alias',players=roster('p'))
        self.other=dict(team_id='1',owner='Matt',team_name='Matt alias',players=roster('o'))
        self.mine['players'].append(player('Chip',points=9,ros=18))
        self.other['players'][3]['points']=5
        self.other['players'][7]=player('Target','RB',13,15,current_index=7)

    def evaluate(self,give=('Chip',),receive=('Target',),capacity=15,calls=()):
        baselines={t['team_id']:optimize_roster(t['players']) for t in (self.mine,self.other)}
        return evaluate_trade(self.mine,self.other,list(give),list(receive),baselines,calls=calls,capacity=capacity)

    def test_starting_lineup_delta_uses_bench_replacement(self):
        r=self.evaluate()
        self.assertAlmostEqual(r['lineup_delta'],3)
        self.assertNotEqual(r['lineup_delta'],13-9)
        self.assertIn('Target',r['acquired_starters'])
        self.assertEqual(len(r['after']['lineup']),10)

    def test_counterparty_roster_fit(self):
        r=self.evaluate()
        # The incoming WR gains one point but the outgoing FLEX RB costs five.
        self.assertEqual(r['counterparty_delta'],-4)
        self.assertIn(r['tier'],('Reject','Only if discounted'))
        self.assertFalse(r['approved'])
        self.assertIn('Chip',r['counterparty_fit'])
        self.assertNotIn('wants',r['counterparty_fit'])

    def test_counterparty_can_benefit_with_real_bench_replacement(self):
        self.other['players'].append(player('Reserve RB','RB',13,15))
        r=self.evaluate()
        self.assertEqual(r['counterparty_delta'],1)
        self.assertEqual(r['lineup_delta'],3)

    def test_missing_bench_ros_cannot_be_priced_as_zero(self):
        self.mine['players'].append(player('Unknown bench',points=5,ros=None))
        r=self.evaluate()
        self.assertFalse(r['approved'])
        self.assertEqual(r['tier'],'Research only')

    def test_already_played_opponent_asset_rejected(self):
        self.other['players'][7]['trade_locked']=True
        self.assertEqual(self.evaluate()['tier'],'Reject')

    def test_bench_cost_is_separate_from_asset_and_starting_value(self):
        r=self.evaluate()
        self.assertIn('Chip',r['bench_before'])
        self.assertNotIn('Chip',r['bench_after'])
        self.assertIsInstance(r['bench_asset_loss'],float)
        self.assertAlmostEqual(r['roster_equity_delta'],r['ros_lineup_delta']-CONFIG['bench_weight']*r['bench_asset_loss'],places=2)

    def test_two_for_one_consolidation_has_no_invented_replacement(self):
        self.mine['players'].append(player('Extra',points=5,ros=60))
        r=self.evaluate(('Chip','Extra'))
        self.assertIsNotNone(r['after']['projection'])
        self.assertEqual(r['depth_change']['WR'],-2)
        self.assertEqual(r['depth_change']['RB'],1)
        self.assertEqual(len(r['bench_after']),len(r['bench_before'])-1)

    def test_over_capacity_counterparty_rejected(self):
        self.mine['players'].append(player('Extra',points=5,ros=60))
        r=self.evaluate(('Chip','Extra'),capacity=10)
        self.assertEqual(r['tier'],'Reject')
        self.assertIn('capacity',r['reason'])
        self.assertIsNone(r['after'])

    def test_fair_but_unnecessary(self):
        self.other['players'].append(player('Bench',points=8,ros=18))
        r=self.evaluate(receive=('Bench',))
        self.assertEqual(r['lineup_delta'],0)
        self.assertEqual(r['tier'],'Fair but unnecessary')
        self.assertFalse(r['approved'])

    def test_protected_starter_requires_more_than_tiny_gain(self):
        self.mine['players'][-1]['protected']=True
        self.other['players'][7]['points']=11
        r=self.evaluate()
        self.assertEqual(r['lineup_delta'],1)
        self.assertNotIn(r['tier'],('Strong target','Worth offering'))
        self.assertIn('Chip',r['protected_sent'])

    def test_invalid_lineup_missing_te_rejected(self):
        r=self.evaluate(give=('p6',))
        self.assertEqual(r['tier'],'Reject')
        self.assertIsNone(r['after']['projection'])

    def test_unknown_ros_does_not_become_a_bargain(self):
        self.mine['players'][-1].update(ros_rank=None,asset_value=None)
        r=self.evaluate()
        self.assertEqual(r['tier'],'Research only')
        self.assertIsNone(r['asset_ratio'])
        self.assertFalse(r['approved'])

    def test_locked_streaming_and_duplicate_assets_rejected(self):
        for give in (('p8',),('Chip','Chip')):
            self.assertEqual(self.evaluate(give=give)['tier'],'Reject')
        self.mine['players'][-1]['locked']=True
        self.assertEqual(self.evaluate()['tier'],'Reject')

    def test_proposals_do_not_mutate_current_rosters(self):
        before=copy.deepcopy((self.mine,self.other))
        self.evaluate()
        self.assertEqual((self.mine,self.other),before)


class IdentityAndEvidenceTests(unittest.TestCase):
    def test_canonical_owner_exact_alias_and_unknown(self):
        h=load_history()
        team=dict(team_id='3',team_name='Take It To The (Doll)house')
        self.assertEqual(owner_identity(team,h)['owner'],'Dave')
        self.assertEqual(owner_identity(dict(team,team_name='Unknown alias'),h)['owner'],'Unconfirmed owner (team 3)')
        self.assertEqual(owner_identity(dict(team_id='6',team_name='anything',is_user_team=True),h)['owner'],'Pat')

    def test_explicit_owner_mapping_must_match_current_alias(self):
        team=dict(team_id='1',team_name='New alias')
        mappings={'teams':{'1':dict(team_name='New alias',owner='George',source='User confirmation')}}
        self.assertEqual(owner_identity(team,load_history(),mappings)['owner'],'George')
        self.assertIn('Unconfirmed',owner_identity(dict(team,team_name='Changed'),load_history(),mappings)['owner'])

    def call(self,kind='analyzer'):
        return dict(kind=kind,arguments={'players_trade_away':'Send','players_trade_for':'Receive'},
                    data=dict(team2='Other',team1GetsDetails=[{'id':'Receive'}],
                              team2GetsDetails=[{'id':'Send'}],rest_of_season_power_ranking_increase='+1.2%'))

    def test_finder_exact_values_never_quoted_without_analyzer(self):
        give,receive=[player('Send')],[player('Receive')]
        other=dict(team_name='Other')
        self.assertFalse(verification([self.call('finder')],give,receive,other)['verified'])
        result=verification([self.call()],give,receive,other)
        self.assertTrue(result['verified'])
        self.assertEqual(result['ros_gain'],1.2)
        self.assertIn('Live analyzer',result['verdict'])

    def test_analyzer_wrong_players_or_partner_rejected(self):
        for change in ('players','team'):
            call=self.call()
            if change=='players':
                call['data']['team1GetsDetails']=[{'id':'Someone else'}]
            else:
                call['data']['team2']='Foreign team'
            self.assertFalse(verification([call],[player('Send')],[player('Receive')],{'team_name':'Other'})['verified'])

    def test_analyzer_nonfinite_gain_and_wrong_user_team_rejected(self):
        for value in ('nan%','inf%','-inf%'):
            call=self.call()
            call['data']['rest_of_season_power_ranking_increase']=value
            self.assertFalse(verification([call],[player('Send')],[player('Receive')],{'team_name':'Other'})['verified'])
        self.assertFalse(verification([self.call()],[player('Send')],[player('Receive')],
                                      {'team_name':'Other'},'Pat alias')['verified'])

    def test_stale_wrong_snapshot_or_league_verification_rejected(self):
        s=build_snapshot(fixture_bundle())
        evidence=dict(snapshot_id=s['snapshot_id'],league_key=s['league_key'],calls=[dict(kind='analyzer',
            arguments={'league_key':s['league_key']},fetched_at='2020-01-01T00:00:00Z',raw={})])
        self.assertEqual(trusted_calls(s,evidence,utc_now()),[])
        evidence['snapshot_id']='wrong'
        self.assertEqual(trusted_calls(s,evidence,utc_now()),[])

    def test_opening_fair_walkaway_order_and_negative_verdict_exclusion(self):
        rows=[dict(id=str(i),asset_give=cost,effective_target_cost=cost,asset_ratio=cost/100,lineup_delta=5-i,
                   roster_equity_delta=1,tier='Worth offering',approved=True,plausible=True,
                   verification={'verified':True}) for i,cost in enumerate((90,100,110))]
        levels=package_levels(rows)
        self.assertEqual([levels[k]['asset_give'] for k in ('opening','fair','walk_away')],[90,100,110])
        rows[-1].update(tier='Only if discounted',approved=False)
        self.assertEqual(package_levels(rows)['walk_away']['asset_give'],100)


class IntegrationTests(unittest.TestCase):
    def test_all_twelve_rosters_and_missing_inputs_audited(self):
        s=build_snapshot(fixture_bundle())
        a=analyze_trades(s,evidence={},mappings={})
        self.assertEqual(len(a['all_roster_inputs']),12)
        self.assertEqual(len(a['league_profiles']),12)
        self.assertFalse(any(r['approved'] for r in a['candidates_considered']))
        with tempfile.TemporaryDirectory() as folder:
            save_trade_analysis(a,folder)
            saved=json.loads((Path(folder)/'latest_trade_analysis.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['version'],'0.5')

    def test_dashboard_trade_sections_and_rostered_streaming_label(self):
        from streamlit.testing.v1 import AppTest
        s=build_snapshot(fixture_bundle())
        # Include both benchmarks even when projection/availability data is absent.
        s['sections']['my_roster']['data']['roster'].update(K=['Bench kicker'],DST=['Bench defense'])
        # Use the unmodified validated snapshot for the dashboard and inject analysis
        # only for the isolated benchmark presentation assertion below.
        s=build_snapshot(fixture_bundle())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            persist(s,root/'data')
            with patch.object(gm_schema,'BASE',root):
                app=AppTest.from_file(str(Path(__file__).parent/'dashboard.py')).run(timeout=45)
            self.assertEqual(len(app.exception),0)
            expected=('Verified Trade Board','Research Candidates','Trade Chips',
                      'Core / untouchable at normal market value','League Positional Needs')
            for title in expected:
                self.assertIn(title,[r.value for r in app.subheader])
            actual=[r.value for r in app.subheader if r.value in expected]
            self.assertEqual(actual,list(expected))
            self.assertTrue(any('Not send-ready' in c.value for c in app.caption))
            self.assertTrue((root/'data'/'latest_trade_analysis.json').exists())
        # The streaming formatter must label existing options as Rostered.
        source=(Path(__file__).parent/'dashboard.py').read_text(encoding='utf-8')
        self.assertIn("'Availability':'Rostered' if p['current']",source)


if __name__=='__main__':
    unittest.main()
