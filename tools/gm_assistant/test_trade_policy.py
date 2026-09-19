"""ROS-first approval, protection, generation and price-level regressions."""
import copy
import unittest
from unittest.mock import patch

from gm_trades import (CONFIG, construct_rosters, grade_trade, is_protected,
                       package_levels, roster_values, generate_candidates, analyze_trades)
from gm_faab import load_history
from gm_snapshot import build_snapshot
from test_gm import fixture_bundle
from test_trades import player, roster


def plausible(**updates):
    row=dict(id='offer',owner='Matt',ros_assessment={'complete':True},asset_ratio=1,
             asset_give=100,effective_target_cost=100,ros_lineup_delta=1,roster_equity_delta=.8,
             counterparty_ros_delta=.5,counterparty_equity_delta=.4,counterparty_delta=-.5,
             lineup_delta=.1,protected_sent=[],verification=dict(verified=True,ros_gain=1,counterparty_ros_gain=.2))
    row.update(updates)
    return row


class ApprovalPolicyTests(unittest.TestCase):
    def test_provisional_cannot_be_worth_offering(self):
        r=grade_trade(plausible(verification={'verified':False}))
        self.assertEqual(r['tier'],'Research only')
        self.assertFalse(r['approved'])
        self.assertNotIn('Worth offering',r['local_tier'])

    def test_lopsided_opening_rejected_despite_positive_verification(self):
        r=grade_trade(plausible(asset_ratio=.8038))
        self.assertEqual(r['tier'],'Reject')
        self.assertFalse(r['approved'])
        self.assertIn('Lopsided',r['reason'])

    def test_counterparty_requires_concrete_roster_fit(self):
        for changes in ({'counterparty_equity_delta':-.01},
                        {'counterparty_ros_delta':-.5},
                        {'counterparty_delta':-3},
                        {'counterparty_delta':0,'counterparty_ros_delta':0}):
            with self.subTest(changes=changes):
                r=grade_trade(plausible(**changes))
                self.assertEqual(r['tier'],'Reject')
                self.assertFalse(r['approved'])

    def test_counterparty_analyzer_loss_cannot_be_ignored(self):
        r=grade_trade(plausible(verification=dict(verified=True,ros_gain=.7,counterparty_ros_gain=-1.8)))
        self.assertEqual(r['tier'],'Reject')

    def test_ros_prioritized_over_tiny_weekly_gain(self):
        positive=grade_trade(plausible(lineup_delta=.1))
        self.assertTrue(positive['approved'])
        negative=grade_trade(plausible(lineup_delta=10,roster_equity_delta=-.1))
        self.assertFalse(negative['approved'])
        self.assertIn('ROS roster value declines',negative['reason'])
        trivial=grade_trade(plausible(lineup_delta=.2,ros_lineup_delta=.1,roster_equity_delta=.1))
        self.assertEqual(trivial['tier'],'Fair but unnecessary')

    def test_incomplete_ros_keeps_low_confidence_and_no_weekly_substitution(self):
        r=grade_trade(plausible(ros_assessment={'complete':False},ros_lineup_delta=None,
                                roster_equity_delta=None,lineup_delta=30))
        self.assertEqual((r['tier'],r['confidence']),('Research only','Low'))
        self.assertIsNone(r['roster_equity_delta'])
        self.assertFalse(r['approved'])

    def test_unknown_owner_is_not_guessed_from_analyzer(self):
        r=grade_trade(plausible(owner='Unconfirmed owner (team 1)'))
        self.assertEqual(r['owner'],'Unconfirmed owner (team 1)')
        self.assertEqual(r['tier'],'Research only')
        self.assertFalse(r['approved'])

    def test_ranking_based_verified_offer_discloses_incomplete_ros_context(self):
        r=grade_trade(plausible(ros_assessment={'complete':True,'context_incomplete':True}))
        self.assertTrue(r['approved'])
        self.assertEqual(r['confidence'],'Low')

    def test_core_starter_requires_material_ros_upgrade(self):
        r=grade_trade(plausible(protected_sent=['GM starter'],lineup_delta=5))
        self.assertFalse(r['approved'])
        self.assertIn('Protected GM starter',r['reason'])

    def test_opening_fair_walkaway_all_plausible_and_verified(self):
        rows=[grade_trade(plausible(id=str(cost),asset_give=cost,effective_target_cost=cost,asset_ratio=cost/100))
              for cost in (80,90,100,110,130)]
        rows.append(grade_trade(plausible(id='unverified',effective_target_cost=85,verification={'verified':False})))
        levels=package_levels(rows)
        self.assertEqual([levels[k]['asset_give'] for k in ('opening','fair','walk_away')],[90,100,110])
        self.assertTrue(all(r['approved'] and r['plausible'] and r['verification']['verified'] for r in levels.values()))

    def test_single_verified_price_is_not_invented_into_three_different_offers(self):
        row=grade_trade(plausible())
        levels=package_levels([row])
        self.assertEqual({v['id'] for v in levels.values()},{row['id']})

    def test_net_target_cost_accounts_for_secondary_assets_received(self):
        first=grade_trade(plausible(id='larger',asset_give=110,effective_target_cost=70,asset_ratio=1))
        second=grade_trade(plausible(id='smaller',asset_give=100,effective_target_cost=90,asset_ratio=1))
        self.assertEqual(package_levels([first,second])['opening']['id'],'larger')


class ProtectionAndGenerationTests(unittest.TestCase):
    def test_gm_recommended_starter_protected_even_with_missing_waiver_flag(self):
        s=build_snapshot(fixture_bundle())
        lineup=dict(players=[],recommended_lineup=[{'name':'QB 6'}])
        waivers={'drop_hierarchy':[dict(name='QB 6',classification='Useful depth',gm_recommended_starter=False)]}
        teams=construct_rosters(s,lineup,waivers,1,load_history(),{},CONFIG,season=2026)
        p=next(t for t in teams if t['team_id']=='6')['players'][0]
        self.assertTrue(p['protected'])
        self.assertTrue(p['gm_recommended_starter'])
        self.assertEqual(p['trade_classification'],'Core / protected starter')
        self.assertTrue(is_protected({'protected':False,'gm_recommended_starter':True}))

    def test_ros_values_independent_of_weekly_lineup_and_points(self):
        rows=roster('p')+[player('Bench','WR',2,12)]
        before=roster_values(rows)
        for i,p in enumerate(rows):
            p.update(points=1000-i*20,current_index=None,locked=True)
        self.assertEqual(roster_values(rows),before)

    def test_missing_ros_asset_does_not_get_zero_bench_value(self):
        rows=roster('p')+[player('Unknown','WR',30,None)]
        result=roster_values(rows)
        self.assertIsNone(result['bench_assets'])
        self.assertIsNone(result['asset_total'])
        self.assertEqual(result['missing_ros'],['Unknown'])

    def test_repeated_package_generation_is_explainable_and_deduplicated(self):
        mine={'players':[player('Chip A'),player('Chip B')]}
        other={'team_id':'1','team_name':'Other','players':[player('Target A'),player('Target B')]}
        with patch('gm_trades.quick_total',return_value=100),patch('gm_trades.roster_values',return_value={'starter_ros':100}):
            rows=generate_candidates(mine,[other],[],CONFIG,[])
        self.assertEqual(len(rows),len({r['team_id']+'|'+','.join(sorted(r['give']))+'|'+','.join(sorted(r['receive'])) for r in rows}))
        paired=next(r for r in rows if len(r['give'])==2 and len(r['receive'])==2)
        self.assertEqual(len(paired['generation_paths']),2)
        self.assertIn('not a fixed offer template',paired['explanation'])
        self.assertTrue(all('screened_ros_delta' in p for p in paired['generation_paths']))

    def test_core_players_never_mixed_into_trade_chips(self):
        s=build_snapshot(fixture_bundle())
        teams=[dict(team_id=str(i),team_name='Team '+str(i),owner='Owner',identity_source='fixture',players=roster(str(i))) for i in range(1,13)]
        mine=teams[5]
        for p in mine['players']:
            p.update(protected=True,trade_classification='Core / protected starter',team=None)
        mine['players'].append(dict(player('Depth'),trade_classification='Useful depth',team=None))
        for t in teams:
            for p in t['players']: p.setdefault('team',None)
        lineup=dict(players=mine['players'],strategy='Neutral',recommended_lineup=[],recommended_projection=100)
        stream=dict(current_week={'week':1,'season':2026},calls=[])
        with patch('gm_trades.construct_rosters',return_value=teams),patch('gm_trades.analyze_lineup',return_value=lineup),patch('gm_trades.generate_candidates',return_value=[]):
            result=analyze_trades(s,waiver_analysis={'drop_hierarchy':[]},streaming_analysis=stream,evidence={},mappings={})
        self.assertEqual([p['player'] for p in result['trade_chips']],['Depth'])
        self.assertTrue(result['core_players'])


if __name__=='__main__':
    unittest.main()
