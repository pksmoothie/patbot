"""Behavioral regressions for owner identity, opportunity costs, bids and UI."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gm_schema
from gm_faab import load_history, owner_priors, canonical_owner, pressure
from gm_waivers import (ownership, evaluate, scarcity, drop_hierarchy, bid_range,
    claim_plan, analyze_waivers, save_waiver_analysis, collect_candidates, enrich)
from gm_waivers import confirmed_budget, apply_drop_preference
from gm_snapshot import build_snapshot, persist, utc_now
from test_gm import fixture_bundle


def player(label, pos='WR', ros=60, points=6, **extra):
    return dict(name=label,position=pos,ros_rank=ros,points=points,
        weekly_comparable=False,current_index=None,locked=False,injury=None,role=None,
        availability={'confirmed':True,'status':'Confirmed free agent'},
        **extra)


def bid(**kwargs):
    args = dict(tier='Strong roster upgrade',gain=5,starter_gain=3,scarce=0,
                fp_bid=None,prior_fraction=0,remaining=100)
    args.update(kwargs)
    return bid_range(**args)


class OwnerTests(unittest.TestCase):
    def setUp(self):
        self.history = load_history()
        self.priors = {p['owner']:p for p in owner_priors(self.history)}

    def test_canonical_alias_mapping(self):
        self.assertEqual(canonical_owner(self.history,'Pancake Drawers',2024),'Lou')
        self.assertEqual(canonical_owner(self.history,'The Coin',2025),'Paul')
        self.assertEqual(canonical_owner(self.history,'Pat',2023),'Pat')
        with self.assertRaises(ValueError):
            canonical_owner(self.history,'Pancake Drawers',2023)

    def test_aggregation_across_changing_names(self):
        owners = [canonical_owner(self.history,n,y) for n,y in
                  [('Reverse Jinx',2023),('Bandit',2024),('Remember The Face',2025)]]
        self.assertEqual(set(owners),{'George'})
        self.assertEqual(self.priors['George']['median_spent'],92)
        self.assertEqual(self.priors['George']['seasons'],3)

    def test_all_three_year_totals(self):
        expected = {'Pat':(95,100,92),'Matt':(100,36,40),'Lou':(40,100,45),
          'Dave':(41,20,27),'Sarlo':(98,83,49),'Faherty':(72,40,51),
          'Potter':(100,79,100),'George':(100,70,92),'James':(2,66,35),
          'Frank':(0,12,0),'Vinnie':(5,14,80),'Paul':(83,37,53)}
        self.assertEqual(len(self.priors),12)
        for owner,spends in expected.items():
            p = self.priors[owner]
            self.assertEqual(p['average_spent'],round(sum(spends)/3,2))
            self.assertEqual(p['minimum_spent'],min(spends))
            self.assertEqual(p['maximum_spent'],max(spends))
            self.assertEqual(p['median_spent'],sorted(spends)[1])
        self.assertEqual(self.priors['Pat']['average_moves'],36.67)
        self.assertEqual(self.priors['Pat']['consistency'],'Consistent')

    def test_duplicate_histories_and_seasons_rejected(self):
        for duplicate in ('owner','season','alias'):
            h = copy.deepcopy(self.history)
            if duplicate=='owner':
                h['owners'].append(h['owners'][0])
            elif duplicate=='season':
                h['owners'][0]['seasons'].append(h['owners'][0]['seasons'][0])
            else:
                h['owners'][0]['seasons'][0]['aliases']=['TheMateoFootballTeam']
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'history.json'
                path.write_text(json.dumps(h))
                with self.assertRaises(ValueError):
                    load_history(path)

    def test_spending_and_activity_are_distinct(self):
        self.assertEqual(self.priors['Dave']['activity'],'High activity')
        self.assertEqual(self.priors['Dave']['spending'],'Conservative spender')
        self.assertEqual(self.priors['Potter']['activity'],'High activity')
        self.assertEqual(self.priors['Potter']['spending'],'Aggressive spender')
        altered = copy.deepcopy(self.history)
        for o in altered['owners']:
            for s in o['seasons']:
                s['moves'] = 999
        self.assertEqual(pressure(owner_priors(altered))['fraction'],pressure(list(self.priors.values()))['fraction'])

    def test_insufficient_history_is_not_aggressive(self):
        self.history['owners'][0]['seasons'] = self.history['owners'][0]['seasons'][:1]
        self.assertEqual(owner_priors(self.history)[0]['spending'],'Insufficient history')


class BidTests(unittest.TestCase):
    def test_bounded_historical_prior_even_extreme_input(self):
        for fraction in (-100,100):
            result = bid(prior_fraction=fraction)['audit']
            self.assertLessEqual(abs(result['historical_dollars']),result['before_history']*.05+1e-9)

    def test_whole_dollars_order_and_budget_cap(self):
        for remaining in (0,1,12,100):
            b = bid(remaining=remaining)
            self.assertTrue(0<=b['conservative']<=b['recommended']<=b['aggressive_ceiling']<=remaining)
            self.assertTrue(all(type(b[k]) is int for k in ('conservative','recommended','aggressive_ceiling')))

    def test_fp_suggestion_cannot_dominate(self):
        self.assertEqual(bid(fp_bid=999)['audit']['fp_fraction'],.2)
        self.assertEqual(bid(fp_bid=0)['audit']['fp_fraction'],-.2)

    def test_league_changing_can_spend_early(self):
        self.assertGreaterEqual(bid(tier='League-changing / potential long-term starter')['recommended'],40)
        self.assertLessEqual(bid(tier='Streamer')['aggressive_ceiling'],5)


class RecommendationTests(unittest.TestCase):
    def test_owner_release_precedes_wr_drop_without_reclassifying_stash(self):
        roster = [player('RB1','RB',ros=10),player('RB2','RB',ros=20),
                  player('Sampson','RB',ros=56,owner_expendable=True),
                  player('WR1',ros=5),player('WR2',ros=10),player('WR3',ros=20),
                  player('Flex',ros=30),player('Flournoy',ros=81)]
        roster[2]['injury']='IR'
        hierarchy=drop_hierarchy(roster)
        self.assertEqual(hierarchy[0]['name'],'Sampson')
        self.assertEqual(hierarchy[0]['classification'],'Upside stash')
        c=player('Harris',ros=63)
        result=evaluate(c,roster,[c],{'fraction':0},100)
        self.assertEqual(result['drop'],'Sampson')
        self.assertEqual(result['starter_gain'],0)
        qb=player('Backup QB','QB',ros=15)
        self.assertEqual(evaluate(qb,roster+[player('QB1','QB',ros=4)],[qb],{'fraction':0},100)['recommendation'],'NO CLAIM')
        roster[2]['retain_on_ir']=True
        self.assertEqual(evaluate(c,roster,[c],{'fraction':0},100)['drop'],'Flournoy')
        roster[2]['retain_on_ir']=False
        roster[2]['locked']=True
        self.assertEqual(evaluate(c,roster,[c],{'fraction':0},100)['drop'],'Flournoy')

    def test_owner_release_does_not_force_poor_rb_claim(self):
        roster=[player('Starter1','RB',ros=9),player('Starter2','RB',ros=23),
                player('Depth','RB',ros=31),player('Sampson','RB',ros=56,owner_expendable=True)]
        roster[-1]['injury']='IR'
        for ros in (51,57,65,68):
            c=player('Available','RB',ros=ros)
            self.assertEqual(evaluate(c,roster,[c],{'fraction':0},100)['recommendation'],'NO CLAIM')

    def test_clear_upgrade_independent_of_fp(self):
        c = player('Upgrade',ros=15)
        result = evaluate(c,[player('Weak',ros=70)],[c],{'fraction':0})
        self.assertEqual(result['recommendation'],'CLAIM')
        self.assertEqual(result['drop'],'Weak')
        self.assertGreater(result['faab']['recommended'],0)

    def test_no_claim_when_no_upgrade(self):
        c = player('Worse',ros=70)
        self.assertEqual(evaluate(c,[player('Better',ros=15)],[c],{'fraction':0})['recommendation'],'NO CLAIM')

    def test_positional_scarcity_is_bounded_and_separate(self):
        c = player('Candidate','TE',ros=10)
        self.assertEqual(scarcity(c,[player('Only TE','TE',ros=20)],[c]),.1)
        deep = [player(str(i),'TE',ros=20) for i in range(3)]
        self.assertEqual(scarcity(c,deep,[c]),0)
        self.assertGreater(bid(scarce=.1)['recommended'],bid(scarce=0)['recommended'])

    def test_drop_hierarchy_preserves_starters_and_injured_stashes(self):
        core = player('Core',ros=5)
        stash = player('Injured',ros=70)
        stash['injury']='PUP'
        rows = drop_hierarchy([core,stash,player('Replaceable',ros=70)])
        self.assertEqual(rows[0]['name'],'Replaceable')
        self.assertEqual(rows[1]['classification'],'Upside stash')
        self.assertEqual(rows[-1]['classification'],'Core hold')

    def test_recommended_starter_cannot_be_dropped_even_for_same_position_upgrade(self):
        for attributes in ({}, {'role':'released'}, {'injury':'PUP'}, {'ros_rank':None}):
            protected = player('Promoted',ros=70,gm_recommended_starter=True)
            protected.update(attributes)
            self.assertEqual(drop_hierarchy([protected])[0]['classification'],'Core hold')
            candidate = player('Elite available',ros=1)
            result = evaluate(candidate,[protected],[candidate],{'fraction':0})
            self.assertEqual(result['recommendation'],'NO CLAIM')
            self.assertIsNone(result['drop'])

    def test_lineup_promotion_flows_into_waiver_protection(self):
        from test_lineup import snapshot as lineup_snapshot
        snapshot = lineup_snapshot()
        snapshot.update(league_key='test',refreshed_at=utc_now(),refresh_started_at=utc_now())
        lineup = snapshot['sections']['lineup']['data']
        for row in lineup['starters']:
            if row['player_pos']=='RB':
                row['original_proj'] = 20
        lineup['bench'] = [dict(fp_id='promoted',player_name='Promoted',player_pos='WR',
                                 original_proj=10.05,locked=False)]
        result = analyze_waivers(snapshot,evidence={},context={})
        promoted = next(p for p in result['drop_hierarchy'] if p['name']=='Promoted')
        self.assertTrue(promoted['gm_recommended_starter'])
        self.assertEqual(promoted['classification'],'Core hold')
        self.assertIsNone(promoted['current_index'])
        outgoing = next(p for p in result['drop_hierarchy']
                        if p['position']=='WR' and not p['gm_recommended_starter'])
        self.assertEqual(outgoing['classification'],'Core hold')
        self.assertFalse(any(r['drop']=='Promoted' for r in result['board']))

    def test_unavailable_filtered(self):
        c = player('Owned',ros=1)
        c['availability']['confirmed']=False
        self.assertEqual(evaluate(c,[player('Weak',ros=70)],[c],{'fraction':0})['recommendation'],'NO CLAIM')

    def test_missing_ros_does_not_become_zero_rank(self):
        c = player('Unknown',ros=None,points=None)
        result = evaluate(c,[player('Known',ros=70)],[c],{'fraction':0})
        self.assertEqual(result['recommendation'],'NO CLAIM')
        self.assertTrue(any('ROS data missing' in w for w in result['warnings']))

    def test_streamer_requires_matching_week(self):
        c = player('Kicker','K',ros=1)
        r = evaluate(c,[player('K','K',ros=20)],[c],{'fraction':0})
        self.assertEqual(r['recommendation'],'NO CLAIM')

    def test_weekly_only_candidate_uses_common_basis(self):
        c = player('Weekly only',ros=None,points=10)
        c['weekly_comparable']=True
        d = player('Strong weekly',ros=70,points=20)
        d['weekly_comparable']=True
        self.assertEqual(evaluate(c,[d],[c],{'fraction':0})['recommendation'],'NO CLAIM')

    def test_unknown_drop_ros_cannot_look_like_cheap_opportunity(self):
        c = player('Ranked',ros=10)
        d = player('Missing ROS',ros=None,points=2)
        d['weekly_comparable']=True
        self.assertEqual(evaluate(c,[d],[c],{'fraction':0})['recommendation'],'NO CLAIM')

    def test_first_useful_rb_backup_is_depth(self):
        roster = [player('RB1','RB',ros=10),player('RB2','RB',ros=20),player('Backup','RB',ros=33)]
        self.assertEqual(next(p for p in drop_hierarchy(roster) if p['name']=='Backup')['classification'],'Useful depth')

    def test_cannot_drop_only_te_for_wr_depth(self):
        c = player('WR','WR',ros=1)
        r = evaluate(c,[player('Only TE','TE',ros=70)],[c],{'fraction':0})
        self.assertEqual(r['recommendation'],'NO CLAIM')

    def test_locked_drop_is_blocked(self):
        d = player('Locked',ros=70)
        d['locked']=True
        c = player('Upgrade',ros=1)
        self.assertEqual(evaluate(c,[d],[c],{'fraction':0})['recommendation'],'NO CLAIM')

    def test_sequential_claim_conflicts_and_total_budget(self):
        board = [dict(add=a,drop=d,recommendation='CLAIM',faab={'recommended':b}) for a,d,b in
                 [('A','X',21),('B','X',13),('C','Y',6),('D','Z',20)]]
        plan = claim_plan(board,27)
        self.assertEqual([r['add'] for r in plan],['A','B','C'])
        self.assertEqual(plan[1]['requires_failed_claims'],[1])
        self.assertEqual(plan[2]['requires_failed_claims'],[])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = build_snapshot(fixture_bundle())
        self.now = utc_now()
        self.evidence = dict(league_key=self.snapshot['league_key'],
            league_lookup={'leagues':[{'key':self.snapshot['league_key'],'name':'Test League'}]},
            calls=[dict(player_name='Candidate',fetched_at=self.now,
                        raw={'Candidate (CLE)':[{'league_name':'Test League','ownership_info':'Not owned - free agent'}]})])

    def test_explicit_league_confirmation_and_owned(self):
        self.assertTrue(ownership(self.snapshot,self.evidence,'Candidate',self.now)['confirmed'])
        self.evidence['calls'][0]['raw']['Candidate (CLE)'][0]['ownership_info']='Owned by somebody'
        self.assertFalse(ownership(self.snapshot,self.evidence,'Candidate',self.now)['confirmed'])

    def test_user_budget_survives_reanalysis_without_changing_snapshot(self):
        original = copy.deepcopy(self.snapshot)
        confirmation = dict(source='user', league_key=self.snapshot['league_key'],
            snapshot_id=self.snapshot['snapshot_id'], confirmed_at=self.now,
            team_balances={self.snapshot['team_id']:7})
        context = dict(faab_confirmation=confirmation)
        for _ in range(2):
            result = analyze_waivers(self.snapshot, evidence={}, context=context)
            self.assertEqual(result['remaining_faab'],7)
            self.assertFalse(any('Current remaining FAAB unavailable' in w for w in result['warnings']))
        self.assertEqual(self.snapshot, original)
        confirmation['team_balances'][self.snapshot['team_id']] = 0
        self.assertEqual(analyze_waivers(self.snapshot,evidence={},context=context)['remaining_faab'],0)

    def test_drop_preference_scope_ir_and_starter_protection(self):
        preference=dict(source='user',league_key=self.snapshot['league_key'],
            snapshot_id=self.snapshot['snapshot_id'],confirmed_at=self.now,
            preferred_drop='Sampson',ir_move_confirmed=False)
        roster=[player('Sampson','RB',ros=56)]
        context=dict(drop_preference=preference)
        self.assertTrue(apply_drop_preference(self.snapshot,context,roster)[0]['owner_expendable'])
        for change in (dict(snapshot_id='other'),dict(confirmed_at='2000-01-01T00:00:00Z')):
            self.assertNotIn('owner_expendable',apply_drop_preference(self.snapshot,
                dict(drop_preference=dict(preference,**change)),roster)[0])
        for flags in (dict(gm_recommended_starter=True),dict(current_index=0),dict(locked=True)):
            self.assertFalse(apply_drop_preference(self.snapshot,context,[dict(roster[0],**flags)])[0]['owner_expendable'])
        preference['ir_move_confirmed']=True
        result=apply_drop_preference(self.snapshot,context,roster)[0]
        self.assertTrue(result['retain_on_ir'])
        self.assertFalse(result['owner_expendable'])

    def test_budget_confirmation_rejects_stale_mismatched_or_invalid_input(self):
        confirmation = dict(source='user', league_key=self.snapshot['league_key'],
            snapshot_id=self.snapshot['snapshot_id'], confirmed_at=self.now,
            team_balances={self.snapshot['team_id']:100})
        for change in (dict(league_key='foreign'),dict(snapshot_id='old'),
                       dict(confirmed_at='2000-01-01T00:00:00Z'),
                       dict(confirmed_at='2100-01-01T00:00:00Z'),
                       dict(team_balances={self.snapshot['team_id']:-1}),
                       dict(team_balances={self.snapshot['team_id']:True}),
                       dict(team_balances={})):
            with self.subTest(change=change):
                self.assertIsNone(confirmed_budget(self.snapshot,
                    dict(faab_confirmation=dict(confirmation,**change)),self.now))

    def test_other_league_or_wrong_player_cannot_confirm(self):
        self.assertFalse(ownership(self.snapshot,self.evidence,'Different',self.now)['confirmed'])
        self.evidence['league_key']='foreign'
        self.assertFalse(ownership(self.snapshot,self.evidence,'Candidate',self.now)['confirmed'])

    def test_duplicate_league_names_fail_closed(self):
        self.evidence['league_lookup']['leagues'].append({'key':'other','name':'Test League'})
        self.assertFalse(ownership(self.snapshot,self.evidence,'Candidate',self.now)['confirmed'])

    def test_stale_evidence_fails_closed(self):
        self.evidence['calls'][0]['fetched_at']='2020-01-01T00:00:00Z'
        self.assertFalse(ownership(self.snapshot,self.evidence,'Candidate',self.now)['confirmed'])

    def test_candidate_inclusion_does_not_confirm_availability(self):
        pool = collect_candidates(self.snapshot,{})
        self.assertTrue(all(not c['availability']['confirmed'] for c in pool))

    def test_save_audit_and_incomplete_roster_no_claim(self):
        result = analyze_waivers(self.snapshot,evidence={},context={})
        self.assertEqual(result['overall'],'NO CLAIM')
        with tempfile.TemporaryDirectory() as directory:
            save_waiver_analysis(result,directory)
            saved = json.loads((Path(directory)/'latest_waiver_analysis.json').read_text())
            self.assertEqual(saved['historical_owner_prior']['owners'][0]['owner'],'Pat')
            self.assertEqual(saved['snapshot_id'],self.snapshot['snapshot_id'])

    def test_dashboard_board_and_collapsed_weekly_warning(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            persist(self.snapshot,folder/'data')
            with patch.object(gm_schema,'BASE',folder):
                app = AppTest.from_file(str(Path(__file__).parent/'dashboard.py')).run(timeout=45)
            self.assertEqual(len(app.exception),0)
            self.assertIn('GM Waiver Board',[s.value for s in app.subheader])
            self.assertIn('Scoring and data limitation details',[e.label for e in app.expander])
            self.assertIn('Raw FantasyPros waiver data / diagnostics',[e.label for e in app.expander])
            self.assertTrue((folder/'data'/'latest_waiver_analysis.json').exists())
            self.assertEqual(len(app.tabs[0].warning),1)

    def test_best_candidates_never_displays_unavailable_or_no_claim_references(self):
        from streamlit.testing.v1 import AppTest
        analysis = analyze_waivers(self.snapshot,evidence={},context={})
        rows = []
        for i,(label,status,confirmed,claim) in enumerate([
            ('Unconfirmed reference','Unconfirmed',False,True),
            ('Rostered reference','Owned / unavailable',False,True),
            ('Available reference','Confirmed free agent',True,False),
            ('Available upgrade','Confirmed free agent',True,True)],1):
            c = player(label,ros=1)
            row = evaluate(c,[player('Weak',ros=70)],[c],{'fraction':0})
            row.update(priority=i,recommendation='CLAIM' if claim else 'NO CLAIM')
            row['position'] = ['RB','QB','TE','WR'][i-1]
            row['availability'] = dict(confirmed=confirmed,status=status)
            rows.append(row)
        analysis.update(board=rows)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            persist(self.snapshot,folder/'data')
            with patch.object(gm_schema,'BASE',folder), patch('gm_waivers.analyze_waivers',return_value=analysis):
                app = AppTest.from_file(str(Path(__file__).parent/'dashboard.py')).run(timeout=45)
            self.assertEqual(len(app.exception),0)
            positional = [m.value for m in app.markdown if m.value.startswith(('RB:', 'QB:', 'WR:', 'TE:'))]
            self.assertEqual(len(positional),4)
            self.assertTrue(any('WR: Available upgrade / drop Weak' in m for m in positional))
            for pos in ('RB','QB','TE'):
                self.assertIn(pos+': NO CLAIM — no confirmed available candidate improves the roster',positional)
            self.assertFalse(any('reference' in m for m in positional))

    def test_dashboard_populated_claim_columns(self):
        from streamlit.testing.v1 import AppTest
        analysis = analyze_waivers(self.snapshot,evidence={},context={})
        c = player('Upgrade',ros=15)
        row = evaluate(c,[player('Weak',ros=70)],[c],{'fraction':0})
        row['priority']=1
        analysis.update(board=[row],claim_plan=claim_plan([row]),overall='CLAIM')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            persist(self.snapshot,folder/'data')
            with patch.object(gm_schema,'BASE',folder), patch('gm_waivers.analyze_waivers',return_value=analysis):
                app = AppTest.from_file(str(Path(__file__).parent/'dashboard.py')).run(timeout=45)
            self.assertEqual(len(app.exception),0)
            board = next(d.value for d in app.dataframe if 'Aggressive ceiling' in d.value.columns)
            self.assertEqual(board.iloc[0]['Add'],'Upgrade')
            self.assertEqual(board.iloc[0]['Drop'],'Weak')
            self.assertGreater(board.iloc[0]['Recommended bid'],0)


if __name__=='__main__':
    unittest.main()
