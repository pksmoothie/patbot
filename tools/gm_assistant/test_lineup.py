"""Offline optimizer contracts; no live services or draft engine imports."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from gm_lineup import (BASE, MODEL, SLOTS, adjustment, analyze, build_players, eligible,
                      join, legal_lineups, optimize, save_analysis, score_projection, strategy, total)
from gm_views import opponent_rows


def player(pid, pos, points=10, index=None, **kwargs):
    return dict(id=pid, name=pid, position=pos, eligibility=[pos], points=points,
                current_index=index, locked=False, excluded=False, stats={}, **kwargs)


def roster():
    return [player(str(i), 'RB' if pos == 'FLEX' else pos, 10, i) for i, pos in enumerate(SLOTS)]


def snapshot():
    rows = [dict(fp_id=p['id'], player_name=p['name'], player_pos=p['position'],
                 slot=SLOTS[p['current_index']], original_proj=p['points'], locked=False) for p in roster()]
    data = {'lineup': {'week': None, 'starters': rows, 'bench': []},
            'settings': {'scoring': 'PPR'}, 'matchup': {'win_probability': .5},
            'start_sit': {}, 'injuries': {}}
    return {'snapshot_id': 'synthetic', 'sections': {k: {'data': v, 'status': 'ok'} for k,v in data.items()}}


class ConstructionTests(unittest.TestCase):
    def test_legal_slots_flex_duplicates_and_completeness(self):
        players = roster() + [player('rb', 'RB'), player('wr', 'WR'), player('te', 'TE'), player('qb', 'QB')]
        alternatives = list(legal_lineups(players))
        self.assertGreater(len(alternatives), 1)
        self.assertEqual({a[7]['position'] for a in alternatives}, {'RB', 'WR', 'TE'})
        for a in alternatives:
            self.assertEqual(len(a), 10)
            self.assertEqual(len({p['id'] for p in a}), 10)
            self.assertTrue(all(eligible(p, slot) for p, slot in zip(a, SLOTS)))
        # Independently enumerate starting sets: every eligible ten-player set
        # must have an assignment iff its positional counts can fill the roster.
        import itertools
        expected = set()
        for subset in itertools.combinations(players, 10):
            counts = {pos: sum(p['position'] == pos for p in subset) for pos in ('QB','RB','WR','TE','K','DST')}
            if (counts['QB'] == counts['K'] == counts['DST'] == 1 and
                counts['RB'] >= 2 and counts['WR'] >= 3 and counts['TE'] >= 1):
                expected.add(frozenset(p['id'] for p in subset))
        self.assertEqual({frozenset(p['id'] for p in a) for a in alternatives}, expected)

    def test_locked_starter_and_locked_bench(self):
        players = roster() + [player('better', 'RB', 50), player('bench', 'WR', 80)]
        players[1]['locked'] = True
        players[-1]['locked'] = True
        players[1]['excluded'] = True  # Locked inactive starter is still fixed.
        for a in legal_lineups(players):
            self.assertEqual(a[1]['id'], '1')
            self.assertNotIn('bench', [p['id'] for p in a])

    def test_locked_flex_cannot_move(self):
        players = roster() + [player('better', 'RB', 50)]
        players[7]['locked'] = True
        best, _, _ = optimize(players, 'Neutral')
        self.assertEqual(best[7]['id'], '7')

    def test_multi_position_eligibility(self):
        p = player('dual', 'TE')
        p['eligibility'] = ['WR', 'TE']
        self.assertTrue(eligible(p, 'WR'))
        self.assertTrue(eligible(p, 'FLEX'))
        self.assertFalse(eligible(p, 'QB'))

    def test_no_legal_lineup(self):
        self.assertIsNone(optimize(roster()[:-1], 'Neutral')[1])


class ModelTests(unittest.TestCase):
    def test_clear_upgrade(self):
        best, rec, _ = optimize(roster() + [player('new', 'WR', 15)], 'Neutral')
        self.assertEqual(total(best), 105)
        self.assertIn('new', [p['id'] for p in rec])

    def close_pair(self):
        players = roster()
        # Single TE contest; FLEX RB remains substantially better.
        players[6]['points'] = 5
        players.append(player('alternative', 'TE', 4.9))
        return players

    def test_favorite_workload_tiebreak(self):
        players = self.close_pair()
        players[6]['stats'] = {'rush_att': 0, 'rec': 1}
        players[-1]['stats'] = {'rush_att': 0, 'rec': 15}
        self.assertEqual(strategy(.7), 'Floor-Leaning')
        self.assertIn('alternative', [p['id'] for p in optimize(players, strategy(.7))[1]])
        self.assertNotIn('alternative', [p['id'] for p in optimize(players, 'Neutral')[1]])

    def test_underdog_td_variance_tiebreak(self):
        players = self.close_pair()
        players[6]['stats'] = {'rush_tds': 0, 'rec_tds': .05}
        players[-1]['stats'] = {'rush_tds': 0, 'rec_tds': .7}
        self.assertEqual(strategy(.3), 'Ceiling-Leaning')
        self.assertIn('alternative', [p['id'] for p in optimize(players, strategy(.3))[1]])

    def test_material_gap_cannot_be_overridden_even_with_extreme_weights(self):
        players = self.close_pair()
        players[-1]['points'] = 3
        players[-1]['stats'] = {'rush_att': 25, 'rec': 25, 'rush_tds': 1, 'rec_tds': 1}
        model = {**MODEL, 'strategy_weight': 100}
        for mode in ('Floor-Leaning', 'Ceiling-Leaning'):
            self.assertNotIn('alternative', [p['id'] for p in optimize(players, mode, model)[1]])

    def test_injury_is_not_upside(self):
        p = player('x', 'WR', injury='Q')
        self.assertLess(sum(adjustment(p, 'Ceiling-Leaning').values()), 0)

    def test_missing_fields_not_zero_or_fake_proxies(self):
        self.assertEqual(adjustment(player('x', 'WR'), 'Ceiling-Leaning'), {})
        s = snapshot()
        s['sections']['lineup']['data']['starters'][0].pop('original_proj')
        result = analyze(s)
        self.assertIsNone(result['recommended_projection'])
        self.assertEqual(result['lineup_changes'], [])
        self.assertTrue(result['warnings'])

    def test_reproducible_audit_and_actual_changes(self):
        s = snapshot()
        s['sections']['lineup']['data']['bench'] = [dict(fp_id='new', player_name='New WR', player_pos='WR', original_proj=15)]
        first = analyze(s, timestamp='fixed')
        self.assertEqual(first, analyze(copy.deepcopy(s), timestamp='fixed'))
        self.assertEqual(len(first['lineup_changes']), 1)
        self.assertEqual(first['lineup_changes'][0]['gm_recommendation'], 'New WR')
        with tempfile.TemporaryDirectory() as directory:
            save_analysis(first, directory)
            save_analysis(first, directory)
            self.assertEqual(len(list((Path(directory)/'analyses').glob('*.json'))), 1)
            self.assertEqual(json.loads((Path(directory)/'latest_analysis.json').read_text()), first)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.rules = json.loads((BASE/'scoring_rules.json').read_text())

    def test_custom_qb_completion_and_bonus_scoring(self):
        missing = score_projection({}, 'QB', self.rules, 20)['unavailable_components']
        stats = dict.fromkeys(missing, 0)
        stats.update(pass_cmp=20, pass_yds=300, pass_tds=2, pass_ints=1,
                     pass_yds_320=.4, pass_yds_350=.2, pass_yds_380=.1)
        result = score_projection(stats, 'QB', self.rules, 20)
        self.assertEqual(result['method'], 'Yahoo custom scoring')
        self.assertAlmostEqual(result['points'], 24.1)

    def test_mean_yards_does_not_create_bonus_and_fumbles_not_lost(self):
        result = score_projection({'pass_yds': 400, 'fumbles': .5}, 'QB', self.rules, 20)
        self.assertEqual(result['points'], 20)
        self.assertIn('pass_yds_320', result['unavailable_components'])
        self.assertIn('fumbles_lost', result['unavailable_components'])

    def test_native_fp_component_names(self):
        result = score_projection({'rec_rec': 5, 'rec_yds': 50}, 'WR', self.rules, 12)
        self.assertEqual(result['calculated_components']['Receptions'], 5)
        self.assertEqual(result['calculated_components']['rec yards'], 5)
        result = score_projection({'def_sack': 3, 'def_int': 1, 'def_fr': .5}, 'DST', self.rules, 7)
        self.assertEqual(result['calculated_components']['sack'], 3)
        self.assertEqual(result['calculated_components']['interception'], 2)
        self.assertEqual(result['calculated_components']['fumble_recovery'], 1)
        self.assertEqual(result['points'], 7)

    def test_missing_and_zero_rank_safe(self):
        self.assertEqual(adjustment(player('x', 'WR', ecr='WR0'), 'Neutral'), {})

    def test_kicker_distance_miss_scoring_and_fallback(self):
        missing = score_projection({}, 'K', self.rules, 8)['unavailable_components']
        stats = dict.fromkeys(missing, 0)
        stats.update(fg_50_plus=1, fg_missed_20_29=.5, xpt=2, xpt_missed=.1)
        self.assertAlmostEqual(score_projection(stats, 'K', self.rules, 8)['points'], 6.4)
        self.assertEqual(score_projection({'fg': 2, 'fga': 3, 'xpt': 2}, 'K', self.rules, 8)['points'], 8)

    def test_week_mismatch_rejects_projection_components(self):
        s = snapshot()
        s['sections']['lineup']['data']['week'] = 1
        s['sections']['projection_weekly_QB'] = {'status': 'ok', 'data': {'week': 2,
            'projections': [{'fpid': '0', 'stats': {'pass_cmp': 50}}]}}
        players, _ = build_players(s, self.rules)
        self.assertEqual(players[0]['stats'], {})
        s['sections']['projection_weekly_QB']['data']['week'] = 1
        self.assertEqual(build_players(s, self.rules)[0][0]['stats'], {'pass_cmp': 50})

    def test_identity_join_rejects_ambiguity_and_conflict(self):
        self.assertEqual(join({'name': 'Test'}, [{'name': 'Test'}, {'name': 'Test'}]), {})
        self.assertEqual(join({'name': 'Test', 'fp_id': '1'}, [{'name': 'Test', 'fp_id': '2'}]), {})

    def test_opponent_join_and_dash(self):
        s = snapshot()
        s['sections']['opponent'] = {'status': 'ok', 'data': {'team_id': '3', 'roster': {'QB': ['Opponent', 'Unknown']}}}
        s['sections']['lineup']['data']['raw_payload'] = {'matchup': {'team2': {'id': 3, 'starters': [
            {'fpId': 42, 'full': 'Opponent', 'real_position': 'QB', 'position': 'QB', 'real_team': 'BUF', 'injuryStatus': 'Q'}]}}}
        rows = opponent_rows(s)
        self.assertEqual(rows[0]['NFL team'], 'BUF')
        self.assertEqual(rows[0]['Injury status'], 'Q')
        self.assertEqual(rows[1]['NFL team'], '—')
        s['sections']['lineup']['data']['raw_payload']['matchup']['team2']['id'] = 4
        self.assertEqual(opponent_rows(s)[0]['NFL team'], '—')


if __name__ == '__main__':
    unittest.main()
