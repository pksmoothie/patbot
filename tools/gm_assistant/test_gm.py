"""Offline tests: isolated synthetic fixtures, no MCP/Yahoo calls or draft imports."""
import copy
import json
import tempfile
import subprocess
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import gm_schema
from gm_schema import BASE, LEAGUE_KEY, TEAM_ID, SCHEMA
from gm_snapshot import build_snapshot, load_snapshot, persist, source_section, stamp, utc_now, validate
from gm_views import candidates, lineup_rows, matchup_metrics, obvious_warnings, projection_rows, roster_rows
from refresh import core_requests, enrichment_requests, run_codex


def fixture_bundle():
    now = utc_now()
    teams = [{'team_id': str(i), 'team_name': f'Team {i}', 'roster': {'QB': [f'QB {i}']}}
             for i in range(1, 13)]
    raw = {
        'resync': {'synced': True, 'league_key': LEAGUE_KEY},
        'settings': {'n_teams': 12, 'scoring': 'PPR'},
        'all_rosters': {'league_key': LEAGUE_KEY, 'teams': teams},
        'my_roster': {'league_key': LEAGUE_KEY, **teams[5]},
        'lineup': {'league_key': LEAGUE_KEY, 'week': None,
            'starters': [{'fp_id': '1', 'player_name': 'QB 6', 'player_pos': 'QB', 'slot': 'QB'}],
            'bench': [], 'raw_payload': {'teamId': '6', 'matchup': {
                'team1': {'id': 6, 'starters': [{'fpId': 1, 'full': 'QB 6', 'injuryStatus': 'Q'}]},
                'team2': {'id': 3, 'name': 'Team 3', 'starters': [], 'bench': []}}}},
        'matchup': {'win_probability': 0, 'user_lineup': {'current_projection': 0,
                    'starters': [{'position': 'K', 'warning': 'no player currently in position slot'}]},
                    'opponent_lineup': {'current_projection': 100}},
        'schedule_DST': {'current_week': 1}, 'schedule_K': {'current_week': 1},
    }
    return {'run_id': 'test_snapshot', 'started_at': now, 'league_key': LEAGUE_KEY, 'team_id': TEAM_ID,
            'calls': [{**r, 'fetched_at': now, 'raw': raw[r['id']]} for r in core_requests() if r['id'] in raw]}


def add_source(bundle, identifier, tool, arguments, data):
    bundle['calls'].append({'id': identifier, 'tool': 'mcp__fantasypros__' + tool,
                            'arguments': arguments, 'fetched_at': utc_now(), 'raw': data})


class SnapshotTests(unittest.TestCase):
    def test_schema_file_matches_definition(self):
        self.assertEqual(json.loads((BASE / 'snapshot.schema.json').read_text()), SCHEMA)

    def test_valid_partial_snapshot(self):
        snapshot = build_snapshot(fixture_bundle())
        self.assertEqual(snapshot['refresh_status'], 'partial')
        self.assertEqual(snapshot['sections']['opponent']['data']['team_id'], '3')
        self.assertIsNone(snapshot['sections']['lineup']['data']['week'])

    def test_foreign_league_or_implicit_league_rejected(self):
        for modification in ('top', 'args', 'raw'):
            bundle = fixture_bundle()
            if modification == 'top':
                bundle['league_key'] = 'nfl~other'
            elif modification == 'args':
                del bundle['calls'][0]['arguments']['league_key']
            else:
                bundle['calls'][2]['raw']['league_key'] = 'nfl~other'
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                build_snapshot(bundle)

    def test_core_failure_and_missing_team_rejected(self):
        for failure in ('sync', 'team'):
            bundle = fixture_bundle()
            if failure == 'sync':
                bundle['calls'][0]['raw']['synced'] = False
            else:
                bundle['calls'][2]['raw']['teams'].pop()
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                build_snapshot(bundle)

    def test_empty_vs_unavailable_vs_error(self):
        cases = [([], 'empty'), ({}, 'unavailable'),
                 ({'rankings': []}, 'empty'),
                 ({'status': 'premium_required', 'required_user_disclosure': 'Paid feature.'}, 'unavailable'),
                 ({'projections': [], 'note': 'No rest-of-season projections available (published in-season).'}, 'unavailable'),
                 ({'isError': True, 'content': []}, 'error')]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(source_section({'id': 'example', 'raw': raw})['status'], expected)

    def test_source_tampering_rejected(self):
        snapshot = build_snapshot(fixture_bundle())
        snapshot['sections']['matchup']['data']['win_probability'] = .99
        with self.assertRaises(ValueError):
            validate(snapshot)

    def test_wrong_team_selector_and_account_wide_tool_rejected(self):
        bundle = fixture_bundle()
        bundle['calls'][3]['arguments']['team_id'] = '3'
        with self.assertRaises(ValueError):
            build_snapshot(bundle)
        bundle = fixture_bundle()
        bundle['calls'][3]['tool'] = 'mcp__fantasypros__get_player_ownership'
        with self.assertRaises(ValueError):
            build_snapshot(bundle)

    def test_long_collection_window_rejected(self):
        bundle = fixture_bundle()
        bundle['started_at'] = (stamp(bundle['started_at']) - timedelta(days=2)).isoformat()
        with self.assertRaisesRegex(ValueError, 'two hours'):
            build_snapshot(bundle)

    def test_history_is_exclusive_and_latest_survives_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            first = build_snapshot(fixture_bundle())
            first_path = persist(first, folder)
            before = (folder / 'latest_snapshot.json').read_bytes()
            self.assertEqual(load_snapshot(folder / 'latest_snapshot.json')['snapshot_id'], 'test_snapshot')
            with self.assertRaises(FileExistsError):
                persist(first, folder)
            self.assertEqual(before, (folder / 'latest_snapshot.json').read_bytes())
            invalid = copy.deepcopy(first)
            invalid['team_id'] = '3'
            with self.assertRaises(ValueError):
                persist(invalid, folder)
            self.assertEqual(before, (folder / 'latest_snapshot.json').read_bytes())
            second = build_snapshot({**fixture_bundle(), 'run_id': 'second_snapshot'})
            second_path = persist(second, folder)
            self.assertNotEqual(first_path, second_path)
            self.assertTrue(first_path.exists())
            self.assertEqual(len(list((folder / 'snapshots').glob('*.json'))), 2)

    def test_corrupt_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'broken.json'
            path.write_text('{broken')
            with self.assertRaises(ValueError):
                load_snapshot(path)


class ViewTests(unittest.TestCase):
    def test_zero_metrics_unknown_week_and_explicit_warnings(self):
        snapshot = build_snapshot(fixture_bundle())
        metrics = matchup_metrics(snapshot)
        self.assertEqual(metrics['my_projection'], 0)
        self.assertEqual(metrics['win_probability'], 0)
        self.assertIsNone(metrics['week'])
        self.assertIn('Empty starting slot: K', obvious_warnings(snapshot))
        self.assertIn('Starter QB 6: Q', obvious_warnings(snapshot))

    def test_roster_does_not_infer_slots(self):
        rows = roster_rows({'RB': ['Example Player']})
        self.assertIsNone(rows[0]['Slot'])
        self.assertIsNone(rows[0]['NFL team'])

    def test_candidates_do_not_imply_availability(self):
        bundle = fixture_bundle()
        add_source(bundle, 'waiver_RB', 'waiver_finder', {'league_key': LEAGUE_KEY, 'position': 'RB'},
                   {'recommended_adds': [], 'top_starts_this_week': {'RB': [{'full': 'Candidate A', 'ecr': 50}]}})
        rows = candidates(build_snapshot(bundle), 'RB')
        self.assertEqual(rows[0]['Availability'], 'Not independently established')

    def test_projection_uses_explicit_ppr_stat(self):
        bundle = fixture_bundle()
        add_source(bundle, 'projection_weekly_RB', 'get_projections', {'position': 'RB', 'projection_type': 'weekly', 'week': 1},
                   {'season': '2026', 'week': '1', 'projections': [{'name': 'Player', 'stats': {'points': 4, 'points_ppr': 9}}]})
        rows = projection_rows(build_snapshot(bundle), 'weekly', 'RB')
        self.assertEqual(rows[0]['Points (PPR)'], 9)

    def test_enrichment_never_guesses_week(self):
        bundle = fixture_bundle()
        bundle['calls'] = [c for c in bundle['calls'] if not c['id'].startswith('schedule_')]
        tasks, skipped = enrichment_requests(bundle,calendar={})
        self.assertFalse(any(t['id'].startswith('projection_weekly_') for t in tasks))
        self.assertEqual(len([k for k in skipped if k.startswith('projection_weekly_')]), 6)


class DashboardTests(unittest.TestCase):
    def test_complete_lineup_recommendation_rendering(self):
        from streamlit.testing.v1 import AppTest
        from test_lineup import snapshot as lineup_fixture
        bundle = fixture_bundle()
        lineup = lineup_fixture()['sections']['lineup']['data']
        lineup['league_key'] = LEAGUE_KEY
        lineup['bench'] = [{'fp_id': 'new', 'player_name': 'Upgrade WR', 'player_pos': 'WR',
                            'slot': 'BN', 'original_proj': 15, 'locked': False}]
        lineup['raw_payload'] = {'teamId': '6', 'matchup': {
            'team1': {'id': 6}, 'team2': {'id': 3, 'name': 'Team 3'}}}
        for source in bundle['calls']:
            if source['id'] == 'lineup':
                source['raw'] = lineup
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            persist(build_snapshot(bundle), folder / 'data')
            with patch.object(gm_schema, 'BASE', folder):
                app = AppTest.from_file(str(BASE / 'dashboard.py')).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.metric[0].value, '105.00')
            self.assertEqual(app.metric[3].value, '1')
            self.assertIn('Recommended lineup', [s.value for s in app.subheader])
            audit = json.loads((folder / 'data' / 'latest_analysis.json').read_text())
            self.assertEqual(audit['lineup_changes'][0]['gm_recommendation'], 'Upgrade WR')

    def test_dashboard_missing_and_populated_snapshots(self):
        from streamlit.testing.v1 import AppTest
        for populated in (False, True):
            with self.subTest(populated=populated), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                if populated:
                    persist(build_snapshot(fixture_bundle()), folder / 'data')
                with patch.object(gm_schema, 'BASE', folder):
                    app = AppTest.from_file(str(BASE / 'dashboard.py')).run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual([tab.label for tab in app.tabs], ['This Week', 'My Roster', 'Waivers', 'Streaming', 'Trades', 'League'])
                if populated:
                    self.assertEqual(app.metric[0].value, 'Unavailable')
                    self.assertIn('GM Final Call', [s.value for s in app.subheader])
                    self.assertIn('Raw source data / diagnostics', [e.label for e in app.expander])
                    self.assertTrue((folder / 'data' / 'latest_analysis.json').exists())


class RunnerTests(unittest.TestCase):
    def test_missing_mcp_does_not_start_collection(self):
        responses = [subprocess.CompletedProcess([], 0, '--sandbox --cd --output-last-message'),
                     subprocess.CompletedProcess([], 0, '[]')]
        with patch('refresh.shutil.which', return_value='codex.cmd'), \
             patch('refresh.subprocess.run', side_effect=responses) as runner, \
             patch('refresh.prepare_run') as prepare:
            with self.assertRaisesRegex(ValueError, 'no enabled fantasypros'):
                run_codex()
            self.assertEqual(runner.call_count, 2)
            prepare.assert_not_called()

    def test_failed_codex_does_not_publish_or_bypass_security(self):
        responses = [subprocess.CompletedProcess([], 0, '--sandbox --cd --output-last-message'),
                     subprocess.CompletedProcess([], 0, '[{"name":"fantasypros","enabled":true}]'),
                     subprocess.CompletedProcess([], 1)]
        with tempfile.TemporaryDirectory() as directory, \
             patch('refresh.shutil.which', return_value='codex.cmd'), \
             patch('refresh.subprocess.run', side_effect=responses) as runner, \
             patch('refresh.prepare_run', return_value=Path(directory)), \
             patch('refresh.persist') as publish:
            with self.assertRaisesRegex(ValueError, 'latest was not changed'):
                run_codex()
            publish.assert_not_called()
            args = runner.call_args.args[0]
            self.assertIn('workspace-write', args)
            self.assertNotIn('--dangerously-bypass-approvals-and-sandbox', args)
            self.assertNotIn('--ignore-user-config', args)
            self.assertEqual(args[-1], '-')

    def test_without_dated_schedule_conflicting_source_weeks_skip_projections(self):
        bundle = fixture_bundle()
        for source in bundle['calls']:
            if source['id'] == 'schedule_K':
                source['raw']['current_week'] = 2
        tasks, skipped = enrichment_requests(bundle,calendar={})
        self.assertFalse(any(t['id'].startswith('projection_weekly_') for t in tasks))
        self.assertIn('projection_weekly_K', skipped)


if __name__ == '__main__':
    unittest.main()
