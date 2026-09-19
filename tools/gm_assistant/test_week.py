"""Dated slate regressions: 2026 Wednesday opener and NFL week transitions."""
import copy
import unittest
from datetime import timedelta

from gm_snapshot import build_snapshot, stamp
from gm_week import establish_week
from gm_streaming import analyze_streaming
from refresh import enrichment_requests
from test_gm import fixture_bundle


def calendar_at(now='2026-09-10T00:30:00Z'):
    return dict(season=2026,season_type='REG',verified_at=(stamp(now)-timedelta(hours=1)).isoformat(),weeks=[
        dict(week=1,complete=True,source_url='https://www.nfl.com/schedules/2026/by-week/week-1',games=[
            dict(away='NE',home='SEA',kickoff='2026-09-09T20:20:00-04:00'),
            dict(away='SF',home='LAR',kickoff='2026-09-11T10:35:00+10:00'),
            dict(away='ATL',home='PIT',kickoff='2026-09-13T13:00:00-04:00'),
            dict(away='DEN',home='KC',kickoff='2026-09-14T20:15:00-04:00')]),
        dict(week=2,complete=True,source_url='https://www.nfl.com/schedules/2026/by-week/week-2',games=[
            dict(away='DET',home='BUF',kickoff='2026-09-17T20:15:00-04:00'),
            dict(away='NYG',home='LAR',kickoff='2026-09-21T20:15:00-04:00')])])


PROVIDER = [dict(id='FantasyPros',data=dict(current_week=2,season_state='in_season'))]


class PlayingWeekTests(unittest.TestCase):
    def resolve(self,now):
        return establish_week(PROVIDER,now=now,calendar=calendar_at(now))

    def test_wednesday_opener_overrides_provider_week_two(self):
        result=self.resolve('2026-09-10T00:30:00Z')
        self.assertEqual((result['week'],result['playing_week'],result['next_week']),(1,1,2))
        self.assertTrue(result['provider_conflict'])

    def test_before_and_at_opener_kickoff(self):
        before=self.resolve('2026-09-10T00:19:59Z')
        at=self.resolve('2026-09-10T00:20:00Z')
        self.assertEqual(before['week'],1)
        self.assertIsNone(before['playing_week'])
        self.assertEqual(at['playing_week'],1)

    def test_australia_local_friday_and_utc_midnight_stay_week_one(self):
        for now in ('2026-09-10T23:59:59Z','2026-09-11T00:35:00Z','2026-09-11T10:35:00+10:00'):
            self.assertEqual(self.resolve(now)['playing_week'],1)

    def test_sunday_and_monday_games_stay_week_one(self):
        for now in ('2026-09-13T20:00:00Z','2026-09-15T02:00:00Z','2026-09-15T09:59:59Z'):
            self.assertEqual(self.resolve(now)['playing_week'],1)

    def test_tuesday_rollover_and_next_thursday(self):
        between=self.resolve('2026-09-15T10:00:00Z')
        self.assertEqual((between['week'],between['playing_week'],between['next_week']),(2,None,2))
        self.assertEqual(self.resolve('2026-09-18T00:15:00Z')['playing_week'],2)

    def test_rescheduled_tuesday_game_delays_rollover(self):
        now='2026-09-15T12:00:00Z'
        cal=calendar_at(now)
        cal['weeks'][0]['games'][-1]['kickoff']='2026-09-15T20:15:00-04:00'
        self.assertEqual(establish_week(PROVIDER,now,cal)['playing_week'],1)

    def test_no_calendar_never_trusts_provider_week(self):
        self.assertIsNone(establish_week(PROVIDER,'2026-09-10T01:00:00Z',{})['week'])

    def test_future_only_schedule_cannot_hide_current_slate(self):
        now='2026-09-13T01:00:00Z'
        cal=calendar_at(now)
        cal['weeks']=cal['weeks'][1:]
        self.assertIsNone(establish_week(PROVIDER,now,cal)['week'])

    def test_stale_or_future_verification_rejected(self):
        now='2026-09-10T01:00:00Z'
        for verified in ('2026-09-01T00:00:00Z','2026-09-11T00:00:00Z'):
            cal=calendar_at(now)
            cal['verified_at']=verified
            self.assertIsNone(establish_week(PROVIDER,now,cal)['week'])

    def test_missing_dates_incomplete_conflicting_and_wrong_season_rejected(self):
        now='2026-09-10T01:00:00Z'
        for kind in ('missing','incomplete','duplicate','overlap','season','naive'):
            cal=calendar_at(now)
            if kind=='missing': del cal['weeks'][0]['games'][0]['kickoff']
            if kind=='incomplete': cal['weeks'][0]['complete']=False
            if kind=='duplicate': cal['weeks'].append(copy.deepcopy(cal['weeks'][0]))
            if kind=='overlap': cal['weeks'][1]['games'][0]['kickoff']='2026-09-13T20:00:00-04:00'
            if kind=='season': cal['season']=2024
            if kind=='naive': cal['weeks'][0]['games'][0]['kickoff']='2026-09-09T20:20:00'
            with self.subTest(kind=kind):
                self.assertIsNone(establish_week(PROVIDER,now,cal)['week'])

    def test_preseason_and_exhausted_capture_do_not_guess(self):
        for now in ('2026-08-10T00:00:00Z','2026-09-22T10:00:00Z','2027-02-01T00:00:00Z'):
            self.assertIsNone(self.resolve(now)['week'])

    def test_january_uses_explicit_season_and_dst_offset(self):
        now='2027-01-04T03:00:00Z'
        cal=dict(season=2026,season_type='REG',verified_at='2027-01-03T00:00:00Z',weeks=[
            dict(week=18,complete=True,source_url='fixture',games=[
                dict(away='A',home='B',kickoff='2027-01-03T20:20:00-05:00')])])
        self.assertEqual(establish_week(PROVIDER,now,cal)['playing_week'],18)

    def test_projection_refresh_uses_dates_despite_conflicting_provider_labels(self):
        bundle=fixture_bundle()
        for call in bundle['calls']:
            if call['id'].startswith('schedule_'): call['raw']['current_week']=2
        tasks,_=enrichment_requests(bundle,now='2026-09-10T00:30:00Z',calendar=calendar_at())
        weekly=[r for r in tasks if r['id'].startswith('projection_weekly_')]
        self.assertEqual(len(weekly),6)
        self.assertTrue(all(r['arguments']['week']==1 and r['arguments']['season']==2026 for r in weekly))

    def test_streaming_does_not_relabel_next_week_projections(self):
        now='2026-09-10T00:30:00Z'
        s=build_snapshot(fixture_bundle())
        s['refreshed_at']=s['refresh_started_at']=now
        s['sections']['my_roster']['data']['roster']={'DST':['Current']}
        s['sections']['projection_weekly_DST'].update(status='ok',data=dict(week=2,season=2026,
            projections=[dict(name='Current',team='PIT',stats={'points':20})]))
        a=analyze_streaming(s,waiver_analysis={'drop_hierarchy':[dict(name='Current',position='DST')]},
                            sources={},evidence={},now=now,calendar=calendar_at())
        self.assertEqual(a['current_week']['week'],1)
        self.assertEqual(a['current_week']['projection_weeks']['DST'],2)
        self.assertIsNone(next(p for p in a['candidate_pool'] if p['name']=='Current')['projection'])
        self.assertFalse(any(c['transaction_required'] for c in a['calls']))

    def test_matching_week_wrong_season_excluded_before_scoring(self):
        from gm_streaming import score_candidate, CONFIG, RULES
        s={'sections':{'projection_weekly_K':{'status':'ok','data':{'week':1,'season':2025}}}}
        p=dict(name='K',position='K',team='PIT',current=True,
               projection_source={'stats':{'points':99}},ecr_source={})
        result=score_candidate(p,s,1,[],CONFIG,RULES,season=2026)
        self.assertFalse(result['projection_context']['aligned'])
        self.assertIsNone(result['projection'])
        self.assertIsNone(result['scoring']['points'])


if __name__=='__main__':
    unittest.main()
