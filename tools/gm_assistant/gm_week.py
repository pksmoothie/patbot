"""Resolve a playing slate from dated NFL games; provider labels are diagnostic.

The slate remains current through 06:00 Eastern after its last scheduled game
date. This conservative rollover preserves Monday night games across UTC midnight.
Between slates, week is the upcoming planning week and playing_week is None.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from gm_snapshot import stamp, utc_now

EASTERN = ZoneInfo('America/New_York')


def load_calendar():
    try:
        return json.loads((Path(__file__).parent/'nfl_schedule.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def establish_week(sources, now=None, calendar=None):
    now = stamp(now or utc_now())
    calendar = load_calendar() if calendar is None else calendar
    evidence = [dict(source=s['id'], current_week=s['data'].get('current_week'),
                     season_state=s['data'].get('season_state')) for s in sources]
    result = dict(week=None, playing_week=None, next_week=None, season=None,
                  phase='Unavailable', sources=evidence, date_evidence=calendar,
                  provider_conflict=False, status='Unavailable: verified dated NFL schedule required')
    try:
        if not 0 <= (now-stamp(calendar['verified_at'])).total_seconds() <= 7*86400:
            return result
        season = calendar['season']
        if type(season) is not int or calendar.get('season_type') != 'REG':
            return result
        windows = []
        for slate in calendar['weeks']:
            week = slate['week']
            games = slate['games']
            if (type(week) is not int or not 1 <= week <= 18 or not slate.get('source_url')
                    or slate.get('complete') is not True or not games):
                return result
            times = [stamp(g['kickoff']) for g in games]
            teams = [t for g in games for t in (g['away'], g['home'])]
            if len(teams) != len(set(teams)) or any(t.year not in (season, season+1) for t in times):
                return result
            last = max(times).astimezone(EASTERN)
            end = datetime.combine(last.date()+timedelta(days=1), datetime.min.time(), EASTERN)+timedelta(hours=6)
            windows.append((week, min(times), end))
        windows.sort()
        if len({w[0] for w in windows}) != len(windows):
            return result
        if any(a[2] > b[1] or a[0]+1 != b[0] for a,b in zip(windows,windows[1:])):
            return result
        active = [w for w in windows if w[1] <= now < w[2]]
        upcoming = next((w for w in windows if now < w[1]), None)
        chosen = active[0] if len(active)==1 else upcoming
        # Do not turn preseason or an exhausted schedule capture into a current week.
        if chosen is None or (not active and (chosen[1]-now).total_seconds() > 7*86400):
            return result
        if not active and chosen[0] != 1 and not any(w[0]==chosen[0]-1 and w[2]<=now for w in windows):
            return result
        result.update(week=chosen[0], playing_week=chosen[0] if active else None,
                      next_week=upcoming[0] if upcoming else None, season=season,
                      phase='Playing' if active else 'Upcoming',
                      first_kickoff=chosen[1].isoformat(), rollover_at=chosen[2].isoformat(),
                      status='Available: explicit NFL game dates')
        result['provider_conflict'] = any(e['current_week'] != chosen[0] for e in evidence
                                          if e['current_week'] is not None)
    except (KeyError, ValueError, TypeError, AttributeError):
        pass
    return result


def kickoff_for(calendar, team, week):
    times = {g['kickoff'] for slate in calendar.get('weeks',[]) if slate.get('week')==week
             for g in slate.get('games',[]) if team in (g.get('home'),g.get('away'))}
    return next(iter(times)) if len(times)==1 else None
