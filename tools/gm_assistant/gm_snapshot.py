"""Validate and persist audited snapshots. Network access belongs exclusively to Codex MCP."""
import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from gm_schema import BASE, LEAGUE_KEY, TEAM_ID, POSITIONS, SECTION_NAMES, SCHEMA

SCOPED = {'resync_league', 'get_league_settings', 'get_roster', 'get_current_starters',
          'matchup_analyzer', 'league_analyzer', 'start_sit_assistant', 'waiver_finder'}
GLOBAL = {'get_ecr', 'get_projections', 'injury_status', 'get_schedule_and_strength'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamps must include a timezone.')
    return result


def decode(raw):
    if isinstance(raw, dict) and 'content' in raw:
        if raw.get('isError'):
            raise ValueError('MCP returned isError=true; inspect source.raw.')
        texts = [c['text'] for c in raw['content'] if c.get('type') == 'text']
        if len(texts) != 1:
            raise ValueError('Expected one JSON text payload; raw MCP result retained.')
        return json.loads(texts[0])
    return raw  # A decoded, otherwise unmodified MCP JSON payload is also accepted.


def check_identity(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('league_key', 'key') and isinstance(item, str) and item.startswith('nfl~') and item != LEAGUE_KEY:
                raise ValueError('Foreign league data rejected.')
            check_identity(item)
    elif isinstance(value, list):
        for item in value:
            check_identity(item)


def unavailable(reason, status='unavailable', source_ids=None):
    return {'status': status, 'data': None, 'source_ids': source_ids or [], 'warnings': [reason]}


def source_section(source):
    try:
        data = decode(source['raw'])
    except (ValueError, TypeError, KeyError) as exc:
        return unavailable(str(exc), 'error', [source['id']])
    check_identity(data)
    if isinstance(data, dict):
        status = data.get('status')
        if status in ('premium_required', 'no_data', 'league_sync_required', 'league_not_found',
                      'unavailable', 'not_available', 'error', 'failed'):
            message = data.get('required_user_disclosure') or data.get('message') or str(status)
            return unavailable(str(message), 'error' if status in ('error', 'failed') else 'unavailable', [source['id']])
        note = str(data.get('note', ''))
        if 'no rest-of-season projections available' in note.lower():
            return unavailable(note, source_ids=[source['id']])
    if data is None or data == {}:
        return unavailable('No usable data returned; not a confirmed empty result.', source_ids=[source['id']])
    empty = data == []
    if isinstance(data, dict):
        for key in ('rankings', 'projections'):
            if key in data and data[key] == []:
                empty = True
        if data.get('practice_details') == [] and data.get('player_news') == []:
            empty = True
        if data.get('recommended_adds') == [] and isinstance(data.get('top_starts_this_week'), dict) and all(v == [] for v in data['top_starts_this_week'].values()):
            empty = True
    return {'status': 'empty' if empty else 'ok', 'data': [] if empty else copy.deepcopy(data),
            'source_ids': [source['id']], 'warnings': []}


def check_source_contract(source):
    identifier, args = source['id'], source['arguments']
    fixed = {'resync': 'resync_league', 'settings': 'get_league_settings',
             'all_rosters': 'get_roster', 'my_roster': 'get_roster', 'lineup': 'get_current_starters',
             'matchup': 'matchup_analyzer', 'league_analysis': 'league_analyzer',
             'start_sit': 'start_sit_assistant', 'injuries': 'injury_status'}
    expected = fixed.get(identifier)
    for prefix, tool in [('waiver_', 'waiver_finder'), ('ranking_', 'get_ecr'),
                         ('projection_', 'get_projections'), ('schedule_', 'get_schedule_and_strength')]:
        if identifier.startswith(prefix):
            expected = tool
            if args.get('position') != identifier.rsplit('_', 1)[1]:
                raise ValueError('Position does not match source section.')
    if identifier not in SECTION_NAMES or expected is None or source['tool'] != 'mcp__fantasypros__' + expected:
        raise ValueError('Source tool does not match the section contract.')
    if identifier == 'all_rosters' and (args.get('all_teams') is not True or 'team_id' in args or 'team_name' in args):
        raise ValueError('All-rosters request must explicitly select all teams.')
    if identifier == 'my_roster' and (str(args.get('team_id')) != TEAM_ID or args.get('all_teams') or 'team_name' in args):
        raise ValueError('Own-roster request must explicitly select team 6.')
    if identifier.startswith('ranking_') and args.get('ranking_type') != identifier.split('_')[1]:
        raise ValueError('Ranking timeframe mismatch.')
    if identifier.startswith('projection_') and args.get('projection_type') != identifier.split('_')[1]:
        raise ValueError('Projection timeframe mismatch.')


def section_data(snapshot, name):
    section = snapshot['sections'][name]
    return section['data'] if section['status'] in ('ok', 'empty') else None


def opponent_section(sections):
    lineup = sections['lineup']['data'] or {}
    matchup = lineup.get('raw_payload', {}).get('matchup', {})
    user, opponent = matchup.get('team1', {}), matchup.get('team2', {})
    if str(user.get('id')) != TEAM_ID or opponent.get('id') is None:
        return unavailable('Current opponent identity was not explicitly returned for team 6.')
    teams = (sections['all_rosters']['data'] or {}).get('teams', [])
    roster = next((t for t in teams if str(t.get('team_id')) == str(opponent['id'])), None)
    data = {'team_id': str(opponent['id']), 'team_name': opponent.get('name'),
            'roster': roster.get('roster') if roster else None,
            'starters': opponent.get('starters'), 'bench': opponent.get('bench'),
            'week': lineup.get('week')}
    return {'status': 'ok', 'data': data, 'source_ids': ['lineup', 'all_rosters'],
            'warnings': [] if roster else ['Opponent roster unavailable.']}


def validate(snapshot):
    errors = sorted(Draft202012Validator(SCHEMA, format_checker=FormatChecker()).iter_errors(snapshot),
                    key=lambda e: str(list(e.path)))
    if errors:
        raise ValueError('; '.join(f'{list(e.path)}: {e.message}' for e in errors[:5]))
    check_identity(snapshot)
    sources = {s['id']: s for s in snapshot['sources']}
    if len(sources) != len(snapshot['sources']):
        raise ValueError('Duplicate source IDs.')
    start, end = stamp(snapshot['refresh_started_at']), stamp(snapshot['refreshed_at'])
    if start > end:
        raise ValueError('Refresh timestamps are reversed.')
    if (end - start).total_seconds() > 7200:
        raise ValueError('Collection window exceeds two hours. Start a new refresh; do not mix sessions.')
    for source in sources.values():
        check_source_contract(source)
        tool = source['tool'].removeprefix('mcp__fantasypros__')
        args = source['arguments']
        if tool not in SCOPED | GLOBAL:
            raise ValueError(f'Tool outside the GM read/refresh allowlist: {tool}')
        if tool in SCOPED and args.get('league_key') != LEAGUE_KEY:
            raise ValueError('Every league-scoped call must explicitly pass the authoritative key.')
        if args.get('sport', 'nfl').lower() != 'nfl':
            raise ValueError('Non-NFL call rejected.')
        if not start <= stamp(source['fetched_at']) <= end:
            raise ValueError('Source timestamp outside this refresh window.')
        try:
            check_identity(decode(source['raw']))
        except json.JSONDecodeError:
            pass
        except ValueError as exc:
            if 'Foreign league' in str(exc):
                raise
    for name, section in snapshot['sections'].items():
        if not set(section['source_ids']) <= sources.keys():
            raise ValueError(f'{name}: dangling source reference.')
        if section['status'] in ('ok', 'empty') and not section['source_ids']:
            raise ValueError(f'{name}: data has no source.')
        if name in sources and section != source_section(sources[name]):
            raise ValueError(f'{name}: section differs from the preserved source.')
    if snapshot['sections']['opponent'] != opponent_section(snapshot['sections']):
        raise ValueError('Opponent differs from the explicit lineup and roster sources.')
    expected_status = 'partial' if any(s['status'] not in ('ok', 'empty') for s in snapshot['sections'].values()) else 'complete'
    if snapshot['refresh_status'] != expected_status:
        raise ValueError('Refresh status does not match source coverage.')
    for name in ('resync', 'settings', 'all_rosters', 'my_roster'):
        if snapshot['sections'][name]['status'] != 'ok':
            raise ValueError(f'Core refresh failed: {name}. Keep the previous latest snapshot.')
    if section_data(snapshot, 'resync').get('synced') is not True:
        raise ValueError('A confirmed resync is required before publishing.')
    sync_time = stamp(sources['resync']['fetched_at'])
    if any(stamp(s['fetched_at']) < sync_time for s in sources.values()):
        raise ValueError('Data was collected before the confirmed resync.')
    settings = section_data(snapshot, 'settings')
    all_rosters = section_data(snapshot, 'all_rosters')
    if all_rosters.get('user_team_id') is not None and str(all_rosters['user_team_id']) != TEAM_ID:
        raise ValueError('The authenticated user team is not team 6.')
    teams = all_rosters.get('teams', [])
    ids = [str(t.get('team_id')) for t in teams]
    if settings.get('n_teams') != 12 or len(teams) != 12 or len(set(ids)) != 12 or TEAM_ID not in ids:
        raise ValueError('Expected all 12 distinct team rosters, including team 6.')
    if any(not isinstance(t.get('roster'), dict) for t in teams):
        raise ValueError('Missing team roster; cannot publish a full-league snapshot.')
    if str(section_data(snapshot, 'my_roster').get('team_id')) != TEAM_ID:
        raise ValueError('Own roster is not team 6.')
    lineup = section_data(snapshot, 'lineup') or {}
    if isinstance(lineup, dict):
        own_id = lineup.get('raw_payload', {}).get('teamId')
        if own_id is not None and str(own_id) != TEAM_ID:
            raise ValueError('Lineup is not team 6.')
        matchup_id = lineup.get('raw_payload', {}).get('matchup', {}).get('team1', {}).get('id')
        if matchup_id is not None and str(matchup_id) != TEAM_ID:
            raise ValueError('Matchup lineup is not team 6.')
    probability = (section_data(snapshot, 'matchup') or {}).get('win_probability')
    if probability is not None and (type(probability) not in (int, float) or not 0 <= probability <= 1):
        raise ValueError('Invalid source win probability.')
    return snapshot


def build_snapshot(bundle):
    if bundle.get('league_key') != LEAGUE_KEY or str(bundle.get('team_id')) != TEAM_ID:
        raise ValueError('Bundle identity mismatch.')
    sections = {name: unavailable('Not returned in this refresh.', 'not_requested') for name in SECTION_NAMES}
    ids = set()
    for source in bundle['calls']:
        if source['id'] not in sections or source['id'] == 'opponent' or source['id'] in ids:
            raise ValueError(f'Unexpected/duplicate source ID: {source["id"]}')
        ids.add(source['id'])
        sections[source['id']] = source_section(source)
    for name, reason in bundle.get('skipped', {}).items():
        if name in sections and name not in ids:
            sections[name] = unavailable(reason)
    sections['opponent'] = opponent_section(sections)
    warnings = [f'{name}: {warning}' for name, sec in sections.items() for warning in sec['warnings']]
    if (sections['lineup']['data'] or {}).get('week') is None:
        warnings.append('The current matchup/lineup week is not explicitly supplied; do not infer it from the calendar or schedule.')
    warnings.append('Waiver Finder is recommendation/candidate coverage, not the full available-player pool.')
    snapshot = {'schema_version': '0.1', 'snapshot_id': bundle.get('run_id') or uuid.uuid4().hex,
                'refresh_started_at': bundle['started_at'], 'refreshed_at': utc_now(),
                'league_key': LEAGUE_KEY, 'team_id': TEAM_ID,
                'refresh_status': 'partial' if any(s['status'] not in ('ok', 'empty') for s in sections.values()) else 'complete',
                'sections': sections, 'sources': bundle['calls'], 'warnings': warnings}
    return validate(snapshot)


def load_snapshot(path=None):
    path = Path(path) if path else BASE / 'data' / 'latest_snapshot.json'
    return validate(json.loads(path.read_text(encoding='utf-8-sig')))


def persist(snapshot, data_dir=None):
    validate(snapshot)
    folder = Path(data_dir) if data_dir else BASE / 'data'
    history = folder / 'snapshots'
    history.mkdir(parents=True, exist_ok=True)
    lock = folder / '.publish.lock'
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    temporary = folder / f'.latest-{uuid.uuid4().hex}.tmp'
    try:
        latest = folder / 'latest_snapshot.json'
        if latest.exists() and stamp(load_snapshot(latest)['refreshed_at']) > stamp(snapshot['refreshed_at']):
            raise ValueError('Refusing to replace latest with an older snapshot.')
        text = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        filename = stamp(snapshot['refreshed_at']).astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        destination = history / f'{filename}_{snapshot["snapshot_id"]}.json'
        # Exclusive creation: historical snapshots can never be overwritten.
        with destination.open('x', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, latest)
        return destination
    finally:
        temporary.unlink(missing_ok=True)
        os.close(lock_fd)
        lock.unlink()
