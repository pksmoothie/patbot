"""Codex orchestration and offline snapshot publishing; never talks to FantasyPros directly."""
import argparse
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from gm_schema import BASE, LEAGUE_KEY, TEAM_ID, POSITIONS
from gm_snapshot import build_snapshot, decode, persist, stamp, utc_now
from gm_week import establish_week


def request(identifier, tool, **arguments):
    return {'id': identifier, 'tool': f'mcp__fantasypros__{tool}', 'arguments': arguments}


def core_requests():
    scoped = {'league_key': LEAGUE_KEY, 'sport': 'nfl'}
    return [
        request('resync', 'resync_league', **scoped),
        request('settings', 'get_league_settings', **scoped),
        request('all_rosters', 'get_roster', all_teams=True, **scoped),
        request('my_roster', 'get_roster', team_id=TEAM_ID, **scoped),
        request('lineup', 'get_current_starters', **scoped),
        request('matchup', 'matchup_analyzer', league_key=LEAGUE_KEY),
        request('league_analysis', 'league_analyzer', **scoped),
        request('start_sit', 'start_sit_assistant', **scoped),
        *(request(f'schedule_{p}', 'get_schedule_and_strength', position=p) for p in ('DST', 'K')),
        *(request(f'waiver_{p}', 'waiver_finder', league_key=LEAGUE_KEY, position=p) for p in POSITIONS),
    ]


def json_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare_run():
    run_id = 'gm_' + uuid.uuid4().hex
    folder = BASE / 'data' / 'incoming' / run_id
    folder.mkdir(parents=True)
    json_write(folder / 'manifest.json', {'run_id': run_id, 'league_key': LEAGUE_KEY,
        'team_id': TEAM_ID, 'started_at': utc_now(), 'requests': core_requests(), 'skipped': {}})
    return folder


def checked_run(path):
    folder = Path(path).resolve()
    if not folder.is_relative_to((BASE / 'data' / 'incoming').resolve()):
        raise ValueError('Run directory must be inside gm_assistant/data/incoming/.')
    return folder


def bundle_from_run(folder):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    calls = []
    for item in manifest['requests']:
        path = folder / f'{item["id"]}.json'
        if path.exists():
            result = json.loads(path.read_text(encoding='utf-8-sig'))
            stamp(result['fetched_at'])
            calls.append({**item, 'fetched_at': result['fetched_at'], 'raw': result['raw']})
    return {**manifest, 'calls': calls}


def enrichment_requests(bundle, now=None, calendar=None):
    data = {}
    for call in bundle['calls']:
        try:
            data[call['id']] = decode(call['raw'])
        except (ValueError, TypeError, KeyError):
            data[call['id']] = None
    settings = data.get('settings') or {}
    scoring = settings.get('scoring')
    week_info = establish_week([dict(id=name,data=data[name]) for name in ('schedule_DST','schedule_K')
                               if isinstance(data.get(name),dict)],now=now,calendar=calendar)
    week = week_info['week']
    tasks, skipped = [], {}
    for timeline in ('WEEKLY', 'ROS'):
        for position in POSITIONS:
            identifier = f'ranking_{timeline}_{position}'
            if scoring in ('STD', 'HALF', 'PPR'):
                tasks.append(request(identifier, 'get_ecr', sport='NFL', position=position,
                                     ranking_type=timeline, scoring=scoring, limit=100))
            else:
                skipped[identifier] = 'League scoring not explicitly recognized; no scoring assumption.'
    for timeline in ('weekly', 'ros'):
        for position in POSITIONS:
            identifier = f'projection_{timeline}_{position}'
            if timeline == 'weekly' and (week is None or not 1 <= week <= 18):
                skipped[identifier] = 'An unambiguous current NFL week was not returned by the schedule sources.'
                continue
            tasks.append(request(identifier, 'get_projections', sport='nfl', position=position,
                                 projection_type=timeline, limit=100, **({'week': week, 'season':week_info['season']} if timeline == 'weekly' else {})))
    names = set()
    for position, players in (data.get('my_roster') or {}).get('roster', {}).items():
        if position != 'DST':
            names.update(p for p in players if isinstance(p, str))
    for position in POSITIONS:
        if position == 'DST':
            continue
        waiver = data.get(f'waiver_{position}') or {}
        for candidate in waiver.get('top_starts_this_week', {}).get(position, []):
            if name := candidate.get('full') or candidate.get('player_name'):
                names.add(name)
        for candidate in waiver.get('recommended_adds', []):
            if isinstance(candidate, dict) and (name := candidate.get('full') or candidate.get('player_name')):
                names.add(name)
            if isinstance(candidate, dict) and (name := candidate.get('Player to Add', {}).get('player_name')):
                names.add(name)
    if names:
        tasks.append(request('injuries', 'injury_status', player_names=', '.join(sorted(names))))
    else:
        skipped['injuries'] = 'No explicit relevant player names available.'
    return tasks, skipped


def enrich(folder):
    bundle = bundle_from_run(folder)
    # The same core validation as publication; optional sections may still be missing.
    build_snapshot(bundle)
    tasks, skipped = enrichment_requests(bundle)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    manifest['requests'] = core_requests() + tasks
    manifest['skipped'] = skipped
    json_write(folder / 'manifest.json', manifest)
    print(json.dumps({'requests': tasks, 'skipped': skipped}, indent=2))


def codex_prompt(folder):
    return f'''Perform a GM v0.1 FantasyPros refresh. Read REFRESH.md and follow it exactly.
Existing prepared run directory: {folder}
Python executable for helper commands: {sys.executable}
Use ONLY the existing authenticated FantasyPros MCP. League {LEAGUE_KEY}, team 6 only.
Write only files inside this run directory. Do not edit implementation or publish snapshots yourself.
Run the resync request first and confirm success before collecting anything else.
Execute the manifest core requests, save each exact MCP result with fetched_at,
then run the helper: "{sys.executable}" refresh.py enrich "{folder}"
Execute the newly generated enrichment requests and save those exact results too.
Do not call get_leagues, get_player_ownership, or active-league tools: they can span other leagues.
Never access Yahoo or make any lineup/waiver/trade/account changes.
If a capability fails, preserve the actual failure. Never invent or summarize raw data.
Finish with a short collection status report; the parent command validates and publishes.
'''


def run_codex():
    executable = shutil.which('codex.cmd') or shutil.which('codex')
    if not executable:
        raise ValueError('Codex CLI not found. Use the one-prompt interactive workflow in README.md.')
    # Fail closed if this installation does not support the flags or MCP connection.
    help_text = subprocess.run([executable, 'exec', '--help'], capture_output=True, text=True, check=True).stdout
    for flag in ('--sandbox', '--cd', '--output-last-message'):
        if flag not in help_text:
            raise ValueError(f'Installed Codex lacks {flag}; use the documented interactive refresh.')
    listed = subprocess.run([executable, 'mcp', 'list', '--json'], capture_output=True, text=True, check=True)
    servers = json.loads(listed.stdout)
    if not any(s.get('name') == 'fantasypros' and s.get('enabled') for s in servers):
        raise ValueError('This CLI has no enabled fantasypros MCP. Use a Codex session with the authenticated connector.')
    folder = prepare_run()
    prompt = codex_prompt(folder)
    (folder / 'prompt.txt').write_text(prompt, encoding='utf-8')
    print(f'Collecting through Codex. Run evidence: {folder}', flush=True)
    completed = subprocess.run([executable, 'exec', '--sandbox', 'workspace-write', '--cd', str(BASE),
                                '--output-last-message', str(folder / 'codex_summary.txt'), '-'],
                               input=prompt, text=True, encoding='utf-8', cwd=BASE)
    if completed.returncode:
        raise ValueError(f'Codex exited {completed.returncode}; latest was not changed. Run evidence retained in {folder}.')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if not any(r['id'].startswith('projection_ros_') for r in manifest['requests']):
        raise ValueError('Codex did not complete the enrichment phase; latest was not changed.')
    snapshot = build_snapshot(bundle_from_run(folder))
    destination = persist(snapshot)
    print(f'Published {snapshot["refresh_status"]} snapshot: {destination}')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest='command', required=True)
    sub.add_parser('run', help='Run a fresh noninteractive Codex MCP collection, validate, and publish')
    sub.add_parser('prepare', help='Prepare an interactive Codex collection run')
    for name in ('enrich', 'publish'):
        sub.add_parser(name).add_argument('run_dir')
    sub.add_parser('import-bundle', help='Validate/publish an exact MCP capture bundle from a Codex session').add_argument('bundle')
    args = cli.parse_args()
    try:
        if args.command == 'run':
            run_codex()
        elif args.command == 'prepare':
            folder = prepare_run()
            print(folder)
            print(codex_prompt(folder))
        elif args.command == 'enrich':
            enrich(checked_run(args.run_dir))
        else:
            bundle = (json.loads(Path(args.bundle).read_text(encoding='utf-8-sig')) if args.command == 'import-bundle'
                      else bundle_from_run(checked_run(args.run_dir)))
            snapshot = build_snapshot(bundle)
            print(f'Published {snapshot["refresh_status"]}: {persist(snapshot)}')
        return 0
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
        print(f'Refresh stopped: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
