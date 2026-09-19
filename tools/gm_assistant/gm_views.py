"""Pure display transformations. Missing fields stay missing."""
import json
from gm_schema import POSITIONS
from gm_snapshot import section_data


def payload(snapshot, name):
    if name not in snapshot.get('sections', {}):
        return {}
    data = section_data(snapshot, name)
    return data if isinstance(data, dict) else {}


def player_row(player):
    return {
        'Player': player.get('player_name') or player.get('full') or player.get('name'),
        'NFL team': player.get('player_team') or player.get('real_team') or player.get('team'),
        'Position': player.get('player_pos') or player.get('real_position'),
        'Slot': player.get('slot') or player.get('position'),
        'Injury status': player.get('injuryStatus') or player.get('injury_status'),
        'Projected points': player.get('original_proj'),
    }


def lineup_rows(snapshot, group='starters'):
    lineup = payload(snapshot, 'lineup')
    raw_players = lineup.get('raw_payload', {}).get('matchup', {}).get('team1', {}).get(group, [])
    indexed = {str(p.get('fpId')): p for p in raw_players if p.get('fpId') is not None}
    injuries = {p.get('name'): p.get('status') for p in payload(snapshot, 'injuries').get('practice_details', [])}
    result = []
    for p in lineup.get(group, []) or []:
        merged = {**indexed.get(str(p.get('fp_id')), {}), **p}
        if injuries.get(merged.get('player_name')):
            merged['injury_status'] = injuries[merged['player_name']]
        result.append(player_row(merged))
    # Empty slot warnings are explicit source data, not deductions from league settings.
    if group == 'starters':
        for p in payload(snapshot, 'matchup').get('user_lineup', {}).get('starters', []):
            if p.get('warning') == 'no player currently in position slot':
                result.append({'Player': '(empty slot)', 'Slot': p.get('position'), 'NFL team': None,
                               'Position': None, 'Injury status': None, 'Projected points': None})
    return result


def roster_rows(roster, details=None):
    indexed = {p['Player']: p for p in details or [] if p.get('Player')}
    rows = []
    for position, players in (roster or {}).items():
        for player in players:
            if isinstance(player, str):
                rows.append({'Player': player, 'Position': position if position in POSITIONS else None,
                             'NFL team': None, 'Slot': None, 'Injury status': None,
                             **indexed.get(player, {})})
    return rows


def opponent_rows(snapshot):
    # Import here to avoid a module cycle. Shared joins reject conflicting IDs
    # and ambiguous names; roster names can join unique source identities.
    from gm_lineup import join
    raw = payload(snapshot, 'lineup').get('raw_payload', {}).get('matchup', {}).get('team2', {})
    opponent = payload(snapshot, 'opponent')
    if str(raw.get('id')) != str(opponent.get('team_id')):
        raw = {}
    raw_players = sum([raw.get(g, []) or [] for g in ('starters', 'bench', 'ir')], [])
    reference = []
    for pos in POSITIONS:
        reference.extend(payload(snapshot, 'projection_weekly_' + pos).get('projections', []) or [])
    rows = []
    for row in roster_rows(opponent.get('roster')):
        probe = {'name': row['Player']}
        detail = join(probe, raw_players)
        ref = join(detail or probe, reference)
        injury = join(detail or probe, payload(snapshot, 'injuries').get('practice_details', []) or [])
        sourced = player_row(detail)
        sourced['NFL team'] = sourced['NFL team'] or ref.get('team')
        sourced['Injury status'] = sourced['Injury status'] or injury.get('status')
        row.update({k: v for k, v in sourced.items() if v is not None})
        rows.append({k: ('—' if v is None or v == '' else v) for k, v in row.items()})
    return rows


def matchup_metrics(snapshot):
    matchup = payload(snapshot, 'matchup')
    opponent = payload(snapshot, 'opponent')
    return {'opponent': opponent.get('team_name'),
            'my_projection': matchup.get('user_lineup', {}).get('current_projection'),
            'opponent_projection': matchup.get('opponent_lineup', {}).get('current_projection'),
            'win_probability': matchup.get('win_probability'),
            'week': payload(snapshot, 'lineup').get('week')}


def obvious_warnings(snapshot):
    warnings = []
    for player in lineup_rows(snapshot):
        if player['Player'] == '(empty slot)':
            warnings.append(f'Empty starting slot: {player["Slot"]}')
        if player.get('Injury status'):
            warnings.append(f'Starter {player["Player"]}: {player["Injury status"]}')
    return warnings


def injury_rows(snapshot):
    return [{'Player': p.get('name'), 'Status': p.get('status'),
             'Injury': p.get('injury_type'), 'Updated (source)': p.get('injury_update_date')}
            for p in payload(snapshot, 'injuries').get('practice_details', [])]


def candidates(snapshot, position):
    data = payload(snapshot, f'waiver_{position}')
    rows = []
    for item in data.get('recommended_adds', []):
        if not isinstance(item, dict):
            continue
        add = item.get('Player to Add', {})
        drop = item.get('Player to Drop', {})
        rows.append({'Player': add.get('player_name') or item.get('player_name') or item.get('full'),
                     'Suggested drop': drop.get('player_name'), 'FAB bid': item.get('FAB Bid'),
                     'Priority value': item.get('priority_value'), 'Weekly ECR': None,
                     'Coverage': 'FantasyPros recommended add',
                     'Availability': 'Not independently established',
                     'Reasons': ', '.join(item.get('Reasons', []))})
    for item in data.get('top_starts_this_week', {}).get(position, []):
        explicit = item.get('available') is True or item.get('is_available') is True
        rows.append({'Player': item.get('full') or item.get('player_name'),
                     'Suggested drop': None, 'FAB bid': None, 'Priority value': None,
                     'Weekly ECR': item.get('ecr'), 'Coverage': 'FantasyPros weekly candidate',
                     'Availability': 'Explicitly available in league response' if explicit else 'Not independently established',
                     'Reasons': None})
    return rows


def ranking_rows(snapshot, timeline, position):
    return payload(snapshot, f'ranking_{timeline}_{position}').get('rankings', [])


def projection_rows(snapshot, timeline, position):
    data = payload(snapshot, f'projection_{timeline}_{position}')
    scoring = payload(snapshot, 'settings').get('scoring')
    points_key = {'PPR': 'points_ppr', 'HALF': 'points_half', 'STD': 'points'}.get(scoring)
    return [{'Player': p.get('name'), 'NFL team': p.get('team'), 'Position': p.get('position'),
             'Season (source)': data.get('season'), 'Week (source)': data.get('week'),
             f'Points ({scoring or "unknown scoring"})': p.get('stats', {}).get(points_key) if points_key else None}
            for p in data.get('projections', [])]


def schedule_rows(snapshot, position):
    data = payload(snapshot, f'schedule_{position}')
    schedules = data.get('easiest schedules', {}).get('schedules', [])
    return [{'NFL team': item.get('team'), **item.get('Strength of Schedule', {}),
             'Matchups': json.dumps(item.get('Matchups', []), ensure_ascii=False)} for item in schedules]


def source_rows(snapshot):
    sources = {s['id']: s for s in snapshot['sources']}
    return [{'Section': name, 'Status': sec['status'],
             'Fetched at': ', '.join(sources[i]['fetched_at'] for i in sec['source_ids']),
             'Warnings': '; '.join(sec['warnings'])} for name, sec in snapshot['sections'].items()]
