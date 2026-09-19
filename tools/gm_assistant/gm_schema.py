"""GM v0.1 schema; no imports from PatBot."""
import json
from pathlib import Path

LEAGUE_KEY = 'nfl~8922f34b-15ab-48d6-b51a-472098c3ae5a'
TEAM_ID = '6'
POSITIONS = ('QB', 'RB', 'WR', 'TE', 'DST', 'K')
BASE = Path(__file__).resolve().parent
SECTION_NAMES = (
    'resync', 'settings', 'all_rosters', 'my_roster', 'lineup', 'opponent',
    'matchup', 'league_analysis', 'start_sit', 'injuries', 'schedule_DST', 'schedule_K',
    *(f'waiver_{p}' for p in POSITIONS),
    *(f'ranking_{t}_{p}' for t in ('WEEKLY', 'ROS') for p in POSITIONS),
    *(f'projection_{t}_{p}' for t in ('weekly', 'ros') for p in POSITIONS),
)
ENVELOPE = {
    'type': 'object', 'additionalProperties': False,
    'required': ['status', 'data', 'source_ids', 'warnings'],
    'properties': {
        'status': {'enum': ['ok', 'empty', 'unavailable', 'error', 'not_requested']},
        'data': {},
        'source_ids': {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True},
        'warnings': {'type': 'array', 'items': {'type': 'string'}},
    },
    'allOf': [
        {'if': {'properties': {'status': {'enum': ['unavailable', 'error', 'not_requested']}}},
         'then': {'properties': {'data': {'type': 'null'}}}},
        {'if': {'properties': {'status': {'const': 'ok'}}},
         'then': {'properties': {'data': {'not': {'type': 'null'}}}}},
        {'if': {'properties': {'status': {'const': 'empty'}}},
         'then': {'properties': {'data': {'const': []}}}},
    ],
}
SCHEMA = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'title': 'FantasyPros GM snapshot v0.1', 'type': 'object', 'additionalProperties': False,
    'required': ['schema_version', 'snapshot_id', 'refresh_started_at', 'refreshed_at',
                 'league_key', 'team_id', 'refresh_status', 'sections', 'sources', 'warnings'],
    'properties': {
        'schema_version': {'const': '0.1'}, 'snapshot_id': {'type': 'string', 'pattern': '^[a-zA-Z0-9_-]+$'},
        'refresh_started_at': {'type': 'string', 'format': 'date-time'},
        'refreshed_at': {'type': 'string', 'format': 'date-time'},
        'league_key': {'const': LEAGUE_KEY}, 'team_id': {'const': TEAM_ID},
        'refresh_status': {'enum': ['complete', 'partial']},
        'sections': {'type': 'object', 'additionalProperties': False,
                     'required': list(SECTION_NAMES),
                     'properties': {name: {'$ref': '#/$defs/section'} for name in SECTION_NAMES}},
        'sources': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['id', 'tool', 'arguments', 'fetched_at', 'raw'],
            'properties': {
                'id': {'type': 'string'}, 'tool': {'type': 'string', 'pattern': '^mcp__fantasypros__'},
                'arguments': {'type': 'object'}, 'fetched_at': {'type': 'string', 'format': 'date-time'},
                'raw': {},
            }}},
        'warnings': {'type': 'array', 'items': {'type': 'string'}},
    },
    '$defs': {'section': ENVELOPE},
}

if __name__ == '__main__':
    (BASE / 'snapshot.schema.json').write_text(json.dumps(SCHEMA, indent=2) + '\n', encoding='utf-8')
