"""Owner-keyed, descriptive FAAB priors; season totals are not bid observations."""
import json
import statistics
from pathlib import Path

HISTORY = Path(__file__).parent / 'config' / 'faab_history.json'
THRESHOLDS = dict(aggressive_spend=75, conservative_spend=35,
                  high_activity=25, low_activity=15, minimum_seasons=3,
                  consistent_range=20, variable_range=50, max_prior_fraction=.05)


def load_history(path=HISTORY):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    seen, aliases = set(), set()
    for owner in data['owners']:
        key = owner['owner'].casefold()
        if key in seen:
            raise ValueError('Duplicate canonical owner')
        seen.add(key)
        years = set()
        for row in owner['seasons']:
            year = row['season']
            if year in years:
                raise ValueError('Duplicate owner season')
            years.add(year)
            if not 0 <= row['remaining'] <= 100 or row['spent'] != 100-row['remaining'] or row['moves'] < 0:
                raise ValueError('Invalid season totals')
            for alias in row['aliases']:
                alias_key = (year, alias.casefold())
                if alias_key in aliases:
                    raise ValueError('Ambiguous owner alias')
                aliases.add(alias_key)
    return data


def canonical_owner(history, value, season=None):
    hits = {o['owner'] for o in history['owners'] if o['owner'].casefold() == value.casefold()
            or any(value.casefold() == a.casefold() for s in o['seasons']
                   if season is None or s['season'] == season for a in s['aliases'])}
    if len(hits) != 1:
        raise ValueError('Unknown or ambiguous owner identity')
    return hits.pop()


def owner_priors(history):
    result = []
    for owner in history['owners']:
        rows = owner['seasons']
        spend = [r['spent'] for r in rows]
        avg = statistics.mean(spend) if spend else 0
        moves = statistics.mean(r['moves'] for r in rows) if rows else 0
        spread = max(spend)-min(spend) if spend else 0
        label = ('Insufficient history' if len(rows) < 3 else 'Aggressive spender' if avg >= 75
                 else 'Conservative spender' if avg <= 35 else 'Moderate spender')
        result.append(dict(owner=owner['owner'], seasons=len(rows), average_spent=round(avg,2),
            median_spent=statistics.median(spend) if spend else None,
            minimum_spent=min(spend) if spend else None, maximum_spent=max(spend) if spend else None,
            spend_range=spread, spend_stddev=round(statistics.pstdev(spend),2) if spend else None,
            consistency='Insufficient history' if len(rows)<3 else 'Consistent' if spread<=20 else 'Variable' if spread>50 else 'Mixed',
            average_moves=round(moves,2), spending=label,
            activity='High activity' if moves>=25 else 'Low activity' if moves<15 else 'Moderate activity'))
    return result


def pressure(priors):
    peers = [p for p in priors if p['owner'] != 'Pat' and p['spending'] != 'Insufficient history']
    mean = statistics.mean(p['average_spent'] for p in peers) if peers else 50
    return dict(fraction=max(-.05,min(.05,(mean-50)/500)),
        label='Elevated' if mean>=65 else 'Low' if mean<=35 else 'Moderate',
        possible_aggressive_competitors=[p['owner'] for p in peers if p['spending']=='Aggressive spender'],
        historical_budget_preservers=[p['owner'] for p in peers if p['spending']=='Conservative spender'],
        active_low_spenders=[p['owner'] for p in peers if p['activity']=='High activity' and p['spending']=='Conservative spender'],
        caveat='Season totals cannot identify bids, losing bids, acquisitions, timing, or claim versus free-agent activity. Competitors and current budgets are unconfirmed.')
