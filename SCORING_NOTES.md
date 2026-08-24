# PatBot League Scoring Notes

## Kicker scoring

- FG 0–19 yards: +3
- FG 20–29 yards: +3
- FG 30–39 yards: +3
- FG 40–49 yards: +4
- FG 50+ yards: +5
- Missed FG 0–19: -1
- Missed FG 20–29: -1
- Missed FG 30–39: -1
- Missed FG 40–49: -1
- PAT made: +1
- PAT missed: -1

No missed-50+-yard-FG value was supplied, so PatBot intentionally does not assume one.

## Defense / Special Teams scoring

- Sack: +1
- Interception: +2
- Fumble recovery: +2
- Touchdown: +6
- Safety: +2
- Blocked kick: +2
- Extra point returned: +2

Points allowed:
- 0: +10
- 1–6: +7
- 7–13: +4
- 14–20: +1
- 21–27: 0
- 28–34: -1
- 35+: -4

## Implementation status

These rules are now stored in `config/league.yaml`.

PatBot v0.2 still uses provider-projected K/DEF fantasy points in the live projection feed.
A future scoring pass will calculate K and D/ST directly from raw projected stats where
the data source exposes enough category detail.
