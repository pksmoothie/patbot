# GM v0.4 D/ST and kicker streaming

The Streaming tab owns D/ST and kicker decisions. Yahoo actions remain manual.
The v0.2 lineup engine, v0.3.1 waiver/FAAB engine, draft engine and Yahoo history
exporter are unchanged. Shared scoring and availability helpers are reused.

Run `python tools/gm_assistant/gm_streaming.py` using the repository virtualenv
to regenerate `data/latest_streaming_analysis.json`. The dashboard also regenerates
it and routes D/ST/K waiver rows to See Streaming. The original generic waiver
engine remains callable independently; its D/ST/K output is superseded in the UI.

## Sources and freshness

`gm_week.py` establishes the slate from explicit, timezone-aware NFL game dates
in `nfl_schedule.json`. Each complete week includes games, season, source URL and
verification time. Refresh this evidence within seven days, extending the horizon
to include the preceding/current and next slate; an exhausted, stale, conflicting,
incomplete, or missing capture disables analysis. No week is invented from a
provider counter, a weekday alone, or a fixed season-start formula.

The first kickoff starts `playing_week`. The slate remains current until 06:00
America/New_York on the day after its final scheduled game date, keeping Monday
night games in the same week across UTC midnight. After that, `week` selects the
next dated slate for preparation and `playing_week` is null until kickoff.
`next_week`, `phase`, projection weeks and provider counters are recorded separately.
This is a conservative scheduling boundary, not a live final-whistle feed;
rescheduled games require updated dates. January games retain their explicit season.

On September 9-10, 2026, the playing slate is Week 1 despite the old FantasyPros
Week 2 counter. Week 1 runs September 9-14 and Week 2 starts September 17.
The current capture contains complete Weeks 1-2. A future-only Week 2 capture
cannot establish that Week 1 is over. The null matchup/lineup week stays null.

The validated snapshot supplies current roster, all league rosters, Waiver Finder,
weekly projections, ECR and injuries. Candidate expansion includes weekly ranked
and projected D/STs and kickers. Current rostered options always appear, separately
labeled. Only explicit free-agent ownership in the exact league permits an add.
Roster conflicts override positive ownership. The engine reuses the waiver
engine's unique league-name/key mapping and 24-hour ownership expiry.

`data/streaming_sources.json` binds raw schedule calls to `snapshot_id`. Each call
contains `kind: schedule`, exact `arguments`, UTC response-completion `fetched_at`,
and unmodified `raw` MCP response. Sidecar calls must follow the core refresh
start and be no more than 24 hours old. Stale snapshots and incomplete rosters
disable transactions. Wrong-snapshot schedule sidecars are ignored.

`data/streaming_ownership.json` uses the existing waiver ownership format:
`league_key`, raw `league_lookup`, and `calls` containing `player_name`, `tool`,
`fetched_at`, and `raw`. Preserve all evidence, including unavailable results.

For schedule collection, request **one team per call**, no position filter,
`week_start=current_week`, `week_end=min(18,current_week+3)`. This returns opponents
and both positional ratings under `Weekly Matchups`. Multi-team calls can omit
opponents. Multiweek aggregate SOS is diagnostic only; never assign it to each
week. Single-week window SOS is also accepted. The source explicitly states that
high stars mean easier matchups. Ratings are compared at one decimal (common
precision across provider response formats); conflicting opponents/ratings make
the affected week unavailable. BYE is explicit, never inferred from null ratings.

## Model and decisions

All thresholds and weights live in `streaming_config.json`.

Current score = projected points + weekly matchup adjustment + aligned ECR term.
D/ST matchup adjustment is `1.0 * (stars - 3)`; K is `0.35 * (stars - 3)`.
This makes opponent matchup quality a meaningful D/ST input without fabricating
opponent statistics. A dated ECR term is bounded to +/-0.5. Current ECR responses
omit week, so they remain visible context and receive no numeric weight.

Current projections must explicitly match the authoritative week and season. Both positions
reuse the validated custom scorer. Kicker scoring is 3/3/3/4/5 by FG distance,
-1 for missed FG below 50 yards, +1 for PAT and -1 for missed PAT. Missing FG
distance/miss buckets trigger the whole-player FantasyPros point fallback, while
calculable components remain in the audit. FG totals do not imply distances, and
attempts minus makes do not identify the penalty bucket. D/ST missing event or
points-allowed probabilities likewise uses the whole-player point fallback.

Future scores are **schedule outlook utilities, not point forecasts**. D/ST uses
`6 + 1.5 * (stars - 3)`; K uses `7 + 0.75 * (stars - 3)`. These baselines anchor
the heuristic scale and do not extrapolate this week's projected performance.
Explicit byes score zero. Missing matchup ratings/opponents make future scores
unavailable. Current/+1/+2 composite weights are 60/25/15%; weights are configurable
and must sum to one. Missing weeks do not get silently reweighted. Week +3 is
displayed for additional planning but does not enter that composite.

A swap normally requires >=1.25 current-score improvement and >=1 projected
point. D/ST can also qualify with >=1-star matchup improvement while sacrificing
no more than 0.5 projected points. The composite must improve >=0.5 when both
composites are available; a >=2.5 projected-point edge may override a worse future
schedule, with explanation. Incomplete future data permits a supported current
swap at Low confidence. No 0.1/0.2-point-only churn. Available options are ordered
by composite, then current score. Confirmed injured options are excluded.

Swaps always drop the same position. K never uses a bench spot for a second
kicker. No current option or comparable current data means NO ACTION. An explicit
locked current option prevents a transaction; unknown locks must be checked in
Yahoo manually. Strong current needs take precedence over advance stashes.

Early D/ST targets require next-week stars >=4.5, weighted future stars >=4.2,
and future outlook improvement >=2 over the current defense. Future weights are
5/8 and 3/8, matching the composite's remaining 40%. A second defense is unusual:
the current defense must be active, fewer than two defenses may be rostered, and
the existing waiver hierarchy must supply an unlocked, nonstarting Immediate
drop or Replaceable player. Protected GM starters, useful depth and upside
stashes are excluded. A drop reserved in the waiver claim plan is excluded.
Net advance value `0.4 * future_edge - 0.25 * bench_utility` must reach 1.0.
Otherwise the target is only flagged, without a recommended transaction.
An explicitly noncomparable waiver-week utility cannot fund a stash: its zero
may represent unavailable inputs. Scheduled kickoff locks also prevent swaps
involving already-started defenses/kickers. Players without a known opponent
cannot be added from projection totals alone.

HOLD means no supported transaction clears the gates; it is not a season-long
endorsement. Confidence is at most Medium because utilities are uncalibrated and
fallback projections may differ from custom league scoring. Reevaluate after
any acquisition or updated injury/lock information.

## Dashboard and audit

GM Streaming Call leads with action, add/drop, projected edge and composite.
Rankings show only current options and confirmed available alternatives. Every
week's opponents/outlooks, source projections, ECR, SOS, scoring components,
availability evidence, warnings, configuration, and recommendation are retained
in the audit. Unconfirmed/owned references remain collapsed in diagnostics.
Waivers references these same streaming calls instead of issuing competing D/ST/K
claims or NO CLAIM labels. Skill-position bids, owner history, and their claim
sequence are not recalculated by Streaming.

No betting lines, implied totals, unsourced sacks/turnovers, weather, historical
transaction scraping, trade engine, or Yahoo execution is introduced.

## Validation

Run `.\\.venv\\Scripts\\python.exe -m unittest discover -s tools/gm_assistant -p 'test_*.py'`.
Tests cover explicit weeks, ownership, roster inclusion, scoring fallback,
matchups/byes, composite weights, churn, stash cost/protections, kicker swaps,
waiver routing and dashboard rendering.
