# Fantasy Football GM v0.5

Trades now compares both rosters, searches bounded packages, and verifies exact
offers against fresh FantasyPros Trade Analyzer evidence. See [TRADES.md](TRADES.md).
Streaming uses explicit NFL game dates to separate the playing slate from the
next/projection week. See [STREAMING.md](STREAMING.md) and [HANDOFF.md](HANDOFF.md).

The Streaming tab now provides D/ST and kicker HOLD/STREAM/STASH decisions,
confirmed availability, and upcoming matchup outlooks. See [STREAMING.md](STREAMING.md)
for the model, configuration, source refresh and audit contract.

Waivers has a roster-specific GM Waiver Board, owner-level FAAB history,
whole-dollar bid ranges, and an ordered manual Yahoo claim plan. See
[WAIVERS.md](WAIVERS.md) for the complete v0.3.1 methodology, thresholds, evidence
format, refresh workflow, and limitations. This Week keeps its v0.2 lineup engine
and consolidates scoring/data warnings into one expandable notice.

An independent in-season assistant under `tools/gm_assistant/`. It imports no
PatBot draft modules, draft profiles, or production configuration. Yahoo lineup,
waiver, trade, kicker, defense, and account actions are **manual by design**.
This tool reads FantasyPros through Codex, validates snapshots, and displays a
local Streamlit dashboard. Its independent weekly lineup model uses FantasyPros
as evidence; it has no Yahoo execution client.

## Weekly lineup engine (v0.2)

This Week leads with the GM Final Call, recommended lineup, actual start/sit
decisions and FantasyPros comparison. The optimizer enumerates every legal
assignment from the current starters and bench, collapsing identical RB/WR slot
permutations. FLEX placements remain distinct. IR and explicitly inactive
players are excluded unless already locked in a starting slot. Locked starters
stay in their exact slot; locked bench players cannot enter. Unknown locks are
flagged for manual verification. No incomplete lineup is recommended.

`scoring_rules.json` captures only the scoring portions of `config/league.yaml`:
0.25/completion, the custom yardage bonuses, kicker distance and missed-kick
rules, and defense rules. It neither imports nor modifies PatBot. This explicit
local configuration takes precedence over the incomplete FantasyPros settings
(which also disagree on sacks: FP 2, local config 1). Update this GM-local file
when verified Yahoo rules change. Its exact contents are included in every audit.

The scorer multiplies supplied component projections by those rules. Bonuses
require projected threshold event counts/probabilities; mean yards never create
bonus probabilities. Kicker distance/miss buckets and DST points-allowed
probabilities must be supplied. A generic `fumbles` field is not fumbles lost.
If any required category is missing, the **whole player** falls back to FP fantasy
points, with the calculable subtotal and all missing categories retained for
audit. It never adds a partial custom subtotal to a whole-player FP total.

Component projections join by player ID (or unique conservative name) only when
their week explicitly matches the matchup week. The latest development snapshot
does not identify its matchup week, while its separate projections say week 2;
therefore the current analysis uses matchup player-point fallbacks. These are
**not verified custom Yahoo totals**. Matchup positional ECR remains available.

All weights/thresholds are in `gm_lineup.py:MODEL`. Utility starts with projected
points. Expert support adds at most 0.15, positional ECR at most 0.10, and a
reported injury subtracts 0.25. These are heuristic decision weights, not newly
predicted points. At source win probability >=65%, strategy is Floor-Leaning;
<=35% is Ceiling-Leaning; otherwise Neutral. A favorite can gain up to 0.35 utility
from `(projected carries + receptions) / 25`, capped at one. An underdog can gain
up to 0.35 from the projected rushing/receiving TD share of points, capped at one.
These are workload and TD-dependence proxies, not numerical floor/ceiling forecasts.
Missing inputs contribute no adjustment; injuries never become an upside bonus.
SOS and role are displayed as context without numeric weight because their scale
or predictive effect is not established. News is displayed without NLP risk guesses.

Only alternatives within **1.0 total point** of the maximum are considered for
adjustments, and each exchanged player must be within **1.0 point** of a player
with overlapping eligibility. Thus strategy cannot hide a materially worse player
behind other gains. Equal utility favors retaining current starters. High
confidence requires a >1-point advantage with complete scoring; close choices are
Medium, and scoring fallbacks, missing comparisons or <0.25-point edges are Low.
Retained close decisions are conservatively Low. Confidence is not a probability.
Displayed win probability is FP's current-lineup estimate, never a fabricated GM
recalculation; the opponent total is likewise clearly labeled as FP's projection.

Run analysis separately or open/reload the dashboard (which analyzes and saves
the validated snapshot automatically):

```powershell
.\.venv\Scripts\python.exe tools\gm_assistant\gm_lineup.py
```

`data/latest_analysis.json` and immutable content-addressed `data/analyses/*.json`
hold timestamps, source snapshot hash/ID, engine hash, exact rules/weights, all
legal alternatives, current/maximum/recommended lineups, changes, confidence,
FP comparisons, decision inputs, missing inputs and warnings. Reproduce with
`analyze(snapshot, rules=a['scoring_rules'], model=a['model'],
timestamp=a['analysis_timestamp'])` using the matching saved snapshot and engine.
The validated source snapshot/schema and refresh architecture are unchanged.
Opponent details join only an explicitly matching opponent ID and unique player
identity; unsupported display values become a dash.

Files added: `gm_lineup.py`, `scoring_rules.json`, `test_lineup.py`.
Files updated: `dashboard.py`, `gm_views.py`, `test_gm.py`, this README.
No FAAB, trade, streaming model or historical owner behavior is introduced.

Run all existing and new GM tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools/gm_assistant -p 'test_*.py' -v
```

Authoritative league: `nfl~8922f34b-15ab-48d6-b51a-472098c3ae5a`; your team ID: `6`.
The other synced league is never queried. Account-wide `get_player_ownership` and
`get_leagues` are deliberately excluded because they can access other leagues.

## Launch on this machine

The existing repository `.venv` already provides Streamlit and jsonschema. It is
safe to reuse that interpreter without changing its dependencies or importing
PatBot. These batch launchers use a GM-local `.venv` first, then the repository
`.venv`; they never install or upgrade packages automatically.

```powershell
Set-Location 'C:\Users\patri\Documents\patbot'
.\tools\gm_assistant\RUN_GM.bat
```

Open `http://127.0.0.1:8502`. The app binds only to localhost, on a different port
from Streamlit's default. Tabs: This Week, My Roster, Waivers, Streaming, Trades,
League. **Reload local snapshot** rereads the file; it never triggers network calls.
No snapshot produces an honest empty shell, not demo rosters.

For a separate environment on another machine (Python 3.10+):

```powershell
Set-Location 'C:\Users\patri\Documents\patbot\tools\gm_assistant'
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\RUN_GM.bat
```

No PowerShell execution-policy change or activation is needed.

## Refresh through the authenticated Codex MCP

```powershell
Set-Location 'C:\Users\patri\Documents\patbot'
.\tools\gm_assistant\REFRESH_GM.bat
```

The wrapper checks the locally installed CLI help and enabled FantasyPros server,
creates a unique collection directory, and runs `codex exec` with
`--sandbox workspace-write`, `--cd`, `--output-last-message`, and a prompt on stdin.
It preserves the user's existing Codex authentication, model, MCP configuration,
and approval settings. It does not copy credentials, install MCP servers, or
bypass approvals/sandboxing. Codex performs the MCP calls; Python only builds
request plans, checks responses, and writes local files.

The installed CLI inspected during development was **0.153.4**. Its actual
`--help`, `exec --help`, and `mcp list --help` expose these flags, and its MCP list
reports an enabled `fantasypros` streamable-HTTP OAuth server. This establishes
CLI support/configuration, not a guarantee that future OAuth or approval requests
will succeed unattended. If an approval, expired session, or missing connector
blocks exec, use the interactive workflow below. No security settings need to be
weakened. CLI return code 0 alone is insufficient: the wrapper validates data and
checks that enrichment was planned before publishing.

### Interactive fallback: one prompt

In an authenticated Codex session opened on this repo, send:

> Refresh GM v0.2 using tools/gm_assistant/REFRESH.md and publish the validated snapshot.

Or launch that prompt directly from PowerShell:

```powershell
Set-Location 'C:\Users\patri\Documents\patbot'
codex.cmd "Refresh GM v0.2 using tools/gm_assistant/REFRESH.md and publish the validated snapshot."
```

`REFRESH.md` describes the exact repeatable protocol. Codex prepares a run,
explicitly resyncs the authoritative league, obtains settings and all 12 rosters,
then captures lineup, matchup, league analysis, start/sit, six position-specific
waiver calls, and DST/K schedules. The enrichment helper derives requests from
explicit returned scoring/week data: weekly/ROS ECR and projections for all six
positions, plus injuries/news for your roster and named waiver candidates.
No week, scoring, availability, slot, or injury status is guessed. Each exact
response is saved immediately with its tool arguments and UTC collection time.

The first development snapshot was collected through the authenticated Codex
session directly. The subprocess wrapper's CLI contract is verified locally;
an unattended child-Codex end-to-end run has not been used to validate OAuth.

## Files and integrity

- `snapshot.schema.json`: structured JSON Schema, generated from `gm_schema.py`.
- `data/latest_snapshot.json`: most recently published validated snapshot.
- `data/snapshots/<UTC timestamp>_<unique id>.json`: immutable historical snapshots.
- `data/incoming/<run id>/`: manifest, exact per-call results, optional Codex summary.
- `data/bootstrap_capture.json`: exact direct-session MCP evidence for the initial
  validated snapshot, if present. A capture bundle itself is not a published snapshot.

Snapshots have version, collection-start and publication timestamps, league/team identity,
refresh status, named `sections`, `sources`, and warnings. Each section contains
`status`, `data`, `source_ids`, and warnings. Statuses distinguish `ok`, a confirmed
`empty` result, `unavailable`, `error`, and `not_requested`. Unknown fields within a
successful provider response remain null or absent; absence is not an empty list.
Nested data retains FantasyPros' native shape instead of inventing a new provider
contract. `sources` retains the exact JSON payload/envelope and requested filters
for audit; successful displayed sections are checked against those sources.

Required core data: confirmed resync, league settings identifying 12 teams, all
12 distinct team rosters, and your explicitly selected team-6 roster. A failed core
refresh is rejected and latest stays intact. Optional failures publish a **partial**
snapshot with explicit warnings; old data is never silently carried forward.
Collections spanning more than two hours are rejected to prevent mixing sessions.
The current opponent is resolved only from the returned team-6 matchup identity
and matched to the explicit all-rosters team ID. Missing matchup week stays null
even when the schedule endpoint supplies a current NFL week for projections.

Historical files use exclusive creation and are never overwritten. Latest is
atomically replaced only after validation and historical persistence. A publication
lock prevents simultaneous writers; an older snapshot cannot replace a newer one.
If a process is killed during publication, a `.publish.lock` can remain: verify no
refresh is running before manually removing that GM-local lock and retrying.
The dashboard displays the actual per-source collection window separately from
publication time and warns when the oldest source is over 24 hours old.

All private data and local environments are git-ignored within this folder. No
credentials or authentication state are stored here by the GM code. Codex uses
its own existing session storage/settings.

## FantasyPros snapshot limitations

- Waiver Finder returns recommendations/candidates, **not all free agents**. All
  candidate availability remains unverified unless the returned league response
  explicitly says available. Absence from rosters is not used to infer availability.
- Rankings and projections are global references, not availability checks. The
  workflow requests up to 100 rows per position/timeframe and retains returned
  counts and request limits. It does not claim complete player-universe coverage.
- Projections expose STD/HALF/PPR point fields. The dashboard selects the explicit
  league format; it does not re-score custom league settings or sum a new matchup
  forecast. Current matchup scores come directly from the matchup tool.
- `matchup_analyzer` has no arbitrary week selector. Lineup/matchup week may be null.
- ROS rankings may exist while ROS projections are explicitly unavailable; draft
  projections are not substituted. Premium/no-data responses remain unavailable,
  with required disclosures retained and shown.
- Missing injury reports are not proof of health. IR is not inferred from PUP or
  another injury label. Raw news is displayed without generating medical conclusions.
- League analysis may omit playoff odds; no odds or standings are invented.
- Trade suggestions require both local roster-fit gates and exact fresh analyzer
  verification. Owner aliases identify people only; no trading tendencies are inferred.

## Tests

From the repository root, using the already installed environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools/gm_assistant -p 'test_*.py' -v
```

Or from this folder with its own environment:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_gm test_lineup
```

Tests use synthetic snapshots and temporary directories, check validation,
identity, source consistency, unavailable-vs-empty handling, immutable history,
safe view transformations and Streamlit's simulated AppTest. They make no network
calls and import no PatBot code. Regenerate the schema after changing its definition:
`python gm_schema.py` (from this folder).

Official references used to verify the workflow:
[Codex noninteractive mode](https://learn.chatgpt.com/docs/non-interactive-mode),
[Codex MCP](https://learn.chatgpt.com/docs/extend/mcp?surface=cli), and
[Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
