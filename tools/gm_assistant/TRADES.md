# GM v0.5 Trade Engine

Run `python tools/gm_assistant/gm_trades.py` in the repository virtualenv.
The dashboard Trades tab runs the same engine and saves `data/latest_trade_analysis.json`.
All offers are manual alternatives. Recompute after any transaction or kickoff.

## Comparison and price

All 12 rosters are required. Each before/after lineup uses the unchanged v0.2
optimizer on projections explicitly matching the dated NFL schedule week and
season. Opponent starting assignments are modeled. Captured own starter locks
are retained; packages exclude players whose dated kickoff has passed. The
undated validated lineup remains a separate reference, not a mixed-week baseline.
Every team must have a fully projected legal baseline before approval is enabled.

The engine separately reports weekly lineup change, ROS starting utility, bench
insurance, asset exchange, positional depth, core protection and counterparty fit.
ROS starting assignments and bench insurance are now derived solely from ROS
rankings. Changing weekly points, locks or starter assignments cannot change those
long-term values. Rank coverage and missing projection/role/bye context are audited
separately. Ranking-only offers with incomplete context have Low confidence even
when verified; weekly scoring never fills missing ROS value.
Missing skill-player ROS values prevent approval; missing inputs are not free assets.
ROS asset values are positional ranking curves with scarcity weights. They are
heuristics, not market prices or championship probabilities. The source currently
has ROS rankings but no ROS projections. Exact formulas are in gm_trades.py and
all thresholds/search limits are in trade_config.json and the saved audit.

Bounded search considers one-for-one, two-for-one and two-for-two packages.
Cheap screening precedes full optimizer comparisons. It does not enumerate all
possible deals. K/DST never become trade assets; consolidation creates no invented
replacement player. Receiving rosters cannot increase their captured excess over
active capacity. Existing IR ambiguity is disclosed, not silently resolved.

Tiers are Strong target, Worth offering, Fair but unnecessary, Only if discounted,
Reject, and Research only. Strong target and Worth offering are reserved for
send-ready verified packages. Approval additionally requires exact fresh Trade Analyzer evidence.
Negative analyzer ROS impact vetoes an otherwise eligible offer. Opening, fair and
walk-away levels are alternatives among evaluated packages, not guaranteed prices
or a recommendation to execute multiple trades. Unverified packages appear only
as research, without send-ready offer levels.

## Revised offer and protection gates

The validated GM recommended lineup is an independent source of protection,
unioned with waiver Core hold and starter flags. A missing waiver flag cannot
demote a GM starter. Protected players never enter ordinary Trade Chips or the
ordinary two-player chip pool; a protected single can be considered only for a
material ROS upgrade. Trade chip ordering uses marginal ROS starter cost, with
weekly removal cost displayed separately. Waiver classifications are preserved.

All opening/fair/walk-away offers use the same 0.90-1.15 ROS asset-ratio gate.
There is no discounted opening-offer exception. Pat needs at least +0.5 ROS
starter utility and +0.25 depth-adjusted ROS utility, with no more than one weekly
point sacrificed. Protected assets require +1.5 ROS starter utility and +1.0
depth-adjusted ROS utility. These are explicit heuristic thresholds.

The counterparty must retain nonnegative depth-adjusted ROS value, lose no more
than 0.25 ROS starter utility or 2.5 weekly points, and gain either 0.25 ROS starter
utility or 0.5 weekly points with no ROS starter loss. Both exact analyzer ROS
impacts must be known. Negative user ROS impact vetoes approval; a counterparty
analyzer loss greater than 0.25 percentage points also vetoes it. Smaller provider
differences are tolerated only with a positive local roster-fit case. No assumed
willingness, acceptance probability or guessed owner identity is used.

For each target, opening minimizes net target cost (assets sent minus secondary
assets received); fair is closest to a balanced total exchange; walk-away is the
highest verified plausible net cost actually tested. When one package is all that
passes, the three levels intentionally coincide and the UI explains why. No more
expensive untested package is authorized.

The dashboard order is Verified Trade Board, Research Candidates (not send-ready),
Trade Chips, Core / untouchable at normal market value, and League Positional Needs.
Both boards separate weekly lineup delta from ROS assessment and show verification,
tier and confidence. Repeated outgoing-package counts, target lists, generation
paths and screening deltas are retained in generation_audit/candidate_source.
Exact packages are deduplicated; previously analyzer-checked packages are retained
for reevaluation even when revised screening rejects them.

## Evidence and refresh

After publishing a validated snapshot, run local analysis to shortlist packages.
Call authenticated FantasyPros `trade_analyzer` with the explicit league key and
exact comma-separated send/receive names. This is analysis only, not submission.
Save `data/trade_sources.json` with snapshot_id, league_key and calls containing
kind=analyzer, tool, exact arguments, actual response-completion fetched_at and raw
MCP response. Preserve all failures and negative verdicts. Calls must follow the
snapshot refresh start and be no more than 24 hours old. Re-run the engine.

Both returned team names and received/sent player IDs must match the package;
nonfinite ROS gains are rejected. Raw Trade Finder ideas may be collected as
kind=finder, but their cached scores never verify an offer. Do not copy old calls
to a new snapshot or change their timestamps. Required premium disclosures must
be shown if returned. There is no transaction client.

Owner identity uses exact unique saved aliases or explicitly sourced mappings in
data/owner_mappings.json tied to the league key and current team name. Unknown
owners stay unconfirmed. FAAB history is not evidence of trading behavior.

Pending waiver or streaming actions are not assumed executed. The validated
lineup, waiver, FAAB and scoring engine implementations remain unchanged.

## Validation

Run the entire GM suite:
`.\.venv\Scripts\python.exe -m unittest discover -s tools/gm_assistant -p 'test_*.py' -v`

Tests cover legal lineup changes and bench replacements, roster capacity,
counterparty costs, protected assets, missing values, exact analyzer identity,
freshness, offer levels, snapshot persistence and dashboard integration.
