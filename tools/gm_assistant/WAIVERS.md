# GM v0.3.1 waivers and FAAB

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe tools/gm_assistant/gm_waivers.py
.\.venv\Scripts\python.exe -m unittest discover -s tools/gm_assistant -p 'test_*.py' -v
```

The CLI and dashboard save `data/latest_waiver_analysis.json` atomically. All
implementation, configuration and evidence stay inside `tools/gm_assistant/`.
The draft engine, Yahoo history exporter, and validated lineup engine are untouched.

## Evidence and availability

Start with `load_snapshot()` validation of `data/latest_snapshot.json`. Use all
six positional Waiver Finder sections, deduplicate names and retain every source
category and the exact FantasyPros add/drop advice. Waiver Finder is a candidate
set, not an exhaustive available-player list.

Optional expansion uses ROS-ranked players queried through ownership. The first
v0.3 capture checks the first two ROS-ranked names per position not listed on the
12 captured rosters, in addition to the 18 Waiver Finder candidates: 24 distinct
players total. Roster absence is only a shortlist filter, never proof of availability.

`data/waiver_ownership.json` contains `league_key`, the unmodified `get_leagues`
response under `league_lookup`, and `calls` with `player_name`, `tool`, UTC
`fetched_at`, and the exact MCP envelope in `raw`. In the initial v0.3 sidecars timestamps were recorded
at request start and conservatively bound data age; refreshes should timestamp
completed responses. Ownership responses contain
league names rather than keys. Accept their exact `Not owned - free agent` status
only through a unique league-name-to-key mapping for the authoritative league.
Reject wrong player, wrong league, ambiguous names, errors, ownership older than
24 hours or preceding the snapshot refresh, and conflicts with captured rosters.
Snapshot age over 24 hours disables claims. Absence of confirmation means NO CLAIM.

`data/waiver_context.json` preserves depth-chart calls for DAL, CLE, LAC, SEA and
MIN with arguments, tool, UTC fetched time, and exact raw envelope. Depth positions
join by ID or a unique name, using the existing conservative join helper. They
expire after 24 hours and must not precede the snapshot refresh. Conditional paths
behind named teammates are explicitly inferences; they do not establish increased
volume and receive no workload premium. The preserved snapshot provides injuries
and schedule/SOS context. An undocumented SOS scale receives zero numerical weight.

## Roster opportunity costs

Every candidate is evaluated against Pat's complete actual roster. A mismatch
between roster names and lineup-player inputs disables claims. Current starters,
positional minimums (QB1/RB2/WR3/TE1/K1/DST1), and one RB/WR/TE FLEX are considered.
Weekly projections are comparable only when their week matches an explicitly
supplied lineup week. Otherwise they are displayed with week/scoring as reference
only. Current v0.3 capture has no explicit lineup week, so decisions use ROS ranks.

ROS utility is `max(0, 18 - 9 * positional_ROS_rank / cutoff)`, with cutoffs QB12,
RB24, WR36, TE12, DST12, K12. When weekly points are aligned, use 75% ROS utility
and 25% points. Missing ROS is never a zero/worst rank: weekly-only candidates
require aligned points across the entire roster and use weekly-only comparisons.
Unknown drop ROS cannot be compared to blended candidate utility. These utilities
are transparent heuristics, not fantasy-point or winning-bid forecasts.

A sourced, explicitly confirmed workload path can add 0.5 utility. Conditional
depth-chart paths do not. The best starter utility includes a FLEX after filling
positional slots. A claim must improve retained-player utility by at least 0.6 and
not reduce best starter utility. Cross-position swaps also need at least 0.6 starter
gain. Known locked players cannot be dropped. K/DST require aligned weekly data
and a same-position drop, preventing unnecessary duplicate streamers.

Drop classes, in increasing protection:

- Immediate drop: explicitly released, no workload path, and not a core hold.
- Replaceable: non-core, non-stash utility under 6, excluding useful first backups.
- Upside stash: injury context, confirmed upside path, or a conditional RB2/RB3 path.
- Useful depth: unknown valuation (protected), utility at least 6, or utility at
  least 3 within the first backup at that position.
- Core hold: current starter, current GM-recommended starter, or ROS rank inside
  the position's starter cutoff. The unchanged v0.2 lineup analysis supplies the
  recommended player IDs; this protection is additive, so a close lineup swap
  does not demote the outgoing current starter.

Class order and utility order form the displayed non-core drop hierarchy. Core
holds appear separately in a collapsed protected section; the audit retains all
classifications. GM-recommended starters are excluded from all waiver-drop
options, including same-position and weekly-only comparisons. It is contextual:
claim selection separately verifies positional opportunity cost. An injury is not
an automatic drop, and no current player has evidence requiring Immediate drop.

## FAAB calculation and budget strategy

For a current owner-requested correction, `waiver_context.json.drop_preference`
records the explicit statement, user source, league/snapshot identity, confirmation
time, preferred drop, affected positions (RB/WR), and whether a free IR move is
confirmed. It expires after 24 hours or a different snapshot. The preferred
expendable player sorts first without changing anyone's classification or value.
For the requested positions only, this allows a cross-position depth swap with
at least 0.6 retained-player gain and no starter loss. Starter protection, locks,
positional minimums, ownership and valuation requirements still apply. Confirmed
free IR retention excludes that player from claim drops. The preference cannot
force a poor candidate into a claim or change the bid formula.

Starting tier dollars are 40 for league-changing/potential long-term starter,
15 for strong roster upgrade, 3 for useful depth/upside, 1 for speculative stash,
and 1 for streamer. League-changing requires a starter-level ROS rank and at least
5 starter utility gain; strong upgrade requires at least 2 starter gain; useful
depth requires at least 1 retained-player gain, otherwise speculative.

Add bounded fractions of base: roster improvement up to 25% (gain/20), regular
starter value up to 20% (starter gain/15), candidate-set scarcity up to 10%, explicit
lead role 10%, starter-level ROS value 10%, urgent thin-roster upgrade 10%, and likely
market interest 10% for starter-level ROS or 5% for an FP bid of at least $10.
Scarcity is 10% with no healthy backup and at most two similarly valued confirmed
candidates; 5% with no healthy backup but more alternatives; otherwise zero. This
does not estimate scarcity in the complete free-agent pool.

FantasyPros suggested dollar bids adjust resulting valuation by at most +/-20%.
Then history adjusts it by at most +/-5%. Store each intermediate value. Multiply
the final unrounded value by 0.65/1/1.4; floor conservative, round recommended
half-up, ceil aggressive. Cap all amounts at explicitly supplied current remaining
FAAB (`my_roster.faab_remaining`); otherwise use nominal $100 maximum and flag the
missing balance. Streamer ceilings never exceed $5. No mandatory late-season
reserve reduces league-changing opportunities.

When FantasyPros omits the remaining balance, an explicit user confirmation can
be preserved in `data/waiver_context.json` under `faab_confirmation`: `source`
must be `user`, with `league_key`, `snapshot_id`, timezone-aware `confirmed_at`,
and `team_balances` keyed by team ID. It only applies to the matching snapshot,
must follow refresh start, and expires after 24 hours. A valid provider balance
takes precedence. Raw FantasyPros evidence is never altered. The analysis records
the confirmation and uses the same existing bid and sequential-budget formulas.

## Canonical owner priors

`config/faab_history.json` uses permanent owner names; each completed season retains
team-name aliases only as evidence. Pat's unspecified aliases remain empty. Each
row stores remaining FAAB, spent = 100 - remaining, and moves. Duplicate owners,
owner seasons, and ambiguous within-season aliases are rejected.

Three-season aggregates: mean, median, min/max spend, spend range, population
standard deviation, and mean moves. Spending labels require three seasons:
aggressive >=$75 mean, conservative <=$35, moderate between. Activity is independent:
high >=25 mean moves, low <15, moderate otherwise. Consistency uses spend range:
consistent <=$20, variable >$50, mixed otherwise; under three seasons is insufficient.

| Owner | Mean spent | Median | Min–max | Mean moves | Spend / activity |
|---|---:|---:|---:|---:|---|
| Pat | 95.67 | 95 | 92–100 | 36.67 | Aggressive / High |
| Matt | 58.67 | 40 | 36–100 | 11.33 | Moderate / Low |
| Lou | 61.67 | 45 | 40–100 | 18.33 | Moderate / Moderate |
| Dave | 29.33 | 27 | 20–41 | 29.67 | Conservative / High |
| Sarlo | 76.67 | 83 | 49–98 | 28.33 | Aggressive / High |
| Faherty | 54.33 | 51 | 40–72 | 24.33 | Moderate / Moderate |
| Potter | 93.00 | 100 | 79–100 | 31.67 | Aggressive / High |
| George | 87.33 | 92 | 70–100 | 29.67 | Aggressive / High |
| James | 34.33 | 35 | 2–66 | 12.67 | Conservative / Low |
| Frank | 4.00 | 0 | 0–12 | 9.67 | Conservative / Low |
| Vinnie | 33.00 | 14 | 5–80 | 9.33 | Conservative / Low |
| Paul | 57.67 | 53 | 37–83 | 19.00 | Moderate / Moderate |

Market history excludes Pat and insufficient histories. Its bid fraction is
`clamp((peer_mean_spend - 50) / 500, -0.05, 0.05)`. Current fraction is +0.7273%,
usually invisible after rounding small bids. Pressure labels: elevated at peer
mean >=65, low <=35, otherwise moderate. Sarlo, Potter and George are historically
aggressive possible competitors; their current participation/balances are unknown.
Dave's high activity and low total spending do not prove low-cost/free-agent adds.
Individual bids, losing bids, players, timing, and transaction types are unknown.
These labels are descriptive context, not deterministic bidding predictions.

## Manual sequencing and audit

Order by starter gain, then retained-player gain. All claims sharing a drop are
mutually exclusive fallbacks with explicit prior-claim failure dependencies. Reserve
the maximum bid per drop group and enforce the sum against a known budget. With
unknown budget, every claim requires balance verification. Rerun after acquisitions;
the plan applies to the captured roster. No Yahoo action is executed.

The audit records source snapshot identity/time, all candidate inputs and coverage,
explicit availability evidence and league mapping, roster inputs, drop costs,
owner priors and history hash, every bid component/range, confidence, warnings, and
claim dependencies. Raw FantasyPros data is collapsed in dashboard diagnostics.

Initial results: Tre' Harris / Flournoy $2/$3/$5, then Jeudy / Flournoy $1/$3/$4,
then Jennings / Flournoy $1/$3/$4 as alternatives. All are optional low-confidence
WR depth moves with zero starter/FLEX gain. All other positions: NO CLAIM. These
are capture-time recommendations, not a permanent watchlist. Current FAAB, aligned
matchup week, future workload, and complete free-agent coverage remain unavailable.

## v0.3.1 corrective release

The waiver model now consumes the current validated lineup analysis rather than
relying only on captured starting slots and ROS ranks. Brian Thomas Jr. and
Jayden Reed both classify as Core hold. The current non-core drop order is Ryan
Flournoy (Replaceable), Khalil Shakir (Replaceable), Dylan Sampson (Upside stash),
Zach Charbonnet (Upside stash), and Kenny Gainwell (Useful depth).

Best candidates by position names only confirmed available players with a CLAIM
recommendation, in existing board order. Otherwise it says NO CLAIM — no confirmed
available candidate improves the roster. ROS references remain in the audit and
never fill this actionable display. Existing league-specific ownership validation,
expiry and roster-conflict checks still apply.

FAAB formulas, owner names/history priors, claim sequencing, and v0.2 lineup
methodology are unchanged. No D/ST or kicker streaming work is introduced.
