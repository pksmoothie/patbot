# GM v0.5 collection instructions for Codex

Use ONLY the authenticated FantasyPros MCP for league
`nfl~8922f34b-15ab-48d6-b51a-472098c3ae5a`, team ID `6`.
Do not select an active league, use PatBot data, or access Yahoo. No transactions.
The core snapshot contract remains unchanged. For waiver and streaming ownership sidecars,
get_leagues and get_player_ownership are permitted to establish explicit ownership
in the authoritative league; disregard the other league results. Keep their exact
raw evidence outside the core snapshot. Do not modify implementation files,
dependencies, or other repo files during a refresh.

## v0.3 waiver enrichment after publishing the core snapshot

Read WAIVERS.md for the sidecar formats. Preserve get_leagues(sport='nfl') as
league_lookup and the authoritative league_key in data/waiver_ownership.json.
Collect all distinct players in the six waiver sections. Optionally shortlist the
first two ROS-ranked players per position absent from captured rosters; absence
only selects names to check. Call get_player_ownership for each candidate and
preserve player_name, tool, UTC fetched_at and exact raw envelope in calls.
The engine requires explicit free-agent status in the exact league, using the
unique key/name mapping; candidate inclusion alone never confirms availability.
Missing/error/premium responses must remain unconfirmed, and required premium
disclosures must be relayed. Record timestamps when responses arrive.

For relevant add/drop decisions, collect get_nfl_depth_charts with player_name or
team_code, preserving arguments, tool, fetched_at and raw in the calls array of
data/waiver_context.json. Never fabricate roles or workloads. Ownership and depth
evidence older than 24 hours or preceding the snapshot refresh cannot be reused.
Then run gm_waivers.py to save data/latest_waiver_analysis.json. No Yahoo action.
The dashboard also regenerates this local analysis when loaded.

## One-prompt interactive refresh

If no run directory was supplied, run `refresh.py prepare` with an existing Python
environment that has this folder's dependencies (prefer `.venv` here, otherwise
the repository `.venv`). This creates `data/incoming/gm_<unique-id>/manifest.json`.
Use that run directory for the entire refresh. Paths below are relative to this
folder; do not create files in the repo's production data directory.

1. Read the manifest and inspect currently exposed tool schemas. Use the exact
   tools and arguments listed. Do not guess missing tool parameters.
2. Execute **resync first**. It must return `synced: true` before any dependent
   reads. Failure means stop; keep all evidence and the previous latest snapshot.
3. Execute the remaining core requests. At most four independent calls at a time.
   Every league-scoped request must carry the explicit league key, never session
   state. `my_roster` must explicitly select team `6`; `all_rosters` uses
   `all_teams=true`. Do not replace required missing data with a prior snapshot.
4. Immediately save each result to `<run_dir>/<request-id>.json` in this format:
   `{"fetched_at":"<actual UTC ISO timestamp after the call>","raw":<exact MCP result>}`.
   Prefer preserving the whole MCP content envelope. Unmodified decoded JSON is
   also accepted. Preserve nulls, empty lists, raw payloads, warnings and premium
   disclosures; no paraphrasing, trimming, inference, or fake success responses.
   Use tools to get actual timestamps. Preserve actual tool error envelopes too.
   If a tool is absent or never returned, leave its result file absent and report
   that fact. Do not fabricate a response on its behalf.
5. Run `refresh.py enrich <run_dir>`. This validates core identity and resync, then
   generates league-scoring rankings, weekly/ROS projections and relevant injury
   requests from the received data. Execute only the new requests and save results
   in the same format. Before enrichment, verify explicit NFL game dates in
   `nfl_schedule.json` using the official NFL weekly schedule, including the current
   and next slate. Record season, all games, timezone-aware kickoffs, source URL,
   completeness and actual verification time. `gm_week.establish_week` chooses the
   projection week from those dates; provider `current_week` is diagnostic only.
   Matchup week stays unknown if not returned. Do not relabel Week 2 projections
   as Week 1; request Week 1 and retain unavailable responses when necessary.
6. Report any `premium_required` disclosure to the user. No upgrading or routing
   around a premium gate. Never call candidate coverage the entire free-agent pool.
7. When invoked by `refresh.py run`, stop after collection: the parent publishes.
   In the interactive workflow only, run `refresh.py publish <run_dir>`. Report the
   new historical path, refresh status, and important missing-data warnings.

The helper validates core coverage, timestamps and identity before publishing.
It retains raw sources, marks unavailable optional sections, and never overwrites
historical snapshots. Failed core collection must not change latest_snapshot.json.


## v0.4 streaming enrichment after publishing the core snapshot

Read STREAMING.md. Preserve the validated snapshot and waiver sidecars.
Use `gm_week.establish_week` and the verified dated NFL schedule to obtain the
playing/planning week. Preserve conflicting provider counters as diagnostics.
If the dated evidence is unavailable, save an unavailable analysis without guessing.

Expand D/ST and K from Waiver Finder plus weekly ECR/projections. Use the team codes
in those rows, including the current rostered options. For each distinct team,
call get_schedule_and_strength(team_codes=[one_team], week_start=current_week,
week_end=min(18,current_week+3)), with no position filter to obtain both D/ST and K
weekly ratings and opponents. Preserve snapshot_id and exact calls (kind=schedule,
arguments, completion fetched_at, raw) in data/streaming_sources.json. Single-team
requests are necessary because multi-team responses can omit Weekly Matchups.

Preserve get_leagues(sport='nfl') as league_lookup with the explicit league_key in
data/streaming_ownership.json. Query get_player_ownership for each distinct
potential addition; roster absence is only a shortlist filter. Preserve all raw
results, names, tool and completion timestamps. Use at most four independent calls
at a time. Never overwrite either file with invented or silently reused evidence.

Run gm_streaming.py to save data/latest_streaming_analysis.json, then reload the
dashboard. Waivers hands D/ST/K to the same Streaming calls; skill-position claims
and bids remain under v0.3.1. Both sidecars expire after 24 hours; schedule sidecars
must match the new snapshot_id. No Yahoo actions or historical transaction scraping.
