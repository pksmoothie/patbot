# GM recovery handoff — September 10, 2026

## Before the interruption

Git had no tracked or staged changes. The entire GM directory was untracked,
alongside older draft audit outputs; therefore Git cannot reconstruct the exact
interrupted-session diff. File timestamps, source inspection, saved artifacts and
a baseline test run established the following. The pre-edit files and data are
retained in `data/recovery_before/`, with a recursive inventory.

GM v0.5 already had its trade model, bounded package search, configuration, roster
profiles, analyzer matching, dashboard integration and trade tests. It was not
fully validated: the baseline suite ran 122 tests with one failing counterparty
fit assertion. Its saved trade analysis was disabled and referenced an older
snapshot than `latest_snapshot.json`. Trade documentation still described a shell.

The critical Streaming fix had not started: code and documentation explicitly
made FantasyPros current_week authoritative. The saved analysis selected Week 2
and recommended Packers for Steelers and Trey Smack for Evan McPherson. Those
were not validated Week 1 recommendations and are superseded.

## Completed during recovery

- Added `gm_week.py`, verified `nfl_schedule.json`, and opener/transition regressions.
  Dated NFL games now drive Streaming and refresh projection requests. Playing,
  upcoming, provider and projection weeks remain distinct. Wrong weeks/seasons,
  stale or incomplete calendars and exhausted horizons cannot silently become
  current-week inputs. UTC midnight, Melbourne local dates, January season labels,
  the Tuesday rollover, and rescheduled games are covered.
- Added kickoff exclusions and blocked defense stashes funded by noncomparable
  waiver utilities. An unavailable utility represented as zero is not free depth.
- Corrected the trade fixture expectation and tightened missing ROS, comparable
  baseline, duplicate roster, analyzer identity/nonfinite result and kickoff gates.
  Existing trade weights and search limits were retained. Completed TRADES.md,
  Streaming/refresh documentation and dashboard week labels.
- Resynced the authoritative league and published a fresh validated snapshot:
  `gm_eb74c563b26d469d93554736df6a7b93`. Collected Week 1 projections, 32 team
  schedules, 44 streaming ownership checks, and seven exact trade analyzer checks.
  Preserved raw responses under the matching `data/incoming/` directory.
- Regenerated analyses through the dashboard against that snapshot.

Byte comparisons confirm unchanged gm_lineup.py, gm_waivers.py, gm_faab.py,
gm_views.py, gm_snapshot.py, gm_schema.py, scoring_rules.json,
streaming_config.json and trade_config.json. No draft or Yahoo execution changes.

## Current output

Streaming uses **2026 NFL Week 1**, playing September 9-14. Week 2 begins
September 17. Evidence: [NFL Week 1](https://www.nfl.com/schedules/2026/by-week/week-1)
and [NFL Week 2](https://www.nfl.com/schedules/2026/by-week/week-2).
Latest dashboard verification/analysis: September 10 around 8:38 PM Eastern.

- **HOLD Pittsburgh Steelers D/ST** against Atlanta; FP fallback projection 7.39.
- **HOLD Evan McPherson** against Tampa Bay; FP fallback projection 8.01.
- Tampa Bay is a future defense stash candidate against Cleveland/Minnesota;
  no bench drop or stash transaction is approved with current comparable-cost evidence.

**GM v0.5 is implemented and validated.** Of 223 evaluated packages, two are
approved alternatives with James (The DReem Team), targeting DK Metcalf:

| Level | Pat sends | Pat receives | FP ROS impact | Modeled Week 1 lineup change |
|---|---|---|---|---|
| Opening | Kenny Gainwell | DK Metcalf | +0.7% | +1.47 points |
| Fair / tested walk-away | Brian Thomas Jr. + Kenny Gainwell | DK Metcalf + Chuba Hubbard | +0.1% | +1.76 points |

The percentages are FantasyPros ROS power-ranking changes, not title or acceptance
probabilities. Tested negative-ROS packages were downgraded. Other locally eligible
Warren, Goedert and Odunze packages remain explicitly provisional/unverified.
Offers are alternatives; no offer, add/drop or lineup change was submitted.

## Validation and limitations

**141/141 GM tests passed**, including lineup, waiver/FAAB, streaming, trades,
snapshot/refresh and dated-week regressions. Log: `data/recovery_test_results.txt`.
The real refreshed-snapshot Streamlit AppTest rendered all six tabs with zero
exceptions. Final streaming/trade outputs share the current snapshot ID.

The verified calendar currently contains complete Weeks 1-2; refresh its dates
within seven days and extend the horizon before it expires. Week rollover uses
06:00 Eastern after the final scheduled game date, not a live final-whistle feed.
Schedule changes require refreshed evidence. League/ownership/analyzer evidence
expires after 24 hours. ROS projections remain unavailable; ROS rankings supply
heuristics. Custom projection components, some roles/byes and owner identities
remain incomplete. Undated lineup/waiver comparisons retain their existing limits.
Trade search is bounded, and opponent slot assignments are modeled. Manual Yahoo
review is still needed for transactions and roster/IR capacity.

All work remains uncommitted and the GM directory remains untracked, as found.
