# PatBot Fantasy Football GM — v0.4.0

PatBot is a personal 2026 Yahoo fantasy-football draft and team-management project built around the league's exact scoring system.

The repository is the permanent source of truth for PatBot. Local secrets, the paid Athletic workbook and generated live-data snapshots stay on the user's PC and are intentionally excluded from Git.

## Current draft capabilities

- Exact custom offensive scoring for QB/RB/WR/TE projections.
- Sleeper raw projections and ADP fallback.
- FantasyPros Premium full-board PPR ECR/ADP.
- FantasyData PPR rankings.
- Local Athletic custom rankings/VORP workbook.
- Custom VORP, consensus value, tiers and tier-cliff signals.
- Fixed real-manager room model and live roster state for all 12 teams.
- Fast paired Monte Carlo draft simulation through Round 8.
- Early-pick one-step lookahead.
- Model Diagnostics scoring/weight ablations.
- Explicit risk & availability layer using FantasyPros historical games, current injuries and recent risk news.
- Dated manual uncertainty flags for unusual off-field situations that structured feeds cannot represent cleanly.
- Risk-adjusted Monte Carlo outputs including 10th-percentile outcomes, candidate games, 4+ game tail events and off-field-event frequency.
- Partial replacement-player value during missed games rather than treating an injured starter as zero lineup production.
- Pick-immediately-after-us diagnostics to show whether candidate branches quickly reconverge.

## One-time Windows setup

Clone this repository once, then double-click:

`SETUP_ONCE.bat`

The setup script creates `.venv`, installs packages, creates a local `.env` from `.env.example`, runs the tests, refreshes live data and launches Streamlit.

The `.env` file is ignored by Git. Put the FantasyPros key there and never commit or share it.

## Normal use after setup

When ChatGPT has pushed a PatBot update, run:

`UPDATE_AND_RUN.bat`

It will:

1. `git pull` the latest code.
2. Sync Python dependencies.
3. Run the test suite.
4. Refresh live player data.
5. Launch PatBot.

The v0.4 refresh makes several FantasyPros calls for six years of availability history plus current injury/news data and intentionally spaces them to respect the Premium API rate limit, so refresh is a little slower than v0.3.x.

If you only want to launch the existing local version without pulling or refreshing, run:

`RUN_PATBOT.bat`

## Athletic workbook

The current paid Athletic workbook is stored locally at:

`private_sources/athletic.xlsx`

It is Git-ignored. Do not re-upload it unless The Athletic has published changes you want PatBot to use; after replacing it, refresh live data.

## Manual commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe refresh_data.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Risk model philosophy

PatBot does not replace a projection with an arbitrary injury haircut. It keeps the source projection as the production baseline and models availability as a separate distribution.

Historical games provide an availability signal, with recent seasons weighted more heavily and young-player history downweighted because early-career missed games can reflect role rather than injury. Current FantasyPros injury/practice probabilities add near-term information. Recent legal/discipline news can create a temporary off-field tail, but a risk event is sampled as uncertainty rather than assumed to occur.

When a player misses simulated games, PatBot credits a fraction of replacement-level production for those weeks. This is meant to model the roster decision more realistically than assigning zero points.

The risk calibration is intentionally transparent and should be reviewed as new 2026 information arrives; it is not a claim that historical availability perfectly predicts future injury.

## Local environment

Copy `.env.example` to `.env` if the setup script has not already done so.

```text
FANTASYPROS_API_KEY=your_key_here
```

Yahoo OAuth fields are scaffolded in `.env.example` for future API integration. Yahoo approval is not required for the current manual draft-assistant workflow.
