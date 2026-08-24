# PatBot Fantasy Football GM — v0.3.6

PatBot is a personal 2026 Yahoo fantasy-football draft and team-management project built around the league's exact scoring system.

The repository is now the permanent source of truth for PatBot. Local secrets and generated live-data snapshots stay on the user's PC and are intentionally excluded from Git.

## Current draft capabilities

- Exact custom offensive scoring for QB/RB/WR/TE projections.
- Sleeper projections and ADP fallback.
- FantasyPros official API consensus rankings / ADP when a local API key is present.
- FantasyData PPR rankings.
- Custom VORP, consensus value, tiers and tier-cliff signals.
- Fast Monte Carlo draft simulation through Round 8.
- Opponent room model: 4 casual, 3 market, 2 league-aware, 1 sharp, 1 extremely sharp, plus PatBot.
- Lineup-aware roster-construction evaluation.
- Real snake-draft ownership for every overall pick.
- Live roster tracking for all 12 teams.
- Real opponent roster state fed into future simulations.
- Draft log, team-roster view, manager-name setup and undo-last-pick.

## One-time Windows setup

Clone this repository once, then double-click:

`SETUP_ONCE.bat`

The setup script creates `.venv`, installs packages, creates a local `.env` from `.env.example`, runs the tests, refreshes live data and launches Streamlit.

The `.env` file is ignored by Git. Put the FantasyPros key there and never commit or share it.

## Normal use after setup

When ChatGPT has pushed a PatBot update, double-click:

`UPDATE_AND_RUN.bat`

It will:

1. `git pull` the latest code.
2. Sync Python dependencies.
3. Run the test suite.
4. Refresh live player data.
5. Launch PatBot.

If you only want to launch the existing local version without pulling or refreshing, double-click:

`RUN_PATBOT.bat`

## Manual commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe refresh_data.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Local environment

Copy `.env.example` to `.env` if the setup script has not already done so.

```text
FANTASYPROS_API_KEY=your_key_here
```

Yahoo OAuth fields are scaffolded in `.env.example` for future API integration. Yahoo approval is not required for the current manual draft-assistant workflow.

## v0.3.6 — real draft state by team

PatBot records each real draft selection by overall pick and automatically maps it to the correct snake-draft slot. The resulting roster state is passed into simulations, so an opponent who already drafted a QB is less likely to draft another early QB in future simulated picks.

Opponent archetypes are still randomized across the 11 opposing slots. Once the real draft order and manager profiles are available, the next layer is to lock manager identities and tendencies to specific slots.
