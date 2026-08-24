# PatBot Draft Engine Specification

## Objective

Maximize expected championship equity, not merely projected points at the current pick.

For each available player, PatBot should eventually evaluate:

- Value under the exact scoring system.
- Position-specific replacement value.
- Current roster construction.
- Market cost / ADP.
- Probability the player survives to the next PatBot pick.
- Positional tier cliffs.
- Injury and role uncertainty.
- Opponent roster needs.
- Correlations and playoff schedule.
- Expected season outcomes from simulated rosters.

## Core decision question

> How much expected team value do we gain by drafting this player now versus the best portfolio of players likely to remain at our future picks?

## v0.1 composite score

Default weights:

- 35% VORP
- 25% season projection
- 20% urgency (risk player is gone by next PatBot pick)
- 12% positional scarcity
- 8% roster fit
- configurable injury-risk penalty

These are editable in `config/league.yaml`.

## Availability model

v0.1 treats draft position as approximately normal:

```text
mean = ADP
sd = max(6, ADP × 0.15)
```

This gives an approximate probability of the player surviving to the next PatBot selection.

v0.4 should replace this with empirical draft distributions and Monte Carlo simulations.

## Initial replacement levels

- QB12
- RB30
- WR42
- TE14
- K12
- DEF12

RB and WR extend beyond strict starter counts because the FLEX position adds demand.

## Snake picks

For draft slot `s` in a 12-team snake:

Odd round:
```text
pick = (round - 1) × 12 + s
```

Even round:
```text
pick = (round - 1) × 12 + (13 - s)
```

At slot 1.03, the first picks are 3, 22, 27, 46, 51, etc.

## Draft guardrails

- Strong K/DEF penalty before the configured minimum rounds.
- Position-cap penalties once starter/bench needs are saturated.
- Unavailable players are removed immediately.
- Future versions add bye-week, injury-concentration and correlation limits.

## Yahoo architecture

Yahoo Fantasy Sports uses OAuth 2.0 and the Fantasy API base:

```text
https://fantasysports.yahooapis.com/fantasy/v2
```

PatBot should never store the Yahoo password.

### Draft mode
Until a documented live-draft write endpoint is confirmed, PatBot makes the decision and the user physically submits the pick.

### In-season
Expose narrowly scoped actions:
- get roster
- get free agents
- set lineup
- add/drop
- submit waiver
- propose trade

Every write passes through deterministic safety rules before reaching Yahoo.

## AI responsibilities

Use the language model for:
- concise explanations
- news interpretation
- uncertainty/context
- trade reasoning
- natural-language reporting

Do not use the LLM as the sole projection or valuation engine.
