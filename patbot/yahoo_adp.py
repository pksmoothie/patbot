from __future__ import annotations

from io import StringIO
import time

import numpy as np
import pandas as pd
import requests

from .market import _known_name_match, _numeric, normalize_name

YAHOO_DRAFT_ANALYSIS = "https://football.fantasysports.yahoo.com/f1/draftanalysis"
# Yahoo's historical/current draft-analysis tabs use SD for snake/standard-draft
# ADP (Avg Pick / Avg Round) and AD for auction/salary data (Avg Cost). PatBot
# needs SD because Yahoo ADP is being used only as a room-behavior signal.
YAHOO_ADP_TAB = "SD"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 PatBot/0.5.9.1"
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join(
                str(x).strip()
                for x in col
                if str(x).strip() and str(x).lower() != "nan" and "Unnamed" not in str(x)
            ).strip()
            for col in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def _player_column(df: pd.DataFrame, known: dict[str, str]) -> str | None:
    best_col = None
    best_matches = 0
    for col in df.columns:
        matches = sum(
            1
            for value in df[col].head(400).astype(str)
            if _known_name_match(value, known)
        )
        if matches > best_matches:
            best_matches = matches
            best_col = str(col)
    return best_col if best_matches >= 2 else None


def _adp_column(df: pd.DataFrame, player_col: str) -> str | None:
    columns = [str(c) for c in df.columns if str(c) != str(player_col)]
    lowered = {c: c.lower().replace("💎", "") for c in columns}

    preferred = [
        c for c in columns
        if "basic adp" in lowered[c] and "all drafts" in lowered[c]
    ]
    if preferred:
        return preferred[0]

    preferred = [c for c in columns if "avg pick" in lowered[c] or "average pick" in lowered[c]]
    if preferred:
        return preferred[0]

    preferred = [
        c for c in columns
        if "all drafts" in lowered[c] and "plus" not in lowered[c]
    ]
    if preferred:
        return preferred[0]

    preferred = [c for c in columns if lowered[c].strip() in {"adp", "avg", "average"}]
    if preferred:
        return preferred[0]
    return None


def _percent_drafted_column(df: pd.DataFrame, player_col: str) -> str | None:
    for col in df.columns:
        if str(col) == str(player_col):
            continue
        text = str(col).lower()
        if "%drafted" in text.replace(" ", "") or "percent drafted" in text:
            return str(col)
    return None


def parse_yahoo_adp_tables(
    tables: list[pd.DataFrame],
    player_names: list[str],
) -> pd.DataFrame:
    """Parse Yahoo's public snake-draft analysis tables into PatBot ADP rows."""
    known = {normalize_name(x): x for x in player_names}
    best_rows: list[dict] = []

    for raw in tables:
        df = _flatten_columns(raw)
        player_col = _player_column(df, known)
        if not player_col:
            continue
        adp_col = _adp_column(df, player_col)
        if not adp_col:
            continue
        pct_col = _percent_drafted_column(df, player_col)

        rows = []
        for _, row in df.iterrows():
            name = _known_name_match(row[player_col], known)
            adp = _numeric(row[adp_col])
            if not name or np.isnan(adp) or adp <= 0:
                continue
            pct = _numeric(row[pct_col]) if pct_col else np.nan
            rows.append(
                {
                    "name": name,
                    "yahoo_adp": float(adp),
                    "yahoo_percent_drafted": float(pct) if not np.isnan(pct) else np.nan,
                }
            )
        if len(rows) > len(best_rows):
            best_rows = rows

    if not best_rows:
        raise ValueError("Yahoo Draft Analysis page parsed but no usable Avg Pick rows matched PatBot players.")
    return pd.DataFrame(best_rows).drop_duplicates("name", keep="first").reset_index(drop=True)


def _request_params(*, pos: str, count: int) -> dict:
    return {
        "pos": str(pos).upper(),
        "sort": "DA_AP",
        "tab": YAHOO_ADP_TAB,
        "count": int(count),
    }


def _fetch_one_page(
    session: requests.Session,
    player_names: list[str],
    *,
    pos: str,
    count: int,
    timeout: int,
) -> pd.DataFrame:
    params = _request_params(pos=pos, count=count)
    response = session.get(YAHOO_DRAFT_ANALYSIS, params=params, timeout=timeout)
    if response.status_code == 429:
        raise RuntimeError("Yahoo Draft Analysis rate-limited the request (HTTP 429).")
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    return parse_yahoo_adp_tables(tables, player_names)


def fetch_yahoo_adp(
    player_names: list[str],
    *,
    max_players: int = 300,
    page_size: int = 50,
    request_spacing_seconds: float = 0.35,
    timeout: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Fetch Yahoo snake-draft Avg Pick from the public Draft Analysis page.

    Yahoo ADP is a room-behavior input only. This function does not alter PatBot
    projections, VORP, expert rank, or the broader market-value signal.
    """
    session = _session()
    collected: dict[str, dict] = {}
    requests_made = 0
    errors: list[str] = []

    # Yahoo's draft-analysis pagination has historically advanced in 50-player
    # offsets. Try the combined snake-draft board first; position pages provide a
    # fallback if ALL changes layout or becomes sparse.
    for pos in ("ALL", "QB", "RB", "WR", "TE"):
        no_new_pages = 0
        for count in range(0, int(max_players), int(page_size)):
            if requests_made:
                time.sleep(max(0.0, float(request_spacing_seconds)))
            try:
                page = _fetch_one_page(
                    session,
                    player_names,
                    pos=pos,
                    count=count,
                    timeout=int(timeout),
                )
                requests_made += 1
            except Exception as exc:
                errors.append(f"{pos} count={count}: {type(exc).__name__}: {exc}")
                if count == 0:
                    break
                no_new_pages += 1
                if no_new_pages >= 2:
                    break
                continue

            before = len(collected)
            for _, row in page.iterrows():
                collected[str(row["name"])] = row.to_dict()
            if len(collected) == before:
                no_new_pages += 1
            else:
                no_new_pages = 0

            if no_new_pages >= 2:
                break
            if pos == "ALL" and len(collected) >= min(int(max_players), 180):
                break

        if pos == "ALL" and len(collected) >= 150:
            break

    if not collected:
        detail = errors[0] if errors else "no parser detail available"
        raise ValueError(
            "Yahoo snake-draft ADP fetch returned no matched players. "
            f"First page failure: {detail}"
        )

    out = pd.DataFrame(collected.values()).drop_duplicates("name", keep="last")
    out = out.sort_values("yahoo_adp").reset_index(drop=True)
    status = {
        "ok": True,
        "matched": int(len(out)),
        "requests": int(requests_made),
        "endpoint": YAHOO_DRAFT_ANALYSIS,
        "tab": YAHOO_ADP_TAB,
        "note": "Yahoo snake-draft Avg Pick; intended only for opponent behavior and availability modeling.",
    }
    return out, status


def manager_yahoo_weight(archetype: str) -> float:
    """Proposed Yahoo-board anchoring by manager sophistication."""
    return {
        "casual": 0.80,
        "market": 0.55,
        "league_aware": 0.30,
        "sharp": 0.15,
        "extremely_sharp": 0.05,
    }.get(str(archetype), 0.40)


def behavioral_adp(existing_market_adp: pd.Series, yahoo_adp: pd.Series, yahoo_weight: float) -> pd.Series:
    existing = pd.to_numeric(existing_market_adp, errors="coerce")
    yahoo = pd.to_numeric(yahoo_adp, errors="coerce")
    w = max(0.0, min(1.0, float(yahoo_weight)))
    both = existing.notna() & yahoo.notna()
    result = existing.copy()
    result.loc[both] = (1.0 - w) * existing.loc[both] + w * yahoo.loc[both]
    result = result.where(result.notna(), yahoo)
    return result
