from __future__ import annotations

from io import StringIO
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from .market import _known_name_match, _numeric, normalize_name

YAHOO_DRAFT_ANALYSIS = "https://football.fantasysports.yahoo.com/f1/draftanalysis"
# Yahoo's draft-analysis tabs use SD for snake/standard-draft ADP
# (Avg Pick / Avg Round) and AD for auction/salary data (Avg Cost).
YAHOO_ADP_TAB = "SD"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 PatBot/0.5.9.2"
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


def _url_for_page(*, pos: str, count: int) -> str:
    return f"{YAHOO_DRAFT_ANALYSIS}?{urlencode(_request_params(pos=pos, count=count))}"


def _find_chromium_browser() -> str | None:
    """Locate an installed Chromium-family browser without needing WebDriver.

    Windows 10/11 normally ships with Microsoft Edge, so this lets PatBot render
    Yahoo's JavaScript-backed public ADP page without adding Selenium/Playwright
    or touching the user's normal browser profile.
    """
    override = (os.getenv("PATBOT_CHROMIUM_PATH") or "").strip()
    if override and Path(override).exists():
        return override

    for command in ("msedge", "msedge.exe", "chrome", "chrome.exe", "chromium", "chromium.exe"):
        found = shutil.which(command)
        if found:
            return found

    roots = [
        os.getenv("PROGRAMFILES(X86)"),
        os.getenv("PROGRAMFILES"),
        os.getenv("LOCALAPPDATA"),
    ]
    relative_paths = [
        Path("Microsoft/Edge/Application/msedge.exe"),
        Path("Google/Chrome/Application/chrome.exe"),
        Path("Chromium/Application/chrome.exe"),
    ]
    for root in roots:
        if not root:
            continue
        for relative in relative_paths:
            candidate = Path(root) / relative
            if candidate.exists():
                return str(candidate)
    return None


def _browser_command(browser: str, url: str, user_data_dir: str, virtual_time_ms: int = 5000) -> list[str]:
    """Build a side-effect-free headless browser command for Yahoo rendering."""
    return [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--disable-extensions",
        "--disable-default-apps",
        "--no-first-run",
        "--log-level=3",
        "--window-size=1600,1200",
        f"--user-data-dir={user_data_dir}",
        f"--virtual-time-budget={int(virtual_time_ms)}",
        "--dump-dom",
        str(url),
    ]


def _render_html_with_browser(url: str, *, timeout: int) -> tuple[str, str]:
    browser = _find_chromium_browser()
    if not browser:
        raise RuntimeError(
            "Yahoo returned a JavaScript-only page and PatBot could not find Microsoft Edge/Chrome for the headless fallback."
        )

    with tempfile.TemporaryDirectory(prefix="patbot_yahoo_") as tmp:
        command = _browser_command(browser, url, tmp)
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(int(timeout) + 10, 20),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Headless browser timed out while rendering Yahoo Draft Analysis.") from exc

    html = completed.stdout or ""
    if completed.returncode != 0 or len(html) < 200:
        detail = (completed.stderr or "").strip().splitlines()
        detail_text = detail[-1] if detail else f"exit code {completed.returncode}"
        raise RuntimeError(f"Headless browser could not render Yahoo Draft Analysis: {detail_text}")
    return html, Path(browser).name


def _parse_html_page(html: str, player_names: list[str]) -> pd.DataFrame:
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError as exc:
        raise ValueError("No tables found") from exc
    return parse_yahoo_adp_tables(tables, player_names)


def _fetch_one_page(
    session: requests.Session,
    player_names: list[str],
    *,
    pos: str,
    count: int,
    timeout: int,
    force_browser: bool = False,
) -> tuple[pd.DataFrame, str]:
    url = _url_for_page(pos=pos, count=count)
    request_error: Exception | None = None

    if not force_browser:
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 429:
                raise RuntimeError("Yahoo Draft Analysis rate-limited the request (HTTP 429).")
            response.raise_for_status()
            return _parse_html_page(response.text, player_names), "requests"
        except Exception as exc:
            request_error = exc

    # Yahoo increasingly serves Draft Analysis as a JS-backed shell to plain
    # requests. Render it with the already-installed Edge/Chrome executable and
    # dump the final DOM; no browser driver and no Yahoo login are required.
    try:
        html, browser_name = _render_html_with_browser(url, timeout=timeout)
        return _parse_html_page(html, player_names), f"headless-{browser_name}"
    except Exception as browser_exc:
        if request_error is None:
            raise
        raise RuntimeError(
            f"plain HTTP failed ({type(request_error).__name__}: {request_error}); "
            f"headless fallback failed ({type(browser_exc).__name__}: {browser_exc})"
        ) from browser_exc


def fetch_yahoo_adp(
    player_names: list[str],
    *,
    max_players: int = 300,
    page_size: int = 50,
    request_spacing_seconds: float = 0.35,
    timeout: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Fetch Yahoo snake-draft Avg Pick from the public Draft Analysis page.

    Yahoo ADP is a supporting room-behavior input only. It does not alter PatBot
    projections, VORP, expert rank, Athletic/FantasyPros quality signals, or the
    player's intrinsic valuation.
    """
    session = _session()
    collected: dict[str, dict] = {}
    pages_loaded = 0
    browser_pages = 0
    errors: list[str] = []
    force_browser = False
    transports: set[str] = set()

    # Yahoo pagination advances in 50-player offsets. Try the combined board
    # first; position pages are a fallback if ALL changes layout or becomes sparse.
    for pos in ("ALL", "QB", "RB", "WR", "TE"):
        no_new_pages = 0
        for count in range(0, int(max_players), int(page_size)):
            if pages_loaded:
                time.sleep(max(0.0, float(request_spacing_seconds)))
            try:
                page, transport = _fetch_one_page(
                    session,
                    player_names,
                    pos=pos,
                    count=count,
                    timeout=int(timeout),
                    force_browser=force_browser,
                )
                pages_loaded += 1
                transports.add(transport)
                if transport.startswith("headless-"):
                    browser_pages += 1
                    force_browser = True
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
        "requests": int(pages_loaded),
        "pages_loaded": int(pages_loaded),
        "browser_pages": int(browser_pages),
        "transport": ", ".join(sorted(transports)) if transports else "unknown",
        "endpoint": YAHOO_DRAFT_ANALYSIS,
        "tab": YAHOO_ADP_TAB,
        "note": "Yahoo snake-draft Avg Pick; supporting input only for opponent behavior and availability modeling.",
    }
    return out, status


def manager_yahoo_weight(archetype: str) -> float:
    """Supporting Yahoo-board influence by manager sophistication.

    Yahoo is deliberately not a majority behavioral input even for casuals. It
    nudges the existing market/custom/roster-need/randomness model because the
    Yahoo board is visible in the draft room, but it never becomes the room model
    by itself.
    """
    return {
        "casual": 0.35,
        "market": 0.25,
        "league_aware": 0.15,
        "sharp": 0.07,
        "extremely_sharp": 0.03,
    }.get(str(archetype), 0.18)


def behavioral_adp(existing_market_adp: pd.Series, yahoo_adp: pd.Series, yahoo_weight: float) -> pd.Series:
    existing = pd.to_numeric(existing_market_adp, errors="coerce")
    yahoo = pd.to_numeric(yahoo_adp, errors="coerce")
    w = max(0.0, min(1.0, float(yahoo_weight)))
    both = existing.notna() & yahoo.notna()
    result = existing.copy()
    result.loc[both] = (1.0 - w) * existing.loc[both] + w * yahoo.loc[both]
    result = result.where(result.notna(), yahoo)
    return result
