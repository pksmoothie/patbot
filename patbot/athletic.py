from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    tokens = [t for t in value.split() if t not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(tokens)


def _numeric(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
        return float(match.group()) if match else np.nan


def _column_number(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", str(cell_ref).upper())
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    value = 0
    for char in match.group(1):
        value = value * 26 + (ord(char) - 64)
    return value


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    out = []
    for item in root.findall(f"{{{_MAIN_NS}}}si"):
        out.append("".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t")))
    return out


def _sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships.findall(f"{{{_PKG_REL_NS}}}Relationship")
    }
    for sheet in workbook.find(f"{{{_MAIN_NS}}}sheets"):
        if sheet.attrib.get("name") != sheet_name:
            continue
        rid = sheet.attrib.get(f"{{{_REL_NS}}}id")
        target = targets.get(rid, "")
        if target.startswith("/"):
            return target.lstrip("/")
        if target.startswith("xl/"):
            return target
        return f"xl/{target}"
    raise ValueError(f"Worksheet {sheet_name!r} not found in workbook.")


def _read_sheet_rows(path: str | Path, sheet_name: str) -> list[dict[int, object]]:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        root = ET.fromstring(archive.read(_sheet_path(archive, sheet_name)))
        rows: list[dict[int, object]] = []
        for row in root.findall(f".//{{{_MAIN_NS}}}row"):
            values: dict[int, object] = {}
            for cell in row.findall(f"{{{_MAIN_NS}}}c"):
                ref = cell.attrib.get("r", "")
                if not ref:
                    continue
                col = _column_number(ref)
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{{{_MAIN_NS}}}v")
                raw = "" if value_node is None else (value_node.text or "")
                if cell_type == "s" and raw != "":
                    try:
                        value = shared[int(raw)]
                    except (ValueError, IndexError):
                        value = raw
                elif cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iter(f"{{{_MAIN_NS}}}t"))
                elif cell_type == "b":
                    value = raw == "1"
                else:
                    value = raw
                values[col] = value
            if values:
                rows.append(values)
        return rows


def _find_overall_block(rows: list[dict[int, object]]) -> dict[str, int]:
    if not rows:
        raise ValueError("Athletic workbook ranking sheet was empty.")
    header = {col: str(value).strip().upper() for col, value in rows[0].items()}
    player_cols = [col for col, value in header.items() if value == "OVERALL PLAYER"]
    if not player_cols:
        raise ValueError("Could not find the OVERALL PLAYER ranking block.")

    candidates = []
    for player_col in player_cols:
        rank_col = player_col - 1
        nearby = range(player_col + 1, player_col + 7)
        pos_col = next((c for c in nearby if "POS RK" in header.get(c, "")), None)
        bye_col = next((c for c in nearby if header.get(c, "") == "BYE"), None)
        points_col = next((c for c in nearby if header.get(c, "") in {"FPS", "CUSTOM"}), None)
        vorp_col = next((c for c in nearby if header.get(c, "") == "VORP"), None)
        if pos_col is None or points_col is None or vorp_col is None:
            continue

        seen = 0
        sequential = 0
        for row in rows[1:31]:
            name = str(row.get(player_col, "")).strip()
            rank = _numeric(row.get(rank_col))
            if not name or math.isnan(rank):
                continue
            seen += 1
            if int(rank) == seen:
                sequential += 1
        candidates.append(
            (
                sequential,
                seen,
                {
                    "rank": rank_col,
                    "player": player_col,
                    "pos_rank": pos_col,
                    "bye": bye_col or -1,
                    "points": points_col,
                    "vorp": vorp_col,
                },
            )
        )

    if not candidates:
        raise ValueError("Could not identify a usable overall rankings block.")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def extract_rankings_from_rows(
    rows: list[dict[int, object]],
    player_names: list[str],
) -> pd.DataFrame:
    block = _find_overall_block(rows)
    known = {_normalize_name(name): name for name in player_names}
    records = []

    for row in rows[1:]:
        raw_name = str(row.get(block["player"], "")).strip()
        if not raw_name:
            continue
        matched_name = known.get(_normalize_name(raw_name))
        if not matched_name:
            continue

        rank = _numeric(row.get(block["rank"]))
        points = _numeric(row.get(block["points"]))
        vorp = _numeric(row.get(block["vorp"]))
        pos_rank = str(row.get(block["pos_rank"], "")).strip().upper()
        pos_match = re.match(r"([A-Z]+)", pos_rank)
        pos = pos_match.group(1) if pos_match else ""

        # This local source is intentionally offense-only. PatBot keeps its
        # own DEF treatment because the workbook's defense settings were left
        # at source defaults.
        if pos not in {"QB", "RB", "WR", "TE"}:
            continue
        if math.isnan(rank) or math.isnan(vorp):
            continue

        bye = _numeric(row.get(block["bye"])) if block["bye"] > 0 else np.nan
        records.append(
            {
                "name": matched_name,
                "athletic_rank": float(rank),
                "athletic_points": float(points) if not math.isnan(points) else np.nan,
                "athletic_vorp": float(vorp),
                "athletic_pos_rank": pos_rank,
                "athletic_bye": float(bye) if not math.isnan(bye) else np.nan,
            }
        )

    if not records:
        raise ValueError("Athletic workbook parsed but no offensive PatBot players matched.")
    return (
        pd.DataFrame(records)
        .sort_values("athletic_rank")
        .drop_duplicates("name", keep="first")
        .reset_index(drop=True)
    )


def _settings_from_rows(rows: list[dict[int, object]]) -> tuple[dict[str, float], dict[str, float]]:
    scoring = {}
    roster = {}
    for row in rows[1:]:
        left_name = str(row.get(1, "")).strip().upper()
        left_value = _numeric(row.get(2))
        if left_name and not math.isnan(left_value):
            scoring[left_name] = float(left_value)

        right_name = str(row.get(4, "")).strip().upper()
        right_value = _numeric(row.get(5))
        if right_name and not math.isnan(right_value):
            roster[right_name] = float(right_value)
    return scoring, roster


def validate_core_settings(path: str | Path, config: dict) -> list[str]:
    try:
        rows = _read_sheet_rows(path, "Settings")
        scoring, roster = _settings_from_rows(rows)
    except Exception as exc:
        return [f"Could not validate workbook settings: {exc}"]

    expected_scoring = {
        "COMPLETIONS": float(config["scoring"].get("pass_completion", 0.0)),
        "PASS YARDS": 1.0 / float(config["scoring"].get("pass_yards_per_point", 25)),
        "PASS TDS": float(config["scoring"].get("pass_td", 0.0)),
        "INTERCEPTIONS": float(config["scoring"].get("interception", 0.0)),
        "RUSH YARDS": 1.0 / float(config["scoring"].get("rush_yards_per_point", 10)),
        "RUSH TDS": float(config["scoring"].get("rush_td", 0.0)),
        "RECEPTIONS (RB)": float(config["scoring"].get("reception", 0.0)),
        "RECEPTIONS (WR)": float(config["scoring"].get("reception", 0.0)),
        "RECEPTIONS (TE)": float(config["scoring"].get("reception", 0.0)),
        "RECV YARDS": 1.0 / float(config["scoring"].get("rec_yards_per_point", 10)),
        "RECV TDS": float(config["scoring"].get("rec_td", 0.0)),
    }
    expected_roster = {
        "TEAMS": float(config["league"].get("teams", 12)),
        "STARTING QB": float(config["roster"].get("QB", 0)),
        "STARTING RB": float(config["roster"].get("RB", 0)),
        "STARTING WR": float(config["roster"].get("WR", 0)),
        "STARTING TE": float(config["roster"].get("TE", 0)),
        "STARTING FLEX": float(config["roster"].get("FLEX", 0)),
    }

    warnings = []
    for name, expected in expected_scoring.items():
        actual = scoring.get(name)
        if actual is None:
            warnings.append(f"Missing setting {name}")
        elif abs(actual - expected) > 1e-6:
            warnings.append(f"{name}: workbook {actual:g} vs PatBot {expected:g}")
    for name, expected in expected_roster.items():
        actual = roster.get(name)
        if actual is None:
            warnings.append(f"Missing setting {name}")
        elif abs(actual - expected) > 1e-6:
            warnings.append(f"{name}: workbook {actual:g} vs PatBot {expected:g}")
    return warnings


def load_athletic_custom(
    player_names: list[str],
    path: str | Path,
    config: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No local Athletic workbook at {path}. Upload/copy the latest .xlsx into PatBot first."
        )

    rows = _read_sheet_rows(path, "OVR & VORP Ranks")
    rankings = extract_rankings_from_rows(rows, player_names)
    warnings = validate_core_settings(path, config) if config else []
    modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    status = {
        "ok": True,
        "matched": int(rankings["name"].nunique()),
        "file": path.name,
        "modified_local": modified,
        "warning": "; ".join(warnings[:4]) if warnings else "",
    }
    return rankings, status
