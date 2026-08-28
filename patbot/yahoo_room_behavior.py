from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from .market import normalize_name
from .yahoo_adp import manager_yahoo_weight


CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "yahoo_adp_2026.csv"
_ARCHETYPE_KEY = "_yahoo_room_archetype"
_WEIGHT_KEY = "_yahoo_room_weight"
_INSTALLED = False


def load_yahoo_room_cache(
    player_names: list[str],
    *,
    path: str | Path = CACHE_PATH,
    max_age_hours: float = 72.0,
) -> tuple[pd.DataFrame, dict]:
    """Load a recent local Yahoo ADP cache and match it to PatBot names.

    Yahoo is deliberately kept outside the player-valuation dataframe. The cache
    is used only by the opponent/availability simulator, and stale or missing
    data cleanly disables the signal.
    """
    cache = Path(path)
    if not cache.exists():
        return pd.DataFrame(columns=["name", "yahoo_adp"]), {
            "ok": False,
            "reason": "missing",
            "file": str(cache),
        }

    age_hours = max(0.0, (time.time() - cache.stat().st_mtime) / 3600.0)
    if age_hours > float(max_age_hours):
        return pd.DataFrame(columns=["name", "yahoo_adp"]), {
            "ok": False,
            "reason": "stale",
            "file": str(cache),
            "age_hours": round(age_hours, 2),
            "max_age_hours": float(max_age_hours),
        }

    try:
        raw = pd.read_csv(cache)
    except Exception as exc:
        return pd.DataFrame(columns=["name", "yahoo_adp"]), {
            "ok": False,
            "reason": f"read_error: {type(exc).__name__}: {exc}",
            "file": str(cache),
        }

    if "name" not in raw.columns or "yahoo_adp" not in raw.columns:
        return pd.DataFrame(columns=["name", "yahoo_adp"]), {
            "ok": False,
            "reason": "missing_required_columns",
            "file": str(cache),
        }

    known = {normalize_name(name): str(name) for name in player_names}
    rows = []
    for _, row in raw.iterrows():
        name = known.get(normalize_name(row.get("name", "")))
        value = pd.to_numeric(pd.Series([row.get("yahoo_adp")]), errors="coerce").iloc[0]
        if name and pd.notna(value) and float(value) > 0:
            rows.append({"name": name, "yahoo_adp": float(value)})

    out = pd.DataFrame(rows, columns=["name", "yahoo_adp"])
    if not out.empty:
        out = out.drop_duplicates("name", keep="last").reset_index(drop=True)

    return out, {
        "ok": not out.empty,
        "reason": "ok" if not out.empty else "no_matches",
        "file": str(cache),
        "matched": int(len(out)),
        "age_hours": round(age_hours, 2),
        "max_age_hours": float(max_age_hours),
    }


def blend_room_market(
    market_latent: np.ndarray,
    yahoo_adp: np.ndarray,
    yahoo_weight: float,
) -> np.ndarray:
    """Add Yahoo as a supporting opponent-behavior signal, never a replacement."""
    market = np.asarray(market_latent, dtype=float)
    yahoo = np.asarray(yahoo_adp, dtype=float)
    out = market.copy()
    weight = max(0.0, min(0.35, float(yahoo_weight)))
    valid = np.isfinite(market) & np.isfinite(yahoo) & (yahoo > 0)
    out[valid] = (1.0 - weight) * market[valid] + weight * yahoo[valid]
    return out


def attach_yahoo_room_cache(sim) -> dict:
    cfg = sim.cfg.get("yahoo_room_behavior", {}) if isinstance(sim.cfg, dict) else {}
    enabled = bool(cfg.get("enabled", True))
    max_age_hours = float(cfg.get("max_cache_age_hours", 72.0))

    sim.yahoo_adp = np.full(sim.n, np.nan, dtype=float)
    if not enabled:
        status = {"ok": False, "reason": "disabled", "matched": 0}
        sim.yahoo_room_status = status
        return status

    cache, status = load_yahoo_room_cache(
        sim.players["name"].astype(str).tolist(),
        max_age_hours=max_age_hours,
    )
    if not cache.empty:
        mapping = cache.set_index("name")["yahoo_adp"].to_dict()
        sim.yahoo_adp = np.array(
            [float(mapping.get(str(name), np.nan)) for name in sim.names],
            dtype=float,
        )
    status["coverage_pct"] = round(100.0 * float(np.isfinite(sim.yahoo_adp).mean()), 1)
    sim.yahoo_room_status = status
    return status


def install_yahoo_room_behavior_patch() -> None:
    """Inject Yahoo into opponent behavior only, after owner-history patching."""
    global _INSTALLED
    if _INSTALLED:
        return

    from .sim import FastDraftSimulator

    original_init = FastDraftSimulator.__init__
    original_manager_profile = FastDraftSimulator._manager_profile
    original_opponent_pick = FastDraftSimulator.opponent_pick

    def init_with_yahoo_room(self, engine):
        original_init(self, engine)
        attach_yahoo_room_cache(self)

    def manager_profile_with_yahoo_room(self, team_slot: int, archetype: str) -> dict:
        profile = original_manager_profile(self, team_slot, archetype)
        profile[_ARCHETYPE_KEY] = str(archetype)
        profile[_WEIGHT_KEY] = manager_yahoo_weight(str(archetype))
        return profile

    def opponent_pick_with_yahoo_room(
        self,
        available: np.ndarray,
        market_latent: np.ndarray,
        custom_latent: np.ndarray,
        roster_counts: np.ndarray,
        round_no: int,
        profile: dict,
    ) -> int:
        yahoo = getattr(self, "yahoo_adp", None)
        status = getattr(self, "yahoo_room_status", {}) or {}
        weight = float(profile.get(_WEIGHT_KEY, 0.0))
        room_market = market_latent
        if bool(status.get("ok")) and yahoo is not None and weight > 0:
            room_market = blend_room_market(market_latent, yahoo, weight)

        return original_opponent_pick(
            self,
            available,
            room_market,
            custom_latent,
            roster_counts,
            round_no,
            profile,
        )

    FastDraftSimulator.__init__ = init_with_yahoo_room
    FastDraftSimulator._manager_profile = manager_profile_with_yahoo_room
    FastDraftSimulator.opponent_pick = opponent_pick_with_yahoo_room
    _INSTALLED = True
