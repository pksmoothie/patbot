import json
from pathlib import Path

import pandas as pd

from patbot.config import load_config
from patbot.sleeper import refresh_snapshot, SleeperDataError


YAHOO_CACHE_PATH = Path("data/yahoo_adp_2026.csv")


def _print_status_block(title: str, statuses: dict):
    print(f"\n{title}:")
    if not statuses:
        print("  WARN no status returned")
        return
    for source, status in statuses.items():
        if status.get("ok"):
            extra = f" • {status['file']}" if status.get("file") else ""
            seasons = status.get("seasons")
            if seasons:
                extra += f" • seasons {', '.join(str(x) for x in seasons)}"
            matched = status.get("matched")
            matched_text = f": {matched} players matched" if matched is not None else ""
            print(f"  OK   {source}{matched_text}{extra}")
            if status.get("coverage_pct") is not None:
                print(f"       coverage: {status['coverage_pct']}%")
            if status.get("sleeper_weight") is not None:
                print(
                    f"       blend weights: Sleeper {status['sleeper_weight']:.0%} / "
                    f"FantasyPros {status.get('fantasypros_weight', 0):.0%}"
                )
            if status.get("transport"):
                print(f"       transport: {status['transport']}")
            if status.get("warning"):
                print(f"       warning: {status['warning']}")
            if status.get("note"):
                print(f"       {status['note']}")
        else:
            print(f"  WARN {source}: {status.get('error', status.get('reason', 'unavailable'))}")


def _write_meta(meta_path, meta):
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _attach_projection_sources(csv_path, meta_path, meta, cfg):
    """Attach FantasyPros full-stat projections and build production blend."""
    enabled = bool(cfg.get("projection_sources", {}).get("fantasypros_preseason", True))
    if not enabled:
        meta["projection_sources"] = {
            "fantasypros_preseason_projections": {
                "ok": False,
                "error": "Disabled by projection_sources.fantasypros_preseason",
            }
        }
        return meta

    try:
        from patbot.consensus import add_consensus_values
        from patbot.fantasypros_projection import augment_fantasypros_projections
        from patbot.projection_blend import blend_projection_sources

        frame = pd.read_csv(csv_path)
        frame, fp_status = augment_fantasypros_projections(frame, cfg)
        frame, blend_status = blend_projection_sources(frame, cfg)
        frame = add_consensus_values(frame, cfg)
        frame.to_csv(csv_path, index=False)
        meta["projection_sources"] = {**fp_status, **blend_status}
    except Exception as exc:
        meta["projection_sources"] = {
            "projection_pipeline": {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        }

    _write_meta(meta_path, meta)
    return meta


def _refresh_yahoo_room_signal(csv_path, meta_path, meta, cfg):
    """Refresh Yahoo ADP separately for opponent/survival modeling.

    This intentionally does not merge Yahoo into the player valuation snapshot.
    If refresh fails, the previous cache is left untouched so the simulator can
    use it only while it remains within its freshness window.
    """
    enabled = bool(cfg.get("yahoo_room_behavior", {}).get("enabled", True))
    if not enabled:
        meta["yahoo_room_behavior"] = {
            "yahoo_adp": {"ok": False, "reason": "disabled"}
        }
        _write_meta(meta_path, meta)
        return meta

    try:
        from patbot.yahoo_adp import fetch_yahoo_adp

        frame = pd.read_csv(csv_path)
        names = frame["name"].dropna().astype(str).tolist()
        yahoo, status = fetch_yahoo_adp(names)
        YAHOO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        yahoo.to_csv(YAHOO_CACHE_PATH, index=False)
        status = dict(status)
        status["file"] = str(YAHOO_CACHE_PATH)
        status["coverage_pct"] = round(100.0 * len(yahoo) / max(len(frame), 1), 1)
        meta["yahoo_room_behavior"] = {"yahoo_adp": status}
    except Exception as exc:
        prior = YAHOO_CACHE_PATH.exists()
        meta["yahoo_room_behavior"] = {
            "yahoo_adp": {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "warning": "Prior Yahoo cache retained." if prior else "No prior Yahoo cache available.",
            }
        }

    _write_meta(meta_path, meta)
    return meta


def main():
    cfg = load_config()
    print("PatBot v0.6.0 — refreshing projections, market data, risk, owner tendencies and Yahoo room-behavior inputs...")
    print(
        "Yahoo ADP is refreshed only as a supporting opponent/survival signal; "
        "it does not enter PatBot projections, VORP, expert rank, or intrinsic valuation."
    )
    try:
        csv_path, meta_path, meta = refresh_snapshot(cfg)
        meta = _attach_projection_sources(csv_path, meta_path, meta, cfg)
        meta = _refresh_yahoo_room_signal(csv_path, meta_path, meta, cfg)
    except SleeperDataError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print(f"\nSaved player snapshot: {csv_path}")
    print(f"Saved metadata:       {meta_path}")
    print(f"Players:              {meta['draftable_rows']}")
    print(f"Snapshot UTC:         {meta['snapshot_at_utc']}")

    _print_status_block("Independent market/ranking source status", meta.get("market_sources", {}))
    _print_status_block("Projection source / production blend status", meta.get("projection_sources", {}))
    _print_status_block("Risk & availability source status", meta.get("risk_sources", {}))
    _print_status_block("Yahoo supporting room-behavior status", meta.get("yahoo_room_behavior", {}))

    print("\nNext: run  .\\.venv\\Scripts\\python.exe -m streamlit run app.py")


if __name__ == "__main__":
    main()
