import json
from pathlib import Path

import pandas as pd

from patbot.config import load_config
from patbot.sleeper import refresh_snapshot, SleeperDataError


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
            if status.get("warning"):
                print(f"       warning: {status['warning']}")
            if status.get("note"):
                print(f"       {status['note']}")
        else:
            print(f"  WARN {source}: {status.get('error', 'unavailable')}")


def _attach_projection_sources(csv_path, meta_path, meta, cfg):
    """Attach FantasyPros full-stat projections and build production blend.

    v0.5.1 promotes the previously diagnostic FantasyPros stat projection into
    PatBot's base projection at a conservative configured weight. Consensus is
    then recomputed from the blended base projection. Athletic stays separate in
    the consensus layer rather than being counted twice.
    """
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

    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main():
    cfg = load_config()
    print("PatBot v0.5.1 — refreshing projections, market data, risk, championship-strategy, owner-history and quality-aware roster-strategy inputs...")
    print(
        "This checks Sleeper, FantasyPros ECR/ADP plus full preseason stat projections, "
        "builds the configured production projection blend, checks FantasyPros injury/news/history feeds, "
        "the promoted high-confidence league-history tendencies, PatBot's value-aware roster policy, "
        "quality-aware TE2 rules, and any local private Athletic workbook."
    )
    try:
        csv_path, meta_path, meta = refresh_snapshot(cfg)
        meta = _attach_projection_sources(csv_path, meta_path, meta, cfg)
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

    print("\nNext: run  .\\.venv\\Scripts\\python.exe -m streamlit run app.py")


if __name__ == "__main__":
    main()
