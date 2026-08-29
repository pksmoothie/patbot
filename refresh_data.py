from patbot.config import load_config
from patbot.refresh_pipeline import run_full_refresh
from patbot.sleeper import SleeperDataError


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


def main():
    cfg = load_config()
    print("PatBot v0.6.3 — refreshing the full production draft-day data chain...")
    print(
        "Sleeper + FantasyPros production projections, market/ranking sources, risk history, "
        "Athletic local input when present, and Yahoo as a supporting opponent-room signal."
    )
    print(
        "Yahoo does not enter PatBot projections, VORP, expert rank or intrinsic valuation."
    )
    try:
        csv_path, meta_path, meta = run_full_refresh(cfg)
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
