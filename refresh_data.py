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
            if status.get("warning"):
                print(f"       warning: {status['warning']}")
            if status.get("note"):
                print(f"       {status['note']}")
        else:
            print(f"  WARN {source}: {status.get('error', 'unavailable')}")


def main():
    cfg = load_config()
    print("PatBot v0.4.1 — refreshing projections, market data, risk and championship-strategy inputs...")
    print(
        "This checks public feeds, the Premium FantasyPros market/injury/news/history feeds, "
        "and any local private Athletic workbook."
    )
    try:
        csv_path, meta_path, meta = refresh_snapshot(cfg)
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

    _print_status_block("Independent source status", meta.get("market_sources", {}))
    _print_status_block("Risk & availability source status", meta.get("risk_sources", {}))

    print("\nNext: run  .\\.venv\\Scripts\\python.exe -m streamlit run app.py")


if __name__ == "__main__":
    main()
