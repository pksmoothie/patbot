from patbot.config import load_config
from patbot.sleeper import refresh_snapshot, SleeperDataError


def main():
    cfg = load_config()
    print("PatBot v0.3.8 — refreshing 2026 projections, market ADP and ranking sources...")
    print("This checks public feeds plus any local private Athletic workbook.")
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

    print("\nIndependent source status:")
    for source, status in meta.get("market_sources", {}).items():
        if status.get("ok"):
            extra = f" • {status['file']}" if status.get("file") else ""
            print(f"  OK   {source}: {status.get('matched', '?')} players matched{extra}")
            if status.get("warning"):
                print(f"       warning: {status['warning']}")
        else:
            print(f"  WARN {source}: {status.get('error', 'unavailable')}")

    print("\nNext: run  .\\.venv\\Scripts\\python.exe -m streamlit run app.py")


if __name__ == "__main__":
    main()
