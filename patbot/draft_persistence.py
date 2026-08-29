from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile


DEFAULT_SESSION_PATH = Path("data/draft_session_2026.json")


def load_draft_session(path: str | Path = DEFAULT_SESSION_PATH) -> list[dict]:
    """Load a locally persisted draft history; malformed files fail closed."""
    target = Path(path)
    if not target.exists():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, dict):
        payload = payload.get("draft_history", [])
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, dict)]


def save_draft_session(
    draft_history: list[dict],
    path: str | Path = DEFAULT_SESSION_PATH,
) -> Path:
    """Atomically persist manual draft entry so an app restart does not erase it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"draft_history": list(draft_history)}

    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        temp_path = Path(handle.name)

    temp_path.replace(target)
    return target


def clear_draft_session(path: str | Path = DEFAULT_SESSION_PATH) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()
