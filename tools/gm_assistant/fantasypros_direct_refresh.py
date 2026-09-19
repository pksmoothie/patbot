"""Run the GM FantasyPros refresh directly through MCP, without Codex."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import fantasypros_mcp_windows as fpw
import refresh
from gm_snapshot import build_snapshot, decode, persist, utc_now


MAX_CALL_ATTEMPTS = 3


def raw_result(result):
    """Convert the SDK CallToolResult to JSON-compatible data without summarizing it."""
    return json.loads(result.model_dump_json(by_alias=True))


def save_raw(folder: Path, item: dict, raw: dict) -> dict:
    payload = {"fetched_at": utc_now(), "raw": raw}
    refresh.json_write(folder / f"{item['id']}.json", payload)
    return payload


async def call_tool_once(item: dict) -> dict:
    """Use a short-lived MCP session so one dropped stream cannot kill a long refresh."""
    oauth = fpw.fp.oauth_provider()
    async with httpx2.AsyncClient(auth=oauth) as http_client:
        async with streamable_http_client(fpw.fp.MCP_URL, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tool = item["tool"].removeprefix("mcp__fantasypros__")
                result = await session.call_tool(tool, item["arguments"])
                return raw_result(result)


async def collect(folder: Path, item: dict) -> dict:
    """Retry only the failed MCP call, always on a brand-new transport/session."""
    for attempt in range(1, MAX_CALL_ATTEMPTS + 1):
        try:
            raw = await call_tool_once(item)
            return save_raw(folder, item, raw)["raw"]
        except ExceptionGroup:
            if attempt >= MAX_CALL_ATTEMPTS:
                raise
            print(
                f"FantasyPros MCP call {item['id']} lost its connection; retrying "
                f"({attempt + 1}/{MAX_CALL_ATTEMPTS})...",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(2)
    raise RuntimeError("unreachable")


async def run_direct() -> Path:
    folder = refresh.prepare_run()
    print(f"Collecting directly from FantasyPros MCP. Run evidence: {folder}", flush=True)

    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    core = manifest["requests"]

    # Resync must be first and positively confirmed before any other league data is read.
    resync_item = core[0]
    if resync_item["id"] != "resync":
        raise ValueError("Core request order no longer starts with resync.")
    resync_raw = await collect(folder, resync_item)
    resync_data = decode(resync_raw)
    if not isinstance(resync_data, dict) or resync_data.get("synced") is not True:
        raise ValueError("FantasyPros resync was not explicitly confirmed; refresh stopped.")

    for item in core[1:]:
        await collect(folder, item)

    bundle = refresh.bundle_from_run(folder)
    # Validate the core collection before asking for optional enrichment.
    build_snapshot(bundle)
    tasks, skipped = refresh.enrichment_requests(bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["requests"] = core + tasks
    manifest["skipped"] = skipped
    refresh.json_write(manifest_path, manifest)

    for item in tasks:
        await collect(folder, item)

    snapshot = build_snapshot(refresh.bundle_from_run(folder))
    destination = persist(snapshot)
    print(f"Published {snapshot['refresh_status']} snapshot: {destination}")
    return destination


def main() -> int:
    try:
        asyncio.run(run_direct())
        return 0
    except KeyboardInterrupt:
        print("Direct FantasyPros refresh stopped.", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Direct refresh stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Avoid dumping OAuth/MCP request details. The run folder retains non-secret call evidence.
        print(f"Direct refresh failed ({type(exc).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
