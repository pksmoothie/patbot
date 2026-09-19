"""Prove direct FantasyPros MCP can read the authoritative GM roster without Codex."""
from __future__ import annotations

import asyncio
import json

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# Importing this adapter installs the proven native Windows credential storage
# into fantasypros_mcp before we construct the OAuth provider.
import fantasypros_mcp_windows as fpw

LEAGUE_KEY = "nfl~8922f34b-15ab-48d6-b51a-472098c3ae5a"
TEAM_ID = 6


def render_result(result) -> str:
    structured = getattr(result, "structuredContent", None)
    if structured:
        return json.dumps(structured, ensure_ascii=False, indent=2)
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


async def main_async() -> int:
    oauth = fpw.fp.oauth_provider()
    async with httpx2.AsyncClient(auth=oauth) as http_client:
        async with streamable_http_client(fpw.fp.MCP_URL, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "get_roster",
                    {"league_key": LEAGUE_KEY, "sport": "nfl", "team_id": TEAM_ID},
                )
                if getattr(result, "isError", False):
                    print("Direct FantasyPros roster read returned a tool error.")
                    return 1
                text = render_result(result)
                if not text:
                    print("Direct FantasyPros roster read succeeded but returned no printable roster payload.")
                    return 1
                print("Direct FantasyPros roster read succeeded.")
                print("Kamara present:", "YES" if "Kamara" in text else "NO")
                print("--- roster payload ---")
                print(text[:6000])
                return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Direct FantasyPros roster test failed ({type(exc).__name__}).")
        raise SystemExit(1)
