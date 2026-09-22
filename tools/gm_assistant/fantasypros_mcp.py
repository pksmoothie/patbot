"""Direct FantasyPros MCP OAuth probe for the GM assistant.

This client bypasses Codex. It uses the official MCP Python SDK, performs OAuth
in the user's normal browser, stores OAuth client registration and tokens in
Windows Credential Manager, then opens a read-only MCP session and lists tools.
No access token, refresh token, client registration secret, or authorization
code is printed or written to the repository.
"""
from __future__ import annotations

import asyncio
import json
import sys
import webbrowser
from urllib.parse import parse_qs, urlsplit

import httpx2
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)
from pydantic import AnyUrl

MCP_URL = "https://api.fantasypros.com/mcp"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 8766
CALLBACK_PATH = "/callback"
CALLBACK_URL = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
SERVICE = "patbot.gm_assistant.fantasypros.mcp.v1"
TOKENS_ACCOUNT = "oauth_tokens"
CLIENT_ACCOUNT = "oauth_client_info"

# FantasyPros currently proxies its public MCP endpoint to an AWS Bedrock
# AgentCore runtime. The MCP SDK correctly rejects unrelated protected-resource
# URLs by default, so keep that protection and allow only the specific backend
# shape FantasyPros is using rather than disabling validation globally.
BEDROCK_RESOURCE_HOST = "bedrock-agentcore.us-east-1.amazonaws.com"
BEDROCK_RESOURCE_PATH_PREFIX = (
    "/runtimes/arn%3Aaws%3Abedrock-agentcore%3Aus-east-1%3A100709682036%3Aruntime%2Ffp_mcp_server-"
)


class FantasyProsMCPError(Exception):
    """Safe fixed-message errors for the local CLI."""


async def validate_fantasypros_resource(server_url: str, prm_resource: str | None) -> None:
    """Accept only FantasyPros' public MCP resource or its known Bedrock proxy."""
    if server_url != MCP_URL:
        raise FantasyProsMCPError("Unexpected FantasyPros MCP server URL.")
    if not prm_resource or prm_resource == MCP_URL:
        return
    parsed = urlsplit(prm_resource)
    params = parse_qs(parsed.query)
    if (
        parsed.scheme == "https"
        and parsed.hostname == BEDROCK_RESOURCE_HOST
        and parsed.path.startswith(BEDROCK_RESOURCE_PATH_PREFIX)
        and params.get("qualifier") == ["DEFAULT"]
    ):
        return
    raise FantasyProsMCPError("FantasyPros returned an unexpected protected resource URL.")


class WindowsTokenStorage:
    """Persist MCP OAuth state only in Windows Credential Manager."""

    def __init__(self):
        if sys.platform != "win32":
            raise FantasyProsMCPError("This workflow currently requires Windows Credential Manager.")
        try:
            import keyring
            from keyring.backends.Windows import WinVaultKeyring

            backend = keyring.get_keyring()
            if not isinstance(backend, WinVaultKeyring):
                raise RuntimeError("Unexpected keyring backend")
            self.backend = backend
        except Exception:
            raise FantasyProsMCPError(
                "Windows Credential Manager is unavailable. Install the GM requirements and retry."
            ) from None

    def _get(self, account: str) -> str | None:
        try:
            return self.backend.get_password(SERVICE, account)
        except Exception:
            raise FantasyProsMCPError("Windows Credential Manager read failed.") from None

    def _set(self, account: str, value: str) -> None:
        try:
            self.backend.set_password(SERVICE, account, value)
        except Exception:
            raise FantasyProsMCPError("Windows Credential Manager write failed.") from None

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._get(TOKENS_ACCOUNT)
        if raw is None:
            return None
        try:
            return OAuthToken.model_validate_json(raw)
        except Exception:
            raise FantasyProsMCPError("Saved FantasyPros OAuth tokens are malformed.") from None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._set(TOKENS_ACCOUNT, tokens.model_dump_json())

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._get(CLIENT_ACCOUNT)
        if raw is None:
            return None
        try:
            return OAuthClientInformationFull.model_validate_json(raw)
        except Exception:
            raise FantasyProsMCPError("Saved FantasyPros OAuth client registration is malformed.") from None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._set(CLIENT_ACCOUNT, client_info.model_dump_json())


async def open_browser(auth_url: str) -> None:
    print("Opening FantasyPros authorization in your normal browser...")
    if not webbrowser.open(auth_url):
        print("Browser did not open automatically. Open this URL manually:")
        print(auth_url)


async def wait_for_callback() -> AuthorizationCodeResult:
    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[AuthorizationCodeResult] = loop.create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = (await asyncio.wait_for(reader.readline(), timeout=10)).decode(
                "ascii", errors="replace"
            ).strip()
            parts = request_line.split()
            target = parts[1] if len(parts) >= 2 else ""
            parsed = urlsplit(target)
            params = parse_qs(parsed.query)

            # Consume request headers before replying.
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                if line in (b"\r\n", b"\n", b""):
                    break

            ok = parsed.path == CALLBACK_PATH and bool(params.get("code"))
            body = (
                "FantasyPros authorization completed. You can close this tab and return to PowerShell."
                if ok
                else "FantasyPros authorization callback was invalid. Return to PowerShell."
            )
            payload = body.encode("utf-8")
            status = "200 OK" if ok else "400 Bad Request"
            writer.write(
                (f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
                 f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode("ascii")
                + payload
            )
            await writer.drain()

            if not result_future.done():
                if ok:
                    result_future.set_result(
                        AuthorizationCodeResult(
                            code=params["code"][0],
                            state=params.get("state", [None])[0],
                            iss=params.get("iss", [None])[0],
                        )
                    )
                else:
                    result_future.set_exception(
                        FantasyProsMCPError("FantasyPros returned an invalid OAuth callback.")
                    )
        except Exception as exc:
            if not result_future.done():
                result_future.set_exception(
                    exc if isinstance(exc, FantasyProsMCPError)
                    else FantasyProsMCPError("FantasyPros OAuth callback failed.")
                )
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    try:
        server = await asyncio.start_server(handle, CALLBACK_HOST, CALLBACK_PORT)
    except OSError:
        raise FantasyProsMCPError(
            f"Local callback port {CALLBACK_PORT} is unavailable. Close the process using it and retry."
        ) from None

    print("Waiting for FantasyPros authorization to return to this PC...")
    async with server:
        try:
            return await asyncio.wait_for(result_future, timeout=300)
        except TimeoutError:
            raise FantasyProsMCPError("FantasyPros authorization timed out after 5 minutes.") from None


def oauth_provider() -> OAuthClientProvider:
    return OAuthClientProvider(
        server_url=MCP_URL,
        client_metadata=OAuthClientMetadata(
            client_name="PatBot Fantasy GM",
            redirect_uris=[AnyUrl(CALLBACK_URL)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            scope="user:read offline_access",
            token_endpoint_auth_method="none",
            application_type="native",
        ),
        storage=WindowsTokenStorage(),
        redirect_handler=open_browser,
        callback_handler=wait_for_callback,
        validate_resource_url=validate_fantasypros_resource,
    )


async def test_connection() -> None:
    oauth = oauth_provider()
    async with httpx2.AsyncClient(auth=oauth) as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = sorted(tool.name for tool in tools.tools)
                print("FantasyPros MCP authentication succeeded.")
                print(f"Available tools: {len(names)}")
                print(", ".join(names))


def main() -> int:
    try:
        asyncio.run(test_connection())
        return 0
    except KeyboardInterrupt:
        print("FantasyPros MCP test stopped.", file=sys.stderr)
        return 130
    except FantasyProsMCPError as exc:
        print(f"FantasyPros MCP test stopped: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Do not dump exception repr/response bodies: OAuth failures can contain
        # sensitive request context. The exception class is sufficient for this probe.
        print(f"FantasyPros MCP test failed ({type(exc).__name__}).", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
