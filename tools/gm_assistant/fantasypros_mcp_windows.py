"""Windows Credential Manager adapter for the direct FantasyPros MCP probe.

This reuses the same native credential write path already proven by the Yahoo
client. It exists separately so the OAuth/MCP probe can stay focused on protocol
behavior while Windows persistence is validated.
"""
from __future__ import annotations

import sys

import fantasypros_mcp as fp
from keyring.backends.Windows import WinVaultKeyring
from yahoo_client import write_credential_blob


class NativeWindowsTokenStorage:
    """MCP TokenStorage backed by native Windows Generic Credentials."""

    def __init__(self):
        if sys.platform != "win32":
            raise fp.FantasyProsMCPError("This workflow currently requires Windows Credential Manager.")
        try:
            self.backend = WinVaultKeyring()
            self.backend.persist = "local machine"
        except Exception:
            raise fp.FantasyProsMCPError("Windows Credential Manager is unavailable.") from None

    @staticmethod
    def _target(account: str) -> str:
        return fp.SERVICE + ":" + account

    def _get(self, account: str) -> str | None:
        try:
            credential = self.backend._read_credential(self._target(account))
            return credential["CredentialBlob"].decode("utf-8") if credential else None
        except Exception:
            raise fp.FantasyProsMCPError("Windows Credential Manager read failed.") from None

    def _set(self, account: str, value: str) -> None:
        try:
            encoded = value.encode("utf-8")
            if len(encoded) > 2560:
                raise ValueError("credential too large")
            write_credential_blob(self._target(account), account, encoded)
        except Exception:
            raise fp.FantasyProsMCPError("Windows Credential Manager write failed.") from None

    async def get_tokens(self):
        raw = self._get(fp.TOKENS_ACCOUNT)
        if raw is None:
            return None
        try:
            return fp.OAuthToken.model_validate_json(raw)
        except Exception:
            raise fp.FantasyProsMCPError("Saved FantasyPros OAuth tokens are malformed.") from None

    async def set_tokens(self, tokens) -> None:
        self._set(fp.TOKENS_ACCOUNT, tokens.model_dump_json())

    async def get_client_info(self):
        raw = self._get(fp.CLIENT_ACCOUNT)
        if raw is None:
            return None
        try:
            return fp.OAuthClientInformationFull.model_validate_json(raw)
        except Exception:
            raise fp.FantasyProsMCPError("Saved FantasyPros OAuth client registration is malformed.") from None

    async def set_client_info(self, client_info) -> None:
        self._set(fp.CLIENT_ACCOUNT, client_info.model_dump_json())


fp.WindowsTokenStorage = NativeWindowsTokenStorage

if __name__ == "__main__":
    raise SystemExit(fp.main())
