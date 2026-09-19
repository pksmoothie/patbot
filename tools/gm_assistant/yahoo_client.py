"""Isolated public-client OAuth and GET-only Yahoo Fantasy connection probe."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlencode, urlsplit
import webbrowser
import xml.etree.ElementTree as ET

AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
API_URL = "https://fantasysports.yahooapis.com/fantasy/v2"
ROOT = Path(__file__).resolve().parent
SERVICE = "patbot.gm_assistant.yahoo.public.v1"
REDIRECT = "oob"
TARGET = "Fantasy Premier League"


class YahooError(Exception):
    """Only fixed, sanitized messages may cross the CLI boundary."""


def pkce_challenge(verifier):
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).decode().rstrip("=")


def generate_pkce():
    verifier = secrets.token_urlsafe(64)
    return verifier, pkce_challenge(verifier)


def authorization_url(client_id, challenge, state):
    # API permissions come from the approved YDN app; no OpenID/profile request.
    return AUTH_URL + "?" + urlencode(dict(client_id=client_id, redirect_uri=REDIRECT,
        response_type="code", code_challenge=challenge, code_challenge_method="S256", state=state))


def hidden_prompt(label, *, strip=True):
    # Fail closed: getpass can otherwise fall back to echoed stdin.
    if not sys.stdin.isatty():
        raise YahooError("Run this command in your own interactive local terminal.")
    value = getpass.getpass(label)
    return value.strip() if strip else value


class WindowsStore:
    def __init__(self):
        if sys.platform != "win32":
            raise YahooError("This workflow requires Windows Credential Manager.")
        try:
            from keyring.backends.Windows import WinVaultKeyring
            self.backend = WinVaultKeyring()  # Never select a plaintext fallback.
            self.backend.persist = "local machine"
        except Exception:
            raise YahooError("Install requirements-yahoo.txt to enable Windows Credential Manager.") from None

    def get(self, account):
        try:
            credential = self.backend._read_credential(SERVICE + ":" + account)
            return credential["CredentialBlob"].decode("utf-8") if credential else None
        except Exception:
            raise YahooError("Windows Credential Manager read failed.") from None

    def set(self, account, value):
        try:
            # A single CredWrite replaces the whole token bundle, including rotation.
            write_credential_blob(SERVICE + ":" + account, account, value.encode("utf-8"))
        except Exception:
            raise YahooError("Windows Credential Manager write failed; stop and reauthorize if needed.") from None


def write_credential_blob(target, account, value):
    """One native write; avoid keyring backups and its UTF-16 blob conversion."""
    import ctypes
    from win32ctypes.core.ctypes._authentication import CREDENTIAL, PCREDENTIAL, _CredWrite
    if len(value) > 2560:
        raise YahooError("Token bundle exceeds Windows credential capacity.")
    credential = CREDENTIAL.fromdict(dict(Type=1, TargetName=target,
        UserName=account, Comment="GM assistant Yahoo OAuth", Persist=2))
    blob = ctypes.create_string_buffer(value)
    credential.CredentialBlobSize = len(value)
    credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
    _CredWrite(PCREDENTIAL(credential), 0)


@contextmanager
def single_process():
    """Serialize auth/refresh across launchers, avoiding refresh rotation races."""
    import msvcrt
    with (ROOT / ".yahoo.lock").open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise YahooError("Another Yahoo command is running; finish or close it first.") from None
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def valid_secret(value):
    return isinstance(value, str) and bool(value) and not any(c.isspace() for c in value)


class Transport:
    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.trust_env = False  # No ambient netrc credentials or proxy auth.

    def request(self, method, url, **kwargs):
        try:
            return self.session.request(method, url, timeout=(10, 30), allow_redirects=False, **kwargs)
        except Exception:
            raise YahooError("Yahoo network request failed. Check connectivity and retry.") from None


class OAuth:
    def __init__(self, client_id, store, transport, clock=time.time):
        self.client_id, self.store, self.transport, self.clock = client_id, store, transport, clock
        self.account = "tokens:" + hashlib.sha256(client_id.encode()).hexdigest()
        self.sensitive = {client_id}

    def load(self):
        raw = self.store.get(self.account)
        if raw is None:
            return None
        try:
            token = json.loads(raw)
            if not isinstance(token, dict) or not all(valid_secret(token.get(k)) for k in ("access_token", "refresh_token")):
                raise ValueError()
            expiry = token["expires_at"]
            if isinstance(expiry, bool) or not isinstance(expiry, (int, float)) or not math.isfinite(expiry):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise YahooError("Saved Yahoo tokens are malformed. Run YAHOO_AUTH.bat --reauthorize.") from None
        self.sensitive.update((token["access_token"], token["refresh_token"]))
        return token

    def exchange(self, params, previous=None):
        response = self.transport.request("POST", TOKEN_URL, data={"client_id": self.client_id,
            "redirect_uri": REDIRECT, **params}, headers={"Accept": "application/json"})
        if response.status_code != 200:
            raise YahooError("Yahoo token exchange rejected. Code/refresh token may be expired or revoked. "
                "Run YAHOO_AUTH.bat --reauthorize. If Yahoo rejects oob or the public client, stop; do not change the redirect or add a secret.")
        try:
            body = response.json()
            access = body["access_token"]
            refresh = body.get("refresh_token", previous)
            lifetime = body["expires_in"]
            if isinstance(lifetime, bool):
                raise ValueError()
            lifetime = float(lifetime)
            if not math.isfinite(lifetime) or lifetime <= 0 or not valid_secret(access) or not valid_secret(refresh):
                raise ValueError()
            if str(body.get("token_type", "")).lower() != "bearer":
                raise ValueError()
        except (ValueError, KeyError, TypeError, AttributeError):
            raise YahooError("Yahoo returned malformed tokens; nothing was saved. Reauthorize.") from None
        token = dict(access_token=access, refresh_token=refresh, expires_at=self.clock() + lifetime)
        self.sensitive.update((access, refresh))
        self.store.set(self.account, json.dumps(token))
        return access

    def access_token(self, force_refresh=False):
        token = self.load()
        if token is None:
            raise YahooError("Authenticate first with YAHOO_AUTH.bat.")
        if not force_refresh and token["expires_at"] > self.clock() + 60:
            return token["access_token"]
        return self.exchange(dict(grant_type="refresh_token", refresh_token=token["refresh_token"]), token["refresh_token"])

    def authorize(self):
        verifier, challenge = generate_pkce()
        self.sensitive.add(verifier)
        url = authorization_url(self.client_id, challenge, secrets.token_urlsafe(32))
        print("Opening Yahoo in your normal browser. Authorize only the approved read-only Fantasy app.")
        print("If Yahoo rejects oob/callback, stop here (Ctrl+C). No redirect registration will be changed.")
        if not webbrowser.open(url):
            raise YahooError("Could not open your browser. Set your Windows default browser and retry.")
        code = hidden_prompt("Yahoo short-lived authorization code (hidden; blank to stop): ")
        if not code:
            raise YahooError("Authorization stopped; no token exchange attempted.")
        if not valid_secret(code) or len(code) > 2048 or "://" in code:
            raise YahooError("Enter only the short-lived code displayed by Yahoo in this local terminal.")
        self.sensitive.add(code)
        self.exchange(dict(grant_type="authorization_code", code=code, code_verifier=verifier))


def parse_xml(content):
    try:
        if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
            raise ValueError()
        root = ET.fromstring(content)
        for node in root.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        if root.tag != "fantasy_content":
            raise ValueError()
        return root
    except (ET.ParseError, ValueError, TypeError):
        raise YahooError("Yahoo returned an unexpected Fantasy XML response.") from None


class FantasyClient:
    def __init__(self, oauth, transport):
        self.oauth, self.transport = oauth, transport

    def get(self, path):
        # Fixed origin and a narrow resource allowlist; no generic write interface.
        if not (path == "users;use_login=1/games;game_codes=nfl/leagues" or
                re.fullmatch(r"league/\d+\.l\.\d+/(teams|settings)", path)):
            raise YahooError("Fantasy resource is outside this read-only connection milestone.")
        for attempt in range(2):
            token = self.oauth.access_token(force_refresh=bool(attempt))
            response = self.transport.request("GET", API_URL + "/" + path,
                headers={"Authorization": "Bearer " + token, "Accept": "application/xml"})
            if response.status_code == 401 and attempt == 0:
                continue
            if response.status_code != 200:
                if response.status_code == 401:
                    raise YahooError("Yahoo rejected refreshed access. Run YAHOO_AUTH.bat --reauthorize.")
                raise YahooError("Yahoo Fantasy GET failed (HTTP %d)." % response.status_code)
            return parse_xml(response.content)
        raise YahooError("Yahoo authentication failed.")


def fields(node, names):
    return {name: node.findtext(name) for name in names if node.findtext(name) is not None}


def discover_leagues(root):
    games, leagues, seen = [], [], set()
    for game in root.findall("./users/user/games/game"):
        metadata = fields(game, ("game_key", "game_id", "code", "name", "season", "is_game_over"))
        if metadata.get("code") != "nfl":
            continue
        games.append(metadata)
        for league in game.findall("./leagues/league"):
            item = fields(league, ("league_key", "league_id", "name", "season", "num_teams", "scoring_type"))
            key = item.get("league_key", "")
            if not re.fullmatch(r"\d+\.l\.\d+", key):
                raise YahooError("Yahoo returned a malformed league identifier.")
            item.setdefault("season", metadata.get("season"))
            item["game_key"] = metadata.get("game_key")
            if key not in seen:
                leagues.append(item)
                seen.add(key)
    # Yahoo identifies NFL season by year; January still belongs to the prior season.
    now = datetime.now(timezone.utc)
    season = str(now.year if now.month >= 3 else now.year - 1)
    for item in leagues:
        item["current_season"] = item.get("season") == season
    return games, leagues, season


SETTINGS_FIELDS = frozenset("draft_type scoring_type uses_playoff has_playoff_consolation has_multiweek_championship waiver_type waiver_rule uses_faab draft_time post_draft_players free_agents_lock waiver_time waiver_pickup max_teams num_playoff_teams playoff_start_week uses_playoff_reseeding uses_lock_eliminated_teams num_playoff_consolation_teams has_multiweek_consolation max_weekly_adds max_season_adds max_season_trades trade_end_date trade_ratify_type trade_reject_time player_pool cant_cut_list_provider roster_positions roster_position position position_type count stat_categories stats stat stat_id enabled name display_name sort_order stat_position_types stat_position_type is_only_display_stat stat_modifiers value is_composite_stat base_stat bonuses bonus target points".split())


def settings_metadata(node):
    """Project only known settings fields; omit URLs, managers, messages and unknowns."""
    result = {}
    for child in node:
        if child.tag not in SETTINGS_FIELDS:
            continue
        value = settings_metadata(child) if len(child) else child.text
        result.setdefault(child.tag, []).append(value)
    return result


def connection_test(client):
    games, leagues, season = discover_leagues(client.get("users;use_login=1/games;game_codes=nfl/leagues"))
    for league in leagues:
        if not league["current_season"]:
            continue
        key = league["league_key"]
        teams = client.get("league/" + key + "/teams")
        league["teams"] = [fields(team, ("team_key", "team_id", "name", "is_owned_by_current_login"))
                           for team in teams.findall("./league/teams/team")]
        try:
            settings = client.get("league/" + key + "/settings").find("./league/settings")
            league["settings_status"] = "ok" if settings is not None else "absent"
            league["settings"] = settings_metadata(settings) if settings is not None else {}
        except YahooError as exc:
            # Only settings availability is optional. Record fixed error, never response bodies.
            league["settings_status"] = "unavailable"
            league["settings_error"] = str(exc)
    matches = [x["league_key"] for x in leagues if x["current_season"] and x.get("name", "").casefold() == TARGET.casefold()]
    return dict(retrieved_at=datetime.now(timezone.utc).isoformat(), status="read_only_access_proven",
        season=season, season_basis="NFL season year (March-February); compared with Yahoo season metadata",
        games=games, leagues=leagues, target_name=TARGET, target_league_keys=matches,
        target_status="unique_match" if len(matches) == 1 else "ambiguous" if matches else "not_found")


def write_audit(path, result, sensitive):
    forbidden = {"access_token", "refresh_token", "authorization_code", "code", "code_verifier", "pkce_verifier", "client_secret", "client_id", "id_token"}
    def scrub(value):
        if isinstance(value, dict):
            # game 'code' (nfl) is deliberately omitted too.
            return {k: scrub(v) for k, v in value.items() if k.lower() not in forbidden}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str):
            for secret in sorted(sensitive, key=len, reverse=True):
                if secret:
                    value = value.replace(secret, "[REDACTED]")
        return value
    safe = scrub(result)
    encoded = json.dumps(safe, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    return safe


def replace_consumer_key(store):
    """Replace only the client ID, then verify its storage and URL offline."""
    entered = hidden_prompt("Replacement Yahoo Consumer Key (hidden): ", strip=False)
    if not valid_secret(entered) or not entered.isascii() or len(entered) < 9:
        raise YahooError("Consumer Key is empty, too short, or contains whitespace/non-ASCII characters; nothing saved.")
    store.set("consumer_key", entered)
    stored = store.get("consumer_key")
    if stored != entered:
        raise YahooError("Consumer Key read-back verification failed. No authentication attempted.")
    _, challenge = generate_pkce()
    url = authorization_url(stored, challenge, secrets.token_urlsafe(32))
    parameters = parse_qs(urlsplit(url).query, strict_parsing=True)
    if parameters.get("client_id") != [stored]:
        raise YahooError("Offline client_id verification failed. No authentication attempted.")
    return dict(character_length=len(stored), first_4=stored[:4], last_4=stored[-4:],
        sha256=hashlib.sha256(stored.encode("utf-8")).hexdigest(),
        offline_client_id_matches_stored_value=True)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("auth", "test", "replace-key"))
    parser.add_argument("--reauthorize", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command != "auth" and args.reauthorize:
            raise YahooError("--reauthorize is supported only by the auth command.")
        store = WindowsStore()
        with single_process():
            if args.command == "replace-key":
                print(json.dumps(replace_consumer_key(store), indent=2))
                return 0
            client_id = store.get("consumer_key")
            if not client_id:
                if args.command != "auth":
                    raise YahooError("Run YAHOO_AUTH.bat first to enter your Consumer Key locally.")
                client_id = hidden_prompt("Yahoo Consumer Key / OAuth client_id (hidden): ")
                if not valid_secret(client_id):
                    raise YahooError("Consumer Key was empty or malformed; nothing saved.")
                store.set("consumer_key", client_id)
            transport = Transport()
            oauth = OAuth(client_id, store, transport)
            if args.command == "auth":
                if args.reauthorize or oauth.load() is None:
                    oauth.authorize()
                else:
                    oauth.access_token(force_refresh=True)
                print("Yahoo authentication succeeded. Tokens saved in Windows Credential Manager.")
                print("Run YAHOO_TEST.bat to prove Fantasy API access.")
            else:
                # Refresh on every validation run to exercise persistent authentication.
                oauth.access_token(force_refresh=True)
                result = connection_test(FantasyClient(oauth, transport))
                safe = write_audit(ROOT / "data" / "yahoo_connection_test.json", result, oauth.sensitive)
                print(json.dumps(safe, indent=2, ensure_ascii=True))
                print("Saved sanitized data/yahoo_connection_test.json. FantasyPros/GM sources unchanged.")
        return 0
    except YahooError as exc:
        print(str(exc), file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print("Yahoo workflow stopped.", file=sys.stderr)
    except Exception:
        # Never expose tracebacks containing credentials, bodies or request URLs.
        print("Yahoo workflow failed locally; no diagnostic secrets were printed. Check setup and retry.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
