"""Explicit three-step diagnostic; never authorizes or retries Fantasy calls."""
import json
import logging
import re
from urllib.parse import quote, quote_plus
import xml.etree.ElementTree as ET

import yahoo_client as y

PATH = "users;use_login=1/games;game_codes=nfl/leagues"


def sanitize(value, sensitive):
    text = str(value)
    for secret in sorted(sensitive, key=len, reverse=True):
        if secret:
            for encoded in (secret, quote(secret, safe=""), quote_plus(secret)):
                text = text.replace(encoded, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:client_id|access_token|refresh_token|code_verifier|authorization_code)\s*[=:]\s*)[^\s&<,;]+", r"\1[REDACTED]", text)
    text = re.sub(r"[A-Za-z0-9_+/.=-]{64,}", "[REDACTED]", text)
    return " ".join(text.split())[:1500]


def response_summary(response, sensitive):
    result = {"http_status": response.status_code,
              "content_type": sanitize(response.headers.get("Content-Type", ""), sensitive)}
    if 200 <= response.status_code < 300:
        return result
    allowed = {"error", "error_description", "code", "message", "description"}
    errors = []
    try:
        body = response.json()
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in allowed and isinstance(item, (str, int)):
                        errors.append({"field": key, "value": sanitize(item, sensitive)})
                    elif isinstance(item, (dict, list)):
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(body)
    except (ValueError, TypeError):
        try:
            content = response.content
            if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
                raise ValueError()
            root = ET.fromstring(content)
            for node in root.iter():
                tag = node.tag.rsplit("}", 1)[-1]
                if tag in allowed and node.text and node.text.strip():
                    errors.append({"field": tag, "value": sanitize(node.text, sensitive)})
        except (ET.ParseError, ValueError, TypeError):
            pass
    result["yahoo_errors"] = errors
    if not errors:
        result["error_body_note"] = "No structured error fields; raw body withheld."
    return result


class ObservedRefresh:
    def __init__(self, transport):
        self.transport = transport
        self.response = None
        self.called = False

    def request(self, method, url, **kwargs):
        data = kwargs.get("data", {})
        if self.called or method != "POST" or url != y.TOKEN_URL or set(data) != {
                "client_id", "redirect_uri", "grant_type", "refresh_token"}:
            raise y.YahooError("Diagnostic refresh request guard failed.")
        if data["grant_type"] != "refresh_token" or data["redirect_uri"] != "oob":
            raise y.YahooError("Diagnostic public-client parameter guard failed.")
        self.called = True
        self.response = self.transport.request(method, url, **kwargs)
        return self.response


def direct_get(transport, access, sensitive):
    result = {"endpoint_purpose": "Discover the signed-in user's NFL fantasy leagues",
              "method": "GET", "path": PATH}
    try:
        response = transport.request("GET", y.API_URL + "/" + PATH,
            headers={"Authorization": "Bearer " + access, "Accept": "application/xml"})
        result.update(response_summary(response, sensitive))
    except y.YahooError:
        result["local_error"] = "Network request failed; no HTTP response available."
    return result


def diagnose(store, transport):
    client_id = store.get("consumer_key")
    if not client_id:
        raise y.YahooError("No saved Consumer Key; diagnostic stopped.")
    observer = ObservedRefresh(transport)
    oauth = y.OAuth(client_id, store, observer)
    token = oauth.load()
    if token is None:
        raise y.YahooError("No saved token bundle; diagnostic stopped.")
    now = y.time.time()
    result = {"stored_token_state": {
        "access_token_exists": bool(token.get("access_token")),
        "refresh_token_exists": bool(token.get("refresh_token")),
        "checked_at_utc": y.datetime.fromtimestamp(now, y.timezone.utc).isoformat(),
        "expires_at_utc": y.datetime.fromtimestamp(token["expires_at"], y.timezone.utc).isoformat(),
        "seconds_until_expiration": round(token["expires_at"] - now),
        "issued_at_or_age": "Not stored; issuance age cannot be determined"}}
    if token["expires_at"] > now:
        result["direct_access_token_test"] = direct_get(transport, token["access_token"], oauth.sensitive)
    else:
        result["direct_access_token_test"] = {"skipped": "Stored access token has expired"}
    refreshed = None
    refresh_error = None
    try:
        # Existing implementation validates and atomically saves any returned rotation.
        refreshed = oauth.exchange(dict(grant_type="refresh_token", refresh_token=token["refresh_token"]),
                                   previous=token["refresh_token"])
    except y.YahooError:
        refresh_error = "Refresh exchange or atomic storage did not complete; post-refresh GET skipped."
    refresh = {"method": "POST", "endpoint_purpose": "OAuth public-client refresh",
               "grant_type": "refresh_token", "redirect_uri": "oob", "client_secret_sent": False}
    if observer.response is not None:
        # Redact even malformed token responses before processing their error fields.
        try:
            body = observer.response.json()
            if isinstance(body, dict):
                for key in ("access_token", "refresh_token", "id_token"):
                    if isinstance(body.get(key), str):
                        oauth.sensitive.add(body[key])
        except ValueError:
            pass
        refresh.update(response_summary(observer.response, oauth.sensitive))
    else:
        refresh["local_error"] = "No HTTP response available."
    refresh["exchange_and_atomic_storage_succeeded"] = refreshed is not None
    if refresh_error:
        refresh["local_error"] = refresh_error
    if refreshed is not None:
        saved = oauth.load()
        refresh["refresh_token_rotated"] = saved["refresh_token"] != token["refresh_token"]
        refresh["expires_at_utc"] = y.datetime.fromtimestamp(saved["expires_at"], y.timezone.utc).isoformat()
    result["refresh_test"] = refresh
    result["post_refresh_test"] = direct_get(transport, refreshed, oauth.sensitive) if refreshed else {
        "skipped": "Refresh did not complete successfully"}
    return result


def main():
    logging.disable(logging.CRITICAL)
    try:
        with y.single_process():
            result = diagnose(y.WindowsStore(), y.Transport())
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    except Exception:
        print("Diagnostic stopped locally; no secret-bearing exception details emitted.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
