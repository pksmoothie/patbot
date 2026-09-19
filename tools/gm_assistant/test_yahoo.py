"""Synthetic fixtures only. No Yahoo calls, browser, or real credential access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import yahoo_client as y


class Store:
    def __init__(self):
        self.data = {}
        self.writes = []

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.writes.append((key, value))
        self.data[key] = value


class Response:
    def __init__(self, status=200, body=None, content=b""):
        self.status_code, self.body, self.content = status, body, content

    def json(self):
        return self.body


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected request")
        return self.responses.pop(0)


def token_body(**changes):
    return dict(access_token="synthetic-access", refresh_token="synthetic-refresh", expires_in=3600,
                token_type="bearer", **changes)


def discovery_xml():
    now = y.datetime.now(y.timezone.utc)
    season = now.year if now.month >= 3 else now.year - 1
    return ('''<fantasy_content xmlns="http://fantasysports.yahooapis.com/fantasy/v2/base.rng">
      <users count="1"><user><guid>PRIVATE</guid><games count="2">
      <game><game_key>999</game_key><code>nfl</code><season>%s</season><leagues count="1">
       <league><league_key>999.l.123</league_key><league_id>123</league_id>
       <name>Fantasy Premier League</name><num_teams>12</num_teams></league>
      </leagues></game><game><game_key>998</game_key><code>nfl</code><season>2020</season><leagues>
       <league><league_key>998.l.123</league_key><league_id>123</league_id>
       <name>Fantasy Premier League</name></league></leagues></game>
      </games></user></users></fantasy_content>''' % season).encode()


class YahooTests(unittest.TestCase):
    def setUp(self):
        # Any accidental real request is an immediate failure, even on online machines.
        self.network = patch("requests.sessions.Session.request", side_effect=AssertionError("Offline test"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.store = Store()

    def oauth(self, transport):
        return y.OAuth("synthetic-client", self.store, transport, clock=lambda: 1000)

    def save(self, oauth, expiry=900):
        self.store.data[oauth.account] = json.dumps(dict(access_token="old-access", refresh_token="old-refresh", expires_at=expiry))

    def test_pkce_rfc7636_vector(self):
        self.assertEqual(y.pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
                         "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")

    def test_pkce_randomness_and_shape(self):
        values = [y.generate_pkce() for _ in range(20)]
        self.assertEqual(len({v for v, _ in values}), 20)
        for verifier, challenge in values:
            self.assertTrue(43 <= len(verifier) <= 128)
            self.assertRegex(verifier, r"^[A-Za-z0-9_-]+$")
            self.assertEqual(challenge, y.pkce_challenge(verifier))

    def test_authorization_parameters(self):
        url = urlsplit(y.authorization_url("client", "challenge", "state"))
        args = parse_qs(url.query)
        self.assertEqual(args["redirect_uri"], ["oob"])
        self.assertEqual(args["code_challenge_method"], ["S256"])
        self.assertEqual(args["state"], ["state"])
        self.assertNotIn("client_secret", args)
        self.assertNotIn("code_verifier", args)

    def test_public_code_exchange_no_secret_or_basic_auth(self):
        transport = FakeTransport(Response(body=token_body()))
        self.oauth(transport).exchange(dict(grant_type="authorization_code", code="short-code", code_verifier="verifier"))
        method, url, kwargs = transport.calls[0]
        self.assertEqual((method, url), ("POST", y.TOKEN_URL))
        self.assertEqual(kwargs["data"]["client_id"], "synthetic-client")
        self.assertEqual(kwargs["data"]["redirect_uri"], "oob")
        self.assertNotIn("client_secret", kwargs["data"])
        self.assertNotIn("Authorization", kwargs["headers"])
        self.assertNotIn("auth", kwargs)

    def test_refresh_rotation_is_one_bundle_replacement(self):
        transport = FakeTransport(Response(body=token_body()))
        oauth = self.oauth(transport)
        self.save(oauth)
        self.assertEqual(oauth.access_token(), "synthetic-access")
        self.assertEqual(len(self.store.writes), 1)
        self.assertEqual(oauth.load()["refresh_token"], "synthetic-refresh")
        self.assertNotIn("old-refresh", self.store.get(oauth.account))
        data = transport.calls[0][2]["data"]
        self.assertEqual(data["refresh_token"], "old-refresh")
        self.assertNotIn("client_secret", data)

    def test_refresh_without_rotation_preserves_refresh(self):
        body = token_body()
        del body["refresh_token"]
        oauth = self.oauth(FakeTransport(Response(body=body)))
        self.save(oauth)
        oauth.access_token()
        self.assertEqual(oauth.load()["refresh_token"], "old-refresh")

    def test_valid_token_avoids_network(self):
        transport = FakeTransport()
        oauth = self.oauth(transport)
        self.save(oauth, 2000)
        self.assertEqual(oauth.access_token(), "old-access")
        self.assertFalse(transport.calls)

    def test_malformed_saved_tokens(self):
        oauth = self.oauth(FakeTransport())
        for raw in ("{", "null", "[]", '{}', '{"access_token":"x","refresh_token":"y","expires_at":"never"}'):
            self.store.data[oauth.account] = raw
            with self.assertRaises(y.YahooError):
                oauth.access_token()

    def test_invalid_token_responses_leave_old_bundle(self):
        bodies = [None, [], {}, {**token_body(), "expires_in": "bad"},
                  {**token_body(), "expires_in": -1}, {**token_body(), "expires_in": float("nan")},
                  {**token_body(), "access_token": "contains whitespace"},
                  {**token_body(), "refresh_token": None}, {**token_body(), "token_type": "other"}]
        for body in bodies:
            oauth = self.oauth(FakeTransport(Response(body=body)))
            self.save(oauth)
            old = self.store.get(oauth.account)
            with self.assertRaises(y.YahooError):
                oauth.access_token()
            self.assertEqual(self.store.get(oauth.account), old)

    def test_expired_refresh_error_is_sanitized(self):
        oauth = self.oauth(FakeTransport(Response(400, {"error_description": "SECRET"})))
        self.save(oauth)
        with self.assertRaises(y.YahooError) as caught:
            oauth.access_token()
        self.assertNotIn("SECRET", str(caught.exception))
        self.assertIn("reauthorize", str(caught.exception))

    def test_get_only_and_path_restriction(self):
        oauth = Mock()
        oauth.access_token.return_value = "fake"
        transport = FakeTransport(Response(content=b"<fantasy_content/>"))
        client = y.FantasyClient(oauth, transport)
        client.get("league/999.l.123/teams")
        self.assertEqual(transport.calls[0][0], "GET")
        for path in ("https://evil.test/", "//evil.test", "../team", "league/999.l.123/settings?x=y", "team/999.l.123.t.1/roster"):
            with self.assertRaises(y.YahooError):
                client.get(path)
        for method in ("post", "put", "delete", "patch", "request"):
            self.assertFalse(hasattr(client, method))
        self.assertEqual(len(transport.calls), 1)

    def test_401_refresh_once(self):
        oauth = Mock()
        oauth.access_token.return_value = "fake"
        transport = FakeTransport(Response(401), Response(content=b"<fantasy_content/>"))
        y.FantasyClient(oauth, transport).get("league/999.l.123/teams")
        self.assertEqual([c.kwargs for c in oauth.access_token.call_args_list],
                         [{"force_refresh": False}, {"force_refresh": True}])

    def test_repeated_401_stops(self):
        oauth = Mock()
        oauth.access_token.return_value = "fake"
        transport = FakeTransport(Response(401), Response(401))
        with self.assertRaises(y.YahooError):
            y.FantasyClient(oauth, transport).get("league/999.l.123/teams")
        self.assertEqual(len(transport.calls), 2)

    def test_redirect_not_followed(self):
        transport = y.Transport()
        transport.session = Mock()
        transport.request("GET", y.API_URL)
        self.assertFalse(transport.session.request.call_args.kwargs["allow_redirects"])

    def test_discovery_namespace_and_season(self):
        games, leagues, season = y.discover_leagues(y.parse_xml(discovery_xml()))
        self.assertEqual(len(games), 2)
        self.assertEqual(leagues[0]["league_key"], "999.l.123")
        self.assertEqual(leagues[0]["season"], season)
        self.assertTrue(leagues[0]["current_season"])
        self.assertFalse(leagues[1]["current_season"])

    def test_malformed_xml_and_league_key(self):
        for content in (b"<html/>", b"broken", b'<!DOCTYPE x><fantasy_content/>'):
            with self.assertRaises(y.YahooError):
                y.parse_xml(content)
        with self.assertRaises(y.YahooError):
            y.discover_leagues(y.parse_xml(discovery_xml().replace(b"999.l.123", b"../evil")))

    def test_discovery_and_settings_projection(self):
        client = Mock()
        client.get.side_effect = [y.parse_xml(discovery_xml()),
            y.parse_xml(b'<fantasy_content><league><teams><team><team_key>999.l.123.t.1</team_key><team_id>1</team_id><name>Team A</name><managers><email>PRIVATE</email></managers></team></teams></league></fantasy_content>'),
            y.parse_xml(b'<fantasy_content><league><settings><scoring_type>head</scoring_type><roster_positions><roster_position><position>QB</position><count>1</count></roster_position></roster_positions><access_token>SECRET</access_token></settings></league></fantasy_content>')]
        result = y.connection_test(client)
        self.assertEqual(result["target_league_keys"], ["999.l.123"])
        self.assertEqual(result["target_status"], "unique_match")
        self.assertEqual(len(client.get.call_args_list), 3)
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(result["leagues"][0]["teams"][0]["team_id"], "1")

    def test_secrets_never_written_including_nested_and_echoed_values(self):
        payload = {"access_token": "A", "nested": [{"refresh_token": "B", "code_verifier": "C",
                   "authorization_code": "D", "name": "Team synthetic-access"}], "league_id": "123"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.json"
            y.write_audit(path, payload, {"synthetic-access"})
            output = path.read_text(encoding="utf-8")
            for secret in ("access_token", "refresh_token", "code_verifier", "authorization_code", "synthetic-access"):
                self.assertNotIn(secret, output)
            self.assertEqual(json.loads(output)["league_id"], "123")
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_windows_store_uses_single_write_without_backup(self):
        store = y.WindowsStore.__new__(y.WindowsStore)
        store.backend = Mock()
        with patch.object(y, "write_credential_blob") as write:
            store.set("tokens:test", "bundle")
            write.assert_called_once_with(y.SERVICE + ":tokens:test", "tokens:test", b"bundle")
        store.backend.set_password.assert_not_called()

    @unittest.skipUnless(y.sys.platform == "win32", "Windows ABI")
    def test_native_credential_write_preserves_bytes_and_size(self):
        import ctypes
        with patch("win32ctypes.core.ctypes._authentication._CredWrite") as write:
            def inspect(pointer, flags):
                credential = pointer.contents
                self.assertEqual(credential.CredentialBlobSize, 2000)
                self.assertEqual(ctypes.string_at(credential.CredentialBlob, 2000), b"x" * 2000)
                self.assertEqual(credential.Persist, 2)
                self.assertEqual(flags, 0)
            write.side_effect = inspect
            y.write_credential_blob("synthetic-target", "synthetic-account", b"x" * 2000)
            write.assert_called_once()
            with self.assertRaises(y.YahooError):
                y.write_credential_blob("synthetic-target", "synthetic-account", b"x" * 2561)
            write.assert_called_once()

    def test_windows_store_decodes_exact_utf8_blob(self):
        store = y.WindowsStore.__new__(y.WindowsStore)
        store.backend = Mock()
        store.backend._read_credential.return_value = {"CredentialBlob": b'{"refresh_token":"fake"}'}
        self.assertEqual(store.get("tokens:test"), '{"refresh_token":"fake"}')

    def test_empty_leagues_are_not_target_success(self):
        client = Mock()
        client.get.return_value = y.parse_xml(b'<fantasy_content><users><user><games count="0"/></user></users></fantasy_content>')
        result = y.connection_test(client)
        self.assertEqual(result["target_status"], "not_found")
        self.assertEqual(result["leagues"], [])
        client.get.assert_called_once()

    def test_settings_failure_is_recorded(self):
        client = Mock()
        client.get.side_effect = [y.parse_xml(discovery_xml()), y.parse_xml(b'<fantasy_content><league><teams/></league></fantasy_content>'), y.YahooError("Yahoo Fantasy GET failed (HTTP 403).")]
        result = y.connection_test(client)
        self.assertEqual(result["leagues"][0]["settings_status"], "unavailable")

    def test_credential_write_failure_stops_refresh(self):
        oauth = self.oauth(FakeTransport(Response(body=token_body())))
        self.save(oauth)
        old = self.store.get(oauth.account)
        self.store.set = Mock(side_effect=y.YahooError("Credential write failed"))
        with self.assertRaises(y.YahooError):
            oauth.access_token()
        self.assertEqual(self.store.get(oauth.account), old)

    def test_no_interactive_input_fallback(self):
        with patch("sys.stdin.isatty", return_value=False), patch("getpass.getpass") as prompt:
            with self.assertRaises(y.YahooError):
                y.hidden_prompt("hidden")
            prompt.assert_not_called()

    def test_replace_key_cli_is_offline_and_changes_only_consumer_key(self):
        from contextlib import nullcontext, redirect_stdout
        import io
        replacement = "synthetic-new-consumer-key"
        self.store.data = {"consumer_key": "old-key", "tokens:old": "old-token-bundle", "other": "untouched"}
        output = io.StringIO()
        with patch.object(y, "WindowsStore", return_value=self.store), \
             patch.object(y, "single_process", return_value=nullcontext()), \
             patch.object(y, "hidden_prompt", return_value=replacement) as prompt, \
             patch.object(y, "Transport", side_effect=AssertionError("No transport allowed")), \
             patch.object(y, "OAuth", side_effect=AssertionError("No token handling allowed")), \
             patch.object(y.webbrowser, "open", side_effect=AssertionError("No browser allowed")), \
             redirect_stdout(output):
            self.assertEqual(y.main(["replace-key"]), 0)
        prompt.assert_called_once_with("Replacement Yahoo Consumer Key (hidden): ", strip=False)
        self.assertEqual(self.store.writes, [("consumer_key", replacement)])
        self.assertEqual(self.store.data, {"consumer_key": replacement, "tokens:old": "old-token-bundle", "other": "untouched"})
        report = json.loads(output.getvalue())
        self.assertEqual(report, dict(character_length=len(replacement), first_4=replacement[:4],
            last_4=replacement[-4:], sha256=y.hashlib.sha256(replacement.encode()).hexdigest(),
            offline_client_id_matches_stored_value=True))
        self.assertNotIn(replacement, output.getvalue())

    def test_replace_key_rejects_bad_input_without_writing(self):
        for value in ("", "short", " padded-key ", "key-with\nnewline", "non-ascii-key-\u00e9"):
            with patch.object(y, "hidden_prompt", return_value=value):
                with self.assertRaises(y.YahooError):
                    y.replace_consumer_key(self.store)
        self.assertEqual(self.store.writes, [])

    def test_replace_key_detects_readback_or_url_mismatch(self):
        with patch.object(y, "hidden_prompt", return_value="synthetic-new-key"):
            store = Mock()
            store.get.return_value = "mismatched-key"
            with self.assertRaisesRegex(y.YahooError, "read-back"):
                y.replace_consumer_key(store)
            with patch.object(y, "authorization_url", return_value=y.AUTH_URL + "?client_id=wrong"):
                with self.assertRaisesRegex(y.YahooError, "Offline client_id"):
                    y.replace_consumer_key(self.store)


if __name__ == "__main__":
    unittest.main()
