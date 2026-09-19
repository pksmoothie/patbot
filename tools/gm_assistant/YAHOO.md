# Yahoo read-only connection milestone

This standalone probe does not feed or change the GM engines. FantasyPros remains
the league-state source. No roster, lineup, waiver, trade, or settings writes exist.

From a normal PowerShell terminal in the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r tools\gm_assistant\requirements-yahoo.txt
.\tools\gm_assistant\YAHOO_AUTH.bat
.\tools\gm_assistant\YAHOO_TEST.bat
```

The launchers prefer the GM assistant's own `.venv` if present; install requirements
using that environment's Python instead if you have created it.

On first authentication, enter the Consumer Key (OAuth client_id) at the **hidden
local terminal prompt**. The App ID is not the OAuth client_id. Do not put either
tokens or authorization codes into source files, command arguments, or chat.
No client secret is requested or sent. The default browser opens Yahoo; approve
the existing read-only Fantasy permission, then enter only Yahoo's short-lived
authorization code at the second hidden terminal prompt. A blank code stops.
Keep that terminal open while approving: the PKCE verifier exists only in memory.

Both Consumer Key and the combined access/refresh token record are stored in
Windows Credential Manager, under targets beginning `patbot.gm_assistant.yahoo.public.v1`.
No `.env.local` is needed. Environment files, common token files, the lock, and
the generated audit are ignored by Git. The Windows backend is selected explicitly,
with local-machine persistence and no plaintext fallback. Token rotation replaces
the entire bundle with one Windows CredWrite using pinned pywin32-ctypes bindings
(keyring's public method retains a backup, which we deliberately avoid).
Keyring's Windows backend reads the raw blob. A UTF-8 credential blob preserves
Windows' 2,560-byte capacity; oversized bundles fail closed. No token is split
across independently updated records. A process lock prevents concurrent launcher refreshes. If credential storage fails
after Yahoo rotates a token, reauthorization may be necessary.

Subsequent auth/test commands refresh without browser approval. Expired access
tokens refresh automatically, with one retry after an API 401. Revoked or malformed
tokens require an explicit new browser approval:

```powershell
.\tools\gm_assistant\YAHOO_AUTH.bat --reauthorize
```

The test requests XML from `https://fantasysports.yahooapis.com/fantasy/v2`:

- `users;use_login=1/games;game_codes=nfl/leagues`: your NFL games/seasons and leagues.
- `league/{discovered_league_key}/teams`: team IDs, keys, names, and ownership flag
  for each current-season league.
- `league/{discovered_league_key}/settings`: selected scoring, roster, playoff,
  waiver and trade settings metadata, when available.

Current season means the NFL season year (March through February), compared against
Yahoo's returned season. Historical league metadata is retained; detailed team/settings
calls are limited to current-season leagues. The audit explicitly reports no match or
multiple matches for `Fantasy Premier League`; it never assumes a Yahoo league ID.

The sanitized result is displayed and saved to
`tools/gm_assistant/data/yahoo_connection_test.json`, with a UTC timestamp. It preserves
Yahoo's returned strings for selected fields, including nested settings (lists preserve
repeated XML elements). It is a field projection, not a raw response dump. Managers,
emails, URLs, unknown settings fields, and OAuth data are excluded. Settings failures
are reported as unavailable. Failure before completing discovery/teams leaves any old
audit untouched, so check its timestamp. An empty league list proves API access but
does not prove that the target league was found.

## OAuth and redirect limitation

To replace only the stored Consumer Key without contacting Yahoo, run from the
repository root in your own local terminal:

```powershell
.\tools\gm_assistant\YAHOO_REPLACE_KEY.bat
```

The command uses hidden input, rejects whitespace rather than trimming it, overwrites
only the Consumer Key credential, and verifies its read-back. It reports only length,
first/last four characters, SHA-256 fingerprint, and whether an offline-generated
authorization URL contains exactly the newly stored client_id. It never prints the
URL, opens a browser, creates a network transport, or reads/changes tokens or other
credentials. If another Yahoo command still holds the lock, stop that command with
Ctrl+C in its terminal first. Existing tokens remain stored under their original
client-ID-specific credential targets.

Yahoo's [current sign-in documentation](https://developer.yahoo.com/sign-in-with-yahoo/)
documents S256 PKCE for public clients, explicitly says to omit `client_secret`, and
still lists `redirect_uri=oob`. The
[authorization-code guide](https://developer.yahoo.com/oauth2/guide/flows_authcode/)
also documents OOB, although its older secret-based examples are for confidential
clients. This implementation uses the public-client rules. It relies on the app's
approved Fantasy API permissions without requesting OpenID/profile scopes.

Live OOB acceptance for this specific app cannot be confirmed without the owner's
interactive Yahoo approval. If Yahoo displays `INVALID_CALLBACK_URL`, `BAD_REDIRECT_URI`,
or rejects OOB, stop with Ctrl+C. The program never changes app registration, falls
back to another URI, or adds a secret. The documented alternative is an exactly
registered HTTPS callback under your control that receives the code (and validates
state) for the same PKCE exchange. A trusted localhost certificate is not necessary
when using a hosted HTTPS callback. Review that option before changing registration;
an HTTP loopback exception is not assumed. Manual OOB provides no returned state to
validate; PKCE binds the pasted code to the verifier kept in this terminal session.

## Offline verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools\gm_assistant -p test_yahoo.py -v
```

Tests use synthetic fixtures, in-memory credentials, and blocked HTTP calls. They
cover PKCE, public-client requests, expiration and refresh rotation, single credential
replacement, read-only paths, redirect blocking, discovery, settings projection,
and audit secret exclusion. A real connection audit is only produced by your
authenticated validation run; offline tests never impersonate a successful connection.
