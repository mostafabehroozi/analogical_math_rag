"""Kaggle Benchmarks model proxy credentials from a Kaggle API token.

Inside a Kaggle *Benchmark Task* notebook, Kaggle sets ``MODEL_PROXY_URL``
and ``MODEL_PROXY_API_KEY`` itself. Anywhere else (a normal Kaggle notebook,
a laptop, a server) the same proxy is reachable with a short-lived key that
Kaggle mints for the account owning a Kaggle API token. This is what
``kaggle benchmarks auth`` does; this module does it without the CLI:

    POST https://api.kaggle.com/v1/models.ModelProxyApiService/CreateDefaultModelProxyToken
    -> {"token": ..., "baseUri": ..., "expiryTime": ...}

The minted key is short-lived, so :class:`KaggleProxyCredentials` mints a new
one shortly before it expires, or at once after the proxy rejects the key.
Calls spend the account's Model Proxy quota (``kaggle benchmarks quota``).

Settings (config.py):

* ``KAGGLE_PROXY_USE_API_TOKEN``: False keeps the notebook's own proxy
  (the default, unchanged behaviour); True mints keys from a Kaggle API token.
* ``KAGGLE_API_TOKEN``: the token, or a path to a file holding it. Empty means
  look in turn at the ``KAGGLE_API_TOKEN`` environment variable, a Kaggle
  Secret named ``KAGGLE_API_TOKEN``, ``~/.kaggle/access_token`` and
  ``~/.kaggle/kaggle.json`` (legacy username + key).
* ``KAGGLE_PROXY_TOKEN_REFRESH_SECONDS``: mint a new key when the current one
  has less than this many seconds left.

The token and the minted key are never printed or logged.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

KAGGLE_API_URL = "https://api.kaggle.com/v1"
TOKEN_ENDPOINT = "models.ModelProxyApiService/CreateDefaultModelProxyToken"
QUOTA_ENDPOINT = "models.ModelProxyApiService/GetModelProxyQuotas"
ENABLED_KEY = "KAGGLE_PROXY_USE_API_TOKEN"
TOKEN_KEY = "KAGGLE_API_TOKEN"
REFRESH_KEY = "KAGGLE_PROXY_TOKEN_REFRESH_SECONDS"
DEFAULT_REFRESH_SECONDS = 300.0
# kaggle_benchmarks builds its default model when it is imported with a proxy
# configured; these roles' model names give it one that the run really uses.
_MODEL_NAME_KEYS = ("KAGGLE_MODEL_NAME_FINAL_SOLVER", "KAGGLE_MODEL_NAME_EVALUATOR",
                    "KAGGLE_MODEL_NAME_ADAPTATION", "KAGGLE_MODEL_NAME_SIMPLIFICATION")
_FALLBACK_DEFAULT_MODEL = "google/gemini-3-flash-preview"


class KaggleProxyAuthError(RuntimeError):
    """No usable Kaggle API token, or Kaggle refused to mint a proxy key."""


def api_token_enabled(config: Mapping[str, Any]) -> bool:
    value = config.get(ENABLED_KEY, False)
    if not isinstance(value, bool):
        raise ValueError(f"{ENABLED_KEY} must be True or False, got {value!r}.")
    return value


# ---------------------------------------------------------------------------
# Finding the Kaggle API token
# ---------------------------------------------------------------------------

def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _kaggle_secret(name: str) -> Optional[str]:
    """A Kaggle Secret attached to this notebook, or None outside Kaggle."""
    try:
        from kaggle_secrets import UserSecretsClient  # Kaggle notebooks only
    except ImportError:
        return None
    try:
        return UserSecretsClient().get_secret(name) or None
    except Exception:  # Not attached, or no secrets service in this session.
        return None


def _token_value(raw: str) -> str:
    """A token given directly, or read from the file it names."""
    raw = raw.strip()
    if raw and os.path.isfile(os.path.expanduser(raw)):
        return _read_text(os.path.expanduser(raw)) or ""
    return raw


@dataclass(frozen=True)
class KaggleCredential:
    """How to authenticate to the Kaggle API: a bearer token or a legacy username + key."""

    source: str
    token: Optional[str] = None
    username: Optional[str] = None
    key: Optional[str] = None

    def request_auth(self) -> Tuple[Dict[str, str], Optional[Tuple[str, str]]]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}, None
        return {}, (self.username or "", self.key or "")

    def __repr__(self) -> str:  # Never show the secret.
        return f"KaggleCredential(source={self.source!r})"


def resolve_kaggle_credential(
    config: Mapping[str, Any],
    secret_fn: Callable[[str], Optional[str]] = _kaggle_secret,
    home: Optional[str] = None,
) -> KaggleCredential:
    """The first Kaggle API credential found, in the documented order."""
    configured = config.get(TOKEN_KEY)
    if configured not in (None, ""):
        if not isinstance(configured, str):
            raise ValueError(f"{TOKEN_KEY} must be a string (the token or a path to it).")
        token = _token_value(configured)
        if not token:
            raise KaggleProxyAuthError(f"{TOKEN_KEY} names an empty or unreadable file.")
        return KaggleCredential(f"config {TOKEN_KEY}", token=token)

    environment = os.environ.get("KAGGLE_API_TOKEN", "")
    if environment.strip():
        token = _token_value(environment)
        if token:
            return KaggleCredential("environment variable KAGGLE_API_TOKEN", token=token)

    secret = secret_fn("KAGGLE_API_TOKEN")
    if secret and secret.strip():
        return KaggleCredential("Kaggle Secret KAGGLE_API_TOKEN", token=secret.strip())

    kaggle_dir = os.path.join(home or os.path.expanduser("~"), ".kaggle")
    for name in ("access_token", "access_token.txt"):
        token = _read_text(os.path.join(kaggle_dir, name))
        if token:
            return KaggleCredential(f"~/.kaggle/{name}", token=token)

    username, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if username and key:
        return KaggleCredential("environment variables KAGGLE_USERNAME/KAGGLE_KEY", username=username, key=key)
    legacy = _read_text(os.path.join(kaggle_dir, "kaggle.json"))
    if legacy:
        try:
            data = json.loads(legacy)
        except json.JSONDecodeError:
            data = {}
        if data.get("username") and data.get("key"):
            return KaggleCredential("~/.kaggle/kaggle.json", username=data["username"], key=data["key"])

    raise KaggleProxyAuthError(
        f"{ENABLED_KEY} is True but no Kaggle API token was found. Create one at "
        "https://www.kaggle.com/settings/api and give it as a Kaggle Secret named "
        f"KAGGLE_API_TOKEN (Add-ons > Secrets), the KAGGLE_API_TOKEN environment variable, "
        f"CONFIG['{TOKEN_KEY}'], or ~/.kaggle/access_token."
    )


# ---------------------------------------------------------------------------
# Talking to the Kaggle API
# ---------------------------------------------------------------------------

def _post(endpoint: str, credential: KaggleCredential, timeout: float,
          post_fn: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    if post_fn is None:
        import requests
        post_fn = requests.post
    headers, auth = credential.request_auth()
    # Kaggle answers 404 to requests' default "python-requests/x.y" User-Agent
    # (checked 2026-11-23); its own SDK sends an empty one, which is accepted.
    headers = {"User-Agent": "", **headers}
    response = post_fn(f"{KAGGLE_API_URL}/{endpoint}", json={}, headers=headers, auth=auth, timeout=timeout)
    status = getattr(response, "status_code", 0)
    if status == 404:
        raise KaggleProxyAuthError(
            "Kaggle answered 404 for the model proxy token: Kaggle Benchmarks may not be "
            "enabled for this account yet (it is a beta).")
    if status in (401, 403):
        raise KaggleProxyAuthError(
            f"Kaggle refused the API token from {credential.source} ({status}). The token may "
            "be wrong or expired (make a new one at https://www.kaggle.com/settings/api), or the "
            "account may lack phone/identity verification (https://www.kaggle.com/settings).")
    if status >= 400:
        raise KaggleProxyAuthError(f"Kaggle API {endpoint} failed with HTTP {status}.")
    try:
        return response.json()
    except ValueError as error:
        raise KaggleProxyAuthError(f"Kaggle API {endpoint} returned no JSON.") from error


def _parse_expiry(value: Any) -> Optional[float]:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    # Kaggle may send more than the 6 fractional digits fromisoformat accepts.
    if "." in text:
        head, rest = text.split(".", 1)
        digits = rest[:len(rest) - len(rest.lstrip("0123456789"))]
        zone = rest[len(digits):]
        text = f"{head}.{digits[:6]}{zone}"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


@dataclass(frozen=True)
class ProxyKey:
    url: str
    api_key: str
    expires_at: Optional[float]  # Unix time; None when Kaggle gives no expiry.

    def seconds_left(self, now: Optional[float] = None) -> Optional[float]:
        if self.expires_at is None:
            return None
        return self.expires_at - (time.time() if now is None else now)

    def __repr__(self) -> str:
        return f"ProxyKey(url={self.url!r}, expires_at={self.expires_at!r})"


def mint_proxy_key(credential: KaggleCredential, timeout: float = 60.0,
                   post_fn: Optional[Callable[..., Any]] = None) -> ProxyKey:
    """Ask Kaggle for a new Model Proxy key for the token's account."""
    data = _post(TOKEN_ENDPOINT, credential, timeout, post_fn)
    token = data.get("token")
    url = data.get("baseUri") or data.get("base_uri")
    if not token or not url:
        raise KaggleProxyAuthError("Kaggle returned no model proxy key or URL.")
    return ProxyKey(url=url, api_key=token, expires_at=_parse_expiry(data.get("expiryTime") or data.get("expiry_time")))


def fetch_proxy_quota(credential: KaggleCredential, timeout: float = 60.0,
                      post_fn: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """The account's Model Proxy spend quota, as Kaggle returns it (``kaggle benchmarks quota``)."""
    return _post(QUOTA_ENDPOINT, credential, timeout, post_fn)


# ---------------------------------------------------------------------------
# Keeping a fresh key in the environment kaggle_benchmarks reads
# ---------------------------------------------------------------------------

class KaggleProxyCredentials:
    """Mints proxy keys from a Kaggle API token and keeps the current one in os.environ.

    ``generation`` grows with every new key, so a caller that cached model
    clients built with an older key knows to rebuild them. Thread-safe.
    """

    def __init__(self, credential: KaggleCredential, refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
                 default_model: Optional[str] = None,
                 mint_fn: Optional[Callable[[KaggleCredential], ProxyKey]] = None,
                 clock: Callable[[], float] = time.time, environ: Optional[Dict[str, str]] = None):
        if refresh_seconds < 0:
            raise ValueError(f"{REFRESH_KEY} must be >= 0.")
        self.credential = credential
        self.refresh_seconds = float(refresh_seconds)
        self.default_model = default_model
        # Looked up at each call, so tests can replace mint_proxy_key.
        self._mint = mint_fn or (lambda credential: mint_proxy_key(credential))
        self._clock = clock
        self._environ = os.environ if environ is None else environ
        self._lock = threading.Lock()
        self._key: Optional[ProxyKey] = None
        self._stale = False
        self.generation = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> "KaggleProxyCredentials":
        refresh = config.get(REFRESH_KEY, DEFAULT_REFRESH_SECONDS)
        if isinstance(refresh, bool) or not isinstance(refresh, (int, float)):
            raise ValueError(f"{REFRESH_KEY} must be a number of seconds, got {refresh!r}.")
        default_model = next((config[key] for key in _MODEL_NAME_KEYS if config.get(key)), None)
        secret_fn = kwargs.pop("secret_fn", _kaggle_secret)
        return cls(resolve_kaggle_credential(config, secret_fn=secret_fn), refresh, default_model, **kwargs)

    @property
    def key(self) -> Optional[ProxyKey]:
        return self._key

    def _needs_new_key(self) -> bool:
        if self._key is None or self._stale:
            return True
        left = self._key.seconds_left(self._clock())
        return left is not None and left <= self.refresh_seconds

    def ensure_fresh(self) -> int:
        """Mint a key if there is none, it is about to expire, or it was rejected; return the generation."""
        with self._lock:
            if self._needs_new_key():
                key = self._mint(self.credential)
                self._key, self._stale = key, False
                self.generation += 1
                self._install(key)
                left = key.seconds_left(self._clock())
                logger.info("Kaggle model proxy key #%d minted from %s%s.", self.generation, self.credential.source,
                            "" if left is None else f"; valid for {left / 60:.0f} min")
            return self.generation

    def invalidate(self, generation: Optional[int] = None) -> None:
        """The proxy rejected a key: mint a new one on the next call.

        A rejection reported for an older generation is ignored, so many calls
        failing on one expired key mint only one replacement.
        """
        with self._lock:
            if generation is None or generation == self.generation:
                self._stale = True

    def _install(self, key: ProxyKey) -> None:
        self._environ["MODEL_PROXY_URL"] = key.url
        self._environ["MODEL_PROXY_API_KEY"] = key.api_key
        if key.expires_at is not None:
            self._environ["MODEL_PROXY_EXPIRY_TIME"] = (
                datetime.fromtimestamp(key.expires_at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        else:
            self._environ.pop("MODEL_PROXY_EXPIRY_TIME", None)
        # kaggle_benchmarks loads a default model at import when a proxy is set;
        # without LLM_DEFAULT that import fails (a Benchmark Task notebook sets it).
        self._environ.setdefault("LLM_DEFAULT", self.default_model or _FALLBACK_DEFAULT_MODEL)

    def describe(self) -> str:
        """One line for the console, without any secret."""
        key = self._key
        if key is None:
            return f"Kaggle model proxy via API token ({self.credential.source}): no key yet"
        left = key.seconds_left(self._clock())
        expiry = "no expiry given" if left is None else f"key valid for {left / 60:.0f} more min"
        return f"Kaggle model proxy via API token ({self.credential.source}): {key.url}, {expiry}"
