"""Per-provider and per-model API settings, and roles served by several endpoints.

Configuration helpers only (no SDK or project imports), shared by the API
managers, the orchestrator's preflight and the distributed manifest.

Scoped settings
---------------
The settings in ``SCOPED_SETTINGS`` take a plain value, exactly as before, or
a dict that refines it per provider and, where allowed, per model::

    "GLOBAL_API_CALL_DELAY_SECONDS": 5.0                    # each provider, 5 s
    "GLOBAL_API_CALL_DELAY_SECONDS": {"gemini": 5.0, "kaggle": 0.0, "default": 1.0}
    "GLOBAL_API_CALL_DELAY_SECONDS": {
        "openrouter": {"default": 1.0, "openai/gpt-4o-mini": 3.0},
        "default": 0.5,
    }

The most specific entry wins: the model's own entry, then the provider's
number (or its ``"default"``), then the top-level ``"default"``, then the
setting's built-in default. Every provider is limited on its own, so calls to
two providers never wait for each other.

Provider pools
--------------
``API_PROVIDER_POOLS`` names groups of (provider, model) endpoints. A role
whose ``API_PROVIDER_<ROLE>`` is a pool name spreads its calls over the pool::

    "API_PROVIDER_POOLS": {
        "judges": {
            "routing": "spread",          # or "failover"
            "endpoints": [
                {"provider": "gemini", "model": "models/gemma-3-27b-it"},
                {"provider": "openrouter", "model": "openai/gpt-4o-mini", "weight": 2},
            ],
        },
    },
    "API_PROVIDER_EVALUATOR": "judges",

A pool may also be given as the bare endpoint list (routing "spread").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

DEFAULT_ENTRY = "default"
POOLS_KEY = "API_PROVIDER_POOLS"
POOL_ROUTINGS = ("spread", "failover")
POOL_LABEL_PREFIX = "pool:"

# Setting -> whether a provider entry may refine it per model. The others are
# read once per provider (a client's timeout, a provider's error pause).
SCOPED_SETTINGS: Dict[str, bool] = {
    "GLOBAL_API_CALL_DELAY_SECONDS": True,
    "API_MAX_CONCURRENT_CALLS": True,
    "MAX_API_RETRIES": True,
    "API_RETRY_DELAY_SECONDS": True,
    "API_KEY_ERROR_COOLDOWN_SECONDS": True,
    "API_REQUEST_TIMEOUT_SECONDS": False,
    "GLOBAL_CONSECUTIVE_ERROR_LIMIT": False,
    "GLOBAL_CONSECUTIVE_ERROR_PAUSE_SECONDS": False,
}

ROLE_PROVIDER_KEYS: Dict[str, str] = {
    "adaptation": "API_PROVIDER_ADAPTATION",
    "final_solver": "API_PROVIDER_SOLVER",
    "evaluator": "API_PROVIDER_EVALUATOR",
    "simplification": "API_PROVIDER_SIMPLIFICATION",
}


# --- Scoped settings -------------------------------------------------------

def resolve_scoped(value: Any, provider: Optional[str], model: Optional[str] = None) -> Tuple[Any, bool]:
    """Return ``(value, model_specific)`` for one provider and model.

    ``value`` is ``None`` when nothing applies, so the caller's default is used.
    ``model_specific`` is True only when the model has its own entry; limits
    such as the call gap then keep a clock for that model alone.
    """
    if not isinstance(value, Mapping):
        return value, False
    entry = None
    if provider is not None:
        for key, item in value.items():
            if str(key).strip().lower() == provider:
                entry = item
                break
    if isinstance(entry, Mapping):
        if model is not None and entry.get(model) is not None:
            return entry[model], True
        if entry.get(DEFAULT_ENTRY) is not None:
            return entry[DEFAULT_ENTRY], False
    elif entry is not None:
        return entry, False
    return value.get(DEFAULT_ENTRY), False


def scoped_value(config: Mapping[str, Any], key: str, provider: Optional[str],
                 model: Optional[str] = None, default: Any = None) -> Any:
    value, _ = resolve_scoped(config.get(key), provider, model)
    return default if value is None else value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_scoped_settings(config: Mapping[str, Any], providers: Iterable[str]) -> None:
    """Raise ValueError for a scoped setting that names an unknown provider or is malformed.

    A misspelled provider name would otherwise leave that provider on the
    general value without a word.
    """
    known = {str(provider).lower() for provider in providers}
    for key, model_level in SCOPED_SETTINGS.items():
        value = config.get(key)
        if value is None or _is_number(value):
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} must be a number or a dict per provider, not {value!r}.")
        for name, entry in value.items():
            name_text = str(name).strip().lower()
            if name_text != DEFAULT_ENTRY and name_text not in known:
                raise ValueError(
                    f"{key} names unknown provider {name!r}; use 'default' or one of {sorted(known)}."
                )
            if entry is None or _is_number(entry):
                continue
            if name_text == DEFAULT_ENTRY or not isinstance(entry, Mapping):
                raise ValueError(f"{key}[{name!r}] must be a number, not {entry!r}.")
            if not model_level:
                raise ValueError(
                    f"{key}[{name!r}] must be a number: {key} applies per provider, not per model."
                )
            bad = {model: item for model, item in entry.items() if item is not None and not _is_number(item)}
            if bad:
                raise ValueError(f"{key}[{name!r}] values must be numbers, not {bad!r}.")


# --- Provider pools --------------------------------------------------------

@dataclass(frozen=True)
class PoolEndpoint:
    provider: str
    model: str
    weight: float = 1.0

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True)
class PoolSpec:
    name: str
    endpoints: Tuple[PoolEndpoint, ...]
    routing: str = "spread"

    @property
    def label(self) -> str:
        return pool_label(self.name)


def pool_label(name: str) -> str:
    """The model name a pool's calls are logged under (each response names its real model)."""
    return f"{POOL_LABEL_PREFIX}{name}"


def _pools(config: Mapping[str, Any]) -> Mapping[str, Any]:
    pools = config.get(POOLS_KEY) or {}
    if not isinstance(pools, Mapping):
        raise ValueError(f"{POOLS_KEY} must be a dict of pool name -> endpoints, not {pools!r}.")
    return pools


def is_pool(config: Mapping[str, Any], name: Any) -> bool:
    return isinstance(name, str) and name in _pools(config)


def pool_spec(config: Mapping[str, Any], name: str, providers: Optional[Iterable[str]] = None) -> PoolSpec:
    """Parse one pool, raising ValueError with the fix when it is malformed."""
    pools = _pools(config)
    if name not in pools:
        raise ValueError(f"{name!r} is not defined in {POOLS_KEY}.")
    raw = pools[name]
    where = f"{POOLS_KEY}[{name!r}]"
    routing = "spread"
    if isinstance(raw, Mapping):
        unknown = sorted(set(raw) - {"endpoints", "routing"})
        if unknown:
            raise ValueError(f"{where} has unknown key(s) {unknown}; use 'endpoints' and 'routing'.")
        routing = str(raw.get("routing", "spread")).strip().lower()
        raw = raw.get("endpoints")
    if routing not in POOL_ROUTINGS:
        raise ValueError(f"{where} routing must be one of {list(POOL_ROUTINGS)}, not {routing!r}.")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{where} needs a non-empty list of endpoints.")
    known = {str(provider).lower() for provider in providers} if providers is not None else None
    endpoints: List[PoolEndpoint] = []
    for position, item in enumerate(raw):
        at = f"{where} endpoint {position}"
        if not isinstance(item, Mapping):
            raise ValueError(f"{at} must be a dict such as {{'provider': 'gemini', 'model': '...'}}, not {item!r}.")
        unknown = sorted(set(item) - {"provider", "model", "weight"})
        if unknown:
            raise ValueError(f"{at} has unknown key(s) {unknown}; use 'provider', 'model' and 'weight'.")
        provider = str(item.get("provider") or "").strip().lower()
        model = item.get("model")
        weight = item.get("weight", 1.0)
        if not provider:
            raise ValueError(f"{at} needs a 'provider'.")
        if provider in _pools(config):
            raise ValueError(f"{at} names pool {provider!r}; a pool lists providers, not other pools.")
        if known is not None and provider not in known:
            raise ValueError(f"{at} names unknown provider {provider!r}; the providers are {sorted(known)}.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"{at} needs a 'model' (the model ID that provider serves).")
        if not _is_number(weight) or weight <= 0:
            raise ValueError(f"{at} weight must be a positive number, not {weight!r}.")
        endpoint = PoolEndpoint(provider, model.strip(), float(weight))
        if any(other.label == endpoint.label for other in endpoints):
            raise ValueError(f"{at} repeats {endpoint.label}; give it a larger weight instead.")
        endpoints.append(endpoint)
    return PoolSpec(name, tuple(endpoints), routing)


def role_provider(config: Mapping[str, Any], role: str) -> str:
    """The provider or pool name a role uses (simplification follows adaptation)."""
    key = ROLE_PROVIDER_KEYS["final_solver" if role == "solver" else role]
    default = config.get("API_PROVIDER_ADAPTATION", "gemini") if role == "simplification" else "gemini"
    return config.get(key) or default


def used_pool_names(config: Mapping[str, Any]) -> List[str]:
    """Pools a role of this configuration selects, in role order."""
    names: List[str] = []
    for role in ROLE_PROVIDER_KEYS:
        name = role_provider(config, role)
        if is_pool(config, name) and name not in names:
            names.append(name)
    return names


def providers_in_use(config: Mapping[str, Any]) -> Set[str]:
    """Real providers the roles call, pool members included."""
    providers: Set[str] = set()
    for role in ROLE_PROVIDER_KEYS:
        name = role_provider(config, role)
        if is_pool(config, name):
            providers.update(endpoint.provider for endpoint in pool_spec(config, name).endpoints)
        elif isinstance(name, str):
            providers.add(name)
    return providers


def config_uses_provider(config: Mapping[str, Any], provider: str) -> bool:
    """Whether a role calls ``provider``, directly or through a pool."""
    for role in ROLE_PROVIDER_KEYS:
        name = config.get(ROLE_PROVIDER_KEYS[role])
        if name == provider:
            return True
        if is_pool(config, name):
            try:
                endpoints = pool_spec(config, name).endpoints
            except ValueError:
                return True  # Malformed: count it as used rather than drop its settings.
            if any(endpoint.provider == provider for endpoint in endpoints):
                return True
    return False


def validate_provider_pools(config: Mapping[str, Any], providers: Iterable[str]) -> List[PoolSpec]:
    """Parse every pool a role uses; reject pools that shadow a provider name."""
    known = {str(provider).lower() for provider in providers}
    for name in _pools(config):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{POOLS_KEY} pool names must be non-empty strings, not {name!r}.")
        if name.strip().lower() in known:
            raise ValueError(f"{POOLS_KEY} pool {name!r} has a provider's name; give it another name.")
    return [pool_spec(config, name, known) for name in used_pool_names(config)]
