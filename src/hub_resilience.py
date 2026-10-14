"""Retry transient Hugging Face Hub failures without hiding real errors.

Kaggle sessions run for hours, so one 503 or rate-limit response must not end a
run. Only failures that a later attempt can fix are retried; authentication,
permission, missing-repository, and local file errors propagate immediately.
"""

import time
from typing import Callable, Optional, Sequence, TypeVar

T = TypeVar("T")

# Seconds waited before each retry: four attempts, at most 85 seconds in total.
DEFAULT_RETRY_DELAYS = (5.0, 20.0, 60.0)
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 412, 425, 429})
_TRANSIENT_MODULES = ("httpcore", "httpx", "requests", "urllib3")


def hub_error_status(error: BaseException) -> Optional[int]:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_transient_hub_error(error: BaseException) -> bool:
    """True for rate limits, server errors, and network failures worth repeating."""
    status = hub_error_status(error)
    if status is not None:
        return status in _TRANSIENT_STATUS_CODES or status >= 500
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return type(error).__module__.startswith(_TRANSIENT_MODULES)


def call_with_retries(
    operation: Callable[[], T], what: str, *,
    delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> T:
    """Run ``operation``; after a transient failure wait and try again.

    The last failure propagates unchanged once ``delays`` is used up, so callers
    keep their existing error handling. ``operation`` must be safe to repeat.
    """
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as error:
            if attempt >= len(delays) or not is_transient_hub_error(error):
                raise
            wait = float(delays[attempt])
            attempt += 1
            log(
                f"{what} failed transiently ({type(error).__name__}: {str(error)[:160]}); "
                f"retry {attempt}/{len(delays)} in {wait:g}s."
            )
            sleep(wait)
