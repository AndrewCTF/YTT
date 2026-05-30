"""Shared HTTP layer: pooled session, realistic headers, retry with backoff.

Centralising network access here is what keeps YouTube rate limits at bay:

* every request carries a realistic browser/app ``User-Agent`` and language
  headers (the default ``python-requests`` UA is the #1 cause of 429s),
* a single pooled :class:`requests.Session` reuses TCP connections,
* transient failures (429 / 5xx / connection resets) are retried with
  exponential backoff + jitter, honouring the ``Retry-After`` header,
* an optional proxy (``YTT_PROXY``) routes traffic to escape IP-level blocks.
"""

import random
import time

import requests

from .config import BROWSER_USER_AGENT, config
from .exceptions import RateLimitError

# Status codes worth retrying — transient throttling / server hiccups.
_RETRY_STATUS = {429, 500, 502, 503, 504}


def new_session() -> requests.Session:
    """Create a session pre-loaded with browser-like default headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": BROWSER_USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "*/*",
        }
    )
    if config.proxies:
        session.proxies.update(config.proxies)
    return session


def _sleep_for(attempt: int, retry_after: float | None) -> float:
    """Backoff delay for a given attempt, honouring Retry-After when present."""
    if retry_after is not None:
        return min(retry_after, config.BACKOFF_MAX)
    # Exponential backoff with full jitter.
    base = min(config.BACKOFF_BASE**attempt, config.BACKOFF_MAX)
    return random.uniform(0, base)


def request(
    method: str,
    url: str,
    *,
    session: requests.Session | None = None,
    headers: dict | None = None,
    json: dict | None = None,
    params: dict | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    allow_status: tuple[int, ...] = (),
) -> requests.Response:
    """Perform an HTTP request with retry/backoff on transient failures.

    Args:
        method: HTTP method ("GET"/"POST").
        url: Target URL.
        session: Optional session to reuse (a throwaway one is made otherwise).
        headers: Per-request header overrides (merged over session headers).
        json: JSON body for POST.
        params: Query params.
        timeout: Per-request timeout (defaults to ``config.REQUEST_TIMEOUT``).
        max_retries: Override retry count (defaults to ``config.MAX_RETRIES``).
        allow_status: Status codes to return without raising (e.g. 404).

    Returns:
        The final :class:`requests.Response`.

    Raises:
        RateLimitError: If still rate-limited after exhausting retries.
        requests.HTTPError: For non-retryable 4xx not in ``allow_status``.
        requests.RequestException: For network errors after retries.
    """
    own_session = session is None
    session = session or new_session()
    timeout = timeout or config.REQUEST_TIMEOUT
    retries = config.MAX_RETRIES if max_retries is None else max_retries

    try:
        for attempt in range(retries + 1):
            try:
                resp = session.request(
                    method, url, headers=headers, json=json, params=params, timeout=timeout
                )
            except requests.RequestException:
                if attempt >= retries:
                    raise
                time.sleep(_sleep_for(attempt, None))
                continue

            if resp.status_code in allow_status or resp.status_code < 400:
                return resp

            if resp.status_code in _RETRY_STATUS and attempt < retries:
                retry_after_hdr = resp.headers.get("Retry-After")
                retry_after = None
                if retry_after_hdr:
                    try:
                        retry_after = float(retry_after_hdr)
                    except ValueError:
                        retry_after = None
                time.sleep(_sleep_for(attempt, retry_after))
                continue

            if resp.status_code == 429:
                retry_after_hdr = resp.headers.get("Retry-After")
                ra = (
                    int(float(retry_after_hdr))
                    if retry_after_hdr and retry_after_hdr.isdigit()
                    else None
                )
                raise RateLimitError(f"Rate limited by YouTube on {url}", retry_after=ra)

            resp.raise_for_status()
            return resp

        # Retries exhausted on a retryable status.
        raise RateLimitError(f"Exhausted retries (still throttled) for {url}")
    finally:
        if own_session:
            session.close()
