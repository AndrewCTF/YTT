"""Local-LLM transcript summarization to save downstream token usage.

Summarizing a transcript with a *local* model (Ollama by default, or any
OpenAI-compatible endpoint) means the agent that consumes the result ingests a
short summary instead of the full transcript — large token savings with no data
leaving the machine.

All of this is opt-in: nothing here runs unless you ask for the ``summary``
format / ``summarize=True``. The default model is ``qwen3.6:27b``; override with
``YTT_SUMMARY_MODEL`` (smaller options: ``qwen3:8b``, ``qwen3:4b``,
``qwen3.5:2b``).
"""

import re
from urllib.parse import urlparse

import requests

from .config import config
from .exceptions import SummarizerError

_DEFAULT_PROMPT = (
    "You are summarizing a YouTube video transcript. Write a faithful, concise "
    "summary that captures the main topics, key points, arguments, and any "
    "conclusions or takeaways. Use short bullet points. Do not invent details "
    "that are not present in the transcript.\n\nTranscript:\n{text}\n\nSummary:"
)

_REDUCE_PROMPT = (
    "The following are partial summaries of consecutive sections of one video "
    "transcript. Combine them into a single coherent summary with short bullet "
    "points, removing redundancy and preserving all distinct key points.\n\n"
    "Partial summaries:\n{text}\n\nCombined summary:"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove reasoning ``<think>...</think>`` blocks some models emit.

    Substitution is repeated until stable so nested same-name blocks collapse.
    """
    prev = None
    while prev != text:
        prev = text
        text = _THINK_RE.sub("", text)
    # Remove any orphan tags left by malformed/nested output.
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _validate_url(url: str) -> str:
    """Only allow http/https endpoints (defensive; endpoint is operator config)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SummarizerError(f"Unsupported summarizer endpoint scheme: {url!r}")
    return url.rstrip("/")


def _session() -> requests.Session:
    """A session that ignores ambient HTTP(S)_PROXY (the LLM is local)."""
    s = requests.Session()
    s.trust_env = False  # don't route localhost LLM traffic through YTT_PROXY etc.
    return s


def _chunk(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most ``max_chars`` on whitespace boundaries.

    A single token longer than ``max_chars`` (rare for transcripts) is hard-split
    so the contract holds and no oversized prompt is ever sent.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for w in text.split():
        # Hard-split any single word that exceeds the limit.
        while len(w) > max_chars:
            if cur:
                chunks.append(" ".join(cur))
                cur, size = [], 0
            chunks.append(w[:max_chars])
            w = w[max_chars:]
        if size + len(w) + 1 > max_chars and cur:
            chunks.append(" ".join(cur))
            cur, size = [], 0
        cur.append(w)
        size += len(w) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _ollama_generate(prompt: str, model: str, session: requests.Session) -> str:
    """One-shot generation via the Ollama /api/generate endpoint."""
    base = _validate_url(config.OLLAMA_URL)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": config.SUMMARY_KEEP_ALIVE,
        "options": {
            "temperature": config.SUMMARY_TEMPERATURE,
            "num_ctx": config.SUMMARY_NUM_CTX,
        },
    }

    def _call():
        return session.post(f"{base}/api/generate", json=payload, timeout=config.SUMMARY_TIMEOUT)

    try:
        resp = _call()
        if resp.status_code == 404:
            # 404 here means the model is not pulled (the server IS reachable).
            if config.SUMMARY_AUTO_PULL:
                _ollama_pull(model, session)
                resp = _call()
                if resp.status_code == 404:
                    raise SummarizerError(
                        f"Model {model!r} is still unavailable after an auto-pull "
                        "attempt; the pull may have failed. Check `ollama list`, "
                        "disk space, and network."
                    )
            else:
                raise SummarizerError(
                    f"Model {model!r} is not available in Ollama. Pull it with "
                    f"`ollama pull {model}`, set YTT_SUMMARY_AUTO_PULL=1, or pick a "
                    "smaller model via YTT_SUMMARY_MODEL (e.g. qwen3:8b)."
                )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise SummarizerError(
            f"Could not reach Ollama at {base} ({e}). Is it running? "
            "Start it with `ollama serve` or set YTT_OLLAMA_URL."
        )
    text = data.get("response", "")
    if not text:
        raise SummarizerError(f"Empty summary from model {model!r}")
    return _strip_think(text)


def _ollama_pull(model: str, session: requests.Session) -> None:
    """Pull a model on demand (only called when SUMMARY_AUTO_PULL is enabled)."""
    base = _validate_url(config.OLLAMA_URL)
    try:
        resp = session.post(
            f"{base}/api/pull",
            json={"model": model, "stream": False},
            timeout=max(config.SUMMARY_TIMEOUT, 1800),
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        raise SummarizerError(f"Failed to pull model {model!r}: {e}")


def _openai_chat(prompt: str, model: str, session: requests.Session) -> str:
    """One-shot generation via an OpenAI-compatible /chat/completions endpoint."""
    base = _validate_url(config.SUMMARY_OPENAI_BASE)
    headers = {}
    if config.SUMMARY_API_KEY:
        headers["Authorization"] = f"Bearer {config.SUMMARY_API_KEY}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.SUMMARY_TEMPERATURE,
        "stream": False,
    }
    try:
        resp = session.post(
            f"{base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=config.SUMMARY_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise SummarizerError(f"Could not reach OpenAI-compatible endpoint at {base} ({e})")
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise SummarizerError("Unexpected response from OpenAI-compatible endpoint")
    if not text:
        raise SummarizerError(f"Empty summary from model {model!r}")
    return _strip_think(text)


def _generate(prompt: str, model: str, provider: str, session: requests.Session) -> str:
    if provider == "ollama":
        return _ollama_generate(prompt, model, session)
    if provider == "openai":
        return _openai_chat(prompt, model, session)
    raise SummarizerError(f"Unknown summary provider: {provider!r} (use 'ollama' or 'openai')")


def summarize_text(
    text: str,
    model: str | None = None,
    provider: str | None = None,
    prompt_template: str | None = None,
) -> str:
    """Summarize transcript text with a local LLM (map-reduce for long input).

    Args:
        text: The (ideally already-cleaned) transcript text.
        model: Model name; defaults to ``config.SUMMARY_MODEL``.
        provider: ``"ollama"`` or ``"openai"``; defaults to ``config.SUMMARY_PROVIDER``.
        prompt_template: Optional override containing ``{text}``.

    Returns:
        The summary string.

    Raises:
        SummarizerError: If the provider is unreachable or returns nothing.
    """
    text = (text or "").strip()
    if not text:
        raise SummarizerError("Nothing to summarize (empty transcript)")

    model = model or config.SUMMARY_MODEL
    provider = (provider or config.SUMMARY_PROVIDER).lower()
    template = prompt_template or _DEFAULT_PROMPT

    session = _session()
    try:
        chunks = _chunk(text, config.SUMMARY_MAX_INPUT_CHARS)
        partials = [_generate(template.format(text=c), model, provider, session) for c in chunks]

        if len(partials) == 1:
            return partials[0]

        # Reduce: combine partial summaries into one (recurse if still huge).
        combined = "\n\n".join(f"- Section {i + 1}:\n{p}" for i, p in enumerate(partials))
        if len(combined) > config.SUMMARY_MAX_INPUT_CHARS:
            combined = "\n\n".join(
                _generate(_REDUCE_PROMPT.format(text=c), model, provider, session)
                for c in _chunk(combined, config.SUMMARY_MAX_INPUT_CHARS)
            )
        return _generate(_REDUCE_PROMPT.format(text=combined), model, provider, session)
    finally:
        session.close()


def preload(model: str | None = None, provider: str | None = None) -> bool:
    """Hot-load the model into memory (Ollama) so the first summary is fast.

    Returns True on success; raises SummarizerError if the provider is down.
    No-op for non-Ollama providers.
    """
    model = model or config.SUMMARY_MODEL
    provider = (provider or config.SUMMARY_PROVIDER).lower()
    if provider != "ollama":
        return False
    base = _validate_url(config.OLLAMA_URL)
    session = _session()
    try:
        resp = session.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": config.SUMMARY_KEEP_ALIVE},
            timeout=config.SUMMARY_TIMEOUT,
        )
        if resp.status_code == 404 and config.SUMMARY_AUTO_PULL:
            _ollama_pull(model, session)
            return True
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        raise SummarizerError(f"Could not preload model {model!r} at {base} ({e})")
    finally:
        session.close()


def is_available(provider: str | None = None) -> bool:
    """Best-effort check that the configured provider is reachable."""
    provider = (provider or config.SUMMARY_PROVIDER).lower()
    session = _session()
    try:
        if provider == "ollama":
            url = f"{_validate_url(config.OLLAMA_URL)}/api/tags"
        else:
            url = f"{_validate_url(config.SUMMARY_OPENAI_BASE)}/models"
        return session.get(url, timeout=5).ok
    except Exception:
        return False
    finally:
        session.close()
