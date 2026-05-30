"""Pluggable, *fully local* text embeddings for semantic search.

The whole point of this module is that semantic search works out of the box with
**zero extra dependencies and no network** — and transparently gets better when a
real local embedding model is available. Backends, in resolution order:

* ``hash``  — a dependency-free hashing embedder (feature hashing of token
  uni/bi-grams with the signed-hash trick). Deterministic, offline, instant.
  Captures lexical overlap, so it is a sane fallback and makes tests hermetic.
* ``sentence-transformers`` — if the optional package is installed, a genuine
  neural model runs locally on CPU/GPU (no server, no network after download).
* ``ollama`` — POST to a local Ollama embedding model (e.g. ``nomic-embed-text``).
* ``openai`` — any OpenAI-compatible ``/v1/embeddings`` endpoint (llama.cpp,
  LM Studio, vLLM …). Still local when you point it at localhost.

Nothing here ever leaves the machine unless you explicitly configure a remote
endpoint. Vectors are plain ``list[float]`` (L2-normalised) so the rest of the
codebase needs no numpy; numpy is used only as an optional speed-up.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

from .config import config

try:  # optional speed-up only; never required.
    import numpy as _np
except Exception:  # pragma: no cover - numpy is optional
    _np = None


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer shared by the lexical and hashing paths."""
    return _TOKEN_RE.findall(text.lower())


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (handles non-normalised input)."""
    if not a or not b:
        return 0.0
    if _np is not None:
        va, vb = _np.asarray(a, dtype=float), _np.asarray(b, dtype=float)
        na, nb = float(_np.linalg.norm(va)), float(_np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(va.dot(vb) / (na * nb))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class Embedder(Protocol):
    """Anything that turns texts into fixed-length normalised vectors."""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class HashingEmbedder:
    """Dependency-free embedder via the signed feature-hashing trick.

    Maps token uni/bi-grams into a fixed-dimension vector. Two independent
    hashes pick the bucket and the sign, which keeps collisions unbiased. The
    result is L2-normalised, so cosine of identical text is 1.0 and lexical
    overlap drives similarity — good enough to be a real offline fallback.
    """

    dim: int = 256
    name: str = "hash"

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        if not toks:
            return vec
        features = list(toks)
        features += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]  # bigrams
        for feat in features:
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[bucket] += sign
        return _l2_normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


@dataclass
class SentenceTransformerEmbedder:
    """Local neural embeddings via the optional ``sentence-transformers`` pkg."""

    model_name: str = "all-MiniLM-L6-v2"
    name: str = "sentence-transformers"
    dim: int = 0

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(self.model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"st:{self.model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


@dataclass
class OllamaEmbedder:
    """Local embeddings via an Ollama embedding model (no data leaves host)."""

    model_name: str = "nomic-embed-text"
    name: str = "ollama"
    dim: int = 0

    def __post_init__(self) -> None:
        self.name = f"ollama:{self.model_name}"

    def _session(self):
        import requests

        s = requests.Session()
        s.trust_env = False  # localhost LLM — don't route through YTT_PROXY
        return s

    def embed(self, texts: list[str]) -> list[list[float]]:
        from .exceptions import SummarizerError

        base = config.OLLAMA_URL.rstrip("/")
        session = self._session()
        try:
            # Prefer the batch /api/embed endpoint; fall back to /api/embeddings.
            try:
                resp = session.post(
                    f"{base}/api/embed",
                    json={"model": self.model_name, "input": texts},
                    timeout=config.SUMMARY_TIMEOUT,
                )
                if resp.status_code == 404:
                    raise FileNotFoundError
                resp.raise_for_status()
                vecs = resp.json().get("embeddings")
                if vecs:
                    out = [list(map(float, v)) for v in vecs]
                    self.dim = len(out[0]) if out else 0
                    return out
            except FileNotFoundError:
                pass
            out = []
            for t in texts:
                resp = session.post(
                    f"{base}/api/embeddings",
                    json={"model": self.model_name, "prompt": t},
                    timeout=config.SUMMARY_TIMEOUT,
                )
                resp.raise_for_status()
                out.append(list(map(float, resp.json().get("embedding", []))))
            self.dim = len(out[0]) if out and out[0] else 0
            return out
        except Exception as e:  # noqa: BLE001 - surface a clear, actionable error
            raise SummarizerError(
                f"Ollama embedding failed at {base} ({e}). Is it running and is "
                f"'{self.model_name}' pulled? `ollama pull {self.model_name}`."
            )
        finally:
            session.close()


@dataclass
class OpenAIEmbedder:
    """Embeddings via any OpenAI-compatible ``/v1/embeddings`` endpoint."""

    model_name: str = "text-embedding-3-small"
    name: str = "openai"
    dim: int = 0

    def __post_init__(self) -> None:
        self.name = f"openai:{self.model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        import requests

        from .exceptions import SummarizerError

        base = config.SUMMARY_OPENAI_BASE.rstrip("/")
        headers = {}
        if config.SUMMARY_API_KEY:
            headers["Authorization"] = f"Bearer {config.SUMMARY_API_KEY}"
        try:
            resp = requests.post(
                f"{base}/embeddings",
                json={"model": self.model_name, "input": texts},
                headers=headers,
                timeout=config.SUMMARY_TIMEOUT,
            )
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out = [list(map(float, d["embedding"])) for d in data]
            self.dim = len(out[0]) if out and out[0] else 0
            return out
        except Exception as e:  # noqa: BLE001
            raise SummarizerError(f"OpenAI-compatible embedding failed at {base} ({e})")


_CACHED: dict[tuple[str, str], Embedder] = {}


def get_embedder(provider: str | None = None, model: str | None = None) -> Embedder:
    """Resolve an embedder.

    ``provider`` precedence: explicit arg > ``YTT_EMBED_PROVIDER`` > ``auto``.
    ``auto`` stays fully offline: it uses ``sentence-transformers`` if installed,
    otherwise the dependency-free hashing embedder. ``ollama``/``openai`` are
    opt-in (they may reach a local server). Instances are cached per process.
    """
    provider = (provider or config.EMBED_PROVIDER or "auto").lower()
    model = model or config.EMBED_MODEL or ""
    key = (provider, model)
    if key in _CACHED:
        return _CACHED[key]

    embedder: Embedder
    if provider == "auto":
        try:
            import sentence_transformers  # noqa: F401

            embedder = SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2")
        except Exception:
            embedder = HashingEmbedder(dim=config.EMBED_HASH_DIM)
    elif provider == "hash":
        embedder = HashingEmbedder(dim=config.EMBED_HASH_DIM)
    elif provider in ("st", "sentence-transformers", "sentence_transformers"):
        embedder = SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2")
    elif provider == "ollama":
        embedder = OllamaEmbedder(model or "nomic-embed-text")
    elif provider == "openai":
        embedder = OpenAIEmbedder(model or "text-embedding-3-small")
    else:
        raise ValueError(
            f"Unknown embed provider {provider!r} "
            "(use auto, hash, sentence-transformers, ollama, or openai)"
        )

    _CACHED[key] = embedder
    return embedder
