"""Tests for the dependency-free embedding layer."""

import math

from ytt.embeddings import HashingEmbedder, cosine, get_embedder, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_cosine_identical_is_one():
    v = [0.1, 0.2, 0.3, 0.4]
    assert math.isclose(cosine(v, v), 1.0, rel_tol=1e-9)


def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_handles_empty():
    assert cosine([], [1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_hashing_embedder_is_normalized_and_deterministic():
    emb = HashingEmbedder(dim=128)
    a1 = emb.embed(["the quick brown fox"])[0]
    a2 = emb.embed(["the quick brown fox"])[0]
    assert a1 == a2  # deterministic
    assert len(a1) == 128
    assert math.isclose(math.sqrt(sum(x * x for x in a1)), 1.0, rel_tol=1e-6)


def test_hashing_embedder_semantics():
    emb = HashingEmbedder(dim=512)
    same = emb.embed(["python async event loop"])[0]
    near = emb.embed(["the python async event loop scheduler"])[0]
    far = emb.embed(["baking sourdough bread at home"])[0]
    q = emb.embed(["python async event loop"])[0]
    assert cosine(q, same) > cosine(q, near) >= cosine(q, far)
    assert cosine(q, near) > cosine(q, far)


def test_get_embedder_hash_and_cache():
    e1 = get_embedder("hash")
    e2 = get_embedder("hash")
    assert isinstance(e1, HashingEmbedder)
    assert e1 is e2  # cached per process


def test_get_embedder_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        get_embedder("does-not-exist")
