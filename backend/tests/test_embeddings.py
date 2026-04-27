import pytest

from app.embeddings import HashingEmbedder, cosine_similarity, tokenize


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dimensions=64)

    first = embedder.embed("Does SleepPilot support Garmin wearables?")
    second = embedder.embed("Does SleepPilot support Garmin wearables?")

    assert first == second
    assert len(first) == 64
    assert cosine_similarity(first, first) == pytest.approx(1.0)


def test_tokenize_adds_domain_phrase_tokens():
    tokens = tokenize("Can SleepPilot help with jet lag and Apple Health?")

    assert "jet_lag" in tokens
    assert "apple_health" in tokens
    assert "sleeppilot" not in tokens
