from pathlib import Path

from app.faq_loader import load_faq_text, chunk_faq_markdown, load_and_chunk_faq


def test_load_faq_text():
    faq_path = Path(__file__).resolve().parent.parent / "data" / "faq.md"

    text = load_faq_text(faq_path)

    assert isinstance(text, str)
    assert len(text) > 0
    assert "SleepPilot" in text


def test_chunk_faq_markdown_returns_chunks():
    faq_text = """
# SleepPilot FAQ

## 1. What is SleepPilot?
SleepPilot is a sleep optimization app.

## 2. Does SleepPilot diagnose sleep disorders?
No. SleepPilot is not a medical device.
"""

    chunks = chunk_faq_markdown(faq_text)

    assert len(chunks) == 2

    assert chunks[0]["id"] == "faq-001"
    assert chunks[0]["question"] == "What is SleepPilot?"
    assert "sleep optimization app" in chunks[0]["answer"]
    assert "Question:" in chunks[0]["text"]
    assert "Answer:" in chunks[0]["text"]

    assert chunks[1]["id"] == "faq-002"
    assert chunks[1]["question"] == "Does SleepPilot diagnose sleep disorders?"


def test_load_and_chunk_faq_has_15_items():
    faq_path = Path(__file__).resolve().parent.parent / "data" / "faq.md"

    chunks = load_and_chunk_faq(faq_path)

    assert len(chunks) == 15

    for chunk in chunks:
        assert "id" in chunk
        assert "question" in chunk
        assert "answer" in chunk
        assert "text" in chunk
        assert "SleepPilot" in chunk["text"]