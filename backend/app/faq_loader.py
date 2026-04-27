import re
from pathlib import Path
from typing import Dict, List


def load_faq_text(file_path: str | Path) -> str:
    """
    Load the FAQ markdown file as plain text.

    Args:
        file_path: Path to the FAQ markdown file.

    Returns:
        The full FAQ file content as a string.

    Raises:
        FileNotFoundError: If the FAQ file does not exist.
        ValueError: If the FAQ file is empty.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"FAQ file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("FAQ file is empty.")

    return text


def chunk_faq_markdown(faq_text: str) -> List[Dict[str, str]]:
    """
    Split a markdown FAQ document into Q&A chunks.

    Expected format:

    ## 1. What is SleepPilot?
    Answer text...

    ## 2. How does SleepPilot improve my sleep?
    Answer text...

    Each FAQ item becomes one chunk.

    Args:
        faq_text: Full FAQ markdown text.

    Returns:
        A list of chunks. Each chunk contains:
        - id
        - question
        - answer
        - text
    """
    if not faq_text.strip():
        raise ValueError("FAQ text is empty.")

    pattern = r"##\s+\d+\.\s+(.*?)\n(.*?)(?=\n##\s+\d+\.|\Z)"
    matches = re.findall(pattern, faq_text, flags=re.DOTALL)

    chunks = []

    for index, match in enumerate(matches, start=1):
        question = match[0].strip()
        answer = match[1].strip()

        if not question or not answer:
            continue

        chunk = {
            "id": f"faq-{index:03d}",
            "question": question,
            "answer": answer,
            "text": f"Question: {question}\nAnswer: {answer}",
        }

        chunks.append(chunk)

    if not chunks:
        raise ValueError(
            "No FAQ chunks found. Make sure questions use the format: ## 1. Question?"
        )

    return chunks


def load_and_chunk_faq(file_path: str | Path) -> List[Dict[str, str]]:
    """
    Load FAQ markdown and convert it into RAG-ready chunks.
    """
    faq_text = load_faq_text(file_path)
    return chunk_faq_markdown(faq_text)


if __name__ == "__main__":
    faq_path = Path(__file__).resolve().parent.parent / "data" / "faq.md"
    chunks = load_and_chunk_faq(faq_path)

    for chunk in chunks:
        print("=" * 80)
        print(chunk["id"])
        print(chunk["question"])
        print(chunk["text"][:300])
