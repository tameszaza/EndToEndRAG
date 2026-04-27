from pathlib import Path
from fastapi import FastAPI

from app.faq_loader import load_and_chunk_faq


app = FastAPI(
    title="SleepPilot RAG Chatbot API",
    description="Backend API for the SleepPilot FAQ chatbot.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "SleepPilot RAG Chatbot API is running.",
        "status": "ok",
    }


@app.get("/faq/chunks")
def get_faq_chunks():
    faq_path = Path(__file__).resolve().parent.parent / "data" / "faq.md"
    chunks = load_and_chunk_faq(faq_path)

    return {
        "count": len(chunks),
        "chunks": chunks,
    }