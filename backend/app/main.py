from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.faq_loader import load_and_chunk_faq
from app.rag_pipeline import RAGPipeline


app = FastAPI(
    title="SleepPilot RAG Chatbot API",
    description="Backend API for the SleepPilot FAQ chatbot.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
FAQ_PATH = BACKEND_DIR / "data" / "faq.md"
VECTOR_DB_PATH = BACKEND_DIR / "data" / "sleeppilot_vectors.sqlite3"
FRONTEND_DIR = REPO_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=700)


class SourceResponse(BaseModel):
    id: str
    question: str
    answer: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    in_scope: bool
    mode: str
    confidence: float


def get_rag_pipeline() -> RAGPipeline:
    if not hasattr(app.state, "rag_pipeline"):
        app.state.rag_pipeline = RAGPipeline(
            faq_path=FAQ_PATH,
            vector_db_path=VECTOR_DB_PATH,
        )
    return app.state.rag_pipeline


@app.get("/")
def landing_page():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "SleepPilot RAG Chatbot API is running.", "status": "ok"}


@app.get("/api/health")
def health():
    pipeline = get_rag_pipeline()
    return {
        "status": "ok",
        "product": "SleepPilot",
        "faq_chunks": pipeline.vector_store.count(),
        "vector_db": str(VECTOR_DB_PATH),
    }


@app.get("/api/faq/chunks")
def get_faq_chunks():
    chunks = load_and_chunk_faq(FAQ_PATH)

    return {
        "count": len(chunks),
        "chunks": chunks,
    }


@app.get("/faq/chunks")
def get_legacy_faq_chunks():
    return get_faq_chunks()


@app.get("/api/retrieve")
def retrieve(question: str, top_k: int = 4) -> dict[str, Any]:
    pipeline = get_rag_pipeline()
    results = pipeline.retrieve(question, top_k=min(max(top_k, 1), 8))
    return {
        "question": question,
        "results": [
            {
                "id": result.chunk.id,
                "question": result.chunk.question,
                "answer": result.chunk.answer,
                "score": result.score,
            }
            for result in results
        ],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    pipeline = get_rag_pipeline()
    result = pipeline.answer(request.question)
    return ChatResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                id=source.id,
                question=source.question,
                answer=source.answer,
                score=source.score,
            )
            for source in result.sources
        ],
        in_scope=result.in_scope,
        mode=result.mode,
        confidence=result.confidence,
    )
