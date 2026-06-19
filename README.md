# SleepPilot RAG Chatbot

SleepPilot is a fictional sleep optimization app with the tagline:

> Smarter nights. Sharper days.

This project is a clean landing page with an embedded FAQ chatbot. The chatbot uses a retrieval-augmented generation pipeline over a small SleepPilot FAQ so answers stay grounded in product information instead of behaving like a general-purpose assistant.

The project was built for the Venture Engineering Intern assignment. It includes the landing page, FAQ knowledge base, chunking, embeddings, a lightweight vector database, retrieval, guardrails, unit tests, integration tests, and setup documentation.

## Product Summary

SleepPilot helps users understand sleep patterns, build healthier bedtime routines, and receive personalized sleep guidance using sleep logs, optional wearable data, and lifestyle inputs.

The assistant is called **SleepPilot Coach**. It answers questions about SleepPilot features, privacy, pricing, wearable integrations, smart alarm support, jet lag support, sleep scores, and bedtime routine guidance.

## Main Features

- Landing page served by FastAPI
- Embedded FAQ chatbot UI
- 15 SleepPilot FAQ Q&A pairs
- Markdown FAQ loader and Q&A chunker
- Self-hosted local embeddings for retrieval by default
- SQLite vector store for lightweight persistence
- Cosine similarity retrieval over document and query embeddings
- OpenAI-compatible LLM answer generation when configured
- Local grounded fallback when no LLM API key is available
- Guardrails for out-of-scope questions
- Unit tests for chunking, embedding configuration, retrieval, and guardrails
- FastAPI integration tests for `/api/chat`

## System Architecture

```mermaid
flowchart TD
    %% =========================
    %% User Interface Layer
    %% =========================
    User["User"] --> UI["Landing Page + Embedded Chatbot<br/>frontend/index.html"]
    UI --> JS["Chat UI Logic<br/>frontend/static/app.js"]
    JS --> ChatAPI["POST /api/chat<br/>FastAPI Backend"]

    %% =========================
    %% API Layer
    %% =========================
    ChatAPI --> MainAPI["API Layer<br/>backend/app/main.py"]
    MainAPI --> Pipeline["RAGPipeline.answer()<br/>backend/app/rag_pipeline.py"]

    %% =========================
    %% Input Validation
    %% =========================
    Pipeline --> Validate{"Validate user question"}
    Validate -->|Empty input| EmptyResponse["Return helper message"]
    Validate -->|Too long| LongResponse["Return too-long message"]
    Validate -->|Valid input| Retrieve["Retrieve relevant FAQ chunks"]

    %% =========================
    %% Knowledge Base Preparation
    %% =========================
    FAQFile["Markdown FAQ<br/>backend/data/faq.md"] --> Loader["FAQ Loader<br/>load_faq_text()"]
    Loader --> Chunker["FAQ Chunker<br/>chunk_faq_markdown()"]
    Chunker --> FAQChunks["Q&A Chunks<br/>faq-001 to faq-015"]

    %% =========================
    %% Embeddings
    %% =========================
    FAQChunks --> LocalEmbed["LocalSentenceTransformerEmbedder<br/>Jina v5 retrieval or E5"]
    LocalEmbed --> DocVectors["Document Embeddings"]

    %% =========================
    %% Vector Store
    %% =========================
    DocVectors --> SQLite["SQLite Vector Store<br/>sleeppilot_vectors.sqlite3"]
    SQLite --> Metadata["Metadata Check<br/>content signature, dimensions, namespace"]
    Metadata -->|FAQ or embedding config changed| Rebuild["Rebuild Vector Store"]
    Metadata -->|No change| Reuse["Reuse Existing Vectors"]
    Rebuild --> SQLite
    Reuse --> SQLite

    %% =========================
    %% Retrieval
    %% =========================
    Retrieve --> QueryEmbed["Embed User Question<br/>same embedding provider"]
    QueryEmbed --> Similarity["Cosine Similarity Search"]
    SQLite --> Similarity

    Similarity --> Rank["Rank FAQ Chunks<br/>sort by cosine similarity"]
    Rank --> TopK["Top FAQ Chunks<br/>top_k = 4"]

    %% =========================
    %% Guardrails and Context
    %% =========================
    TopK --> ScopeCheck{"Scope and confidence check"}
    ScopeCheck -->|Out of scope| Decline["Return polite refusal<br/>no sources"]
    ScopeCheck -->|Low confidence| Unknown["Return not enough FAQ information"]
    ScopeCheck -->|In scope| ContextSelector["Context Selector<br/>select 1 to 3 useful chunks"]

    ContextSelector --> PromptBuilder["Build LLM Prompt<br/>system prompt + FAQ context + user question"]

    %% =========================
    %% Answer Provider
    %% =========================
    PromptBuilder --> AnswerProvider{"Answer provider"}
    AnswerProvider -->|OpenAI-compatible configured| OpenAICompat["OpenAI-compatible Chat API<br/>OpenAI, Typhoon, OpenRouter, LM Studio"]
    AnswerProvider -->|No API or API failure| LocalFallback["Local Grounded Fallback<br/>return best FAQ answer"]

    OpenAICompat --> CleanAnswer["Clean Answer<br/>remove bracket citations"]
    LocalFallback --> CleanAnswer

    %% =========================
    %% Response
    %% =========================
    CleanAnswer --> Response["ChatResponse JSON<br/>answer, sources, in_scope, mode, confidence"]
    Response --> MainAPI
    MainAPI --> JS
    JS --> UI
    UI --> User

    %% =========================
    %% Styling
    %% =========================
    classDef ui fill:#e0f2fe,stroke:#0369a1,stroke-width:1px,color:#0f172a;
    classDef api fill:#dcfce7,stroke:#15803d,stroke-width:1px,color:#0f172a;
    classDef rag fill:#fef3c7,stroke:#b45309,stroke-width:1px,color:#0f172a;
    classDef data fill:#ede9fe,stroke:#6d28d9,stroke-width:1px,color:#0f172a;
    classDef decision fill:#fee2e2,stroke:#b91c1c,stroke-width:1px,color:#0f172a;
    classDef output fill:#f1f5f9,stroke:#475569,stroke-width:1px,color:#0f172a;

    class User,UI,JS ui;
    class ChatAPI,MainAPI api;
    class Pipeline,Retrieve,QueryEmbed,Similarity,Rank,TopK,ContextSelector,PromptBuilder,CleanAnswer rag;
    class FAQFile,Loader,Chunker,FAQChunks,DocVectors,SQLite,Metadata,Rebuild,Reuse data;
    class Validate,ScopeCheck,AnswerProvider decision;
    class EmptyResponse,LongResponse,Decline,Unknown,LocalEmbed,OpenAICompat,LocalFallback,Response output;
```
The system starts from a static landing page with an embedded chatbot. When the user sends a question, the frontend calls the FastAPI `/api/chat` endpoint. The backend passes the question into `RAGPipeline.answer()`, which validates the input, retrieves relevant SleepPilot FAQ chunks from the SQLite vector store, applies guardrails, selects useful context, and generates a grounded answer.

The FAQ knowledge base is stored in `backend/data/faq.md`. Each FAQ question and answer pair is loaded and converted into a chunk. Document and query vectors are generated by the configured embedding provider. The default provider is a local SentenceTransformers model, so FAQ text and user questions are embedded on the same machine instead of being sent to an embedding API. The resulting vectors are stored in SQLite with metadata so the vector store can be reused or rebuilt when the FAQ content or embedding configuration changes.

At query time, the user question is embedded with the same local vector space used for the FAQ chunks. The retriever ranks chunks by cosine similarity between the query embedding and each stored FAQ embedding. The top chunks are passed through scope checks and context selection before being sent to the answer provider. If an OpenAI-compatible API is configured for answer generation, the selected FAQ context is passed to that model. If no answer API is available, the system falls back to a local grounded answer using the best retrieved FAQ chunk.
## Project Structure

```text
.
├── backend
│   ├── app
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── faq_loader.py
│   │   ├── llm_client.py
│   │   ├── main.py
│   │   ├── rag_pipeline.py
│   │   └── vector_store.py
│   ├── data
│   │   └── faq.md
│   ├── tests
│   │   ├── test_api.py
│   │   ├── test_embeddings.py
│   │   ├── test_faq_loader.py
│   │   └── test_rag_pipeline.py
│   ├── .env.example
│   ├── pytest.ini
│   └── requirements.txt
├── frontend
│   ├── index.html
│   └── static
│       ├── app.js
│       ├── favicon.png
│       ├── sleeppilot-hero.png
│       └── styles.css
├── .gitignore
└── README.md
```


## Installation Guide

This project is designed to run locally. Deployment is optional.

### Requirements

- Python 3.11 or newer
- pip
- Enough disk/RAM for a local embedding model

No Node.js installation is required because the frontend is plain HTML, CSS, and JavaScript served by FastAPI.

### 1. Clone or unzip the project

```bash
git clone <your-repo-url>
cd EndToEndRAG
```

Or, if using a zip file:

```bash
unzip EndToEndRAG.zip
cd EndToEndRAG
```

### 2. Create a virtual environment

Linux/macOS:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
cd backend
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

From inside the `backend` folder with the virtual environment activated:

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

Linux/macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

You can keep retrieval fully local, which is the recommended setup for sensitive FAQ or user data. The first local run may download model weights from Hugging Face; after that, embeddings are computed on your machine from the local cache.

#### Option A: Fully local retrieval and answers

Use this for sensitive data and no embedding API calls.

In `backend/.env`:

```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-small-retrieval
LOCAL_EMBEDDING_DIMENSIONS=1024
LOCAL_EMBEDDING_QUERY_PROMPT=query
LOCAL_EMBEDDING_DOCUMENT_PROMPT=document

LLM_PROVIDER=local
```

The default Jina retrieval model is multilingual and strong for Thai/English retrieval. It is licensed CC BY-NC 4.0, so review the license before commercial use. For a smaller local model, set `LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small` and `LOCAL_EMBEDDING_DIMENSIONS=384`.

#### Option B: Local retrieval with OpenAI-compatible answer generation

Keep embeddings local, then send only the selected FAQ context and question to Typhoon, OpenAI, OpenRouter, LM Studio, Ollama-compatible endpoints, or another chat-completions-compatible answer model.

In `backend/.env`:

```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-small-retrieval
LOCAL_EMBEDDING_DIMENSIONS=1024

LLM_PROVIDER=openai-compatible
LLM_API_KEY=your_answer_model_key
LLM_BASE_URL=https://api.opentyphoon.ai/v1
OPENAI_COMPATIBLE_MODEL=typhoon-v2.1-12b-instruct
```

The answer layer can still fall back to the best retrieved FAQ answer if its configured LLM is unavailable.

### 5. Run the app

From inside the `backend` folder with the virtual environment activated:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The FastAPI backend serves the frontend page directly, so you only need one command.


## How The Project Works

### 1. Frontend

The frontend is a static landing page in `frontend/index.html`. It contains the SleepPilot product hero, feature cards, privacy section, FAQ topic strip, and embedded chatbot.

The JavaScript file `frontend/static/app.js` handles the chat interaction. When the user submits a question, it sends a request to:

```text
POST /api/chat
```

with this body:

```json
{
  "question": "Does SleepPilot support Garmin?"
}
```

The frontend then displays the returned answer, source badges, and answer mode such as `local-rag` or the configured OpenAI-compatible model name.

### 2. Backend API

The backend is a FastAPI app in `backend/app/main.py`. It serves both the landing page and the RAG API.

Main routes:

```text
GET  /
GET  /api/health
GET  /api/faq/chunks
GET  /api/retrieve?question=...
POST /api/chat
```

`GET /` returns the landing page. `POST /api/chat` is the main chatbot route. It creates or reuses one `RAGPipeline` instance and sends the user question into the RAG flow.

### 3. FAQ Knowledge Base

The knowledge base is:

```text
backend/data/faq.md
```

It contains 15 Q&A pairs about SleepPilot. This is the source of truth for the assistant.

The loader in `backend/app/faq_loader.py` reads the markdown file and splits sections that look like this:

```md
## 1. What is SleepPilot?
SleepPilot is a sleep optimization app...
```

Each Q&A pair becomes one chunk:

```python
{
    "id": "faq-001",
    "question": "What is SleepPilot?",
    "answer": "SleepPilot is a sleep optimization app...",
    "text": "Question: What is SleepPilot?\nAnswer: SleepPilot is a sleep optimization app..."
}
```

This project uses one Q&A pair per chunk because the FAQ is small and naturally structured. That keeps retrieval simple and makes source display clear.

### 4. Embeddings

Embeddings are handled in `backend/app/embeddings.py`. Retrieval uses:

```text
LocalSentenceTransformerEmbedder
```

It is configured with:

```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-small-retrieval
LOCAL_EMBEDDING_DIMENSIONS=1024
LOCAL_EMBEDDING_QUERY_PROMPT=query
LOCAL_EMBEDDING_DOCUMENT_PROMPT=document
```

`jinaai/jina-embeddings-v5-text-small-retrieval` is a retrieval-targeted multilingual model and uses different prompt names for user questions and documents:

```text
query for user questions
document for FAQ chunks
```

For a smaller model, use:

```env
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
LOCAL_EMBEDDING_DIMENSIONS=384
LOCAL_EMBEDDING_QUERY_PROMPT=query
LOCAL_EMBEDDING_DOCUMENT_PROMPT=passage
```

For document embeddings, the FAQ question is also included with the chunk text. The vector store records the embedding namespace and dimensions, then rebuilds automatically when that configuration changes.

### 5. SQLite Vector Store

The vector database is implemented in `backend/app/vector_store.py` as `SQLiteVectorStore`.

The generated database path is:

```text
backend/data/sleeppilot_vectors.sqlite3
```

The database has two tables:

```text
chunks
metadata
```

`chunks` stores:

```text
id
question
answer
text
embedding
```

`metadata` stores:

```text
content_signature
embedding_dimensions
embedding_namespace
chunk_count
```

The metadata lets the app decide whether the stored vectors are still valid. If the FAQ content, embedding dimensions, or embedding namespace changed, the old vectors are deleted and rebuilt.

The SQLite file is generated automatically. It does not need to be committed or included in the submission zip.

### 6. Retrieval Score

The retriever uses cosine similarity only:

```text
raw_score = cosine_similarity(query_embedding, chunk_embedding)
```

Then it creates a clamped display score:

```text
display_score = clamp(raw_score, 0.0, 1.0)
```

Ranking uses the raw cosine similarity. The displayed source confidence uses the clamped score because it is easier to show in the UI as a value from 0% to 100%.

### 7. Cosine Similarity

The user question is embedded using the same provider and vector space as the FAQ chunks. Then the retriever compares the query vector with every stored chunk vector using cosine similarity.

```text
cosine_similarity = dot(query_vector, chunk_vector) / (query_norm × chunk_norm)
```

A higher value means the question is more similar to that FAQ chunk.

### 8. Context Selection

The vector store returns the top 4 results. Then `RAGPipeline._select_context_results()` chooses how many chunks to send to the LLM.

Rules:

```text
Always include the best chunk.
Include more chunks only if their scores are close enough.
Use at most 3 chunks as LLM context.
```

This keeps the answer grounded without sending too much irrelevant text.

### 9. Guardrails

Guardrails are implemented in `backend/app/rag_pipeline.py`.

The assistant accepts questions related to SleepPilot, sleep, privacy, pricing, wearables, alarms, travel, students, jet lag, sleep scores, and bedtime routines.

It declines unrelated topics such as:

```text
coding
weather
politics
stocks
recipes
homework
movies
sports
```

If the question is unrelated, the system returns a decline message before generating an LLM answer.

If the question seems in-scope but the retrieved score is too low, the system returns an unknown answer instead of hallucinating.

### 10. Answer Generation

Answer generation happens in `RAGPipeline._generate_answer()`.

The selected FAQ chunks are formatted like this:

```text
[1] faq-006 (score 0.912)
Q: Which wearable devices does SleepPilot support?
A: SleepPilot is designed to support common wearable platforms...
```

The system prompt tells the LLM:

```text
You are SleepPilot Coach.
Only answer SleepPilot-related questions.
Use only the provided FAQ context.
Politely decline unrelated requests.
SleepPilot is wellness guidance, not a medical device.
```

The answer provider is selected in this order:

1. OpenAI-compatible API, if `LLM_PROVIDER=openai-compatible` or `LLM_PROVIDER=api`
2. Local grounded fallback if no API is configured or the API call fails

The local fallback returns the best retrieved FAQ answer directly. This keeps the project runnable without an API key.

### 14. Source Display

The API response includes sources:

```json
{
  "answer": "...",
  "sources": [
    {
      "id": "faq-006",
      "question": "Which wearable devices does SleepPilot support?",
      "answer": "...",
      "score": 0.92
    }
  ],
  "in_scope": true,
  "mode": "local-rag",
  "confidence": 0.92
}
```

The frontend displays each source as a badge, such as:

```text
faq-006 - 92%
```

The answer cleaner removes accidental bracket citations like `[1]` or `[2]`, because the UI already displays sources separately.

## File-by-File Explanation

### `backend/app/faq_loader.py`

Loads the markdown FAQ file and splits it into Q&A chunks. It is responsible for turning `faq.md` into RAG-ready data.

Main functions:

```text
load_faq_text()
chunk_faq_markdown()
load_and_chunk_faq()
```

### `backend/app/embeddings.py`

Defines the embedding interface, local SentenceTransformers embedding client, vector normalization, and cosine similarity.

Important parts:

```text
LocalSentenceTransformerEmbedder
normalize_vector()
cosine_similarity()
create_default_embedder()
```

`create_default_embedder()` uses `EMBEDDING_PROVIDER=local` by default.

### `backend/app/vector_store.py`

Implements the lightweight vector database using SQLite. It stores FAQ chunks and embeddings, rebuilds vectors when needed, and retrieves relevant chunks.

Important parts:

```text
SQLiteVectorStore
StoredChunk
SearchResult
similarity_search()
```

This file is the core retrieval engine.

### `backend/app/rag_pipeline.py`

Orchestrates the full RAG process. It loads the FAQ, ensures the vector store exists, runs retrieval, applies guardrails, selects context, calls the LLM or fallback, cleans the answer, and returns sources.

Important parts:

```text
RAGPipeline
ChatResult
ChatSource
SYSTEM_PROMPT
DECLINE_MESSAGE
UNKNOWN_MESSAGE
answer()
_select_context_results()
_generate_answer()
_is_out_of_scope()
```

### `backend/app/llm_client.py`

Contains the optional API client for answer generation.

Important classes:

```text
OpenAICompatibleClient
LLMResponse
```

`OpenAICompatibleClient` supports Typhoon, OpenAI, OpenRouter, LM Studio, Ollama-compatible routers, or other chat-completions-compatible APIs.

### `backend/app/main.py`

Creates the FastAPI application. It serves the frontend and exposes API routes for health checks, FAQ chunk inspection, retrieval debugging, and chat.

Important routes:

```text
GET  /
GET  /api/health
GET  /api/faq/chunks
GET  /api/retrieve
POST /api/chat
```

### `backend/data/faq.md`

The SleepPilot knowledge base. It contains 15 FAQ Q&A pairs and is the only product knowledge source used by the RAG assistant.

### `backend/tests/test_faq_loader.py`

Tests that the FAQ file loads correctly and is split into 15 usable chunks.

### `backend/tests/test_embeddings.py`

Tests local retrieval prompt wiring, vector normalization, provider configuration, and domain phrase tokenization.

### `backend/tests/test_rag_pipeline.py`

Tests retrieval and RAG behavior. It checks that many different user questions retrieve the correct FAQ chunk, that similar questions like wearable support versus no-wearable usage are separated, that in-scope questions get answers, and that out-of-scope requests are declined.

### `backend/tests/test_api.py`

Tests the FastAPI `/api/chat` endpoint using `TestClient`. These tests check that the API returns grounded answers for in-scope questions and applies guardrails for unrelated questions.

### `frontend/index.html`

The landing page and chatbot shell. It contains the product content and chat interface.

### `frontend/static/app.js`

Controls frontend chat behavior. It sends user questions to `/api/chat`, displays answers, shows source badges, updates answer mode, and checks API health.

### `frontend/static/styles.css`

Styles the landing page, product sections, chatbot, messages, source badges, and responsive layout.

## API Usage

### Health check

```bash
curl http://127.0.0.1:8000/api/health
```

Example response:

```json
{
  "status": "ok",
  "product": "SleepPilot",
  "faq_chunks": 15,
  "embedding_provider": "local-sentence-transformers:jinaai/jina-embeddings-v5-text-small-retrieval:1024:q=query:d=document:normalize=True",
  "embedding_dimensions": "1024",
  "vector_db": ".../backend/data/sleeppilot_vectors.sqlite3"
}
```

### Inspect FAQ chunks

```bash
curl http://127.0.0.1:8000/api/faq/chunks
```

### Debug retrieval

```bash
curl "http://127.0.0.1:8000/api/retrieve?question=Does%20SleepPilot%20support%20Garmin%3F"
```

### Chat request

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Does SleepPilot work without a wearable device?"}'
```

## Running Tests

From inside the `backend` folder with the virtual environment activated:

```bash
pytest
```

The tests inject a minimal embedding test double, so they do not download a local embedding model.

Current test coverage includes:

- FAQ loading
- FAQ chunking
- Local embedding prompt wiring
- Domain phrase tokenization
- Vector retrieval ranking
- Similar-topic distinction, such as Garmin/Fitbit support versus no-wearable usage
- In-scope RAG answer behavior
- Context selection
- Citation cleanup
- Out-of-scope guardrails
- FastAPI `/api/chat` integration path

## Manual Test Cases

| Scenario | Question | Expected behavior |
| --- | --- | --- |
| In-scope FAQ | Does SleepPilot work with Garmin or Fitbit? | Retrieves `faq-006` and answers from wearable integrations. |
| In-scope no-wearable case | Can I use SleepPilot without a wearable? | Retrieves `faq-005` and explains manual entry support. |
| Wellness boundary | Does SleepPilot diagnose sleep disorders? | Explains SleepPilot is not a medical device and suggests professional care for concerning symptoms. |
| Privacy | How does SleepPilot protect my privacy? | Retrieves the privacy FAQ and explains user control and data deletion. |
| Out of scope | Write Python code for a todo app. | Politely declines and redirects to SleepPilot topics. |
| Edge case | What if SleepPilot cannot answer my question? | Explains that the assistant only answers from available FAQ/product information. |
| Missing answer-model key | Ask any in-scope question while the answer provider is unavailable. | Returns a grounded FAQ answer with sources after retrieval succeeds. |

## Tools Used

- Python
- FastAPI
- Pydantic
- SQLite
- Requests
- Pytest
- Plain HTML, CSS, and JavaScript
- SentenceTransformers, Transformers, and Torch for self-hosted local embeddings
- OpenAI-compatible chat-completions API support for optional alternative LLMs


## Troubleshooting

### `ModuleNotFoundError: No module named 'app'`

Run commands from inside the `backend` folder. The project includes `backend/pytest.ini` with:

```ini
[pytest]
pythonpath = .
testpaths = tests
```

So this should work:

```bash
cd backend
pytest
```

### The first request is slow

The first request may download the local embedding model and build the vector database by embedding all FAQ chunks. After that, the SQLite vector store is reused.

### Retrieval seems outdated after changing embeddings

Delete the generated SQLite file:

```bash
rm backend/data/sleeppilot_vectors.sqlite3
```

or change the embedding model, prompt names, or dimensions. The stored namespace includes those values, for example:

```text
local-sentence-transformers:jinaai/jina-embeddings-v5-text-small-retrieval:1024:q=query:d=document:normalize=True
```

### The frontend says API offline

Make sure the backend is running:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```
