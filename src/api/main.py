from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import asyncio
import logging
from pathlib import Path

# Ensure the root project directory is in the PYTHONPATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag import SakhiResources, build_graph
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger("sakhi.api")

# ── Global resources ──────────────────────────────────────────────────────────
resources: SakhiResources = None
pipeline: CompiledStateGraph = None


def _load_resources():
    """Synchronous initialization — runs in thread pool so it doesn't block."""
    logger.info("🚀 Starting up Sakhi API...")
    r = SakhiResources()
    p = build_graph(r)
    logger.info("✅ Sakhi API ready to serve requests.")
    return r, p


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan event handler (replaces deprecated on_event)."""
    global resources, pipeline
    try:
        # Run heavy initialization in thread pool — never blocks the event loop
        loop = asyncio.get_event_loop()
        resources, pipeline = await loop.run_in_executor(None, _load_resources)
    except Exception as e:
        logger.error(f"❌ Failed to initialize Sakhi resources: {e}")
        raise
    yield
    # Cleanup (if needed in future)
    logger.info("🛑 Sakhi API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sakhi AI Legal Companion API",
    description="Backend API for the Sakhi RAG engine",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS — only allow known origins ──────────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://sakhi-alpha.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    chat_history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline is still initializing. Try again in a moment.")

    try:
        logger.info(f"  ❓ API received query: {request.query}")

        # Build chat history
        history = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg.dict()
            for msg in request.chat_history
        ]

        # Count trailing clarification questions
        clarification_count = 0
        for msg in reversed(history):
            if msg["role"] == "assistant" and msg["content"].rstrip().endswith("?"):
                clarification_count += 1
            elif msg["role"] == "user":
                continue
            else:
                break

        logger.info(f"  📊 Chat history: {len(history)} messages, {clarification_count} clarifications")

        initial_state = {
            "query":                  request.query,
            "corrected_query":        "",
            "understood_as":          "",
            "needs_clarification":    False,
            "clarification_question": None,
            "clarification_count":    clarification_count,
            "is_legal_query":         True,
            "search_queries":         [],
            "embeddings":             [],
            "chunks":                 [],
            "context":                "",
            "answer":                 "",
            "low_confidence":         False,
            "chat_history":           history,
        }

        # ✅ Run pipeline in thread pool — does NOT block the async event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, pipeline.invoke, initial_state)

        answer = result.get("answer", "I'm sorry, I could not generate an answer.")
        return ChatResponse(answer=answer)

    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status":   "healthy" if pipeline is not None else "initializing",
        "model":    "qwen/qwen3-embedding-8b",
        "llm":      "llama-3.3-70b-versatile",
        "version":  "2.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)
