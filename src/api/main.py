from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from pathlib import Path

# Ensure the root project directory is in the PYTHONPATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag import SakhiResources, build_graph
from langgraph.graph.state import CompiledStateGraph

# Initialize FastAPI App
app = FastAPI(
    title="Sakhi AI Legal Companion API",
    description="Backend API for the Sakhi RAG engine",
    version="1.0.0"
)

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold models and pipeline
resources: SakhiResources = None
pipeline: CompiledStateGraph = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: list[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str

@app.on_event("startup")
async def startup_event():
    global resources, pipeline
    try:
        print("🚀 Starting up Sakhi API...")
        # Load heavy resources once on startup
        resources = SakhiResources()
        pipeline = build_graph(resources)
        print("✅ API ready to serve requests.")
    except Exception as e:
        print(f"❌ Failed to initialize resources: {e}")
        raise e

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if not pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline is not initialized.")

    try:
        print(f"  ❓ API received query: {request.query}")
        
        # Build chat history
        history = [msg.model_dump() if hasattr(msg, "model_dump") else msg.dict() for msg in request.chat_history]
        
        # Compute clarification_count: count consecutive trailing assistant
        # messages that end with '?' — these are clarifying questions
        clarification_count = 0
        for msg in reversed(history):
            if msg["role"] == "assistant" and msg["content"].rstrip().endswith("?"):
                clarification_count += 1
            elif msg["role"] == "user":
                continue  # skip user messages when counting
            else:
                break  # non-question assistant message = real answer was given
        
        print(f"  📊 Chat history: {len(history)} messages, {clarification_count} clarifications so far")
        
        initial_state = {
            "query": request.query,
            "corrected_query": "",
            "understood_as": "",
            "needs_clarification": False,
            "clarification_question": None,
            "clarification_count": clarification_count,
            "is_legal_query": True,
            "search_queries": [],
            "embeddings": [],
            "chunks": [],
            "context": "",
            "answer": "",
            "low_confidence": False,
            "chat_history": history,
        }
        
        # Invoke LangGraph pipeline
        result = pipeline.invoke(initial_state)
        answer = result.get("answer", "I'm sorry, I could not generate an answer.")
        
        return ChatResponse(answer=answer)
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    status = "healthy" if pipeline is not None else "initializing"
    return {"status": status}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=True)
