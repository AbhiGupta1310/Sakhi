"""
Sakhi — RAG Implementation (Optimized v3)
==========================================
LangGraph + LangChain + LangSmith + Groq LLM

Flow:
  User Query → Understand & Clarify → Generate 2 Search Queries → Embed Both
             → Retrieve + Deduplicate → Groq LLM → Answer

New in v3:
  - understand node: fixes typos/slang, detects ambiguity, asks clarifying Q if needed
  - conditional routing: if clarification needed → ask user → skip retrieval
  - otherwise: full rewrite → embed → retrieve → generate pipeline

Setup:
    pip install langchain langchain-groq langgraph langsmith chromadb FlagEmbedding python-dotenv

.env file:
    GROQ_API_KEY=your_groq_key
    LANGCHAIN_API_KEY=your_langsmith_key   # optional
    LANGCHAIN_TRACING_V2=true              # optional
    LANGCHAIN_PROJECT=sakhi                # optional

Usage:
    python rag.py --query "can my landlord evict me without notice"
    python rag.py --interactive
"""

import os
import json
import argparse
from typing import TypedDict, List, Optional

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_PROJECT",    os.getenv("LANGCHAIN_PROJECT", "sakhi"))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import chromadb
from FlagEmbedding import BGEM3FlagModel

from src.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    TOP_K,
    GROQ_MODEL,
    SYSTEM_PROMPT,
)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

# Node 1: Understand the query & decide if we need more context
UNDERSTAND_PROMPT = """You are Sakhi, a highly empathetic, human-like legal companion for everyday Indians.

Your job is to read the user's latest message and the recent conversation history to decide if you have enough information to provide accurate legal advice, OR if you need to ask a clarifying question first.

CRITICAL RULES:
1. DO NOT exceed 2-3 clarifying questions in a row. If the user has already answered several questions, you MUST attempt to answer instead of asking more.
2. If you DO need clarification, the question MUST be highly empathetic, warm, and conversational. 
   BAD: "What is the date of the incident?"
   GOOD: "I'm so sorry you're dealing with this. It sounds really stressful. To help me give you the best advice, could you tell me...?"
3. Fix typos and Indian English slang (e.g., "tolled" -> "towed").

Conversation History:
{chat_history}

Latest User Message: "{query}"

Respond ONLY with a valid JSON object (no markdown, no explanation):
{{
  "corrected_query": "<the user's query with typos fixed>",
  "understood_as": "<one sentence summarizing the legal issue>",
  "needs_clarification": true/false,
  "clarification_question": "<a warm, empathetic question IF needs_clarification is true, else null>"
}}
"""

# Node 2: Generate 2 short retrieval queries from the corrected query
# Minimum cosine similarity score to trust retrieved chunks.
# Below this → Sakhi says "I don't have enough info" instead of hallucinating.
RELEVANCE_THRESHOLD = 0.65

REWRITE_PROMPT = """You are helping a legal RAG system retrieve relevant Indian law sections.

Given the user's question, generate exactly 2 SHORT search queries (5-10 words each) 
that will best retrieve the relevant legal sections from a vector database of Indian laws.

Rules:
- KEEP QUERIES SHORT: 5-10 words max each
- Focus on the legal concept, not the user's story
- Use terminology that would appear in actual Indian law text
- Return ONLY a valid JSON array of exactly 2 strings — no explanation, no markdown

Example:
User: "my vehicle was towed by traffic police wrongfully"
Output: ["vehicle towing wrongful seizure traffic police", "motor vehicle impound release procedure"]

User: "{query}"
Output:"""


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

class SakhiState(TypedDict):
    query:              str               # original user query
    corrected_query:    str               # typo/slang fixed version
    understood_as:      str               # what Sakhi thinks user means
    needs_clarification: bool             # should we ask a clarifying question?
    clarification_question: Optional[str] # the question to ask if needed
    search_queries:     List[str]         # 2 short retrieval queries
    embeddings:         List[List[float]] # one embedding per search query
    chunks:             List[dict]        # retrieved + deduplicated chunks
    context:            str               # formatted context for LLM
    answer:             str               # final answer
    chat_history:       List[dict]        # [{role, content}] multi-turn memory


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCES
# ══════════════════════════════════════════════════════════════════════════════

class SakhiResources:
    def __init__(self):
        print("🔄 Loading BGE-M3 embedding model...")
        self.embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        print("✅ BGE-M3 ready")

        print("🗄️  Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"✅ ChromaDB ready ({self.collection.count():,} chunks)\n")

        print("🤖 Connecting to Groq LLM...")
        self.llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, max_tokens=1024)
        print(f"✅ Groq ({GROQ_MODEL}) ready\n")


# ══════════════════════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════════════════════

def make_understand_node(resources: SakhiResources):
    """
    Node 1: Understand the query.
    - Evaluates chat history and asks highly empathetic clarifying questions if needed.
    """
    def understand_query(state: SakhiState) -> SakhiState:
        print(f"  [1/5] 🧠 Understanding query and assessing context...")

        # Format chat history for context
        recent_context = ""
        if state.get("chat_history"):
            last_turns = state["chat_history"][-6:]
            recent_context = "\n".join(
                f"{m['role'].title()}: {m['content'][:200]}" for m in last_turns
            )
        else:
            recent_context = "No previous context. This is the first message."

        prompt   = UNDERSTAND_PROMPT.format(query=state["query"], chat_history=recent_context)
        response = resources.llm.invoke([HumanMessage(content=prompt)])

        try:
            raw = response.content.strip().replace("```json","").replace("```","").strip()
            data = json.loads(raw)
        except Exception:
            # Safe fallback
            data = {
                "corrected_query":        state["query"],
                "understood_as":          state["query"],
                "needs_clarification":    False,
                "clarification_question": None
            }

        state["corrected_query"]       = data.get("corrected_query", state["query"])
        state["understood_as"]         = data.get("understood_as", "")
        state["needs_clarification"]   = data.get("needs_clarification", False)
        state["clarification_question"]= data.get("clarification_question", None)

        print(f"         Understood as : {state['understood_as']}")
        if state["needs_clarification"]:
            print(f"         Needs clarification: YES → \"{state['clarification_question']}\"")
        else:
            print(f"         Proceeding to retrieval...")

        return state
    return understand_query


def route_after_understand(state: SakhiState) -> str:
    """
    Conditional router after Node 1.
    If clarification needed → go to 'clarify' (short-circuit, skip retrieval)
    Otherwise → go to 'rewrite' (normal RAG flow)
    """
    if state.get("needs_clarification"):
        return "clarify"
    return "rewrite"


def make_clarify_node(resources: SakhiResources):
    """
    Node 1b (conditional): Ask the user a clarifying question.
    Skips embedding, retrieval, and generation entirely.
    """
    def ask_clarification(state: SakhiState) -> SakhiState:
        q = state.get("clarification_question") or "I'm so sorry you're going through this. Could you give me a little more detail so I can give you the right advice?"
        state["answer"] = q
        return state
    return ask_clarification


def make_rewrite_node(resources: SakhiResources):
    """
    Node 2: Generate 2 short, focused search queries from the corrected query.
    Uses corrected_query (typo-fixed) not raw query.
    """
    def rewrite_query(state: SakhiState) -> SakhiState:
        print(f"  [2/5] ✍️  Generating search queries...")

        # Build context-aware input using recent history
        recent_context = ""
        if state["chat_history"]:
            last_turns = state["chat_history"][-4:]
            recent_context = "\n".join(
                f"{m['role'].title()}: {m['content'][:200]}" for m in last_turns
            )

        query_input = state["corrected_query"]
        if recent_context:
            query_input = (
                f"[Recent conversation:\n{recent_context}]\n"
                f"New question: {state['corrected_query']}"
            )

        prompt   = REWRITE_PROMPT.format(query=query_input)
        response = resources.llm.invoke([HumanMessage(content=prompt)])

        try:
            raw     = response.content.strip().replace("```json","").replace("```","").strip()
            queries = json.loads(raw)
            if not isinstance(queries, list) or len(queries) == 0:
                raise ValueError
            queries = [str(q).strip()[:80] for q in queries[:2]]
        except Exception:
            queries = [state["corrected_query"]]

        state["search_queries"] = queries
        for i, q in enumerate(queries, 1):
            print(f"         Query {i}: \"{q}\"")

        return state
    return rewrite_query


def make_embed_node(resources: SakhiResources):
    """Node 3: Embed all search queries in one batch."""
    def embed_queries(state: SakhiState) -> SakhiState:
        n = len(state["search_queries"])
        print(f"  [3/5] 🔢 Embedding {n} search quer{'y' if n == 1 else 'ies'}...")
        output = resources.embed_model.encode(
            state["search_queries"],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        state["embeddings"] = output['dense_vecs'].tolist()
        return state
    return embed_queries


def make_retrieve_node(resources: SakhiResources):
    """
    Node 4: Retrieve for each query, deduplicate by section, sort by score.
    """
    def retrieve_chunks(state: SakhiState) -> SakhiState:
        print(f"  [4/5] 🔍 Retrieving chunks (multi-query)...")

        seen_ids   = set()
        all_chunks = []

        for query, embedding in zip(state["search_queries"], state["embeddings"]):
            results = resources.collection.query(
                query_embeddings=[embedding],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"]
            )
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                dedup_key = f"{meta.get('act_name')}__s{meta.get('section_number')}"
                if dedup_key not in seen_ids:
                    seen_ids.add(dedup_key)
                    all_chunks.append({
                        "text":     doc,
                        "metadata": meta,
                        "score":    round(1 - dist, 4)
                    })

        all_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = all_chunks[:TOP_K]
        state["chunks"] = top_chunks

        # Flag low confidence if best score is below threshold
        best_score = top_chunks[0]["score"] if top_chunks else 0
        state["low_confidence"] = best_score < RELEVANCE_THRESHOLD
        if state["low_confidence"]:
            print(f"         ⚠️  Low confidence (best score={best_score} < {RELEVANCE_THRESHOLD})")
            print(f"            Sakhi will respond honestly about missing data.")

        print(f"         Top {len(top_chunks)} chunks (from {len(all_chunks)} candidates):")
        for i, c in enumerate(top_chunks, 1):
            m = c["metadata"]
            print(f"           #{i}  score={c['score']}  |  "
                  f"{m.get('act_name')} §{m.get('section_number')} — {m.get('section_title')}")

        context_parts = []
        for i, c in enumerate(top_chunks, 1):
            m = c["metadata"]
            context_parts.append(
                f"[Source {i}: {m.get('act_name')}, "
                f"Section {m.get('section_number')} — {m.get('section_title')}]\n"
                f"{c['text']}"
            )
        state["context"] = "\n\n---\n\n".join(context_parts)
        return state
    return retrieve_chunks


def make_generate_node(resources: SakhiResources):
    """
    Node 5: Generate answer with full chat history for multi-turn memory.
    Passes corrected_query to LLM so it responds to the right thing.
    """
    def generate_answer(state: SakhiState) -> SakhiState:
        print(f"  [5/5] 💬 Generating answer with Groq ({GROQ_MODEL})...")

        # If retrieval confidence is low, use a honest fallback prompt
        if state.get("low_confidence"):
            fallback_prompt = (
                "You are Sakhi, a warm and honest AI legal companion for Indians.\n\n"
                "A user asked: \"{query}\"\n\n"
                "You searched your legal database but could not find specific sections "
                "covering this exact situation. The retrieved sections are not relevant enough.\n\n"
                "Be honest that your database doesn't have specific legal information on this, "
                "but still be helpful: explain what area of law this likely falls under, "
                "suggest what kind of lawyer or authority they should approach, "
                "and give 2-3 practical immediate steps they can take.\n"
                "Do NOT cite any section numbers or acts you are not sure about.\n"
                "Be warm, brief, and practical."
            ).format(query=state["corrected_query"])
            messages = [
                SystemMessage(content=fallback_prompt),
                HumanMessage(content=state["corrected_query"])
            ]
            response        = resources.llm.invoke(messages)
            state["answer"] = response.content
            return state

        messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"]))]

        for msg in state["chat_history"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Use corrected query — so LLM answers about "towed" not "tolled"
        messages.append(HumanMessage(content=state["corrected_query"]))

        response        = resources.llm.invoke(messages)
        state["answer"] = response.content
        return state
    return generate_answer


# ══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(resources: SakhiResources):
    graph = StateGraph(SakhiState)

    graph.add_node("understand", make_understand_node(resources))
    graph.add_node("clarify",    make_clarify_node(resources))
    graph.add_node("rewrite",    make_rewrite_node(resources))
    graph.add_node("embed",      make_embed_node(resources))
    graph.add_node("retrieve",   make_retrieve_node(resources))
    graph.add_node("generate",   make_generate_node(resources))

    graph.set_entry_point("understand")

    # Conditional branch: clarify OR continue to retrieval
    graph.add_conditional_edges(
        "understand",
        route_after_understand,
        {
            "clarify": "clarify",
            "rewrite": "rewrite",
        }
    )

    graph.add_edge("clarify",  END)   # clarify short-circuits — no retrieval
    graph.add_edge("rewrite",  "embed")
    graph.add_edge("embed",    "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# RUN QUERY
# ══════════════════════════════════════════════════════════════════════════════

def ask(pipeline, query: str, chat_history: List[dict]) -> str:
    print(f"\n{'═'*62}")
    print(f"  ❓ {query}")
    print(f"{'═'*62}")

    result = pipeline.invoke({
        "query":               query,
        "corrected_query":     "",
        "understood_as":       "",
        "needs_clarification": False,
        "clarification_question": None,
        "search_queries":      [],
        "embeddings":          [],
        "chunks":              [],
        "context":             "",
        "answer":              "",
        "low_confidence":      False,
        "chat_history":        chat_history,
    })

    answer = result["answer"]

    print(f"\n{'─'*62}")
    print(f"  💡 SAKHI:\n")
    for line in answer.strip().split("\n"):
        print(f"  {line}")
    print(f"{'─'*62}\n")

    return answer


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sakhi RAG — Legal Q&A")
    parser.add_argument("--query",       type=str,  default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--top_k",       type=int,  default=TOP_K)
    args = parser.parse_args()

    resources = SakhiResources()
    pipeline  = build_graph(resources)

    if args.query:
        ask(pipeline, args.query, chat_history=[])

    elif args.interactive:
        print("\n🪷  Welcome to Sakhi — Your Legal Companion")
        print("   Type your legal question in any language.")
        print("   Type 'exit' to quit.\n")

        chat_history = []

        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit", "bye"):
                    print("Sakhi: Take care! 🙏")
                    break

                answer = ask(pipeline, query, chat_history)

                chat_history.append({"role": "user",      "content": query})
                chat_history.append({"role": "assistant", "content": answer})
                chat_history = chat_history[-12:]

            except KeyboardInterrupt:
                print("\nSakhi: Take care! 🙏")
                break

    else:
        print("❌ Please provide --query TEXT or --interactive\n")
        parser.print_help()


if __name__ == "__main__":
    main()