"""
Sakhi — RAG Implementation (Optimized v4)
==========================================
LangGraph + LangChain + LangSmith + Groq LLM

Flow:
  User Query → Understand & Clarify → Generate 3 Search Queries → Embed All
             → Retrieve + Deduplicate + Score-Filter → Groq LLM → Answer

v4 Optimizations:
  - Per-node temperature tuning (deterministic for JSON, warm for generation)
  - 3 search queries (legal term + plain language + act-specific)
  - Score-based chunk filtering with configurable RELEVANCE_THRESHOLD
  - MAX_CONTEXT_CHUNKS to avoid overwhelming the LLM
  - Retry + fallback model for Groq API resilience
  - Clarification counter to prevent infinite Q loops
  - Proper Python logging instead of print()
  - Improved prompts tuned for Sakhi's personality

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
import time
import logging
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
    MAX_CONTEXT_CHUNKS,
    RELEVANCE_THRESHOLD,
    GROQ_MODEL,
    FALLBACK_MODEL,
    UNDERSTAND_TEMPERATURE,
    REWRITE_TEMPERATURE,
    GENERATE_TEMPERATURE,
    GENERATE_MAX_TOKENS,
    MAX_CHAT_HISTORY_TURNS,
    MAX_CLARIFICATIONS,
    NUM_SEARCH_QUERIES,
    SYSTEM_PROMPT,
    DEBUG,
)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("sakhi")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

UNDERSTAND_PROMPT = """You are Sakhi, a deeply empathetic, warm, and trusted legal companion for everyday Indians.

Your job is to read the FULL conversation history AND the user's latest message, then decide:
  (a) Do you have enough context to retrieve relevant legal information? → SET needs_clarification to FALSE
  (b) Do you absolutely need ONE specific clarifying detail to give accurate advice? → ask

━━━ ABSOLUTE RULES ━━━

**MEMORY**: You MUST use ALL information from the conversation history. If the user already told you their age, location, situation, relationship details, or any other fact in a PREVIOUS message — DO NOT ask again. Treat the entire conversation as one continuous discussion.

**CLARIFICATION LIMIT**: You have already asked {clarification_count} clarifying question(s). Maximum allowed: {max_clarifications}.
- If {clarification_count} >= {max_clarifications}: You MUST set needs_clarification to FALSE. No exceptions. Work with what you have.
- If the conversation already has 4+ user messages: You almost certainly have enough context. Set needs_clarification to FALSE.

**WHEN TO PROCEED IMMEDIATELY (needs_clarification = FALSE)**:
- The user's question is clear enough to identify a legal domain (even broadly)
- The user has already provided key details across the conversation
- A simple greeting like "hello" or "hi" — just welcome them warmly
- Any situation involving a minor (person under 18) — IMMEDIATELY flag this as a serious legal matter and proceed to retrieval. Do NOT ask more questions.
- When in doubt — PROCEED. It is far better to give a slightly general legal answer than to keep asking questions.

**WHEN TO ASK (needs_clarification = TRUE)**:
- ONLY if a single specific detail would completely change which laws apply (e.g., state jurisdiction, employment sector)
- NEVER ask for details the user already provided in the conversation history
- NEVER ask vague questions like "can you tell me more?" — be specific about what you need

**TYPO/SLANG FIXING**: Fix typos and Indian English slang. Understand Hinglish naturally.
  "tolled" → "towed" | "FIR dala" → "filed an FIR" | "Meri salary nahi mili" → "My salary has not been paid"

**SUMMARIZATION**: In "understood_as", summarize the COMPLETE picture from ALL messages combined, not just the latest message.

━━━ CONTEXT ━━━

Conversation History:
{chat_history}

Latest User Message: "{query}"

━━━ RESPOND ━━━
ONLY a valid JSON object (no markdown, no explanation):
{{
  "corrected_query": "...",
  "understood_as": "...",
  "is_legal_query": true/false,
  "needs_clarification": true/false,
  "clarification_question": "..." or null
}}

Set is_legal_query = false for greetings, small talk, or emotional sharing without legal content."""


REWRITE_PROMPT = """You are a legal search expert helping an Indian legal RAG system retrieve the most relevant law sections.

Given a user's legal question, generate exactly {num_queries} SHORT search queries (5-12 words each) optimized for semantic vector search against a database of Indian laws.

The 3 queries should cover different angles:
1. **Formal Legal Terminology** — Use exact terms from Indian legal acts (BNS, BNSS, IPC, CrPC, etc.)
2. **Plain Language / Concept** — How a layperson would describe the situation
3. **Specific Act / Section** — Target a specific act that likely covers this issue

LEGAL DOMAINS TO CONSIDER:
- Criminal: BNS (Bharatiya Nyaya Sanhita), BNSS (Bharatiya Nagarik Suraksha Sanhita)
- Legacy Criminal: IPC (Indian Penal Code), CrPC (Code of Criminal Procedure)
- Consumer: Consumer Protection Act 2019
- Labour: Industrial Disputes Act, Payment of Wages Act, Factories Act
- Women's Rights: Protection of Women from Domestic Violence Act, POSH Act
- Property: Transfer of Property Act, Rent Control Acts (state-specific)
- Motor Vehicles: Motor Vehicles Act 2019
- Family: Hindu Marriage Act, Special Marriage Act, Hindu Succession Act
- Cyber: IT Act 2000, BNS cyber offence sections
- Constitutional: Fundamental Rights (Articles 14-32)

Rules:
- KEEP QUERIES SHORT: 5-12 words max each
- Focus on the LEGAL CONCEPT, not the user's personal story
- Use terminology that would appear in actual Indian law text
- Return ONLY a valid JSON array of exactly {num_queries} strings — no explanation, no markdown

Example:
User: "my vehicle was towed by traffic police wrongfully"
Output: ["wrongful seizure vehicle towing traffic police authority", "vehicle impound release towing rights India", "Motor Vehicles Act section towing impound procedure"]

User: "{query}"
Output:"""


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

class SakhiState(TypedDict):
    query:                  str               # original user query
    corrected_query:        str               # typo/slang fixed version
    understood_as:          str               # what Sakhi thinks user means
    needs_clarification:    bool              # should we ask a clarifying question?
    clarification_question: Optional[str]     # the question to ask if needed
    clarification_count:    int               # how many times we've asked so far
    is_legal_query:         bool              # ⭐ NEW
    search_queries:         List[str]         # N short retrieval queries
    embeddings:             List[List[float]] # one embedding per search query
    chunks:                 List[dict]        # retrieved + deduplicated chunks
    context:                str               # formatted context for LLM
    answer:                 str               # final answer
    low_confidence:         bool              # retrieval below threshold
    chat_history:           List[dict]        # [{role, content}] multi-turn memory


# ══════════════════════════════════════════════════════════════════════════════
# RETRY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def invoke_llm_with_retry(llm, messages, max_retries=3, fallback_model=FALLBACK_MODEL):
    """
    Retry LLM calls with exponential backoff.
    On persistent failure, try the fallback model.
    On total failure, return a safe fallback string.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning(f"LLM call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)

    # Try fallback model
    if fallback_model:
        logger.warning(f"Primary model failed. Trying fallback: {fallback_model}")
        try:
            fallback_llm = ChatGroq(model=fallback_model, temperature=0.3, max_tokens=1024)
            return fallback_llm.invoke(messages)
        except Exception as e:
            logger.error(f"Fallback model also failed: {e}")

    # Total failure — return a safe static response
    logger.error(f"All LLM attempts failed. Last error: {last_error}")

    class FallbackResponse:
        content = (
            "I'm really sorry — I'm having trouble connecting to my knowledge base right now. "
            "Please try again in a moment. If your situation is urgent, please call the "
            "Women Helpline (181), Police Emergency (112), or contact your nearest "
            "District Legal Services Authority for free legal help. 🙏"
        )
    return FallbackResponse()


def parse_json_safe(raw_text, fallback=None):
    """Robust JSON parsing with multiple cleanup strategies."""
    text = raw_text.strip()

    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code fences
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Find JSON object/array in text
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass

    logger.warning(f"JSON parse failed after all strategies. Raw: {text[:200]}")
    return fallback


# ══════════════════════════════════════════════════════════════════════════════
# RESOURCES
# ══════════════════════════════════════════════════════════════════════════════

class SakhiResources:
    def __init__(self):
        logger.info("🔄 Loading BGE-M3 embedding model...")
        self.embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        logger.info("✅ BGE-M3 ready")

        logger.info("🗄️  Connecting to ChromaDB...")
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = client.get_collection(COLLECTION_NAME)
        logger.info(f"✅ ChromaDB ready ({self.collection.count():,} chunks)")

        logger.info("🤖 Connecting to Groq LLM...")
        # Separate LLMs for different pipeline stages with tuned temperatures
        self.llm_understand = ChatGroq(
            model=GROQ_MODEL,
            temperature=UNDERSTAND_TEMPERATURE,
            max_tokens=512,
        )
        self.llm_rewrite = ChatGroq(
            model=GROQ_MODEL,
            temperature=REWRITE_TEMPERATURE,
            max_tokens=512,
        )
        self.llm_generate = ChatGroq(
            model=GROQ_MODEL,
            temperature=GENERATE_TEMPERATURE,
            max_tokens=GENERATE_MAX_TOKENS,
        )
        logger.info(f"✅ Groq ({GROQ_MODEL}) ready — 3 tuned instances")


# ══════════════════════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════════════════════

def make_understand_node(resources: SakhiResources):
    """
    Node 1: Understand the query.
    - Evaluates chat history and user intent
    - Asks empathetic clarifying questions if needed (max MAX_CLARIFICATIONS)
    - Fixes typos, slang, and Hinglish
    """
    def understand_query(state: SakhiState) -> SakhiState:
        logger.info("[1/5] 🧠 Understanding query and assessing context...")

        clarification_count = state.get("clarification_count", 0)

        # Format chat history for context
        if state.get("chat_history"):
            last_turns = state["chat_history"][-(MAX_CHAT_HISTORY_TURNS * 2):]
            recent_context = "\n".join(
                f"{m['role'].title()}: {m['content'][:500]}" for m in last_turns
            )
        else:
            recent_context = "No previous context. This is the first message."

        prompt = UNDERSTAND_PROMPT.format(
            query=state["query"],
            chat_history=recent_context,
            clarification_count=clarification_count,
            max_clarifications=MAX_CLARIFICATIONS,
        )
        response = invoke_llm_with_retry(
            resources.llm_understand,
            [HumanMessage(content=prompt)]
        )

        fallback_data = {
            "corrected_query":        state["query"],
            "understood_as":          state["query"],
            "is_legal_query":         True,
            "needs_clarification":    False,
            "clarification_question": None,
        }
        data = parse_json_safe(response.content, fallback=fallback_data)

        # SAFETY: ensure data is a dict
        if not isinstance(data, dict):
            logger.warning(f"   parse_json_safe returned {type(data).__name__}, using fallback.")
            data = fallback_data

        # Enforce clarification limit
        needs_clarification = data.get("needs_clarification", False)
        if clarification_count >= MAX_CLARIFICATIONS:
            needs_clarification = False
            logger.info(f"   Clarification limit reached ({MAX_CLARIFICATIONS}). Proceeding to retrieval.")

        state["is_legal_query"] = data.get("is_legal_query", True)  # ⭐ NEW
        state["corrected_query"]        = data.get("corrected_query", state["query"])
        state["understood_as"]          = data.get("understood_as", "")
        state["needs_clarification"]    = needs_clarification
        state["clarification_question"] = data.get("clarification_question") if needs_clarification else None

        if needs_clarification:
            state["clarification_count"] = clarification_count + 1

        logger.info(f"   Understood as: {state['understood_as']}")
        if state["needs_clarification"]:
            logger.info(f"   Needs clarification ({clarification_count + 1}/{MAX_CLARIFICATIONS}): \"{state['clarification_question']}\"")
        else:
            logger.info("   Proceeding to retrieval...")

        return state
    return understand_query


def route_after_understand(state: SakhiState) -> str:
    """
    Decide next step:
      - social → non-legal chat
      - clarify → ask question
      - rewrite → full RAG
    """

    if not state.get("is_legal_query", True):   # ⭐ NEW
        return "social"

    if state.get("needs_clarification"):
        return "clarify"

    return "rewrite"


def make_clarify_node(resources: SakhiResources):
    """
    Node 1b (conditional): Ask the user a clarifying question.
    Skips embedding, retrieval, and generation entirely.
    """
    def ask_clarification(state: SakhiState) -> SakhiState:
        q = state.get("clarification_question") or (
            "I'm really sorry you're dealing with this. "
            "Could you share just a bit more detail so I can give you the most accurate advice? 🙏"
        )
        state["answer"] = q
        return state
    return ask_clarification


def make_rewrite_node(resources: SakhiResources):
    """
    Node 2: Generate N search queries from the corrected query.
    Uses corrected_query (typo-fixed) not raw query.
    Generates queries at different angles: legal terminology, plain language, specific act.
    """
    def rewrite_query(state: SakhiState) -> SakhiState:
        logger.info(f"[2/5] ✍️  Generating {NUM_SEARCH_QUERIES} search queries...")

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

        prompt   = REWRITE_PROMPT.format(query=query_input, num_queries=NUM_SEARCH_QUERIES)
        response = invoke_llm_with_retry(
            resources.llm_rewrite,
            [HumanMessage(content=prompt)]
        )

        queries = parse_json_safe(response.content, fallback=None)

        if not isinstance(queries, list) or len(queries) == 0:
            # Fallback: use the corrected query as-is
            queries = [state["corrected_query"]]
            logger.warning("   Query rewrite failed — using corrected query as fallback")

        # Clean and limit
        queries = [str(q).strip()[:100] for q in queries[:NUM_SEARCH_QUERIES]]

        state["search_queries"] = queries
        for i, q in enumerate(queries, 1):
            logger.info(f"   Query {i}: \"{q}\"")

        return state
    return rewrite_query


def make_embed_node(resources: SakhiResources):
    """Node 3: Embed all search queries in one batch."""
    def embed_queries(state: SakhiState) -> SakhiState:
        n = len(state["search_queries"])
        logger.info(f"[3/5] 🔢 Embedding {n} search quer{'y' if n == 1 else 'ies'}...")
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
    Node 4: Retrieve for each query, deduplicate by section, score-filter, sort.
    Only passes high-confidence chunks to the LLM.
    """
    def retrieve_chunks(state: SakhiState) -> SakhiState:
        logger.info("[4/5] 🔍 Retrieving chunks (multi-query)...")

        seen_ids   = set()
        all_chunks = []

        for query, embedding in zip(state["search_queries"], state["embeddings"]):
            results = resources.collection.query(
                query_embeddings=[embedding],
                n_results=TOP_K,
                include=["documents", "metadatas", "distances"]
            )
            
            # Safe extraction
            docs = results.get("documents", [[]])[0] if results.get("documents") else []
            metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            dists = results.get("distances", [[]])[0] if results.get("distances") else []

            for doc, meta, dist in zip(docs, metas, dists):
                dedup_key = f"{meta.get('act_name')}__s{meta.get('section_number')}"
                if dedup_key not in seen_ids:
                    seen_ids.add(dedup_key)
                    score = round(1 - dist, 4)
                    all_chunks.append({
                        "text":     doc,
                        "metadata": meta,
                        "score":    score,
                    })

        # Sort by score descending
        all_chunks.sort(key=lambda x: x["score"], reverse=True)

        # Filter by relevance threshold — only keep chunks above the bar
        relevant_chunks = [c for c in all_chunks if c["score"] >= RELEVANCE_THRESHOLD]

        # Cap at MAX_CONTEXT_CHUNKS to avoid overwhelming the LLM
        top_chunks = relevant_chunks[:MAX_CONTEXT_CHUNKS] if relevant_chunks else all_chunks[:MAX_CONTEXT_CHUNKS]

        state["chunks"] = top_chunks

        # Flag low confidence if no chunks passed the threshold
        best_score = all_chunks[0]["score"] if all_chunks else 0
        state["low_confidence"] = len(relevant_chunks) == 0

        if state["low_confidence"]:
            logger.warning(f"   ⚠️  Low confidence (best score={best_score} < {RELEVANCE_THRESHOLD})")
            logger.warning("      Sakhi will respond honestly about limited data.")

        logger.info(f"   {len(top_chunks)} chunks selected (from {len(all_chunks)} candidates, {len(relevant_chunks)} above threshold)")
        for i, c in enumerate(top_chunks, 1):
            m = c["metadata"]
            logger.debug(
                f"     #{i}  score={c['score']}  |  "
                f"{m.get('act_name')} §{m.get('section_number')} — {m.get('section_title')}"
            )

        # Format context with confidence indicators
        context_parts = []
        for i, c in enumerate(top_chunks, 1):
            m = c["metadata"]
            confidence = "HIGH" if c["score"] >= 0.75 else "MODERATE" if c["score"] >= RELEVANCE_THRESHOLD else "LOW"
            context_parts.append(
                f"[Source {i} | {m.get('act_name')}, "
                f"Section {m.get('section_number')} — {m.get('section_title')} "
                f"| Relevance: {confidence}]\n"
                f"{c['text']}"
            )
        state["context"] = "\n\n---\n\n".join(context_parts)
        return state
    return retrieve_chunks


def make_generate_node(resources: SakhiResources):
    """
    Node 5: Generate answer with full chat history for multi-turn memory.
    Uses corrected_query and tuned generation parameters.
    """
    def generate_answer(state: SakhiState) -> SakhiState:
        logger.info(f"[5/5] 💬 Generating answer with Groq ({GROQ_MODEL})...")

        # ── Low-confidence fallback ──────────────────────────────────
        if state.get("low_confidence"):
            fallback_prompt = (
                "You are Sakhi, a warm and honest AI legal companion for Indians.\n"
                "You are a LEGAL EDUCATION platform. You MUST NEVER refuse to discuss any legal topic.\n"
                "If something is illegal, explain WHY it's illegal and what the consequences are.\n"
                "NEVER say 'I cannot provide information on illegal activities.'\n\n"
                "A user asked: \"{query}\"\n\n"
                "You searched your legal database but could not find specific sections "
                "covering this exact situation. The retrieved sections are not relevant enough.\n\n"
                "RESPOND WITH:\n"
                "1. Warmly acknowledge their situation (1-2 sentences)\n"
                "2. Honestly explain that your database doesn't have specific legal sections for this\n"
                "3. STILL explain what you know about the legal area — even from general knowledge\n"
                "4. Explain what AREA of law this likely falls under\n"
                "5. Suggest what kind of lawyer or authority they should approach\n"
                "6. Give 2-3 practical immediate steps they can take\n"
                "7. Mention relevant free resources:\n"
                "   - NALSA (National Legal Services Authority) for free legal aid\n"
                "   - District Legal Services Authority (available in every district)\n"
                "   - Women Helpline: 181 | Police Emergency: 112\n"
                "   - Cyber Crime Portal: cybercrime.gov.in\n\n"
                "Be warm, brief, and genuinely helpful. NEVER refuse to discuss the topic."
            ).format(query=state["corrected_query"])

            messages = [
                SystemMessage(content=fallback_prompt),
                HumanMessage(content=state["corrected_query"]),
            ]
            response = invoke_llm_with_retry(resources.llm_generate, messages)
            state["answer"] = response.content
            return state

        # ── Normal generation with full context ──────────────────────
        messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"]))]

        # Add chat history (trimmed to MAX_CHAT_HISTORY_TURNS)
        recent_history = state["chat_history"][-(MAX_CHAT_HISTORY_TURNS):]
        for msg in recent_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Use corrected query — so LLM answers about "towed" not "tolled"
        messages.append(HumanMessage(content=state["corrected_query"]))

        response = invoke_llm_with_retry(resources.llm_generate, messages)
        state["answer"] = response.content
        return state
    return generate_answer

def make_social_node(resources: SakhiResources):
    """
    Node 0: Handle non-legal conversation.
    Skips RAG entirely.
    """
    def social_response(state: SakhiState) -> SakhiState:

        prompt = (
            "You are Sakhi, a warm, supportive friend. "
            "The user has not asked a legal question. "
            "Respond naturally and kindly. "
            "Do NOT mention laws, acts, or legal advice."
        )

        response = invoke_llm_with_retry(
            resources.llm_generate,
            [
                SystemMessage(content=prompt),
                HumanMessage(content=state["query"])
            ]
        )

        state["answer"] = response.content
        return state

    return social_response
# ══════════════════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(resources: SakhiResources):
    graph = StateGraph(SakhiState)

    graph.add_node("social", make_social_node(resources))  # ⭐ NEW
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
            "social": "social",     # ⭐ NEW
            "clarify": "clarify",
            "rewrite": "rewrite",
        }
    )

    graph.add_edge("social", END)   # ⭐ NEW
    graph.add_edge("clarify",  END)   # clarify short-circuits — no retrieval
    graph.add_edge("rewrite",  "embed")
    graph.add_edge("embed",    "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# RUN QUERY
# ══════════════════════════════════════════════════════════════════════════════

def ask(pipeline, query: str, chat_history: List[dict], clarification_count: int = 0) -> tuple:
    """
    Run a single query through the RAG pipeline.
    Returns (answer, updated_clarification_count).
    """
    logger.info(f"\n{'═'*62}")
    logger.info(f"  ❓ {query}")
    logger.info(f"{'═'*62}")

    result = pipeline.invoke({
        "query":                  query,
        "corrected_query":        "",
        "understood_as":          "",
        "needs_clarification":    False,
        "is_legal_query": True,   # ⭐ NEW
        "clarification_question": None,
        "clarification_count":    clarification_count,
        "search_queries":         [],
        "embeddings":             [],
        "chunks":                 [],
        "context":                "",
        "answer":                 "",
        "low_confidence":         False,
        "chat_history":           chat_history,
    })

    answer = result["answer"]
    new_clarification_count = result.get("clarification_count", clarification_count)

    logger.info(f"\n{'─'*62}")
    logger.info(f"  💡 SAKHI:\n")
    for line in answer.strip().split("\n"):
        logger.info(f"  {line}")
    logger.info(f"{'─'*62}\n")

    return answer, new_clarification_count


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sakhi RAG — Legal Q&A")
    parser.add_argument("--query",       type=str,  default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--top_k",       type=int,  default=TOP_K)
    parser.add_argument("--debug",       action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    resources = SakhiResources()
    pipeline  = build_graph(resources)

    if args.query:
        answer, _ = ask(pipeline, args.query, chat_history=[])

    elif args.interactive:
        print("\n🪷  Welcome to Sakhi — Your Trusted Legal Companion")
        print("   Ask me any legal question in English, Hindi, or Hinglish.")
        print("   I'm here for you — no judgment, just help. 🙏")
        print("   Type 'exit' to quit.\n")

        chat_history = []
        clarification_count = 0

        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit", "bye"):
                    print("Sakhi: Take care of yourself! Remember — knowing your rights is your superpower. 🙏✨")
                    break

                answer, clarification_count = ask(pipeline, query, chat_history, clarification_count)

                chat_history.append({"role": "user",      "content": query})
                chat_history.append({"role": "assistant", "content": answer})

                # Keep history manageable
                chat_history = chat_history[-(MAX_CHAT_HISTORY_TURNS * 2):]

                # Reset clarification counter when user gets a real answer (not a clarifying Q)
                if not answer.endswith("?"):
                    clarification_count = 0

            except KeyboardInterrupt:
                print("\nSakhi: Take care! 🙏")
                break

    else:
        print("❌ Please provide --query TEXT or --interactive\n")
        parser.print_help()


if __name__ == "__main__":
    main()