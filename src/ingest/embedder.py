"""
Sakhi Embedder — OpenRouter (Qwen3-Embedding-8B)
=================================================
Replaces the local BGE-M3 model with an API-based embedder.

Why Qwen3-Embedding-8B on OpenRouter:
  - #1 ranked multilingual embedding model (MTEB leaderboard)
  - 7680-dimensional dense vectors (far richer than BGE-M3's 1024)
  - Native Hindi / Hinglish / multilingual support
  - No local model download — pure API call (~1s per batch)
  - OpenAI-compatible endpoint — uses standard openai SDK

Usage:
  embedder = OpenRouterEmbedder()
  vecs = embedder.embed_batch(["Section 354A...", "Domestic violence..."])
  query_vec = embedder.embed_query("police rights india")
"""

import os
import time
import logging
from openai import OpenAI

logger = logging.getLogger("sakhi.embedder")

# ── Model config ──────────────────────────────────────────────────────────────
# Best multilingual embedding model available on OpenRouter (May 2026)
# MTEB leaderboard rank #1 for multilingual tasks
EMBED_MODEL = "qwen/qwen3-embedding-8b"
EMBED_DIM   = 7680   # Qwen3-Embedding-8B output dimension


class OpenRouterEmbedder:
    """
    API-based embedder using Qwen3-Embedding-8B via OpenRouter.

    Drop-in replacement for BGEEmbedder — same .embed_batch() / .embed_query() interface.
    No GPU, no local model files, no torch dependency needed at inference time.
    """

    def __init__(self):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is required. Add it to your .env file.\n"
                "Get one free at: https://openrouter.ai/keys"
            )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = EMBED_MODEL
        logger.info(f"✅ OpenRouterEmbedder ready — model: {self.model}")

    def embed_batch(self, texts: list[str], batch_size: int = 50) -> list[list[float]]:
        """
        Embed a list of texts for STORAGE (document mode).
        Batches to respect API limits. Returns list of float vectors.

        Args:
            texts:      List of strings to embed
            batch_size: Max texts per API call (OpenRouter allows up to 100)

        Returns:
            List of embedding vectors, one per input text
        """
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(
                f"  Embedding batch {i // batch_size + 1}/"
                f"{(len(texts) - 1) // batch_size + 1} "
                f"({len(batch)} texts)..."
            )
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                raise

            # Gentle rate-limit respect between batches
            if i + batch_size < len(texts):
                time.sleep(0.2)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string for RETRIEVAL (query mode).
        Identical to embed_batch([text])[0] but explicit for clarity.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding