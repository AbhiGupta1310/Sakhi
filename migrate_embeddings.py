#!/usr/bin/env python3
"""
migrate_embeddings.py — Re-embed Sakhi's ChromaDB with Qwen3-Embedding-8B
==========================================================================
Run this ONCE after switching from BGE-M3 to OpenRouter embeddings.

What it does:
  1. Reads all 7,617 chunks from your existing ChromaDB (text + metadata)
  2. Re-embeds them using Qwen3-Embedding-8B via OpenRouter API
  3. Drops the old collection (wrong vector space)
  4. Creates a fresh collection with new embeddings

Why needed:
  BGE-M3 vectors (1024-dim) and Qwen3 vectors (7680-dim) live in completely
  different mathematical spaces — you can't mix them. This migrates everything.

Usage:
  python migrate_embeddings.py

Cost estimate:
  ~7,617 chunks × avg 200 tokens = ~1.5M tokens
  At OpenRouter's rate: tiny cost (usually <$0.10 with credits)

Time estimate: 5-10 minutes depending on API rate limits.
"""

import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("migration")

# ── Config ────────────────────────────────────────────────────────────────────
from src.config import CHROMA_DB_PATH, COLLECTION_NAME

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
EMBED_MODEL        = "qwen/qwen3-embedding-8b"
BATCH_SIZE         = 20   # conservative — Qwen3-8B chunks are larger
FETCH_BATCH        = 500  # ChromaDB fetch size


def check_prerequisites():
    if not OPENROUTER_API_KEY:
        logger.error("❌ OPENROUTER_API_KEY not found in .env")
        logger.error("   Add: OPENROUTER_API_KEY=sk-or-... to your .env file")
        sys.exit(1)
    logger.info(f"✅ OPENROUTER_API_KEY found")
    logger.info(f"📦 ChromaDB path: {CHROMA_DB_PATH}")
    logger.info(f"🔢 Embedding model: {EMBED_MODEL}")


def load_all_chunks_from_chroma():
    """Load all documents + metadata from existing ChromaDB collection."""
    import chromadb

    logger.info("\n📂 Connecting to existing ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        logger.error(f"❌ Collection '{COLLECTION_NAME}' not found: {e}")
        logger.error("   Make sure your ChromaDB exists at: " + str(CHROMA_DB_PATH))
        sys.exit(1)

    total = collection.count()
    logger.info(f"📊 Found {total:,} chunks in collection '{COLLECTION_NAME}'")

    all_ids, all_docs, all_metas = [], [], []
    offset = 0

    while offset < total:
        result = collection.get(
            limit=FETCH_BATCH,
            offset=offset,
            include=["documents", "metadatas"],
        )
        batch_ids   = result["ids"]
        batch_docs  = result["documents"]
        batch_metas = result["metadatas"]

        all_ids.extend(batch_ids)
        all_docs.extend(batch_docs)
        all_metas.extend(batch_metas)

        offset += len(batch_ids)
        logger.info(f"   Fetched {min(offset, total):,}/{total:,} chunks...")

        if not batch_ids:
            break  # safety: stop if nothing returned

    logger.info(f"✅ Loaded {len(all_ids):,} chunks from ChromaDB")
    return all_ids, all_docs, all_metas


def embed_all_documents(texts: list[str]) -> list[list[float]]:
    """Re-embed all texts using Qwen3-Embedding-8B via OpenRouter."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    total = len(texts)
    all_embeddings = []
    num_batches = (total - 1) // BATCH_SIZE + 1

    logger.info(f"\n🔢 Embedding {total:,} chunks in {num_batches} batches...")
    logger.info(f"   Model: {EMBED_MODEL} | Batch size: {BATCH_SIZE}")

    for i in range(0, total, BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                )
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)

                progress = min(i + BATCH_SIZE, total)
                logger.info(
                    f"   ✅ Batch {batch_num}/{num_batches} "
                    f"({progress:,}/{total:,} chunks) "
                    f"— dim={len(embeddings[0])}"
                )
                break

            except Exception as e:
                wait = 2 ** (attempt + 1)
                logger.warning(f"   ⚠️  Batch {batch_num} failed (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    logger.info(f"   Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"   ❌ Batch {batch_num} failed after 3 attempts")
                    raise

        # Rate-limit respect
        if i + BATCH_SIZE < total:
            time.sleep(0.3)

    logger.info(f"✅ Generated {len(all_embeddings):,} embeddings")
    return all_embeddings


def rebuild_collection(all_ids, all_docs, all_metas, new_embeddings):
    """Drop old collection and create a fresh one with new embeddings."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    logger.info(f"\n🗑️  Dropping old collection '{COLLECTION_NAME}'...")
    client.delete_collection(COLLECTION_NAME)
    logger.info("   Old collection deleted.")

    logger.info(f"✨ Creating fresh collection '{COLLECTION_NAME}'...")
    new_collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("   Fresh collection created.")

    INSERT_BATCH = 200
    total = len(all_ids)

    logger.info(f"\n💾 Inserting {total:,} chunks with new embeddings...")
    for i in range(0, total, INSERT_BATCH):
        batch_ids   = all_ids[i : i + INSERT_BATCH]
        batch_docs  = all_docs[i : i + INSERT_BATCH]
        batch_metas = all_metas[i : i + INSERT_BATCH]
        batch_embs  = new_embeddings[i : i + INSERT_BATCH]

        new_collection.add(
            ids=batch_ids,
            embeddings=batch_embs,
            documents=batch_docs,
            metadatas=batch_metas,
        )
        logger.info(f"   Inserted {min(i + INSERT_BATCH, total):,}/{total:,}")

    final_count = new_collection.count()
    logger.info(f"✅ Collection ready with {final_count:,} chunks")


def main():
    logger.info("=" * 60)
    logger.info("  Sakhi Embedding Migration: BGE-M3 → Qwen3-Embedding-8B")
    logger.info("=" * 60)

    check_prerequisites()

    # Step 1: Load existing data
    all_ids, all_docs, all_metas = load_all_chunks_from_chroma()

    if not all_ids:
        logger.error("❌ No chunks found! Is ChromaDB populated?")
        sys.exit(1)

    # Step 2: Re-embed with new model
    new_embeddings = embed_all_documents(all_docs)

    # Sanity check
    if len(new_embeddings) != len(all_ids):
        logger.error(
            f"❌ Mismatch: {len(all_ids)} ids vs {len(new_embeddings)} embeddings"
        )
        sys.exit(1)

    # Step 3: Rebuild collection
    rebuild_collection(all_ids, all_docs, all_metas, new_embeddings)

    logger.info("\n" + "=" * 60)
    logger.info("  🎉 Migration Complete!")
    logger.info(f"  Model:   {EMBED_MODEL}")
    logger.info(f"  Chunks:  {len(all_ids):,}")
    logger.info(f"  Dim:     {len(new_embeddings[0])}")
    logger.info("  Now run: ./run_app.sh")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
