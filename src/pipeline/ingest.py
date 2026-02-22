"""
Sakhi — BGE-M3 Embedding + ChromaDB Ingestion
===============================================
Setup:
    pip install chromadb FlagEmbedding torch

Usage:
    python ingest.py
    python ingest.py --input legal_chunks.jsonl --collection sakhi_legal
    python ingest.py --query_only --query "can my landlord evict me without notice"
"""

import json
import time
import argparse
from pathlib import Path
import pickle
from datetime import timedelta

try:
    import chromadb
except ImportError:
    print("❌ chromadb not installed. Run: pip install chromadb")
    exit(1)

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    print("❌ FlagEmbedding not installed. Run: pip install FlagEmbedding")
    exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

from src.config import (
    DEFAULT_INPUT_JSONL,
    COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDINGS_PKL
)

DEFAULT_INPUT      = str(DEFAULT_INPUT_JSONL)
DEFAULT_COLLECTION = COLLECTION_NAME
DEFAULT_DB_PATH    = str(CHROMA_DB_PATH)
BATCH_SIZE         = 32
EMBEDDINGS_PKL_PATH= str(EMBEDDINGS_PKL)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def progress_bar(current, total, width=28):
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = current / total * 100
    return f"[{bar}] {pct:5.1f}%"


def log_progress(batch_num, total_batches, chunks_done, total_chunks,
                 batch_time, total_elapsed, batch_size):
    avg_per_chunk  = total_elapsed / chunks_done if chunks_done else 0
    remaining_chunks = total_chunks - chunks_done
    eta            = remaining_chunks * avg_per_chunk

    bar = progress_bar(chunks_done, total_chunks)

    print(
        f"  Batch {batch_num:>4}/{total_batches}"
        f"  {bar}"
        f"  {chunks_done:>5}/{total_chunks} chunks"
        f"  batch: {batch_time:.2f}s"
        f"  avg: {avg_per_chunk*1000:.0f}ms/chunk"
        f"  ETA: {fmt_time(eta)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════

def load_model():
    print("🔄 Loading BGE-M3 (first run downloads ~2GB, subsequent runs are instant)...")
    t = time.time()
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    print(f"✅ BGE-M3 ready in {fmt_time(time.time() - t)}\n")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# LOAD CHUNKS
# ══════════════════════════════════════════════════════════════════════════════

def load_chunks(filepath):
    path = Path(filepath)
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        exit(1)
    chunks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"📂 Loaded {len(chunks):,} chunks from {filepath}")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# EMBED IN BATCHES
# ══════════════════════════════════════════════════════════════════════════════

def embed_chunks(model, texts, batch_size=BATCH_SIZE):
    total_chunks  = len(texts)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    all_embeddings = []

    print(f"🧮 Embedding {total_chunks:,} chunks in {total_batches} batches (batch_size={batch_size})")
    print(f"{'─' * 85}")

    global_start = time.time()
    chunks_done  = 0

    for i in range(0, total_chunks, batch_size):
        batch     = texts[i : i + batch_size]
        batch_num = i // batch_size + 1

        batch_start = time.time()


        output = model.encode(
            batch,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )

        batch_time = time.time() - batch_start
        all_embeddings.extend(output['dense_vecs'].tolist())

        chunks_done  += len(batch)
        total_elapsed = time.time() - global_start

        log_progress(
            batch_num, total_batches,
            chunks_done, total_chunks,
            batch_time, total_elapsed,
            batch_size
        )

    total_time = time.time() - global_start
    print(f"{'─' * 85}")
    print(f"✅ Embedding done!  {total_chunks:,} chunks in {fmt_time(total_time)}"
          f"  ({total_time/total_chunks*1000:.1f}ms avg/chunk)\n")

    return all_embeddings


# ══════════════════════════════════════════════════════════════════════════════
# CHROMADB
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create_collection(db_path, collection_name):
    client     = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection


def ingest(chunks, embeddings, collection):
    existing_ids = set(collection.get(include=[])['ids'])
    print(f"   Already in DB : {len(existing_ids):,}")

    new_chunks     = []
    new_embeddings = []

    # Only insert chunks that are not in DB
    for chunk, emb in zip(chunks, embeddings):
        if chunk['chunk_id'] not in existing_ids:
            new_chunks.append(chunk)
            new_embeddings.append(emb)

    if not new_chunks:
        print("✅ All chunks already in DB — nothing to do!")
        return

    print(f"   New to insert  : {len(new_chunks):,}\n")

    batch_size = 500
    inserted   = 0
    start      = time.time()

    for i in range(0, len(new_chunks), batch_size):
        bc = new_chunks[i : i + batch_size]
        be = new_embeddings[i : i + batch_size]

        # Remove duplicates within batch
        ids_seen = set()
        filtered_bc, filtered_be = [], []
        for c, e in zip(bc, be):
            if c['chunk_id'] not in ids_seen:
                ids_seen.add(c['chunk_id'])
                filtered_bc.append(c)
                filtered_be.append(e)

        collection.add(
            ids        = [c['chunk_id'] for c in filtered_bc],
            embeddings = filtered_be,
            documents  = [c['text'] for c in filtered_bc],
            metadatas  = [c['metadata'] for c in filtered_bc]
        )
        inserted += len(filtered_bc)
        print(f"  💾 Stored {inserted:,}/{len(new_chunks):,} chunks...", end="\r")

    print(f"\n✅ Ingestion done in {fmt_time(time.time()-start)}  ({inserted:,} chunks added)")


# ══════════════════════════════════════════════════════════════════════════════
# QUERY
# ══════════════════════════════════════════════════════════════════════════════

def test_query(model, collection, query, top_k=5):
    print(f"\n🔍 Query: \"{query}\"")
    print("─" * 60)

    output = model.encode(
        [query],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    query_embedding = output['dense_vecs'][0].tolist()

    results   = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )

    docs      = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        score = round(1 - dist, 4)
        print(f"\n  #{rank}  score={score}  |  {meta.get('act_name')}  §{meta.get('section_number')} — {meta.get('section_title')}")
        print(f"       {meta.get('source_file')}")
        print(f"\n  {doc[:350]}...")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sakhi — BGE-M3 + ChromaDB Ingestion")
    parser.add_argument("--input",      type=str, default=DEFAULT_INPUT)
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION)
    parser.add_argument("--db_path",    type=str, default=DEFAULT_DB_PATH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--query",      type=str, default=None)
    parser.add_argument("--query_only", action="store_true")
    args = parser.parse_args()

    model = load_model()
    client, collection = get_or_create_collection(args.db_path, args.collection)
    print(f"🗄️  ChromaDB: '{args.collection}'  ({collection.count():,} existing chunks)\n")

    # ✅ Load from pickle if exists
    pkl_path = Path(EMBEDDINGS_PKL_PATH)
    if not args.query_only:
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            chunks = data["chunks"]
            embeddings = data["embeddings"]
            print(f"📦 Loaded {len(chunks):,} chunks and embeddings from {EMBEDDINGS_PKL}")
        else:
            chunks = load_chunks(args.input)
            embeddings = embed_chunks(model, [c['text'] for c in chunks], args.batch_size)
            with open(pkl_path, "wb") as f:
                pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
            print(f"💾 Saved {len(chunks):,} chunks and embeddings to {EMBEDDINGS_PKL_PATH}")

        print("💾 Ingesting into ChromaDB...")
        ingest(chunks, embeddings, collection)
        print(f"\n{'═'*55}")
        print(f"  Total in DB : {collection.count():,} chunks")
        print(f"  Collection  : '{args.collection}'")
        print(f"  DB path     : {Path(args.db_path).resolve()}")
        print(f"{'═'*55}\n")

    if args.query:
        test_query(model, collection, args.query)
    elif args.query_only:
        print("❌ --query_only requires --query TEXT")


if __name__ == "__main__":
    main()