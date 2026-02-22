import json
import pickle
import numpy as np
from pathlib import Path
import os
import hashlib

import hashlib

def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    chunks = []
    seen = set()  # safety net (optional)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)

                # 🔥 MAKE ID GLOBALLY UNIQUE USING TEXT HASH
                text_hash = hashlib.sha1(c["text"].encode()).hexdigest()[:12]
                c["chunk_id"] = f"{c['chunk_id']}__{text_hash}"

                # (Optional) Extra guard if original IDs collide badly
                if c["chunk_id"] in seen:
                    continue
                seen.add(c["chunk_id"])

                chunks.append(c)

    return chunks


def hash_chunks(chunks):
    m = hashlib.sha256()
    for c in chunks:
        m.update(c["text"].encode("utf-8"))
    return m.hexdigest()

def atomic_write(file_path, data_bytes):
    """Write file safely (no corruption if crash happens)."""
    tmp_path = str(file_path) + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(data_bytes)
    os.replace(tmp_path, file_path)


def load_or_build_embeddings(chunks, embedder, cache_path, batch_size):
    cache_path = Path(cache_path)
    npz_path = cache_path.with_suffix(".npz")

    # ---------- LOAD CACHE ----------
    if cache_path.exists() and npz_path.exists():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        current_hash = hash_chunks(chunks)

        if data.get("hash") == current_hash:
            npz = np.load(npz_path)
            embeddings = npz["embeddings"]

            print("📦 Loaded embeddings from cache")
            return data["chunks"], embeddings
        else:
            print("⚠️ Cache invalid — input changed, rebuilding...")

    # ---------- BUILD ----------
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_batch(texts, batch_size)
    embeddings_np = np.array(embeddings, dtype=np.float32)
    # Normalize for cosine similarity (safe for Chroma)
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    embeddings_np = embeddings_np / np.clip(norms, 1e-12, None)

    # ---------- SAVE PICKLE (fast reload) ----------
    pickle_bytes = pickle.dumps({
        "chunks": chunks,
        "model": str(type(embedder)),
        "hash": hash_chunks(chunks),
    })
    atomic_write(cache_path, pickle_bytes)

    # ---------- SAVE NPZ (portable backup) ----------
    np.savez_compressed(npz_path, embeddings=embeddings_np)

    print("💾 Saved embeddings to cache (pickle + npz)")
    return chunks, embeddings_np