import chromadb

def get_collection(db_path, name):
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def ingest(collection, chunks, embeddings):
    existing = set(
    collection.get(ids=[c["chunk_id"] for c in chunks], include=[])["ids"]
)

    new_chunks = []
    new_embeds = []

    for c, e in zip(chunks, embeddings):
        if c["chunk_id"] not in existing:
            new_chunks.append(c)
            new_embeds.append(e.tolist() if hasattr(e, "tolist") else e)

    if not new_chunks:
        print("✅ Nothing new to ingest")
        return

    collection.add(
        ids=[c["chunk_id"] for c in new_chunks],
        embeddings=new_embeds,
        documents=[c["text"] for c in new_chunks],
        metadatas=[c.get("metadata", {}) for c in new_chunks],
    )

    print(f"💾 Inserted {len(new_chunks)} chunks")