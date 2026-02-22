from src.ingest.loader import load_jsonl, load_or_build_embeddings
from src.ingest.chroma_store import ingest

def run_ingestion(input_path, embedder, collection, cache_path, batch_size):
    chunks = load_jsonl(input_path)

    chunks, embeddings = load_or_build_embeddings(
        chunks,
        embedder,
        cache_path,
        batch_size,
    )

    ingest(collection, chunks, embeddings)