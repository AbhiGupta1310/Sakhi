import argparse

from src.config import (
    DEFAULT_INPUT_JSONL,
    COLLECTION_NAME,
    CHROMA_DB_PATH,
    EMBEDDINGS_PKL,
)

from src.ingest.embedder import BGEEmbedder
from src.ingest.chroma_store import get_collection
from src.ingest.pipeline import run_ingestion
from src.ingest.query import run_query


def main():
    parser = argparse.ArgumentParser("Sakhi Ingestion")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_JSONL))
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--db_path", default=str(CHROMA_DB_PATH))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--query")
    parser.add_argument("--query_only", action="store_true")

    args = parser.parse_args()

    embedder = BGEEmbedder()
    collection = get_collection(args.db_path, args.collection)

    if not args.query_only:
        run_ingestion(
            args.input,
            embedder,
            collection,
            EMBEDDINGS_PKL,
            args.batch_size,
        )

    if args.query:
        run_query(collection, embedder, args.query)
    elif args.query_only:
        print("❌ query_only requires --query")


if __name__ == "__main__":
    main()