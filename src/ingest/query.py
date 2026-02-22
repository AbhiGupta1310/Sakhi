def run_query(collection, embedder, query, top_k=5):
    q_emb = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        score = 1 - dist
        print(f"\n#{i} score={score:.4f}")
        print(meta.get("act_name"))
        print(doc[:300], "...")