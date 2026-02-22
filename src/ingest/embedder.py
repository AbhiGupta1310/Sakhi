from FlagEmbedding import BGEM3FlagModel
import time
from src.utils.logging_utils import fmt_time

class BGEEmbedder:
    def __init__(self):
        print("🔄 Loading BGE-M3...")
        t = time.time()
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        print(f"✅ Ready in {fmt_time(time.time()-t)}")

    def embed_batch(self, texts, batch_size=32):
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"].tolist()

    def embed_query(self, text, batch_size=1):
        return self.embed_batch([text], batch_size=batch_size)[0]   