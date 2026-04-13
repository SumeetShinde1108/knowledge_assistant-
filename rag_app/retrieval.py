import pickle
from pathlib import Path
import random

import faiss


class Document:
    def __init__(self, page_content: str):
        self.page_content = page_content


class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dim = 384

    def encode(self, texts, convert_to_numpy=True):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            random.seed(hash(text))
            embedding = [random.random() for _ in range(self.dim)]
            embeddings.append(embedding)
        return np.array(embeddings) if convert_to_numpy else embeddings


# Try to import real model, fallback to mock
try:
    from sentence_transformers import SentenceTransformer
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print("Using mock embeddings due to import issue:", e)
    MODEL = MockSentenceTransformer("all-MiniLM-L6-v2")


def load_vectorstore():
    vectorstore_path = Path("vectorstore")
    index_path = vectorstore_path / "index.faiss"
    texts_path = vectorstore_path / "texts.pkl"

    if not index_path.exists() or not texts_path.exists():
        raise FileNotFoundError(
            "Vectorstore not found. Run the ingestion pipeline first to create vectorstore/index.faiss and vectorstore/texts.pkl."
        )

    index = faiss.read_index(str(index_path))
    with open(texts_path, "rb") as handle:
        texts = pickle.load(handle)

    return index, texts


def retrieve(query, k=3):
    index, texts = load_vectorstore()
    query_embedding = MODEL.encode([query], convert_to_numpy=True)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    distances, indexes = index.search(query_embedding.astype("float32"), k)
    results = []
    for idx in indexes[0]:
        if idx >= 0 and idx < len(texts):
            results.append(Document(page_content=texts[idx]))

    return results