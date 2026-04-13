import pickle
from pathlib import Path
import random

import faiss
import numpy as np
from pypdf import PdfReader


class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dim = 384  # Same as all-MiniLM-L6-v2

    def encode(self, texts, convert_to_numpy=True):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            # Generate a deterministic "embedding" based on hash
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


class Document:
    def __init__(self, page_content: str):
        self.page_content = page_content


def load_documents(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    documents = [Document(page_content=text)]
    return documents


def _split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def split_documents(documents):
    chunks = []
    for document in documents:
        for chunk in _split_text(document.page_content, chunk_size=500, chunk_overlap=50):
            chunks.append(Document(page_content=chunk))

    return chunks


def create_vector_store(chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = MODEL.encode(texts, convert_to_numpy=True)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    vectorstore_path = Path("vectorstore")
    vectorstore_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(vectorstore_path / "index.faiss"))

    with open(vectorstore_path / "texts.pkl", "wb") as handle:
        pickle.dump(texts, handle)

    return index