from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def load_vectorstore():
    def embed(texts):
        return model.encode(texts)

    return FAISS.load_local("vectorstore", embed)


def retrieve(query, k=3):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return docs