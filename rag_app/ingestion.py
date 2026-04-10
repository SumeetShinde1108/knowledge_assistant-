from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts):
        return model.encode(texts)

    vectorstore = FAISS.from_texts(
        [chunk.page_content for chunk in chunks],
        embedding=embed
    )

    vectorstore.save_local("vectorstore")
    return vectorstore