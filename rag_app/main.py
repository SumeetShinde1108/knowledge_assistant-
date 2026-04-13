from pathlib import Path

from ingestion import create_vector_store, load_documents, split_documents
from pipeline import generate_answer

PDF_PATH = Path("data") / "sample.txt"
VECTORSTORE_DIR = Path("vectorstore")
VECTOR_INDEX = VECTORSTORE_DIR / "index.faiss"
VECTOR_TEXTS = VECTORSTORE_DIR / "texts.pkl"


def ensure_vectorstore():
    if VECTOR_INDEX.exists() and VECTOR_TEXTS.exists():
        print("Vectorstore already exists. Skipping ingestion.")
        return

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH.resolve()}\nPlease add the file or update the path in main.py."
        )

    print("Building vectorstore from PDF...")
    docs = load_documents(str(PDF_PATH))
    chunks = split_documents(docs)
    create_vector_store(chunks)
    print("Vectorstore created at", VECTORSTORE_DIR.resolve())


def main():
    try:
        ensure_vectorstore()
    except Exception as exc:
        print("Error preparing vectorstore:", exc)
        return

    while True:
        try:
            query = input("Ask something: ")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not query.strip():
            continue

        answer = generate_answer(query)
        print("\n", answer)


if __name__ == "__main__":
    main()
