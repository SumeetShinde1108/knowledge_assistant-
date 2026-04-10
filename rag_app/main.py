from ingestion import load_documents, split_documents, create_vector_store
from pipeline import generate_answer

# Step 1: Load + process docs (run once)
docs = load_documents("data/sample.pdf")
chunks = split_documents(docs)
create_vector_store(chunks)

# Step 2: Query
while True:
    query = input("Ask something: ")
    answer = generate_answer(query)
    print("\n", answer)