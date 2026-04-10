from retrieval import retrieve


def build_prompt(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    return prompt


def generate_answer(query):
    docs = retrieve(query)
    prompt = build_prompt(query, docs)

    # TEMP: no LLM yet
    return prompt