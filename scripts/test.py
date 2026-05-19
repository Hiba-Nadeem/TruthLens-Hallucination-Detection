from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 🔹 Extract main entity (simple heuristic)
def extract_entity(query):
    if " is " in query:
        return query.split(" is ")[0].strip()
    return query


# 🔹 Load vectorstore
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    
    vectorstore = FAISS.load_local(
        "../vectorstore_wiki",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# 🔹 Test retrieval
def test_retrieval(vectorstore, query, k=6):
    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    # Step 1 — extract entity
    entity = extract_entity(query)

    # Step 2 — improved BGE query
    query_text = f"query: information about {entity}. {query}"

    # Step 3 — retrieve more results
    results = vectorstore.similarity_search_with_score(query_text, k=k)

    # Step 4 — entity-aware re-ranking
    entity_key = entity.replace(" ", "_")

    results = sorted(
        results,
        key=lambda x: entity_key in x[0].metadata.get("page_title", ""),
        reverse=True
    )

    # Step 5 — optional strict filtering (only if entity exists)
    filtered = [
        r for r in results
        if entity_key in r[0].metadata.get("page_title", "")
    ]

    if filtered:
        final_results = filtered[:4]
    else:
        final_results = results[:4]

    # Step 6 — print results
    for i, (doc, score) in enumerate(final_results, 1):
        print(f"\nResult {i}")
        print("-" * 50)
        print(f"Score (lower = better): {score:.4f}")
        print(f"Page: {doc.metadata.get('page_title')}")
        print(f"Text: {doc.metadata.get('original_text')}")
        print(f"\nRaw Content: {doc.page_content[:200]}...")

    print("\n" + "=" * 70)


# 🔹 Main
if __name__ == "__main__":
    vectorstore = load_vectorstore()

    # 🔥 Test queries
    test_queries = [
        "Oliver Reed was a film actor.",
        "Roman Atwood is a YouTuber.",
        "Adrienne Bailon is a football player.",
        "Who was Paul walker?"
    ]

    for query in test_queries:
        test_retrieval(vectorstore, query)