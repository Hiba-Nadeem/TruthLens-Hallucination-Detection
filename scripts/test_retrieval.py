"""
Compare old vs new retrieval for the same 10 random claims.
Shows which chunks each approach would send to NLI.
Does NOT run NLI - just tests what gets retrieved and filtered.
"""

import json
import random
import pandas as pd
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
VECTORSTORE_PATH = str(BASE_DIR / "vectorstore_wiki")
CHUNK_META_PATH  = str(BASE_DIR / "data" / "chunk_metadata.json")
ANSWERS_PATH     = str(BASE_DIR / "data" / "llm_answers.csv")

TOP_K = 5

VERB_PATTERNS = [
    " is ", " was ", " are ", " were ", " has ", " have ",
    " had ", " played ", " appeared ", " worked ", " reached ",
    " won ", " composed ", " set ", " play ", " disbanded ",
    " based ", " born ", " died ", " located ", " founded ",
    " released ", " directed ", " starred ", " hosted ",
    " acted ", " hit ", " made ", " wrote ", " sang ", " sold ",
    " married ", " divorced ", " produced ", " created ", " joined ",
    " left ", " signed ", " recorded ", " performed ", " published ",
    " received ", " attended ", " graduated ", " studied ", " moved ",
    " became ", " served ", " represented ", " competed ", " refused ",
]

def extract_entity(claim):
    claim_lower = claim.lower()
    best_idx = len(claim)
    for pattern in VERB_PATTERNS:
        if pattern in claim_lower:
            idx = claim_lower.index(pattern)
            if idx < best_idx:
                best_idx = idx
    if best_idx < len(claim):
        return claim[:best_idx].strip()
    return " ".join(claim.split()[:2]).strip()

def get_entity_chunks(chunk_metadata, entity, max_chunks=3):
    entity_key   = entity.replace(" ", "_").lower()
    entity_plain = entity.lower()

    matches = []

    for c in chunk_metadata:
        page_title = c.get("page_title", "").lower()
        decoded    = c.get("decoded_title", "").lower()

        if (
            entity_key in page_title or
            entity_plain in page_title.replace("_", " ") or
            entity_plain in decoded
        ):
            matches.append(c)

    return matches[:max_chunks]

def retrieve_old(claim, vectorstore, chunk_metadata):
    entity        = extract_entity(claim)
    query_text    = f"query: {claim}"
    faiss_results = vectorstore.similarity_search_with_score(query_text, k=TOP_K)
    entity_chunks = get_entity_chunks(chunk_metadata, entity, max_chunks=3)

    seen, chunks = set(), []
    for c in entity_chunks:
        if c["chunk_text"] not in seen:
            seen.add(c["chunk_text"])
            chunks.append({"page": c["page_title"], "src": "direct", "score": 0.0,
                           "text": c["chunk_text"][:100]})

    for doc, score in faiss_results:
        if float(score) > 0.68:
            continue
        text = doc.metadata.get("original_text", doc.page_content)
        if text not in seen:
            seen.add(text)
            chunks.append({"page": doc.metadata.get("page_title"), "src": "faiss",
                           "score": round(float(score), 4), "text": text[:100]})
    return entity, chunks

def retrieve_new(claim, vectorstore, chunk_metadata):
    entity        = extract_entity(claim)
    entity_lower  = entity.lower()
    query_text    = f"query: {entity}. {claim}"
    faiss_results = vectorstore.similarity_search_with_score(query_text, k=TOP_K)
    entity_chunks = get_entity_chunks(chunk_metadata, entity, max_chunks=3)

    seen, chunks = set(), []
    for c in entity_chunks:
        if c["chunk_text"] not in seen:
            seen.add(c["chunk_text"])
            chunks.append({"page": c["page_title"], "src": "direct", "score": 0.0,
                           "text": c["chunk_text"][:100]})

    for doc, score in faiss_results:
        if float(score) > 0.55:
            continue
        text       = doc.metadata.get("original_text", doc.page_content)
        page_title   = doc.metadata.get("page_title", "Unknown")
        decoded      = doc.metadata.get("decoded_title", "").lower()

        page_match = (
            entity_lower in page_title.replace("_", " ").lower()
            or entity_lower in decoded
        )

        text_match = entity_lower in text.lower()
        if not page_match and not text_match:
            continue
        if text not in seen:
            seen.add(text)
            chunks.append({"page": page_title, "src": "faiss",
                           "score": round(float(score), 4), "text": text[:100]})

    if not chunks:
        for doc, score in faiss_results[:3]:
            text = doc.metadata.get("original_text", doc.page_content)
            if text not in seen:
                seen.add(text)
                chunks.append({"page": doc.metadata.get("page_title", "Unknown"),
                               "src": "faiss(fallback)", "score": round(float(score), 4),
                               "text": text[:100]})
    return entity, chunks

def main():
    print("Loading embeddings and vectorstore...")
    embeddings   = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore  = FAISS.load_local(VECTORSTORE_PATH, embeddings,
                                    allow_dangerous_deserialization=True)

    with open(CHUNK_META_PATH, encoding="utf-8-sig") as f:
        chunk_metadata = json.load(f)

    df     = pd.read_csv(ANSWERS_PATH)
    sample = df.sample(10, random_state=42)

    for _, row in sample.iterrows():
        claim = str(row["claim"]).strip()
        label = row["label"]

        entity_old, old_chunks = retrieve_old(claim, vectorstore, chunk_metadata)
        entity_new, new_chunks = retrieve_new(claim, vectorstore, chunk_metadata)

        print("=" * 90)
        print(f"CLAIM  : {claim}")
        print(f"LABEL  : {label}   |   ENTITY: {entity_old}")
        print()

        print(f"  OLD ({len(old_chunks)} chunks):")
        for c in old_chunks:
            flag = "OK" if entity_old.lower() in c["page"].replace("_"," ").lower() \
                       or entity_old.lower() in c["text"].lower() else "OFF-TOPIC"
            print(f"    [{c['src']:6}] score={c['score']}  page={c['page'][:40]}  {flag}")
            print(f"           {c['text']}")

        print()
        print(f"  NEW ({len(new_chunks)} chunks):")
        for c in new_chunks:
            flag = "OK" if entity_old.lower() in c["page"].replace("_"," ").lower() \
                       or entity_old.lower() in c["text"].lower() else "OFF-TOPIC"
            print(f"    [{c['src']:6}] score={c['score']}  page={c['page'][:40]}  {flag}")
            print(f"           {c['text']}")
        print()

if __name__ == "__main__":
    main()
