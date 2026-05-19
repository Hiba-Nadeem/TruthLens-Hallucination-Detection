import argparse
import json
import pandas as pd
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------------------------------
# Config
# -------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

INPUT_PATH = "../data/llm_answers_googleflan-t5-base.csv"
VECTORSTORE_PATH = "../vectorstore_wiki"
CHUNK_METADATA_PATH = "../data/chunk_metadata.json"
OUTPUT_PATH = "../data/scored_claims_googleflan-t5-base_top3.csv"

TOP_K = 3


# -------------------------------------------------------
# Load data
# -------------------------------------------------------
def load_answers(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["claim", "llm_answer"]).copy()
    return df


# -------------------------------------------------------
# Load chunk metadata for direct entity lookup
# -------------------------------------------------------
def load_chunk_metadata(path=CHUNK_METADATA_PATH):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


# -------------------------------------------------------
# Load embeddings + vectorstore
# -------------------------------------------------------
def load_embeddings_and_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return embeddings, vectorstore


# -------------------------------------------------------
# Load NLI model
# -------------------------------------------------------
def load_nli_pipeline():
    return pipeline(
        "text-classification",
        model=NLI_MODEL,
        top_k=None,
        truncation=True,
        max_length=512
    )


# -------------------------------------------------------
# Extract LLM label from raw answer text
# -------------------------------------------------------
def extract_llm_label(answer):
    answer = str(answer).lower().strip()

    if answer == "true":
        return "TRUE"
    if answer == "false":
        return "FALSE"

    if "false" in answer and "true" not in answer:
        return "FALSE"
    if "true" in answer and "false" not in answer:
        return "TRUE"

    return "UNKNOWN"


# -------------------------------------------------------
# Run NLI on a single (premise, hypothesis) pair
# premise   = evidence chunk from Wikipedia/document
# hypothesis = the original claim
# Returns: {"entailment": float, "contradiction": float, "neutral": float}
# -------------------------------------------------------
def run_nli(nli_pipe, premise, hypothesis):
    result = nli_pipe({"text": premise, "text_pair": hypothesis})
    # result is a flat list of dicts: [{"label": ..., "score": ...}, ...]
    return {item["label"].lower(): item["score"] for item in result}


# -------------------------------------------------------
# Aggregate NLI scores across all retrieved chunks
#
# For each of the TOP_K chunks we have 3 NLI scores.
# Each chunk votes for its dominant stance (entailment / contradiction / neutral).
# The winning stance must beat both others AND have at least one confident
# prediction (>= 0.5). This prevents a single off-topic high-confidence chunk
# from poisoning the result (the failure mode of raw max aggregation).
# -------------------------------------------------------
def aggregate_nli(nli_results):
    votes = {"entailment": 0, "contradiction": 0, "neutral": 0}
    max_entailment    = 0.0
    max_contradiction = 0.0
    max_neutral       = 0.0

    for r in nli_results:
        e = r.get("entailment", 0)
        c = r.get("contradiction", 0)
        n = r.get("neutral", 0)

        max_entailment    = max(max_entailment, e)
        max_contradiction = max(max_contradiction, c)
        max_neutral       = max(max_neutral, n)

        # Each chunk votes for its dominant stance
        if e >= c and e >= n:
            votes["entailment"] += 1
        elif c >= e and c >= n:
            votes["contradiction"] += 1
        else:
            votes["neutral"] += 1

    e_votes = votes["entailment"]
    c_votes = votes["contradiction"]
    n_votes = votes["neutral"]

    # Winning stance must beat both others to avoid ties being miscalled,
    # and must have at least one confident prediction (>= 0.5).
    if e_votes > c_votes and e_votes > n_votes and max_entailment >= 0.5:
        return "ENTAILMENT", round(max_entailment, 4)

    if c_votes > e_votes and c_votes > n_votes and max_contradiction >= 0.5:
        return "CONTRADICTION", round(max_contradiction, 4)

    return "NEUTRAL", round(max_neutral, 4)


# -------------------------------------------------------
# Final verdict: combine LLM label + NLI result
#
#   LLM TRUE  + ENTAILMENT    → Supported           (LLM agreed, evidence backs it)
#   LLM TRUE  + CONTRADICTION → Hallucinated        (LLM agreed, evidence refutes it)
#   LLM FALSE + CONTRADICTION → Supported           (LLM refuted, evidence also refutes claim)
#   LLM FALSE + ENTAILMENT    → Hallucinated        (LLM refuted, but evidence supports claim)
#   LLM TRUE  + NEUTRAL       → True | No Evidence  (LLM said true, evidence inconclusive)
#   LLM FALSE + NEUTRAL       → False | No Evidence (LLM said false, evidence inconclusive)
#   LLM UNKNOWN + anything    → Not Enough Evidence (LLM output unparseable, excluded from eval)
# -------------------------------------------------------
def compute_verdict(llm_label, nli_label):
    if llm_label == "UNKNOWN":
        return "Not Enough Evidence"

    if nli_label == "NEUTRAL":
        return "True | No Evidence" if llm_label == "TRUE" else "False | No Evidence"

    if llm_label == "TRUE":
        return "Supported" if nli_label == "ENTAILMENT" else "Hallucinated"

    if llm_label == "FALSE":
        return "Supported" if nli_label == "CONTRADICTION" else "Hallucinated"


# -------------------------------------------------------
# Extract the main subject/entity from a claim
#
# Splits on common verb patterns to isolate the subject.
# Falls back to the first 3 words if no verb pattern matches.
# -------------------------------------------------------
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
    # Find the leftmost verb match — earlier in the claim = shorter, more specific entity
    best_idx = len(claim)
    for pattern in VERB_PATTERNS:
        if pattern in claim_lower:
            idx = claim_lower.index(pattern)
            if idx < best_idx:
                best_idx = idx
    if best_idx < len(claim):
        return claim[:best_idx].strip()
    # Fallback: first 2 words (avoids including a verb in the entity name)
    return " ".join(claim.split()[:2]).strip()


# -------------------------------------------------------
# Direct entity chunk lookup from chunk_metadata.json
#
# Finds chunks from pages whose title contains the entity.
# This is used as a fallback when FAISS misses the right page.
# -------------------------------------------------------
def get_entity_chunks(chunk_metadata, entity, max_chunks=3):
    entity_key = entity.replace(" ", "_").lower()
    entity_plain = entity.lower()

    matches = []
    for chunk in chunk_metadata:
        page_title = chunk.get("page_title", "").lower()
        decoded_title = chunk.get("decoded_title", "").lower()

        if (
            entity_key in page_title
            or entity_plain in page_title.replace("_", " ")
            or entity_plain in decoded_title
        ):
            matches.append(chunk)

    return matches[:max_chunks]


# -------------------------------------------------------
# Score a single claim
# -------------------------------------------------------
def score_one_claim(claim, llm_answer, embeddings, vectorstore, nli_pipe, chunk_metadata, k=TOP_K):
    entity = extract_entity(claim)
    entity_lower = entity.lower()

    # Step 1 — FAISS semantic retrieval
    # Prepend entity to query so FAISS biases toward the entity's page
    query_text = f"query: {entity}. {claim}"
    faiss_results = vectorstore.similarity_search_with_score(query_text, k=k)

    # Step 2 — Direct entity chunk lookup from metadata (catches what FAISS misses)
    entity_chunks = get_entity_chunks(chunk_metadata, entity, max_chunks=3)

    # Step 3 — Merge: start with entity chunks, then add FAISS results
    # Use chunk_text as dedup key so we don't run NLI on the same text twice
    seen_texts = set()
    merged_chunks = []

    for chunk in entity_chunks:
        text = chunk["chunk_text"]
        if text not in seen_texts:
            seen_texts.add(text)
            merged_chunks.append({
                "page_title":  chunk["page_title"],
                "chunk_text":  text,
                "store_score": 0.0,
                "source":      "direct"
            })

    for doc, store_score in faiss_results:
        # Tightened threshold — drop chunks too dissimilar to the query
        if float(store_score) > 0.55:
            continue
        text = doc.metadata.get("original_text", doc.page_content)
        page_title = doc.metadata.get("page_title", "Unknown")
        decoded_title = doc.metadata.get("decoded_title", "").lower()
        # Drop FAISS chunks where the entity doesn't appear in the page title
        # or chunk text — they're off-topic and poison NLI
        page_match = (
            entity_lower in page_title.replace("_", " ").lower()
            or entity_lower in decoded_title
        )
        text_match = entity_lower in text.lower()
        if not page_match and not text_match:
            continue
        if text not in seen_texts:
            seen_texts.add(text)
            merged_chunks.append({
                "page_title":  page_title,
                "chunk_text":  text,
                "store_score": round(float(store_score), 4),
                "source":      "faiss"
            })

    # Fallback: if entity filtering removed all FAISS chunks and there are no
    # direct chunks either, use top-3 FAISS results unfiltered so NLI has
    # something to work with rather than returning NEUTRAL by default
    if not merged_chunks:
        seen_texts_fallback = set()
        for doc, store_score in faiss_results[:3]:
            text = doc.metadata.get("original_text", doc.page_content)
            if text not in seen_texts_fallback:
                seen_texts_fallback.add(text)
                merged_chunks.append({
                    "page_title":  doc.metadata.get("page_title", "Unknown"),
                    "chunk_text":  text,
                    "store_score": round(float(store_score), 4),
                    "source":      "faiss"
                })

    # Step 4 — run NLI on each merged chunk against the original claim
    evidence_rows = []
    nli_results = []

    for chunk in merged_chunks:
        nli_scores = run_nli(nli_pipe, premise=chunk["chunk_text"], hypothesis=claim)
        nli_results.append(nli_scores)

        evidence_rows.append({
            "page_title":    chunk["page_title"],
            "chunk_text":    chunk["chunk_text"],
            "store_score":   chunk["store_score"],
            "source":        chunk["source"],
            "entailment":    round(nli_scores.get("entailment", 0), 4),
            "contradiction": round(nli_scores.get("contradiction", 0), 4),
            "neutral":       round(nli_scores.get("neutral", 0), 4),
        })

    # Step 5 — aggregate and decide
    if nli_results:
        nli_label, nli_confidence = aggregate_nli(nli_results)
    else:
        nli_label, nli_confidence = "NEUTRAL", 0.0

    llm_label = extract_llm_label(llm_answer)
    decision  = compute_verdict(llm_label, nli_label)

    return {
        "retrieved_evidence": evidence_rows,
        "top_page":           evidence_rows[0]["page_title"] if evidence_rows else "",
        "top_evidence":       evidence_rows[0]["chunk_text"] if evidence_rows else "",
        "nli_label":          nli_label,
        "nli_confidence":     nli_confidence,
        "decision":           decision
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Score LLM answers using retrieval + NLI.")
    parser.add_argument("--input", default=INPUT_PATH, help="Input answers CSV path.")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output scored CSV path.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="FAISS top-k retrieval.")
    args = parser.parse_args()

    print("Loading LLM answers...")
    df = load_answers(args.input)
    print(f"Loaded {len(df)} rows")

    print("Loading vectorstore and embeddings...")
    embeddings, vectorstore = load_embeddings_and_store()

    print("Loading NLI model...")
    nli_pipe = load_nli_pipeline()

    print("Loading chunk metadata...")
    chunk_metadata = load_chunk_metadata()
    print(f"Loaded {len(chunk_metadata)} chunks")

    results = []
    print("Scoring claims...")

    for _, row in df.iterrows():
        claim      = str(row["claim"]).strip()
        llm_answer = str(row["llm_answer"]).strip()
        llm_label  = extract_llm_label(llm_answer)

        scored = score_one_claim(
            claim=claim,
            llm_answer=llm_answer,
            embeddings=embeddings,
            vectorstore=vectorstore,
            nli_pipe=nli_pipe,
            chunk_metadata=chunk_metadata,
            k=args.top_k
        )

        results.append({
            "id":                     row.get("id", ""),
            "verifiable":             row.get("verifiable", ""),
            "label":                  row.get("label", ""),
            "claim":                  claim,
            "llm_answer":             llm_answer,
            "llm_label":              llm_label,
            "llm_model":              row.get("llm_model", ""),
            "top_page":               scored["top_page"],
            "top_evidence":           scored["top_evidence"],
            "nli_label":              scored["nli_label"],
            "nli_confidence":         scored["nli_confidence"],
            "decision":               scored["decision"],
            "final_verdict": (
                "Hallucination"     if scored["decision"] == "Hallucinated"        else
                "Not Hallucination" if scored["decision"] == "Supported"           else
                scored["decision"]
            ),
            "all_retrieved_evidence": str(scored["retrieved_evidence"])
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output, index=False, encoding="utf-8")

    print(f"\nDone. Saved to: {args.output}")
    print(output_df[["claim", "llm_label", "nli_label", "nli_confidence", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
