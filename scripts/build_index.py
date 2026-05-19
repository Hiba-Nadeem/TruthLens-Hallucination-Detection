import json
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "wiki_summaries.json"
STORE_PATH = BASE_DIR / "vectorstore_wiki"
CHUNK_METADATA_PATH = BASE_DIR / "data" / "chunk_metadata.json"


# -----------------------------
# Decode FEVER title (important)
# -----------------------------
def decode_fever_title(title: str) -> str:
    title = title.replace("-LRB-", "(").replace("-RRB-", ")")
    title = title.replace("-LSB-", "[").replace("-RSB-", "]")
    title = title.replace("-LCB-", "{").replace("-RCB-", "}")
    title = title.replace("-COLON-", ":")
    title = title.replace("_", " ")
    return title.strip()


# -----------------------------
# Load summaries
# -----------------------------
def load_wiki_summaries(json_path=DATA_PATH):
    if not json_path.exists():
        print("❌ wiki_summaries.json not found!")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # remove empty summaries
    cleaned = {
        k: v for k, v in data.items()
        if isinstance(v, str) and v.strip()
    }

    return cleaned


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=2, overlap=1):
    sentences = sent_tokenize(text)

    if not sentences:
        return []

    if len(sentences) <= chunk_size:
        return [" ".join(sentences).strip()]

    chunks = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(sentences), step):
        chunk = sentences[i:i + chunk_size]

        if not chunk:
            continue

        chunk_text = " ".join(chunk).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if i + chunk_size >= len(sentences):
            break

    return chunks


# -----------------------------
# Build documents
# -----------------------------
def build_documents(wiki_data, chunk_size=2, overlap=1):
    documents = []
    chunk_metadata = []
    chunk_id = 0

    for page_title, summary in wiki_data.items():

        decoded_title = decode_fever_title(page_title)

        chunks = chunk_text(summary, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk in enumerate(chunks):
            embed_text = f"passage: {chunk}"

            doc = Document(
                page_content=embed_text,
                metadata={
                    "chunk_id": chunk_id,
                    "page_title": page_title,
                    "decoded_title": decoded_title,
                    "chunk_index": idx,
                    "original_text": chunk,
                    "source": f"Wikipedia: {decoded_title}"
                }
            )

            documents.append(doc)

            chunk_metadata.append({
                "chunk_id": chunk_id,
                "page_title": page_title,
                "decoded_title": decoded_title,
                "chunk_index": idx,
                "chunk_text": chunk
            })

            chunk_id += 1

    return documents, chunk_metadata


# -----------------------------
# Build vectorstore
# -----------------------------
def build_vectorstore(documents):
    if not documents:
        print("❌ No documents to index!")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    STORE_PATH.mkdir(exist_ok=True)
    vectorstore.save_local(str(STORE_PATH))

    return vectorstore


# -----------------------------
# Save metadata
# -----------------------------
def save_chunk_metadata(chunk_metadata, output_path=CHUNK_METADATA_PATH):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)


# -----------------------------
# MAIN
# -----------------------------
def main():
    nltk.download("punkt")

    print("Loading wiki summaries...")
    wiki_data = load_wiki_summaries()
    print(f"Loaded {len(wiki_data)} cleaned summaries")

    if not wiki_data:
        print("❌ No data found. Run fetch_wiki.py first.")
        return

    print("\nCreating chunks...")
    documents, chunk_metadata = build_documents(wiki_data)

    print(f"Created {len(documents)} chunks")

    if documents:
        print("\nSample chunk:")
        print(documents[0].metadata)
        print(documents[0].page_content[:200])

    print("\nBuilding FAISS index...")
    vectorstore = build_vectorstore(documents)

    save_chunk_metadata(chunk_metadata)

    print("\n✅ Done.")
    print(f"Vectorstore saved at: {STORE_PATH}")
    print(f"Chunk metadata saved at: {CHUNK_METADATA_PATH}")
    print(f"Total chunks: {len(chunk_metadata)}")


if __name__ == "__main__":
    main()