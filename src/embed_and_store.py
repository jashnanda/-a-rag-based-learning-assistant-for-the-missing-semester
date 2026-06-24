import pickle
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_DIR = str(Path(__file__).parent.parent / "chroma_db")
BM25_CACHE = Path(__file__).parent.parent / "bm25_texts.pkl"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def get_embeddings() -> HuggingFaceEmbeddings:
    print(f"[embed] Loading embedding model: {EMBED_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def build_vectorstore(chunks: list[Document], raw_texts: list[str]) -> Chroma:
    embeddings = get_embeddings()
    print(f"[embed] Embedding {len(chunks)} chunks and storing in ChromaDB at {CHROMA_DIR} ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    # Cache raw texts for BM25
    with open(BM25_CACHE, "wb") as f:
        pickle.dump({"raw_texts": raw_texts, "chunks": chunks}, f)
    print(f"[embed] Done. Saved BM25 text cache to {BM25_CACHE}")
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = get_embeddings()
    print(f"[embed] Loading existing ChromaDB from {CHROMA_DIR}")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def load_bm25_data() -> tuple[list[Document], list[str]]:
    with open(BM25_CACHE, "rb") as f:
        data = pickle.load(f)
    return data["chunks"], data["raw_texts"]


if __name__ == "__main__":
    from load_data import load_and_chunk

    if Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir()):
        print("[embed] ChromaDB already exists. Delete chroma_db/ to rebuild.")
    else:
        chunks, raw_texts = load_and_chunk()
        build_vectorstore(chunks, raw_texts)
        print("[embed] Vector store built successfully.")
