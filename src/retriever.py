from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

RRF_K = 60  
TOP_K = 5   # final number of documents to return

# model to use for re ranking after BM25 and dense
RERANK_MODEL = "BAAI/bge-reranker-base"
_reranker = None

def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(raw_texts: list[str]) -> BM25Okapi:
    tokenized = [tokenize(t) for t in raw_texts]
    return BM25Okapi(tokenized)


def rrf_fusion(
    bm25_ranked: list[int],
    dense_ranked: list[int],
    k: int = RRF_K,
) -> list[int]:
    """
    Given two lists of document indices (ordered by rank),
    return a merged list of indices sorted by RRF score (descending).
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    for rank, idx in enumerate(dense_ranked):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def hybrid_retrieve(
    query: str,
    chunks: list[Document],
    raw_texts: list[str],
    vectorstore: Chroma,
    top_k: int = TOP_K,
    rerank_results: bool = True,
) -> list[Document]:
    n = len(chunks)
    retrieve_n = min(n, max(top_k * 3, 20))  # retrieve more before fusion

    # BM25 retrieval
    bm25 = build_bm25_index(raw_texts)
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranked = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)[:retrieve_n]

    # Dense retrieval
    dense_docs = vectorstore.similarity_search(query, k=retrieve_n)
    content_to_idx = {chunk.page_content: i for i, chunk in enumerate(chunks)}
    dense_ranked = []
    for doc in dense_docs:
        idx = content_to_idx.get(doc.page_content)
        if idx is not None:
            dense_ranked.append(idx)

    # RRF fusion
    fused = rrf_fusion(bm25_ranked, dense_ranked)

    if rerank_results:
        top_indices = rerank(query, fused[:retrieve_n], chunks, top_k)
    else:
        top_indices = fused[:top_k]

    return [chunks[i] for i in top_indices]


def inspect_retrieve(
    query: str,
    chunks: list[Document],
    raw_texts: list[str],
    vectorstore: Chroma,
    top_k: int = TOP_K,
) -> dict:
    """
    Same as hybrid_retrieve but returns intermediate BM25 and dense results
    separately, for inspection/debugging purposes.
    """
    n = len(chunks)
    retrieve_n = min(n, max(top_k * 3, 20))

    bm25 = build_bm25_index(raw_texts)
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranked = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)[:retrieve_n]

    dense_docs = vectorstore.similarity_search(query, k=retrieve_n)
    content_to_idx = {chunk.page_content: i for i, chunk in enumerate(chunks)}
    dense_ranked = []
    for doc in dense_docs:
        idx = content_to_idx.get(doc.page_content)
        if idx is not None:
            dense_ranked.append(idx)

    fused = rrf_fusion(bm25_ranked, dense_ranked)
    reranked = rerank(query, fused[:retrieve_n], chunks, top_k)

    return {
        "bm25": [chunks[i] for i in bm25_ranked[:top_k]],
        "dense": [chunks[i] for i in dense_ranked[:top_k]],
        "fused": [chunks[i] for i in fused[:top_k]],
        "reranked": [chunks[i] for i in reranked],
    }


def get_reranker() -> CrossEncoder:
    """
    Load and return the re ranker model.
    """
    global _reranker
    if _reranker is None:
        print(f"[retriever] Loading re-ranker model: {RERANK_MODEL}")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker

def rerank(
        query: str,
        candidate_indices: list[int],
        chunks: list[Document],
        top_k: int ,
) -> list[int]:
    """re rank candidate documents using the cross encoder model
    returns the top k indices sorted by relevance to the query
    """
    # get the reranker model
    reranker = get_reranker()

    # prepare inputs -> (query, doc) pairs
    pairs = [(query, chunks[i].page_content) for i in candidate_indices]

    # get relevance scores
    scores = reranker.predict(pairs)

    # pair scores with indices
    scored = list(zip(candidate_indices, scores))

    # sort score by descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # return top k indices
    return [idx for idx, score in scored[:top_k]]

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

    from embed_and_store import load_vectorstore, load_bm25_data

    query = "How do I use git to undo a commit?"
    print(f"Query: {query}\n")

    vectorstore = load_vectorstore()
    chunks, raw_texts = load_bm25_data()
    results = hybrid_retrieve(query, chunks, raw_texts, vectorstore)

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"[{i}] Source: {source}")
        print(doc.page_content[:200])
        print()


