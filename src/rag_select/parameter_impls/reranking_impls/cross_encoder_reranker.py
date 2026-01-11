from typing import List, Dict, Any, Optional
from .reranking_base import BaseReranker
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "cross_encoder",
    ComponentCategory.RERANKING,
    default_params={"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
    description="HuggingFace cross-encoder reranker"
)
class CrossEncoderReranker(BaseReranker):
    """
    Reranker using HuggingFace cross-encoder models.

    Cross-encoders jointly encode query and document pairs, providing
    more accurate relevance scores than bi-encoders (embedding similarity).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Args:
            model_name: HuggingFace model name. Common options:
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast)
                - cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced)
                - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest)
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Install `sentence-transformers` to use CrossEncoderReranker: "
                "pip install sentence-transformers"
            )

        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc["text"]) for doc in documents]

        # Get cross-encoder scores
        scores = self._model.predict(pairs)

        # Update documents with new scores
        reranked = []
        for doc, score in zip(documents, scores):
            new_doc = doc.copy()
            new_doc["original_score"] = doc.get("score", 0.0)
            new_doc["rerank_score"] = float(score)
            new_doc["score"] = float(score)  # Update main score
            reranked.append(new_doc)

        # Sort by new score (descending)
        reranked.sort(key=lambda x: x["score"], reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def rerank_batch(
        self,
        queries: List[str],
        documents_per_query: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """Batch reranking - optimized for cross-encoder."""
        # For cross-encoder, batch processing is more efficient
        all_pairs = []
        pair_indices = []  # Track which query each pair belongs to

        for query_idx, (query, docs) in enumerate(zip(queries, documents_per_query)):
            for doc in docs:
                all_pairs.append((query, doc["text"]))
                pair_indices.append(query_idx)

        if not all_pairs:
            return [[] for _ in queries]

        # Get all scores in one batch
        all_scores = self._model.predict(all_pairs)

        # Reconstruct results
        results = [[] for _ in queries]
        pair_idx = 0

        for query_idx, docs in enumerate(documents_per_query):
            query_results = []
            for doc in docs:
                new_doc = doc.copy()
                new_doc["original_score"] = doc.get("score", 0.0)
                new_doc["rerank_score"] = float(all_scores[pair_idx])
                new_doc["score"] = float(all_scores[pair_idx])
                query_results.append(new_doc)
                pair_idx += 1

            query_results.sort(key=lambda x: x["score"], reverse=True)
            if top_k:
                query_results = query_results[:top_k]
            results[query_idx] = query_results

        return results
