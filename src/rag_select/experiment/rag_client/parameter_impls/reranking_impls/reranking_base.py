from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseReranker(ABC):
    """
    Abstract base class for reranking retrieved documents.

    Rerankers take retrieved results from the storage backend and
    re-score them based on more sophisticated relevance models
    (e.g., cross-encoders, Cohere Rerank API).

    This is applied after storage retrieval and before returning top-k results.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query string
            documents: List of documents with at least "text" and "score" keys
            top_k: Optional limit on results to return (if None, return all)

        Returns:
            List of documents with updated "score" values, sorted by relevance.
            Each document dict should preserve original fields and may add:
            - "rerank_score": The new relevance score from the reranker
            - "original_score": The original score from vector search
        """
        pass

    def rerank_batch(
        self,
        queries: List[str],
        documents_per_query: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch reranking for multiple query-document pairs.

        Default implementation calls rerank() for each pair.
        Override for more efficient batch processing.

        Args:
            queries: List of query strings
            documents_per_query: List of document lists, one per query
            top_k: Optional limit on results per query

        Returns:
            List of reranked document lists
        """
        return [
            self.rerank(query, docs, top_k)
            for query, docs in zip(queries, documents_per_query)
        ]
