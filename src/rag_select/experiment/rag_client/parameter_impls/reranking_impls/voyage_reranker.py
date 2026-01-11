from typing import List, Dict, Any, Optional
from .reranking_base import BaseReranker
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "voyage",
    ComponentCategory.RERANKING,
    default_params={"model": "rerank-1"},
    description="Voyage AI Reranker API"
)
class VoyageReranker(BaseReranker):
    """
    Wrapper for Voyage AI's Reranker API.

    High-quality reranking optimized for retrieval.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-1"
    ):
        """
        Args:
            api_key: Voyage API key. If None, uses VOYAGE_API_KEY env var.
            model: Reranker model name
        """
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "Install `voyageai` to use VoyageReranker: "
                "pip install voyageai"
            )

        self._client = voyageai.Client(api_key=api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents using Voyage Reranker API."""
        if not documents:
            return []

        # Extract texts
        texts = [doc["text"] for doc in documents]

        # Call Voyage API
        result = self._client.rerank(
            query=query,
            documents=texts,
            model=self.model,
            top_k=top_k or len(documents)
        )

        # Build reranked results
        reranked = []
        for item in result.results:
            idx = item.index
            original_doc = documents[idx]
            new_doc = original_doc.copy()
            new_doc["original_score"] = original_doc.get("score", 0.0)
            new_doc["original_rank"] = idx
            new_doc["rerank_score"] = item.relevance_score
            new_doc["score"] = item.relevance_score
            reranked.append(new_doc)

        return reranked
