from typing import List, Dict, Any, Optional
from .reranking_base import BaseReranker
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "jina",
    ComponentCategory.RERANKING,
    default_params={"model": "jina-reranker-v2-base-multilingual"},
    description="Jina AI Reranker API"
)
class JinaReranker(BaseReranker):
    """
    Wrapper for Jina AI's Reranker API.

    High-performance multilingual reranking.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-reranker-v2-base-multilingual"
    ):
        """
        Args:
            api_key: Jina API key. If None, uses JINA_API_KEY env var.
            model: Reranker model name
        """
        import os
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        self.model = model
        self._base_url = "https://api.jina.ai/v1/rerank"

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents using Jina Reranker API."""
        if not documents:
            return []

        try:
            import requests
        except ImportError:
            raise ImportError(
                "Install `requests` to use JinaReranker: "
                "pip install requests"
            )

        # Extract texts
        texts = [doc["text"] for doc in documents]

        # Call Jina API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": top_k or len(documents)
        }

        response = requests.post(
            self._base_url,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        result = response.json()

        # Build reranked results
        reranked = []
        for item in result.get("results", []):
            idx = item["index"]
            original_doc = documents[idx]
            new_doc = original_doc.copy()
            new_doc["original_score"] = original_doc.get("score", 0.0)
            new_doc["original_rank"] = idx
            new_doc["rerank_score"] = item["relevance_score"]
            new_doc["score"] = item["relevance_score"]
            reranked.append(new_doc)

        return reranked
