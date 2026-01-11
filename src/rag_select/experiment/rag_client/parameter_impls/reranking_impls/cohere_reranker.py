from typing import List, Dict, Any, Optional
from .reranking_base import BaseReranker
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "cohere",
    ComponentCategory.RERANKING,
    default_params={"model": "rerank-english-v3.0"},
    description="Cohere Rerank API"
)
class CohereReranker(BaseReranker):
    """
    Wrapper for Cohere's Rerank API.

    State-of-the-art neural reranking with minimal latency.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0"
    ):
        """
        Args:
            api_key: Cohere API key. If None, uses CO_API_KEY env var.
            model: Rerank model name (rerank-english-v3.0, rerank-multilingual-v3.0)
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Install `cohere` to use CohereReranker: "
                "pip install cohere"
            )

        self._client = cohere.Client(api_key=api_key)
        self.model = model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents using Cohere Rerank API."""
        if not documents:
            return []

        # Extract texts for Cohere API
        texts = [doc["text"] for doc in documents]

        # Call Cohere Rerank
        response = self._client.rerank(
            query=query,
            documents=texts,
            model=self.model,
            top_n=top_k or len(documents)
        )

        # Build reranked results
        reranked = []
        for result in response.results:
            original_doc = documents[result.index]
            new_doc = original_doc.copy()
            new_doc["original_score"] = original_doc.get("score", 0.0)
            new_doc["original_rank"] = result.index
            new_doc["rerank_score"] = result.relevance_score
            new_doc["score"] = result.relevance_score
            reranked.append(new_doc)

        return reranked
