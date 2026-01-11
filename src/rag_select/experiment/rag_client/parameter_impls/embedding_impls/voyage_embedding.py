from typing import List, Optional
from .embedding_base import BaseEmbedding
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "voyage",
    ComponentCategory.EMBEDDING,
    default_params={"model": "voyage-2", "input_type": "document"},
    description="Voyage AI Embeddings API"
)
class VoyageEmbedding(BaseEmbedding):
    """
    Wrapper for Voyage AI's Embeddings API.

    High-quality embeddings optimized for retrieval.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "voyage-2",
        input_type: str = "document"
    ):
        """
        Args:
            api_key: Voyage API key. If None, uses VOYAGE_API_KEY env var.
            model: Model name (voyage-2, voyage-code-2, voyage-lite-02-instruct)
            input_type: "document" for indexing, "query" for queries
        """
        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "Install `voyageai` to use this embedding method: "
                "pip install voyageai"
            )

        self._client = voyageai.Client(api_key=api_key)
        self.model = model
        self.input_type = input_type

    def embed_text(self, text: str) -> List[float]:
        result = self._client.embed(
            texts=[text],
            model=self.model,
            input_type=self.input_type
        )
        return result.embeddings[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        result = self._client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type
        )
        return result.embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a query (uses query input type)."""
        result = self._client.embed(
            texts=[query],
            model=self.model,
            input_type="query"
        )
        return result.embeddings[0]
