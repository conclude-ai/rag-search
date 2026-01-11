from typing import List, Optional
from .embedding_base import BaseEmbedding
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "cohere",
    ComponentCategory.EMBEDDING,
    default_params={"model": "embed-english-v3.0", "input_type": "search_document"},
    description="Cohere Embed API"
)
class CohereEmbedding(BaseEmbedding):
    """
    Wrapper for Cohere's Embed API.

    Supports different input types for documents vs queries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document"
    ):
        """
        Args:
            api_key: Cohere API key. If None, uses CO_API_KEY env var.
            model: Embedding model name
            input_type: "search_document" for indexing, "search_query" for queries
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Install `cohere` to use this embedding method: "
                "pip install cohere"
            )

        self._client = cohere.Client(api_key=api_key)
        self.model = model
        self.input_type = input_type

    def embed_text(self, text: str) -> List[float]:
        response = self._client.embed(
            texts=[text],
            model=self.model,
            input_type=self.input_type
        )
        return response.embeddings[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type
        )
        return response.embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a query (uses search_query input type)."""
        response = self._client.embed(
            texts=[query],
            model=self.model,
            input_type="search_query"
        )
        return response.embeddings[0]
