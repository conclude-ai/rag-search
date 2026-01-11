from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseStorage(ABC):
    """
    Abstract base class for vector storage backends.

    Implementations should handle:
    - Storing document chunks with their embeddings and metadata
    - Searching for similar documents given a query embedding
    - Supporting different similarity functions (cosine, dot, euclidean)
    """

    @abstractmethod
    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the storage.

        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors (same length as chunks)
            metadata: Optional list of metadata dicts (same length as chunks)
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_func: str = "cosine",
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: The query vector
            top_k: Number of results to return
            similarity_func: Similarity function ("cosine", "dot", "euclidean")
            filter_metadata: Optional metadata filter
            **kwargs: Additional backend-specific parameters

        Returns:
            List of dicts with keys: "text", "score", and any metadata fields
        """
        pass

    def clear(self) -> None:
        """Clear all documents from storage. Optional to implement."""
        raise NotImplementedError("clear() not implemented for this storage backend")

    def count(self) -> int:
        """Return the number of documents in storage. Optional to implement."""
        raise NotImplementedError("count() not implemented for this storage backend")
