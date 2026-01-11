from typing import List, Dict, Any, Optional
import numpy as np

from .storage_base import BaseStorage
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "simple",
    ComponentCategory.STORAGE,
    default_params={},
    description="Simple in-memory vector storage using numpy"
)
class SimpleStorage(BaseStorage):
    """
    Simple in-memory vector storage implementation using numpy.

    Supports cosine, dot product, and euclidean similarity functions.
    Suitable for development and small datasets.
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to the in-memory store."""
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in chunks])

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
            query_embedding: Query vector
            top_k: Number of results
            similarity_func: "cosine", "dot", or "euclidean"
            filter_metadata: Optional filter (exact match on metadata fields)
        """
        if not self.embeddings:
            return []

        query_vec = np.array(query_embedding)
        scores = []

        for idx, doc_embedding in enumerate(self.embeddings):
            # Apply metadata filter if provided
            if filter_metadata:
                if not self._matches_filter(self.metadata[idx], filter_metadata):
                    continue

            doc_vec = np.array(doc_embedding)
            score = self._compute_similarity(query_vec, doc_vec, similarity_func)
            scores.append((idx, score))

        # Sort by score (descending for cosine/dot, ascending for euclidean)
        reverse = similarity_func != "euclidean"
        scores.sort(key=lambda x: x[1], reverse=reverse)
        top_results = scores[:top_k]

        results = []
        for idx, score in top_results:
            result = {
                'text': self.chunks[idx],
                'score': float(score)
            }
            result.update(self.metadata[idx])
            results.append(result)

        return results

    def _compute_similarity(
        self,
        query_vec: np.ndarray,
        doc_vec: np.ndarray,
        similarity_func: str
    ) -> float:
        """Compute similarity between two vectors."""
        if similarity_func == "cosine":
            norm_product = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            if norm_product == 0:
                return 0.0
            return float(np.dot(query_vec, doc_vec) / norm_product)

        elif similarity_func == "dot":
            return float(np.dot(query_vec, doc_vec))

        elif similarity_func == "euclidean":
            return float(np.linalg.norm(query_vec - doc_vec))

        else:
            raise ValueError(
                f"Unknown similarity function: {similarity_func}. "
                f"Supported: cosine, dot, euclidean"
            )

    def _matches_filter(
        self,
        metadata: Dict[str, Any],
        filter_metadata: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches all filter conditions."""
        for key, value in filter_metadata.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def clear(self) -> None:
        """Clear all documents from storage."""
        self.chunks = []
        self.embeddings = []
        self.metadata = []

    def count(self) -> int:
        """Return the number of documents in storage."""
        return len(self.chunks)
