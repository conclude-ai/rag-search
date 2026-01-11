from typing import List, Dict, Any, Optional
from .storage_base import BaseStorage
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "qdrant",
    ComponentCategory.STORAGE,
    default_params={"collection_name": "default"},
    description="Qdrant vector database"
)
class QdrantStorage(BaseStorage):
    """
    Wrapper for Qdrant vector database.

    Qdrant is a high-performance vector similarity search engine.
    """

    def __init__(
        self,
        collection_name: str = "default",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: int = 1536
    ):
        """
        Args:
            collection_name: Name of the collection
            url: Qdrant server URL (None for in-memory)
            api_key: Qdrant API key (for Qdrant Cloud)
            vector_size: Dimension of vectors
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "Install `qdrant-client` to use Qdrant storage: "
                "pip install qdrant-client"
            )

        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(":memory:")

        self.collection_name = collection_name
        self.vector_size = vector_size
        self._id_counter = 0

        # Create collection if it doesn't exist
        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name not in collections:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to Qdrant."""
        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            raise ImportError("Install `qdrant-client` to use Qdrant storage")

        metadata = metadata or [{} for _ in chunks]

        points = []
        for chunk, emb, meta in zip(chunks, embeddings, metadata):
            self._id_counter += 1
            points.append(PointStruct(
                id=self._id_counter,
                vector=emb,
                payload={"text": chunk, **meta}
            ))

        self._client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_func: str = "cosine",
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Search Qdrant collection."""
        query_filter = None
        if filter_metadata:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
            except ImportError:
                raise ImportError("Install `qdrant-client` to use Qdrant storage")

            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter
        )

        output = []
        for point in results:
            payload = point.payload or {}
            output.append({
                "text": payload.pop("text", ""),
                "score": point.score,
                **payload
            })

        return output

    def clear(self) -> None:
        """Clear all documents from the collection."""
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("Install `qdrant-client` to use Qdrant storage")

        self._client.delete_collection(self.collection_name)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )
        self._id_counter = 0

    def count(self) -> int:
        """Return the number of documents."""
        info = self._client.get_collection(self.collection_name)
        return info.points_count
