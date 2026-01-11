from typing import List, Dict, Any, Optional
from .storage_base import BaseStorage
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "weaviate",
    ComponentCategory.STORAGE,
    default_params={"collection_name": "Document"},
    description="Weaviate vector database"
)
class WeaviateStorage(BaseStorage):
    """
    Wrapper for Weaviate vector database.

    Weaviate is an open-source vector search engine with GraphQL support.
    """

    def __init__(
        self,
        collection_name: str = "Document",
        url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Args:
            collection_name: Name of the collection (Weaviate class)
            url: Weaviate server URL (None for embedded)
            api_key: Weaviate API key
        """
        try:
            import weaviate
            from weaviate.classes.config import Configure, Property, DataType
        except ImportError:
            raise ImportError(
                "Install `weaviate-client` to use Weaviate storage: "
                "pip install weaviate-client"
            )

        if url:
            self._client = weaviate.connect_to_custom(
                http_host=url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key) if api_key else None
            )
        else:
            self._client = weaviate.connect_to_embedded()

        self.collection_name = collection_name
        self._id_counter = 0

        # Create collection if it doesn't exist
        if not self._client.collections.exists(collection_name):
            self._client.collections.create(
                name=collection_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                ]
            )

        self._collection = self._client.collections.get(collection_name)

    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to Weaviate."""
        metadata = metadata or [{} for _ in chunks]

        with self._collection.batch.dynamic() as batch:
            for chunk, emb, meta in zip(chunks, embeddings, metadata):
                properties = {
                    "text": chunk,
                    "source": meta.get("source", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "category": meta.get("category", ""),
                }
                batch.add_object(properties=properties, vector=emb)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_func: str = "cosine",
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Search Weaviate collection."""
        try:
            from weaviate.classes.query import MetadataQuery, Filter
        except ImportError:
            raise ImportError("Install `weaviate-client` to use Weaviate storage")

        # Build filter if provided
        weaviate_filter = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(Filter.by_property(key).equal(value))
            if len(conditions) == 1:
                weaviate_filter = conditions[0]
            else:
                weaviate_filter = Filter.all_of(conditions)

        response = self._collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            filters=weaviate_filter,
            return_metadata=MetadataQuery(distance=True)
        )

        results = []
        for obj in response.objects:
            props = obj.properties
            # Convert distance to similarity (Weaviate returns distance)
            distance = obj.metadata.distance if obj.metadata else 0.0
            score = 1.0 - distance if similarity_func == "cosine" else -distance

            results.append({
                "text": props.get("text", ""),
                "score": float(score),
                "source": props.get("source", ""),
                "doc_id": props.get("doc_id", ""),
                "category": props.get("category", ""),
            })

        return results

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self._client.collections.delete(self.collection_name)

        try:
            from weaviate.classes.config import Configure, Property, DataType
        except ImportError:
            raise ImportError("Install `weaviate-client` to use Weaviate storage")

        self._client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="category", data_type=DataType.TEXT),
            ]
        )
        self._collection = self._client.collections.get(self.collection_name)

    def count(self) -> int:
        """Return the number of documents."""
        response = self._collection.aggregate.over_all(total_count=True)
        return response.total_count

    def __del__(self):
        """Close the Weaviate client on cleanup."""
        if hasattr(self, '_client'):
            self._client.close()
