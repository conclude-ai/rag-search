from typing import List, Dict, Any, Optional
from .storage_base import BaseStorage
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "pinecone",
    ComponentCategory.STORAGE,
    default_params={"namespace": ""},
    description="Pinecone managed vector database"
)
class PineconeStorage(BaseStorage):
    """
    Wrapper for Pinecone vector database.

    Requires an existing Pinecone index to be created beforehand.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: str = "default",
        namespace: str = ""
    ):
        """
        Args:
            api_key: Pinecone API key. If None, uses PINECONE_API_KEY env var.
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
        """
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "Install `pinecone-client` to use Pinecone storage: "
                "pip install pinecone-client"
            )

        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)
        self.namespace = namespace
        self._id_counter = 0

    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Upsert documents to Pinecone."""
        metadata = metadata or [{} for _ in chunks]

        vectors = []
        for chunk, emb, meta in zip(chunks, embeddings, metadata):
            self._id_counter += 1
            vectors.append({
                "id": f"chunk_{self._id_counter}",
                "values": emb,
                "metadata": {**meta, "text": chunk}
            })

        # Batch upsert (Pinecone recommends batches of 100)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self._index.upsert(vectors=batch, namespace=self.namespace)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_func: str = "cosine",
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Search Pinecone index."""
        response = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter_metadata
        )

        results = []
        for match in response.matches:
            meta = match.metadata or {}
            result = {
                "text": meta.pop("text", ""),
                "score": match.score,
                **meta
            }
            results.append(result)

        return results

    def clear(self) -> None:
        """Delete all vectors in the namespace."""
        self._index.delete(delete_all=True, namespace=self.namespace)
        self._id_counter = 0
