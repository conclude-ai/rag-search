from typing import List, Dict, Any, Optional
from .storage_base import BaseStorage
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "chroma",
    ComponentCategory.STORAGE,
    default_params={"collection_name": "default"},
    description="ChromaDB embedded vector database"
)
class ChromaStorage(BaseStorage):
    """
    Wrapper for ChromaDB vector database.

    ChromaDB is an open-source embedding database that can run locally
    or in client-server mode.
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None
    ):
        """
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage (None for in-memory)
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "Install `chromadb` to use Chroma storage: "
                "pip install chromadb"
            )

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._id_counter = 0

    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add documents to ChromaDB."""
        metadata = metadata or [{} for _ in chunks]

        ids = []
        for _ in chunks:
            self._id_counter += 1
            ids.append(f"chunk_{self._id_counter}")

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_func: str = "cosine",
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB collection."""
        where_filter = filter_metadata if filter_metadata else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        if results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)

            for doc, meta, dist in zip(documents, metadatas, distances):
                # ChromaDB returns distance, convert to similarity
                score = 1.0 - dist if similarity_func == "cosine" else -dist
                output.append({
                    "text": doc,
                    "score": float(score),
                    **(meta or {})
                })

        return output

    def clear(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        collection_name = self._collection.name
        self._client.delete_collection(collection_name)
        self._collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._id_counter = 0

    def count(self) -> int:
        """Return the number of documents."""
        return self._collection.count()
