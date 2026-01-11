from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .retriever_base import BaseRetriever
from ..storage_impls.storage_base import BaseStorage
from ..embedding_impls.embedding_base import BaseEmbedding
from ..reranking_impls.reranking_base import BaseReranker


@dataclass
class SimpleRetriever(BaseRetriever):
    """
    Simple retriever that searches storage and optionally reranks.

    Args:
        storage: Storage backend to search
        embedding: Embedding model for query encoding
        reranker: Optional reranker for result refinement
        top_k: Number of results to return
        similarity_func: Similarity function (cosine, dot, euclidean)
    """
    storage: BaseStorage
    embedding: BaseEmbedding
    reranker: Optional[BaseReranker] = None
    top_k: int = 5
    similarity_func: str = "cosine"

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query string

        Returns:
            List of results with "text" and "score" keys
        """
        query_embedding = self.embedding.embed_text(query)

        # Get more results if reranking
        search_k = self.top_k * 3 if self.reranker else self.top_k

        results = self.storage.search(
            query_embedding=query_embedding,
            top_k=search_k,
            similarity_func=self.similarity_func
        )

        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=self.top_k)
        else:
            results = results[:self.top_k]

        return results
