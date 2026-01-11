from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..storage_impls.storage_base import BaseStorage
from ..embedding_impls.embedding_base import BaseEmbedding
from ..reranking_impls.reranking_base import BaseReranker


class BaseRetriever(ABC):
    """
    Abstract base class for retrievers.

    A retriever takes a query and returns relevant documents from storage.
    It handles embedding the query, searching storage, and optional reranking.
    """

    @abstractmethod
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query string

        Returns:
            List of results with "text" and "score" keys
        """
        pass
