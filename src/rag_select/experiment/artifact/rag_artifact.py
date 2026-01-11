from typing import Any, Dict, List
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


class RAGArtifact(BaseRetriever):
    experiment_params: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    pipeline: Any = Field(exclude=True)
    k: int = 3

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return [
            Document(page_content=r["text"])
            for r in (self.pipeline.retrieve(query) or [])[: self.k]
        ]
