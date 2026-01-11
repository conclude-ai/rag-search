from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Callable

from rag_select.parameter_impls.ingestion_impls.ingestion_base import BaseIngestion
from rag_select.parameter_impls.chunking_impls.chunking_base import BaseChunking
from rag_select.parameter_impls.embedding_impls.embedding_base import BaseEmbedding
from rag_select.parameter_impls.storage_impls.storage_base import BaseStorage
from rag_select.parameter_impls.retriever_impls.retriever_base import BaseRetriever


@dataclass
class RAGClient:
    """Single RAG client interface.

    - upload_documents: ingestion -> chunking -> embedding -> storage

    - retrieve: query -> retriever (storage search + optional rerank)

    - query (optional): retrieve -> prompt -> llm_generate

    Experiments should treat the client as an executor and produce RAGArtifact outputs.

    """

    ingestion: BaseIngestion
    chunking: BaseChunking
    embedding: BaseEmbedding
    storage: BaseStorage
    retriever: BaseRetriever
    llm_generate: Optional[Callable[[str], str]] = None

    def upload_documents(self, documents: Iterable[Dict[str, Any]]) -> None:
        chunks: List[str] = []
        metas: List[Dict[str, Any]] = []

        for doc in documents:
            text = self.ingestion.ingest(doc)
            doc_meta = doc.get("metadata", {})
            for i, chunk in enumerate(self.chunking.chunk(text)):
                chunks.append(chunk)
                metas.append({**doc_meta, "chunk_id": i, **{k: v for k, v in doc.items() if k in ("doc_id", "source")}})

        if not chunks:
            return

        embeddings = self.embedding.embed_batch(chunks)
        self.storage.add_documents(chunks=chunks, embeddings=embeddings, metadata=metas)

    def retrieve(self, query: str):
        return self.retriever.retrieve(query)

    def query(self, query: str) -> str:
        if self.llm_generate is None:
            raise ValueError("llm_generate is not set on this RAGClient. Use retrieve() or provide llm_generate.")
        results = self.retrieve(query)
        context = "\n\n".join(r.get("text", "") for r in results)
        prompt = (
            "Answer the question using the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}"
        )
        return self.llm_generate(prompt)
