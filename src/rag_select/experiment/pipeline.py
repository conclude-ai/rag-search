from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from rag_select.rag_client import RAGClient
from rag_select.experiment.artifact.rag_artifact import RAGArtifact

from rag_select.parameter_impls.ingestion_impls.ingestion_base import BaseIngestion
from rag_select.parameter_impls.chunking_impls.chunking_base import BaseChunking
from rag_select.parameter_impls.embedding_impls.embedding_base import BaseEmbedding
from rag_select.parameter_impls.storage_impls.storage_base import BaseStorage
from rag_select.parameter_impls.retriever_impls.retriever_base import BaseRetriever


@dataclass(frozen=True)
class PipelineConfig:
    ingestion: BaseIngestion
    chunking: BaseChunking
    embedding: BaseEmbedding
    storage: BaseStorage
    retriever: BaseRetriever
    name: str


def build_client(cfg: PipelineConfig) -> RAGClient:
    return RAGClient(
        ingestion=cfg.ingestion,
        chunking=cfg.chunking,
        embedding=cfg.embedding,
        storage=cfg.storage,
        retriever=cfg.retriever,
    )


def artifact_from_client(
    *,
    client: RAGClient,
    experiment_params: Dict[str, Any],
    metrics: Dict[str, float],
) -> RAGArtifact:
    return RAGArtifact(experiment_params=experiment_params, metrics=metrics, pipeline=client)
