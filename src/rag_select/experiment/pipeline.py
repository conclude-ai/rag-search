from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional

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


def generate_client_variations(
    *,
    ingestion_variants: List[BaseIngestion],
    chunking_variants: List[BaseChunking],
    embedding_variants: List[BaseEmbedding],
    storage_variants: List[BaseStorage],
    retriever_variants: List[BaseRetriever],
) -> List[PipelineConfig]:
    configs: List[PipelineConfig] = []
    for i, (ing, ch, emb, st, ret) in enumerate(product(
        ingestion_variants, chunking_variants, embedding_variants, storage_variants, retriever_variants
    ), 1):
        name = f"Config {i}: {type(ing).__name__} | {type(ch).__name__} | {type(emb).__name__} | {type(st).__name__} | {type(ret).__name__}"
        configs.append(PipelineConfig(ing, ch, emb, st, ret, name))
    return configs


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
