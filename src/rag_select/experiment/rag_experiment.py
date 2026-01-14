"""Main experiment orchestrator for RAG pipeline optimization.

This module provides the RAGExperiment class for automated RAG pipeline
optimization using grid search over component configurations.
"""

from __future__ import annotations

from dataclasses import fields
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Type

from rag_select.experiment.metrics import MetricsCalculator
from rag_select.experiment.pipeline import PipelineConfig, artifact_from_client, build_client
from rag_select.experiment.results import ExperimentResult, ExperimentResults
from rag_select.parameter_impls.ingestion_impls import SimpleIngestion
from rag_select.parameter_impls.storage_impls import SimpleStorage
from rag_select.parameter_impls.retriever_impls import SimpleRetriever
from rag_select.rag_client import RAGClient


class RAGExperiment:
    """Automated RAG pipeline optimization via grid search.

    Accepts lists of component instances and runs all combinations (Cartesian product).

    Example:
        >>> from rag_select.parameter_impls.chunking_impls import SlidingWindowChunking
        >>> from rag_select.parameter_impls.embedding_impls import HuggingFaceEmbedding
        >>>
        >>> experiment = RAGExperiment(
        ...     dataset=eval_set,
        ...     documents=corpus,
        ...     search_space={
        ...         "chunking": [
        ...             SlidingWindowChunking(chunk_size=256, chunk_overlap=20),
        ...             SlidingWindowChunking(chunk_size=512, chunk_overlap=50),
        ...         ],
        ...         "embedding": [
        ...             HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        ...         ],
        ...     }
        ... )
        >>> results = experiment.run()
        >>> results.rank(by="precision@5")
        >>> best = results.get_best_pipeline()
    """

    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        search_space: Dict[str, List[Any]],
        metrics: Optional[List[str]] = None,
    ):
        """Initialize RAG experiment.

        Args:
            dataset: Evaluation dataset with "query" and "relevant_doc_ids" keys
                Example: [{"query": "What is RAG?", "relevant_doc_ids": ["doc_1"]}]
            documents: Document corpus with "doc_id" and "text" or "file_path"
                Example: [{"doc_id": "doc_1", "text": "RAG is..."}]
            search_space: Dict mapping stage names to lists of component instances
                Stages: "ingestion", "chunking", "embedding", "storage", "retriever"
                For retriever, can also pass dicts with params like {"top_k": 5}
            metrics: Metric names to calculate
                Default: ["precision@3", "precision@5", "recall@5", "mrr"]

        Raises:
            ValueError: If dataset or documents have invalid format
        """
        self._validate_dataset(dataset)
        self._validate_documents(documents)

        self.dataset = dataset
        self.documents = documents
        self.metrics = metrics or ["precision@3", "precision@5", "recall@5", "mrr"]

        # Normalize search space with defaults
        self.search_space = self._normalize_search_space(search_space)

        # Generate all configs (Cartesian product)
        self.all_configs = self._generate_configs()

    def _normalize_search_space(self, search_space: Dict[str, List]) -> Dict[str, List]:
        """Fill in defaults for missing stages."""
        defaults = {
            "ingestion": [SimpleIngestion()],
            "storage": [SimpleStorage],  # Class, not instance - fresh per config
        }

        normalized = {}
        for stage in ["ingestion", "chunking", "embedding", "storage", "retriever"]:
            if stage in search_space:
                normalized[stage] = search_space[stage]
            elif stage in defaults:
                normalized[stage] = defaults[stage]
            # chunking, embedding, retriever must be provided by user

        return normalized

    def _generate_configs(self) -> List[Dict[str, Any]]:
        """Generate Cartesian product of all component combinations."""
        stages = ["ingestion", "chunking", "embedding", "storage", "retriever"]
        stage_lists = [self.search_space.get(s, [None]) for s in stages]

        configs = []
        for combo in product(*stage_lists):
            configs.append(dict(zip(stages, combo)))

        return configs

    def _validate_dataset(self, dataset: List[Dict]) -> None:
        """Validate evaluation dataset format."""
        if not dataset:
            raise ValueError("Dataset cannot be empty")

        for i, item in enumerate(dataset):
            if not isinstance(item, dict):
                raise ValueError(f"Dataset item {i} must be a dict, got {type(item)}")

            if "query" not in item:
                raise ValueError(f"Dataset item {i} missing required 'query' field")

            if "relevant_doc_ids" not in item:
                raise ValueError(f"Dataset item {i} missing required 'relevant_doc_ids' field")

            if not isinstance(item["relevant_doc_ids"], list):
                raise ValueError(
                    f"Dataset item {i} 'relevant_doc_ids' must be a list, "
                    f"got {type(item['relevant_doc_ids'])}"
                )

    def _validate_documents(self, documents: List[Dict]) -> None:
        """Validate document corpus format."""
        if not documents:
            raise ValueError("Documents cannot be empty")

        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} must be a dict, got {type(doc)}")

            if "doc_id" not in doc:
                raise ValueError(f"Document {i} missing required 'doc_id' field")

    def run(self, verbose: bool = True, continue_on_error: bool = False) -> ExperimentResults:
        """Execute all experiments.

        Args:
            verbose: Print progress and metrics
            continue_on_error: Skip failed configs instead of raising

        Returns:
            ExperimentResults object with all results
        """
        results = []

        if verbose:
            print(f"Running {len(self.all_configs)} experiments...")
            print()

        for i, config in enumerate(self.all_configs, 1):
            try:
                result = self._run_single_experiment(config, i)
                results.append(result)

                if verbose:
                    print(f"[{i}/{len(self.all_configs)}] {result.config_name}")
                    for metric, value in result.metrics.items():
                        print(f"  {metric}: {value:.4f}")
                    print()

            except Exception as e:
                if continue_on_error:
                    if verbose:
                        print(f"[{i}/{len(self.all_configs)}] FAILED: {e}")
                        print()
                    continue
                else:
                    raise

        return ExperimentResults(results)

    def _run_single_experiment(self, config: Dict, config_id: int) -> ExperimentResult:
        """Execute single configuration and return result."""
        # 1. Build components
        pipeline_config, component_params = self._build_components(config)

        # 2. Build RAGClient
        client = build_client(pipeline_config)

        # 3. Upload documents
        client.upload_documents(self.documents)

        # 4. Evaluate
        avg_metrics = self._evaluate_retrieval(client)

        # 5. Create artifact
        artifact = artifact_from_client(
            client=client, experiment_params=component_params, metrics=avg_metrics
        )

        # 6. Generate config name
        config_name = self._generate_config_name(pipeline_config)

        return ExperimentResult(
            config_id=config_id,
            config_name=config_name,
            artifact=artifact,
            metrics=avg_metrics,
            component_params=component_params,
        )

    def _build_components(self, config: Dict) -> Tuple[PipelineConfig, Dict[str, Any]]:
        """Build component instances from config dict."""
        ingestion = config["ingestion"]
        chunking = config["chunking"]
        embedding = config["embedding"]

        # Fresh storage per config (stateful - holds vectors in memory)
        storage_spec = config["storage"]
        if isinstance(storage_spec, type):
            # Class passed - instantiate it
            storage = storage_spec()
        else:
            # Instance passed - create fresh instance of same type
            storage = type(storage_spec)()

        # Build retriever with storage/embedding injection
        retriever_spec = config["retriever"]
        if isinstance(retriever_spec, dict):
            # Params dict - create SimpleRetriever with params
            retriever = SimpleRetriever(storage=storage, embedding=embedding, **retriever_spec)
        else:
            # Instance passed - need to create new instance with correct storage/embedding
            retriever = type(retriever_spec)(
                storage=storage,
                embedding=embedding,
                top_k=getattr(retriever_spec, "top_k", 5),
                reranker=getattr(retriever_spec, "reranker", None),
            )

        # Create PipelineConfig
        pipeline_config = PipelineConfig(
            ingestion=ingestion,
            chunking=chunking,
            embedding=embedding,
            storage=storage,
            retriever=retriever,
            name="",
        )

        # Extract params for metadata
        component_params = {
            "ingestion": {"type": type(ingestion).__name__},
            "chunking": self._extract_component_params(chunking),
            "embedding": self._extract_component_params(embedding),
            "storage": {"type": type(storage).__name__},
            "retriever": {
                "type": type(retriever).__name__,
                "top_k": getattr(retriever, "top_k", 5),
            },
        }

        return pipeline_config, component_params

    def _extract_component_params(self, component: Any) -> Dict[str, Any]:
        """Extract parameters from a component instance for metadata."""
        params = {"type": type(component).__name__}

        # Try to extract dataclass fields
        if hasattr(component, "__dataclass_fields__"):
            for field in fields(component):
                value = getattr(component, field.name, None)
                # Skip non-serializable fields
                if not callable(value) and not field.name.startswith("_"):
                    params[field.name] = value
        # Fallback to __dict__ for regular classes
        elif hasattr(component, "__dict__"):
            for key, value in component.__dict__.items():
                if not callable(value) and not key.startswith("_"):
                    params[key] = value

        return params

    def _evaluate_retrieval(self, client: RAGClient) -> Dict[str, float]:
        """Run evaluation on dataset and calculate metrics."""
        all_metrics = []

        # Extract k values from metric names (e.g., "precision@3" → 3)
        k_values = []
        for metric in self.metrics:
            if "@" in metric:
                try:
                    k_values.append(int(metric.split("@")[1]))
                except (IndexError, ValueError):
                    pass
        k_values = list(set(k_values)) or [3, 5]

        # Evaluate each query
        for item in self.dataset:
            query = item["query"]
            relevant_ids = item["relevant_doc_ids"]

            # Retrieve
            results = client.retrieve(query)
            retrieved_ids = [r.get("doc_id") for r in results if "doc_id" in r]

            # Calculate metrics for this query
            query_metrics = MetricsCalculator.calculate_all_metrics(
                retrieved_ids, relevant_ids, k_values
            )
            all_metrics.append(query_metrics)

        # Average across all queries
        avg_metrics = {}
        for metric in self.metrics:
            values = [m.get(metric, 0.0) for m in all_metrics]
            avg_metrics[metric] = sum(values) / len(values) if values else 0.0

        return avg_metrics

    def _generate_config_name(self, cfg: PipelineConfig) -> str:
        """Generate human-readable config name."""
        return (
            f"{type(cfg.ingestion).__name__} | "
            f"{type(cfg.chunking).__name__} | "
            f"{type(cfg.embedding).__name__} | "
            f"{type(cfg.storage).__name__} | "
            f"{type(cfg.retriever).__name__}"
        )
