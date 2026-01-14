"""Experiment results container with ranking and analysis capabilities.

This module provides data structures for storing and analyzing experiment results,
including ranking by metrics and extracting the best pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag_select.experiment.artifact.rag_artifact import RAGArtifact


@dataclass
class ExperimentResult:
    """Single experiment result with configuration and metrics."""

    config_id: int  # Numeric ID (1-indexed)
    config_name: str  # Human-readable name
    artifact: RAGArtifact  # LangChain-compatible retriever
    metrics: Dict[str, float]  # Metric scores (e.g., {"precision@5": 0.4, "mrr": 1.0})
    component_params: Dict[str, Any]  # Component configuration details


class ExperimentResults:
    """Container for experiment results with ranking and analysis capabilities."""

    def __init__(self, results: List[ExperimentResult]):
        """Initialize with list of experiment results.

        Args:
            results: List of ExperimentResult objects
        """
        self.results = results
        self._ranked_results: Optional[List[ExperimentResult]] = None
        self._rank_metric: Optional[str] = None

    def rank(self, by: str, ascending: bool = False) -> "ExperimentResults":
        """Rank results by a specific metric.

        Args:
            by: Metric name (e.g., "precision@5", "mrr")
            ascending: Sort order (False = higher is better, True = lower is better)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If metric name not found in results

        Example:
            >>> results.rank(by="precision@5")
            >>> best = results.get_best_pipeline()
        """
        # Validate metric exists
        if not self.results:
            raise ValueError("Cannot rank empty results")

        # Check if metric exists in at least one result
        metric_found = any(by in result.metrics for result in self.results)
        if not metric_found:
            available_metrics = set()
            for result in self.results:
                available_metrics.update(result.metrics.keys())
            raise ValueError(
                f"Metric '{by}' not found in results. "
                f"Available metrics: {', '.join(sorted(available_metrics))}"
            )

        # Sort results by metric
        # Handle missing metrics by treating them as 0.0 for comparison
        self._ranked_results = sorted(
            self.results,
            key=lambda r: r.metrics.get(by, 0.0),
            reverse=not ascending
        )
        self._rank_metric = by

        return self

    def get_best_pipeline(self) -> RAGArtifact:
        """Get the best pipeline (top-ranked artifact).

        Returns:
            RAGArtifact of the best configuration

        Raises:
            RuntimeError: If rank() not called first
            ValueError: If results are empty

        Example:
            >>> results.rank(by="mrr")
            >>> best = results.get_best_pipeline()
        """
        if not self.results:
            raise ValueError("Cannot get best pipeline from empty results")

        if self._ranked_results is None:
            raise RuntimeError(
                "Must call rank() before get_best_pipeline(). "
                "Example: results.rank(by='precision@5').get_best_pipeline()"
            )

        return self._ranked_results[0].artifact

    def get_top_k_pipelines(self, k: int = 5) -> List[RAGArtifact]:
        """Get top K pipelines.

        Args:
            k: Number of top pipelines to return

        Returns:
            List of top K RAGArtifacts

        Raises:
            RuntimeError: If rank() not called first

        Example:
            >>> results.rank(by="precision@5")
            >>> top_3 = results.get_top_k_pipelines(k=3)
        """
        if self._ranked_results is None:
            raise RuntimeError(
                "Must call rank() before get_top_k_pipelines(). "
                "Example: results.rank(by='precision@5').get_top_k_pipelines(k=3)"
            )

        return [r.artifact for r in self._ranked_results[:k]]

    def summary(self, top_k: int = 5) -> str:
        """Format top K results as readable string.

        Args:
            top_k: Number of top results to include

        Returns:
            Formatted summary string

        Raises:
            RuntimeError: If rank() not called first

        Example:
            >>> results.rank(by="mrr")
            >>> print(results.summary(top_k=3))
        """
        if self._ranked_results is None:
            raise RuntimeError(
                "Must call rank() before summary(). "
                "Example: results.rank(by='precision@5').summary()"
            )

        lines = [
            f"Top {min(top_k, len(self._ranked_results))} Results (ranked by {self._rank_metric}):",
            "=" * 80,
        ]

        for i, result in enumerate(self._ranked_results[:top_k], 1):
            lines.append(f"\n{i}. {result.config_name}")
            lines.append(f"   Config ID: {result.config_id}")
            for metric, value in sorted(result.metrics.items()):
                lines.append(f"   {metric}: {value:.4f}")

        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Export results to pandas DataFrame.

        Returns:
            pandas.DataFrame if pandas is available, None otherwise

        Note:
            Requires pandas to be installed. Returns None with a warning if not available.

        Example:
            >>> df = results.to_dataframe()
            >>> if df is not None:
            ...     print(df.head())
        """
        try:
            import pandas as pd
        except ImportError:
            print("Warning: pandas not available. Install pandas to export to DataFrame.")
            return None

        # Flatten results into rows
        rows = []
        for result in self._ranked_results or self.results:
            row = {
                "config_id": result.config_id,
                "config_name": result.config_name,
            }

            # Add metrics
            row.update(result.metrics)

            # Flatten component params
            for stage, params in result.component_params.items():
                if isinstance(params, dict):
                    row[f"{stage}_type"] = params.get("type", "unknown")
                    # Add non-type params
                    for param_key, param_value in params.items():
                        if param_key != "type":
                            row[f"{stage}_{param_key}"] = param_value

            rows.append(row)

        return pd.DataFrame(rows)

    def __len__(self) -> int:
        """Return number of experiments."""
        return len(self.results)

    def __iter__(self):
        """Iterate over results (uses ranked if available, else original order)."""
        return iter(self._ranked_results if self._ranked_results is not None else self.results)

    def __getitem__(self, index: int) -> ExperimentResult:
        """Get result by index (uses ranked if available)."""
        if self._ranked_results is not None:
            return self._ranked_results[index]
        return self.results[index]
