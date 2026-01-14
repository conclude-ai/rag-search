"""Metric calculations for RAG retrieval evaluation.

This module provides metric calculations for evaluating retrieval quality,
including Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
"""

from __future__ import annotations

from typing import Dict, List


class MetricsCalculator:
    """Calculate retrieval evaluation metrics."""

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Calculate Precision@K metric.

        Precision@K measures what fraction of the top-k retrieved documents are relevant.

        Args:
            retrieved_ids: List of retrieved document IDs (in ranking order)
            relevant_ids: List of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision@K score (0.0 to 1.0)

        Example:
            >>> MetricsCalculator.precision_at_k(["doc_1", "doc_2", "doc_3"], ["doc_1"], 3)
            0.3333333333333333
        """
        if k == 0:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        if not retrieved_k:
            return 0.0
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
        return hits / k

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Calculate Recall@K metric.

        Recall@K measures what fraction of relevant documents were found in the top-k results.

        Args:
            retrieved_ids: List of retrieved document IDs (in ranking order)
            relevant_ids: List of ground truth relevant document IDs
            k: Number of top results to consider

        Returns:
            Recall@K score (0.0 to 1.0+, can exceed 1.0 if multiple relevant docs exist)

        Example:
            >>> MetricsCalculator.recall_at_k(["doc_1", "doc_2", "doc_3"], ["doc_1"], 3)
            1.0
        """
        if not relevant_ids:
            return 0.0
        retrieved_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR) metric.

        MRR measures the reciprocal rank of the first relevant document.
        Returns 1/rank where rank is the position of the first relevant doc (1-indexed).

        Args:
            retrieved_ids: List of retrieved document IDs (in ranking order)
            relevant_ids: List of ground truth relevant document IDs

        Returns:
            MRR score (0.0 to 1.0)

        Example:
            >>> MetricsCalculator.mrr(["doc_1", "doc_2", "doc_3"], ["doc_2"])
            0.5
        """
        for idx, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / idx
        return 0.0

    @staticmethod
    def calculate_all_metrics(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k_values: List[int] = [3, 5],
    ) -> Dict[str, float]:
        """Calculate all metrics at once for efficiency.

        Args:
            retrieved_ids: List of retrieved document IDs (in ranking order)
            relevant_ids: List of ground truth relevant document IDs
            k_values: List of k values for precision@k and recall@k (default: [3, 5])

        Returns:
            Dictionary with metric names as keys (e.g., "precision@3", "mrr")
            and scores as values

        Example:
            >>> MetricsCalculator.calculate_all_metrics(
            ...     ["doc_1", "doc_2"], ["doc_1"], k_values=[3, 5]
            ... )
            {'precision@3': 0.33..., 'precision@5': 0.2, 'recall@3': 1.0, 'recall@5': 1.0, 'mrr': 1.0}
        """
        metrics = {}

        # Calculate precision and recall for each k
        for k in k_values:
            metrics[f"precision@{k}"] = MetricsCalculator.precision_at_k(
                retrieved_ids, relevant_ids, k
            )
            metrics[f"recall@{k}"] = MetricsCalculator.recall_at_k(
                retrieved_ids, relevant_ids, k
            )

        # Calculate MRR (doesn't depend on k)
        metrics["mrr"] = MetricsCalculator.mrr(retrieved_ids, relevant_ids)

        return metrics
