"""
Evaluation metrics for information retrieval and ranking.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_ndcg(
    relevance_scores: List[float],
    k: Optional[int] = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k).

    NDCG measures the quality of ranking by considering both relevance
    and position. Higher relevant documents should appear earlier.

    Args:
        relevance_scores: Relevance scores in ranked order
        k: Cut-off rank (None for all)

    Returns:
        NDCG score
    """
    if len(relevance_scores) == 0:
        return 0.0

    if k is not None:
        relevance_scores = relevance_scores[:k]

    # Discounted Cumulative Gain
    dcg = sum(
        (2 ** rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(relevance_scores)
    )

    # Ideal DCG (sort by relevance)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum(
        (2 ** rel - 1) / np.log2(i + 2)
        for i, rel in enumerate(ideal_scores)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_mrr(
    relevance_labels: List[int],
    k: Optional[int] = None
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR@k).

    MRR measures how early the first relevant document appears.
    Higher MRR means relevant documents appear earlier.

    Args:
        relevance_labels: Binary relevance labels in ranked order (1=relevant, 0=not)
        k: Cut-off rank (None for all)

    Returns:
        MRR score (reciprocal rank of first relevant doc)
    """
    if len(relevance_labels) == 0:
        return 0.0

    if k is not None:
        relevance_labels = relevance_labels[:k]

    # Find first relevant document
    for i, label in enumerate(relevance_labels):
        if label > 0:
            return 1.0 / (i + 1)

    return 0.0


def compute_map(
    relevance_labels: List[int],
    k: Optional[int] = None
) -> float:
    """
    Compute Mean Average Precision (MAP@k).

    MAP measures precision at each relevant document position,
    then averages. Rewards ranking all relevant docs highly.

    Args:
        relevance_labels: Binary relevance labels in ranked order
        k: Cut-off rank (None for all)

    Returns:
        AP score
    """
    if len(relevance_labels) == 0:
        return 0.0

    if k is not None:
        relevance_labels = relevance_labels[:k]

    num_relevant = sum(relevance_labels)
    if num_relevant == 0:
        return 0.0

    precision_at_k = []
    num_relevant_seen = 0

    for i, label in enumerate(relevance_labels):
        if label > 0:
            num_relevant_seen += 1
            precision = num_relevant_seen / (i + 1)
            precision_at_k.append(precision)

    return sum(precision_at_k) / num_relevant


def compute_hit_rate(
    relevance_labels: List[int],
    k: Optional[int] = None
) -> float:
    """
    Compute Hit Rate@k (Recall@k).

    Hit rate measures whether at least one relevant document
    appears in the top-k results.

    Args:
        relevance_labels: Binary relevance labels in ranked order
        k: Cut-off rank (None for all)

    Returns:
        1.0 if any relevant doc in top-k, else 0.0
    """
    if len(relevance_labels) == 0:
        return 0.0

    if k is not None:
        relevance_labels = relevance_labels[:k]

    return 1.0 if any(label > 0 for label in relevance_labels) else 0.0


def compute_precision_at_k(
    relevance_labels: List[int],
    k: int
) -> float:
    """
    Compute Precision@k.

    Precision measures the fraction of relevant documents in top-k.

    Args:
        relevance_labels: Binary relevance labels in ranked order
        k: Cut-off rank

    Returns:
        Precision@k score
    """
    if len(relevance_labels) == 0 or k == 0:
        return 0.0

    top_k = relevance_labels[:k]
    return sum(top_k) / k


def compute_recall_at_k(
    relevance_labels: List[int],
    k: int
) -> float:
    """
    Compute Recall@k.

    Recall measures the fraction of all relevant documents found in top-k.

    Args:
        relevance_labels: Binary relevance labels in ranked order
        k: Cut-off rank

    Returns:
        Recall@k score
    """
    if len(relevance_labels) == 0:
        return 0.0

    total_relevant = sum(relevance_labels)
    if total_relevant == 0:
        return 0.0

    top_k = relevance_labels[:k]
    return sum(top_k) / total_relevant


def compute_f1_at_k(
    relevance_labels: List[int],
    k: int
) -> float:
    """
    Compute F1@k.

    F1 is the harmonic mean of precision and recall.

    Args:
        relevance_labels: Binary relevance labels in ranked order
        k: Cut-off rank

    Returns:
        F1@k score
    """
    precision = compute_precision_at_k(relevance_labels, k)
    recall = compute_recall_at_k(relevance_labels, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def compute_all_metrics(
    relevance_labels: List[int],
    relevance_scores: Optional[List[float]] = None,
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Compute all ranking metrics.

    Args:
        relevance_labels: Binary relevance labels in ranked order
        relevance_scores: Graded relevance scores (for NDCG)
        k_values: List of k values to evaluate

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    for k in k_values:
        metrics[f'ndcg@{k}'] = compute_ndcg(
            relevance_scores or relevance_labels, k
        )
        metrics[f'mrr@{k}'] = compute_mrr(relevance_labels, k)
        metrics[f'map@{k}'] = compute_map(relevance_labels, k)
        metrics[f'hit_rate@{k}'] = compute_hit_rate(relevance_labels, k)
        metrics[f'precision@{k}'] = compute_precision_at_k(relevance_labels, k)
        metrics[f'recall@{k}'] = compute_recall_at_k(relevance_labels, k)
        metrics[f'f1@{k}'] = compute_f1_at_k(relevance_labels, k)

    return metrics


def aggregate_metrics(
    query_metrics: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple queries.

    Args:
        query_metrics: List of metric dictionaries, one per query

    Returns:
        Dictionary with averaged metrics
    """
    if len(query_metrics) == 0:
        return {}

    # Get all metric names
    metric_names = query_metrics[0].keys()

    # Average each metric
    aggregated = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in query_metrics if metric_name in m]
        aggregated[metric_name] = np.mean(values) if values else 0.0

    return aggregated


class RankingMetrics:
    """
    Class wrapper for ranking metrics computation.
    Provides a consistent interface for evaluation.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def ndcg_at_k(self, relevance_labels, relevance_scores=None, k=None) -> float:
        """Compute NDCG@k."""
        # For NDCG, use relevance_scores (predicted scores) if available, otherwise labels
        scores_to_use = relevance_scores if relevance_scores is not None else relevance_labels
        return compute_ndcg(scores_to_use, k)

    def mrr_at_k(self, relevance_labels, relevance_scores=None, k=None) -> float:
        """Compute MRR@k."""
        # For MRR, always use relevance_labels (binary relevance)
        return compute_mrr(relevance_labels, k)

    def map_at_k(self, relevance_labels: List[int], k: int) -> float:
        """Compute MAP@k."""
        return compute_map(relevance_labels, k)

    def precision_at_k(self, relevance_labels: List[int], k: int) -> float:
        """Compute Precision@k."""
        return compute_precision_at_k(relevance_labels, k)

    def recall_at_k(self, relevance_labels: List[int], k: int) -> float:
        """Compute Recall@k."""
        return compute_recall_at_k(relevance_labels, k)

    def hit_rate_at_k(self, relevance_labels: List[int], k: int) -> float:
        """Compute Hit Rate@k."""
        return compute_hit_rate(relevance_labels, k)

    def f1_at_k(self, relevance_labels: List[int], k: int) -> float:
        """Compute F1@k."""
        return compute_f1_at_k(relevance_labels, k)

    def compute_all_metrics(
        self,
        relevance_labels: List[int],
        relevance_scores: Optional[List[float]] = None,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Compute all metrics."""
        return compute_all_metrics(relevance_labels, relevance_scores, k_values)
