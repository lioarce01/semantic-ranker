"""
Evaluator for ranking models.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import torch
from tqdm import tqdm

from ..models.cross_encoder import CrossEncoderModel
from .metrics import (
    compute_ndcg,
    compute_mrr,
    compute_map,
    compute_hit_rate,
    compute_all_metrics,
    aggregate_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RankerEvaluator:
    """
    Evaluates ranking model performance.

    Computes standard IR metrics like NDCG@k, MRR@k, MAP, Hit Rate.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[CrossEncoderModel] = None,
        device: Optional[str] = None
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to saved model (if loading from disk)
            model: Loaded model instance (if already in memory)
            device: Device to use
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.model = model
        elif model_path is not None:
            logger.info(f"Loading model from {model_path}")
            self.model = CrossEncoderModel.load(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")

        self.model.model.to(self.device)
        self.model.model.eval()

        logger.info(f"Evaluator initialized on {self.device}")

    def evaluate(
        self,
        test_data: List[Dict[str, Any]],
        metrics: List[str] = ["ndcg@10", "mrr@10", "map@10"],
        batch_size: int = 32,
        query_batch_size: int = 8,
        return_per_query: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate model on test data with optimized batch processing.

        Args:
            test_data: Test dataset
            metrics: List of metrics to compute
            batch_size: Batch size for model inference
            query_batch_size: Number of queries to process simultaneously
            return_per_query: Whether to return per-query metrics

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {len(test_data)} queries (query_batch_size={query_batch_size})")

        # Parse metrics
        metric_configs = self._parse_metrics(metrics)

        # Evaluate queries in batches
        query_metrics = []

        for batch_start in range(0, len(test_data), query_batch_size):
            batch_end = min(batch_start + query_batch_size, len(test_data))
            query_batch = test_data[batch_start:batch_end]

            # Prepare all (query, doc) pairs for this batch
            all_pairs = []
            batch_metadata = []

            for item in query_batch:
                query = item.get('query', '')
                documents = item.get('documents', [])
                labels = item.get('labels', [])

                if not query or not documents:
                    continue

                start_idx = len(all_pairs)
                all_pairs.extend([(query, doc) for doc in documents])
                batch_metadata.append({
                    'start_idx': start_idx,
                    'end_idx': len(all_pairs),
                    'labels': labels
                })

            # Skip if no valid queries in batch
            if not all_pairs:
                continue

            # Single batched inference for all (query, doc) pairs
            queries_list, docs_list = zip(*all_pairs)
            all_scores = self.model.predict(
                list(queries_list),
                list(docs_list),
                batch_size=batch_size
            )

            # Compute metrics for each query in parallel
            with ThreadPoolExecutor(max_workers=query_batch_size) as executor:
                futures = []
                for meta in batch_metadata:
                    scores = all_scores[meta['start_idx']:meta['end_idx']]
                    labels = meta['labels']

                    future = executor.submit(
                        self._compute_metrics_for_query,
                        scores, labels, metric_configs
                    )
                    futures.append(future)

                # Collect results
                for future in futures:
                    query_result = future.result()
                    query_metrics.append(query_result)

        # Aggregate across queries
        aggregated_metrics = aggregate_metrics(query_metrics)

        logger.info("Evaluation results:")
        for metric_name, value in aggregated_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        if return_per_query:
            return {
                'aggregated': aggregated_metrics,
                'per_query': query_metrics
            }

        return aggregated_metrics

    def _compute_metrics_for_query(
        self,
        scores: List[float],
        labels: List[int],
        metric_configs: List[Tuple[str, Optional[int]]]
    ) -> Dict[str, float]:
        """Compute all metrics for a single query (parallelizable)

        Args:
            scores: Prediction scores for documents
            labels: Relevance labels for documents
            metric_configs: List of (metric_name, k) tuples

        Returns:
            Dictionary of metric values
        """
        # Rank documents by score
        ranked_indices = np.argsort(scores)[::-1]
        ranked_labels = [labels[i] for i in ranked_indices]

        # Compute all metrics
        query_result = {}
        for metric_name, k in metric_configs:
            if metric_name == 'ndcg':
                score = compute_ndcg(ranked_labels, k)
            elif metric_name == 'mrr':
                score = compute_mrr(ranked_labels, k)
            elif metric_name == 'map':
                score = compute_map(ranked_labels, k)
            elif metric_name == 'hit_rate':
                score = compute_hit_rate(ranked_labels, k)
            else:
                continue

            metric_key = f"{metric_name}@{k}" if k else metric_name
            query_result[metric_key] = score

        return query_result

    def _parse_metrics(
        self,
        metrics: List[str]
    ) -> List[Tuple[str, Optional[int]]]:
        """
        Parse metric strings like 'ndcg@10' into ('ndcg', 10).

        Args:
            metrics: List of metric strings

        Returns:
            List of (metric_name, k) tuples
        """
        parsed = []

        for metric in metrics:
            if '@' in metric:
                name, k = metric.split('@')
                parsed.append((name, int(k)))
            else:
                parsed.append((metric, None))

        return parsed

    def evaluate_retrieval_rerank(
        self,
        queries: List[str],
        corpus: List[str],
        relevance_qrels: Dict[int, List[int]],
        retriever_top_k: int = 100,
        rerank_top_k: int = 10,
        retriever_model: Optional[str] = None,
        cache_corpus: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate full retrieval + reranking pipeline with corpus caching.

        Args:
            queries: List of queries
            corpus: Document corpus
            relevance_qrels: Ground truth relevance (query_idx -> list of relevant doc indices)
            retriever_top_k: Number of documents to retrieve
            rerank_top_k: Number of documents to rerank
            retriever_model: Bi-encoder model for initial retrieval
            cache_corpus: Cache corpus embeddings for faster evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating retrieval + reranking pipeline")

        # Load retriever if needed
        if retriever_model:
            from sentence_transformers import SentenceTransformer
            retriever = SentenceTransformer(retriever_model)
            logger.info(f"Loaded retriever: {retriever_model}")

            # Encode corpus (with optional caching)
            if cache_corpus and hasattr(self, '_cached_corpus_embeddings'):
                logger.info("Using cached corpus embeddings")
                corpus_embeddings = self._cached_corpus_embeddings
            else:
                logger.info("🔄 Encoding corpus...")
                corpus_embeddings = retriever.encode(
                    corpus,
                    show_progress_bar=True,
                    convert_to_tensor=True
                )
                if cache_corpus:
                    self._cached_corpus_embeddings = corpus_embeddings
                    logger.info("Corpus embeddings cached for future use")
        else:
            # Skip retrieval step, use all documents
            retriever = None

        # Evaluate each query
        all_metrics = []

        for q_idx, query in enumerate(tqdm(queries, desc="Evaluating pipeline")):
            # Retrieval step
            if retriever:
                query_embedding = retriever.encode(
                    query,
                    convert_to_tensor=True
                )

                # Compute similarities
                from sentence_transformers import util
                similarities = util.cos_sim(query_embedding, corpus_embeddings)[0]

                # Get top-k
                top_indices = torch.argsort(similarities, descending=True)[:retriever_top_k]
                candidate_docs = [corpus[i] for i in top_indices.cpu().numpy()]
            else:
                candidate_docs = corpus[:retriever_top_k]
                top_indices = list(range(len(candidate_docs)))

            # Reranking step
            scores = self.model.predict(
                [query] * len(candidate_docs),
                candidate_docs
            )

            # Sort by score
            reranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:rerank_top_k]

            # Map back to corpus indices
            reranked_corpus_indices = [top_indices[i].item() if retriever else top_indices[i]
                                      for i in reranked_indices]

            # Get relevance labels
            if q_idx in relevance_qrels:
                relevant_docs = set(relevance_qrels[q_idx])
                labels = [1 if idx in relevant_docs else 0
                         for idx in reranked_corpus_indices]
            else:
                labels = [0] * len(reranked_corpus_indices)

            # Compute metrics
            query_metrics = compute_all_metrics(
                labels,
                k_values=[1, 3, 5, 10]
            )
            all_metrics.append(query_metrics)

        # Aggregate
        aggregated = aggregate_metrics(all_metrics)

        logger.info("Pipeline evaluation results:")
        for metric_name, value in aggregated.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        return aggregated

    def compare_with_baseline(
        self,
        test_data: List[Dict[str, Any]],
        baseline_scores: List[List[float]],
        metrics: List[str] = ["ndcg@10", "mrr@10"]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model performance with a baseline.

        Args:
            test_data: Test dataset
            baseline_scores: Pre-computed baseline scores
            metrics: Metrics to compute

        Returns:
            Dictionary with 'model' and 'baseline' results
        """
        logger.info("Comparing with baseline")

        # Evaluate model
        model_results = self.evaluate(test_data, metrics)

        # Evaluate baseline
        metric_configs = self._parse_metrics(metrics)
        baseline_metrics = []

        for item, scores in zip(test_data, baseline_scores):
            relevance_labels = item.get('labels', [])

            # Sort by baseline scores
            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )
            ranked_labels = [relevance_labels[i] for i in ranked_indices]

            # Compute metrics
            query_result = {}
            for metric_name, k in metric_configs:
                if metric_name == 'ndcg':
                    score = compute_ndcg(ranked_labels, k)
                elif metric_name == 'mrr':
                    score = compute_mrr(ranked_labels, k)
                elif metric_name == 'map':
                    score = compute_map(ranked_labels, k)
                elif metric_name == 'hit_rate':
                    score = compute_hit_rate(ranked_labels, k)
                else:
                    continue

                metric_key = f"{metric_name}@{k}" if k else metric_name
                query_result[metric_key] = score

            baseline_metrics.append(query_result)

        baseline_results = aggregate_metrics(baseline_metrics)

        # Compare
        comparison = {
            'model': model_results,
            'baseline': baseline_results,
            'improvement': {}
        }

        for metric in model_results.keys():
            if metric in baseline_results:
                model_val = model_results[metric]
                baseline_val = baseline_results[metric]
                improvement = ((model_val - baseline_val) / baseline_val * 100
                              if baseline_val > 0 else 0)
                comparison['improvement'][metric] = improvement

        logger.info("\nComparison results:")
        for metric in model_results.keys():
            logger.info(f"  {metric}:")
            logger.info(f"    Model: {model_results[metric]:.4f}")
            logger.info(f"    Baseline: {baseline_results[metric]:.4f}")
            if metric in comparison['improvement']:
                logger.info(f"    Improvement: {comparison['improvement'][metric]:.2f}%")

        return comparison

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str
    ):
        """
        Save evaluation results to file.

        Args:
            results: Results dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")
