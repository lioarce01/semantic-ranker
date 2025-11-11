"""
RAG pipeline with semantic reranking.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

import torch

from ..models.cross_encoder import CrossEncoderModel
from .retriever import BiEncoderRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline with retrieval and reranking.

    Two-stage approach:
    1. Fast bi-encoder retrieves top-k candidates
    2. Cross-encoder reranks candidates for precision
    """

    def __init__(
        self,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: Optional[str] = None,
        top_k_retrieval: int = 50,
        top_k_rerank: int = 5,
        device: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.

        Args:
            retriever_model: Bi-encoder model for retrieval
            reranker_model: Cross-encoder model for reranking (path or None)
            top_k_retrieval: Number of documents to retrieve
            top_k_rerank: Number of documents to rerank and return
            device: Device to use
        """
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize retriever
        logger.info("Initializing RAG pipeline")
        self.retriever = BiEncoderRetriever(
            model_name=retriever_model,
            device=self.device
        )

        # Initialize reranker
        self.reranker = None
        if reranker_model:
            logger.info(f"Loading reranker from {reranker_model}")
            self.reranker = CrossEncoderModel.load(reranker_model)
            self.reranker.model.to(self.device)
            self.reranker.model.eval()

        logger.info("RAG pipeline initialized")

    def index_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ):
        """
        Index documents for retrieval.

        Args:
            documents: List of documents
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        """
        self.retriever.index_documents(
            documents,
            batch_size=batch_size,
            show_progress=show_progress
        )

    def retrieve_and_rerank(
        self,
        query: str,
        top_k_retrieval: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        return_scores: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank documents for a query.

        Args:
            query: Query string
            top_k_retrieval: Override default retrieval count
            top_k_rerank: Override default rerank count
            return_scores: Whether to return scores

        Returns:
            List of ranked documents
        """
        top_k_retrieval = top_k_retrieval or self.top_k_retrieval
        top_k_rerank = top_k_rerank or self.top_k_rerank

        # Step 1: Retrieve candidates
        candidates = self.retriever.retrieve(query, top_k=top_k_retrieval)

        # Step 2: Rerank if reranker available
        if self.reranker:
            # Extract documents
            docs = [doc for _, doc, _ in candidates]

            # Rerank
            scores = self.reranker.predict([query] * len(docs), docs)

            # Create ranked list
            reranked = [
                (idx, doc, score)
                for (idx, doc, _), score in zip(candidates, scores)
            ]

            # Sort by reranker score
            reranked.sort(key=lambda x: x[2], reverse=True)

            # Take top-k
            results = reranked[:top_k_rerank]
        else:
            # No reranker, use retriever scores
            results = candidates[:top_k_rerank]

        # Format results
        formatted_results = []
        for idx, doc, score in results:
            result = {
                'document': doc,
                'index': idx
            }
            if return_scores:
                result['score'] = score

            formatted_results.append(result)

        return formatted_results

    def batch_retrieve_and_rerank(
        self,
        queries: List[str],
        top_k_retrieval: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        batch_size: int = 32
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve and rerank for multiple queries.

        Args:
            queries: List of queries
            top_k_retrieval: Override default retrieval count
            top_k_rerank: Override default rerank count
            batch_size: Batch size for reranking

        Returns:
            List of results, one per query
        """
        top_k_retrieval = top_k_retrieval or self.top_k_retrieval
        top_k_rerank = top_k_rerank or self.top_k_rerank

        logger.info(f"Processing {len(queries)} queries")

        # Retrieve candidates for all queries
        all_candidates = self.retriever.retrieve_batch(
            queries,
            top_k=top_k_retrieval,
            batch_size=batch_size
        )

        # Rerank each query's candidates
        all_results = []

        for query, candidates in zip(queries, all_candidates):
            if self.reranker:
                # Extract documents
                docs = [doc for _, doc, _ in candidates]

                # Rerank
                scores = self.reranker.predict(
                    [query] * len(docs),
                    docs,
                    batch_size=batch_size
                )

                # Create ranked list
                reranked = [
                    (idx, doc, score)
                    for (idx, doc, _), score in zip(candidates, scores)
                ]

                # Sort by reranker score
                reranked.sort(key=lambda x: x[2], reverse=True)

                # Take top-k
                results = reranked[:top_k_rerank]
            else:
                results = candidates[:top_k_rerank]

            # Format results
            formatted_results = [
                {'document': doc, 'index': idx, 'score': score}
                for idx, doc, score in results
            ]

            all_results.append(formatted_results)

        return all_results

    def get_context_for_llm(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_scores: bool = False,
        separator: str = "\n\n"
    ) -> str:
        """
        Get formatted context string for LLM.

        Args:
            query: Query string
            top_k: Number of documents to include
            include_scores: Whether to include relevance scores
            separator: Separator between documents

        Returns:
            Formatted context string
        """
        results = self.retrieve_and_rerank(
            query,
            top_k_rerank=top_k or self.top_k_rerank,
            return_scores=True
        )

        # Format context
        context_parts = []

        for i, result in enumerate(results, 1):
            doc = result['document']
            score = result.get('score', 0)

            if include_scores:
                context_parts.append(f"[{i}] (score: {score:.4f})\n{doc}")
            else:
                context_parts.append(f"[{i}] {doc}")

        return separator.join(context_parts)

    def augment_prompt(
        self,
        query: str,
        prompt_template: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Augment a prompt with retrieved context.

        Args:
            query: User query
            prompt_template: Template with {context} and {query} placeholders
            top_k: Number of documents to include

        Returns:
            Augmented prompt
        """
        # Get context
        context = self.get_context_for_llm(query, top_k=top_k)

        # Default template
        if prompt_template is None:
            prompt_template = """Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""

        # Format prompt
        augmented = prompt_template.format(context=context, query=query)

        return augmented

    def save_pipeline(self, save_dir: str):
        """
        Save pipeline state.

        Args:
            save_dir: Directory to save pipeline
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save retriever index
        self.retriever.save_index(save_dir / "retriever")

        # Save reranker if present
        if self.reranker:
            self.reranker.save(str(save_dir / "reranker"))

        # Save config
        import json
        config = {
            'top_k_retrieval': self.top_k_retrieval,
            'top_k_rerank': self.top_k_rerank
        }

        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Pipeline saved to {save_dir}")

    @classmethod
    def load_pipeline(
        cls,
        load_dir: str,
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ) -> 'RAGPipeline':
        """
        Load pipeline from disk.

        Args:
            load_dir: Directory containing saved pipeline
            retriever_model: Retriever model name
            device: Device to use

        Returns:
            Loaded pipeline
        """
        load_dir = Path(load_dir)

        # Load config
        import json
        with open(load_dir / "config.json", 'r') as f:
            config = json.load(f)

        # Check if reranker exists
        reranker_path = load_dir / "reranker"
        reranker_model = str(reranker_path) if reranker_path.exists() else None

        # Create pipeline
        pipeline = cls(
            retriever_model=retriever_model,
            reranker_model=reranker_model,
            top_k_retrieval=config['top_k_retrieval'],
            top_k_rerank=config['top_k_rerank'],
            device=device
        )

        # Load retriever index
        pipeline.retriever.load_index(load_dir / "retriever")

        logger.info(f"Pipeline loaded from {load_dir}")
        return pipeline

    def benchmark(
        self,
        queries: List[str],
        num_runs: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark pipeline performance.

        Args:
            queries: Test queries
            num_runs: Number of runs to average

        Returns:
            Performance statistics
        """
        import time

        logger.info(f"Benchmarking with {len(queries)} queries, {num_runs} runs")

        retrieval_times = []
        reranking_times = []
        total_times = []

        for run in range(num_runs):
            for query in queries:
                # Time retrieval
                start = time.time()
                candidates = self.retriever.retrieve(query, top_k=self.top_k_retrieval)
                retrieval_time = time.time() - start
                retrieval_times.append(retrieval_time)

                # Time reranking
                if self.reranker:
                    docs = [doc for _, doc, _ in candidates]
                    start = time.time()
                    _ = self.reranker.predict([query] * len(docs), docs)
                    reranking_time = time.time() - start
                    reranking_times.append(reranking_time)
                    total_time = retrieval_time + reranking_time
                else:
                    reranking_times.append(0)
                    total_time = retrieval_time

                total_times.append(total_time)

        stats = {
            'avg_retrieval_time_ms': np.mean(retrieval_times) * 1000,
            'avg_reranking_time_ms': np.mean(reranking_times) * 1000,
            'avg_total_time_ms': np.mean(total_times) * 1000,
            'throughput_queries_per_sec': 1.0 / np.mean(total_times)
        }

        logger.info("Benchmark results:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.2f}")

        return stats
