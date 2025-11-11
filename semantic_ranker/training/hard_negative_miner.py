"""
Hard negative mining for improved reranker training.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    Mines hard negative examples using a bi-encoder.

    Hard negatives are documents that are semantically similar to the query
    but not relevant. These improve training by providing challenging examples.
    """

    def __init__(
        self,
        bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize hard negative miner.

        Args:
            bi_encoder_model: Bi-encoder model for retrieval
            device: Device to use (None for auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading bi-encoder: {bi_encoder_model}")

        self.bi_encoder = SentenceTransformer(bi_encoder_model, device=self.device)

        logger.info(f"Initialized HardNegativeMiner on {self.device}")

    def mine(
        self,
        queries: List[str],
        documents: List[str],
        positive_pairs: List[Tuple[int, int]],
        top_k: int = 10,
        batch_size: int = 32
    ) -> Dict[int, List[str]]:
        """
        Mine hard negatives for queries.

        Args:
            queries: List of queries
            documents: List of all documents
            positive_pairs: List of (query_idx, doc_idx) positive pairs
            top_k: Number of hard negatives to mine per query
            batch_size: Batch size for encoding

        Returns:
            Dictionary mapping query index to list of hard negative documents
        """
        logger.info(f"Mining hard negatives for {len(queries)} queries")

        # Build set of positive documents for each query
        positive_docs = {}
        for q_idx, d_idx in positive_pairs:
            if q_idx not in positive_docs:
                positive_docs[q_idx] = set()
            positive_docs[q_idx].add(d_idx)

        # Encode queries and documents
        logger.info("Encoding queries...")
        query_embeddings = self.bi_encoder.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True
        )

        logger.info("Encoding documents...")
        doc_embeddings = self.bi_encoder.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True
        )

        # Mine hard negatives
        hard_negatives = {}

        logger.info("Computing similarities and mining negatives...")
        for q_idx in tqdm(range(len(queries)), desc="Mining"):
            # Compute similarities
            query_emb = query_embeddings[q_idx].unsqueeze(0)
            similarities = util.cos_sim(query_emb, doc_embeddings)[0]

            # Get top-k most similar documents
            top_indices = torch.argsort(similarities, descending=True).cpu().numpy()

            # Filter out positives and select hard negatives
            negatives = []
            for doc_idx in top_indices:
                # Skip if this is a positive document
                if q_idx in positive_docs and doc_idx in positive_docs[q_idx]:
                    continue

                negatives.append(documents[doc_idx])

                if len(negatives) >= top_k:
                    break

            hard_negatives[q_idx] = negatives

        logger.info(f"Mined {sum(len(v) for v in hard_negatives.values())} hard negatives")
        return hard_negatives

    def create_training_triples(
        self,
        queries: List[str],
        documents: List[str],
        positive_pairs: List[Tuple[int, int]],
        num_negatives: int = 3,
        batch_size: int = 32
    ) -> List[Tuple[str, str, float]]:
        """
        Create training triples with hard negatives.

        Args:
            queries: List of queries
            documents: List of documents
            positive_pairs: List of (query_idx, doc_idx) positive pairs
            num_negatives: Number of negatives per positive
            batch_size: Batch size for encoding

        Returns:
            List of (query, document, label) triples
        """
        # Mine hard negatives
        hard_negatives = self.mine(
            queries,
            documents,
            positive_pairs,
            top_k=num_negatives,
            batch_size=batch_size
        )

        # Create triples
        triples = []

        for q_idx, d_idx in positive_pairs:
            query = queries[q_idx]
            positive_doc = documents[d_idx]

            # Add positive pair
            triples.append((query, positive_doc, 1.0))

            # Add hard negatives
            if q_idx in hard_negatives:
                for neg_doc in hard_negatives[q_idx]:
                    triples.append((query, neg_doc, 0.0))

        logger.info(f"Created {len(triples)} training triples")
        return triples

    def mine_from_dataset(
        self,
        dataset: List[Dict[str, any]],
        num_negatives: int = 3,
        batch_size: int = 32
    ) -> List[Tuple[str, str, float]]:
        """
        Mine hard negatives from a dataset.

        Expected dataset format:
        [{'query': str, 'positive': str}, ...]

        Args:
            dataset: List of query-positive pairs
            num_negatives: Number of negatives per query
            batch_size: Batch size

        Returns:
            List of training triples with hard negatives
        """
        # Extract queries and documents
        queries = [item['query'] for item in dataset]
        documents = list(set([item['positive'] for item in dataset]))

        # Create document index
        doc_to_idx = {doc: idx for idx, doc in enumerate(documents)}

        # Create positive pairs
        positive_pairs = []
        for q_idx, item in enumerate(dataset):
            d_idx = doc_to_idx[item['positive']]
            positive_pairs.append((q_idx, d_idx))

        # Mine hard negatives
        return self.create_training_triples(
            queries,
            documents,
            positive_pairs,
            num_negatives=num_negatives,
            batch_size=batch_size
        )

    def iterative_mining(
        self,
        queries: List[str],
        documents: List[str],
        positive_pairs: List[Tuple[int, int]],
        cross_encoder,
        num_rounds: int = 2,
        num_negatives: int = 3
    ) -> List[Tuple[str, str, float]]:
        """
        Iterative hard negative mining using trained cross-encoder.

        After initial training, use the cross-encoder to find even harder
        negatives that the model struggles with.

        Args:
            queries: List of queries
            documents: List of documents
            positive_pairs: Positive pairs
            cross_encoder: Trained cross-encoder model
            num_rounds: Number of mining rounds
            num_negatives: Negatives per query

        Returns:
            Training triples with iteratively mined negatives
        """
        logger.info(f"Starting iterative hard negative mining ({num_rounds} rounds)")

        all_triples = []

        for round_num in range(num_rounds):
            logger.info(f"\nRound {round_num + 1}/{num_rounds}")

            # Mine using bi-encoder
            hard_negatives = self.mine(
                queries,
                documents,
                positive_pairs,
                top_k=num_negatives * 3  # Get more candidates
            )

            # Re-rank with cross-encoder to find hardest negatives
            round_triples = []

            for q_idx, d_idx in tqdm(positive_pairs, desc="Re-ranking"):
                query = queries[q_idx]
                positive_doc = documents[d_idx]

                # Add positive
                round_triples.append((query, positive_doc, 1.0))

                # Score negatives with cross-encoder
                if q_idx in hard_negatives:
                    candidates = hard_negatives[q_idx]

                    # Get scores from cross-encoder
                    query_list = [query] * len(candidates)
                    scores = cross_encoder.predict(query_list, candidates)

                    # Sort by score (highest scores are hardest negatives)
                    scored_candidates = list(zip(candidates, scores))
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)

                    # Take top-k hardest
                    for neg_doc, _ in scored_candidates[:num_negatives]:
                        round_triples.append((query, neg_doc, 0.0))

            all_triples.extend(round_triples)
            logger.info(f"Round {round_num + 1}: {len(round_triples)} triples")

        logger.info(f"Iterative mining complete: {len(all_triples)} total triples")
        return all_triples

    def analyze_negatives(
        self,
        queries: List[str],
        hard_negatives: Dict[int, List[str]]
    ) -> Dict[str, float]:
        """
        Analyze quality of mined negatives.

        Args:
            queries: List of queries
            hard_negatives: Mined hard negatives

        Returns:
            Dictionary with statistics
        """
        num_queries_with_negatives = len(hard_negatives)
        total_negatives = sum(len(negs) for negs in hard_negatives.values())
        avg_negatives_per_query = total_negatives / num_queries_with_negatives if num_queries_with_negatives > 0 else 0

        stats = {
            'num_queries': len(queries),
            'num_queries_with_negatives': num_queries_with_negatives,
            'total_negatives': total_negatives,
            'avg_negatives_per_query': avg_negatives_per_query
        }

        logger.info("Hard negative statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return stats
