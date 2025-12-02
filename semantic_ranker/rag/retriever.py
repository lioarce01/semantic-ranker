"""
Bi-encoder retriever for initial document retrieval.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

import torch
from sentence_transformers import SentenceTransformer, util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiEncoderRetriever:
    """
    Bi-encoder retriever for fast initial document retrieval.

    Uses embedding similarity to quickly retrieve candidate documents
    before reranking with a cross-encoder.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize bi-encoder retriever.

        Args:
            model_name: Sentence transformer model name
            device: Device to use (None for auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading bi-encoder: {model_name}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.corpus_embeddings = None
        self.corpus = None

        logger.info(f"Bi-encoder loaded on {self.device}")

    def index_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ):
        """
        Index documents by computing embeddings.

        Args:
            documents: List of documents to index
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        """
        logger.info(f"Indexing {len(documents)} documents")

        self.corpus = documents
        self.corpus_embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
            normalize_embeddings=True  # For cosine similarity
        )

        logger.info("Indexing complete")

    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, str, float]]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of (index, document, score) tuples
        """
        if self.corpus_embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")

        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        # Compute similarities
        similarities = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        # Get top-k
        top_k = min(top_k, len(self.corpus))
        top_indices = torch.argsort(similarities, descending=True)[:top_k]

        # Return results
        results = [
            (idx.item(), self.corpus[idx], similarities[idx].item())
            for idx in top_indices
        ]

        return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[List[Tuple[int, str, float]]]:
        """
        Retrieve documents for multiple queries.

        Args:
            queries: List of queries
            top_k: Number of documents per query
            batch_size: Batch size for encoding queries

        Returns:
            List of results, one per query
        """
        if self.corpus_embeddings is None:
            raise ValueError("No documents indexed. Call index_documents() first.")

        logger.info(f"Retrieving for {len(queries)} queries")

        # Encode all queries
        query_embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        # Compute similarities for all queries at once
        similarities = util.cos_sim(query_embeddings, self.corpus_embeddings)

        # Get top-k for each query
        all_results = []
        top_k = min(top_k, len(self.corpus))

        for q_idx in range(len(queries)):
            query_sims = similarities[q_idx]
            top_indices = torch.argsort(query_sims, descending=True)[:top_k]

            results = [
                (idx.item(), self.corpus[idx], query_sims[idx].item())
                for idx in top_indices
            ]
            all_results.append(results)

        return all_results

    def save_index(self, save_path: str):
        """
        Save indexed corpus and embeddings.

        Args:
            save_path: Path to save index
        """
        if self.corpus_embeddings is None:
            raise ValueError("No index to save")

        import json
        from pathlib import Path

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save corpus as JSON (safer than pickle)
        with open(save_path / "corpus.json", "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=2)

        # Save embeddings
        torch.save(self.corpus_embeddings, save_path / "embeddings.pt")

        logger.info(f"Index saved to {save_path}")

    def load_index(self, load_path: str):
        """
        Load indexed corpus and embeddings.

        Args:
            load_path: Path to load index from
        """
        import json
        from pathlib import Path

        load_path = Path(load_path)

        # Load corpus (with backward compatibility for pickle files)
        json_path = load_path / "corpus.json"
        pkl_path = load_path / "corpus.pkl"

        if json_path.exists():
            # Load from JSON (preferred)
            with open(json_path, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)
        elif pkl_path.exists():
            # Backward compatibility: load from pickle and migrate to JSON
            logger.warning("Loading from legacy pickle format. Migrating to JSON...")
            import pickle
            with open(pkl_path, "rb") as f:
                self.corpus = pickle.load(f)

            # Save as JSON and remove pickle file
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.corpus, f, ensure_ascii=False, indent=2)
            pkl_path.unlink()  # Delete old pickle file
            logger.info("Migration to JSON complete")
        else:
            raise FileNotFoundError(f"No corpus file found in {load_path}")

        # Load embeddings (with weights_only=True for security)
        self.corpus_embeddings = torch.load(
            load_path / "embeddings.pt",
            map_location=self.device,
            weights_only=True
        )

        logger.info(f"Index loaded from {load_path}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)

    def get_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts
            batch_size: Batch size

        Returns:
            Array of embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
