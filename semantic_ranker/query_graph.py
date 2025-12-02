"""
Query graph construction for Query Graph Neural Reranking.

Builds semantic graphs over queries based on embedding similarity,
enabling cross-query knowledge transfer.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from collections import defaultdict


class QueryGraphBuilder:
    """Builds semantic query graphs using sentence embeddings."""

    def __init__(
        self,
        embedding_model: str = 'all-mpnet-base-v2',
        similarity_threshold: float = 0.7,
        max_neighbors: int = 10,
        device: Optional[str] = None
    ):
        """
        Args:
            embedding_model: SentenceTransformer model name
            similarity_threshold: Minimum cosine similarity for edge creation
            max_neighbors: Maximum neighbors per query node
            device: Device for embeddings (cuda/cpu)
        """
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.query_embeddings = None
        self.query_texts = []
        self.edges = []
        self.edge_weights = []

    def build_graph(
        self,
        queries: List[str],
        query_doc_relevance: Optional[Dict[int, List[int]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build query graph from query texts.

        Args:
            queries: List of query strings
            query_doc_relevance: Optional dict mapping query idx to relevant doc indices

        Returns:
            edge_index: [2, num_edges] tensor of edge indices
            edge_weights: [num_edges] tensor of edge weights (cosine similarities)
            query_embeddings: [num_queries, embedding_dim] query embeddings
        """
        self.query_texts = queries

        # Encode queries
        embeddings = self.encoder.encode(
            queries,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # Clone to make trainable (SentenceTransformer uses inference_mode)
        self.query_embeddings = embeddings.clone().detach().requires_grad_(True)

        # Compute pairwise cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            self.query_embeddings.unsqueeze(1),
            self.query_embeddings.unsqueeze(0),
            dim=2
        )

        # Build edges
        edges = []
        edge_weights = []

        for i in range(len(queries)):
            # Get similarities for query i
            sim_scores = similarities[i]

            # Mask self-similarity
            sim_scores[i] = -1

            # Apply threshold
            valid_neighbors = (sim_scores >= self.similarity_threshold).nonzero(as_tuple=True)[0]

            if len(valid_neighbors) == 0:
                continue

            # Sort by similarity and take top-k
            neighbor_sims = sim_scores[valid_neighbors]
            sorted_indices = torch.argsort(neighbor_sims, descending=True)
            top_neighbors = valid_neighbors[sorted_indices[:self.max_neighbors]]
            top_sims = neighbor_sims[sorted_indices[:self.max_neighbors]]

            # Add edges
            for j, sim in zip(top_neighbors.tolist(), top_sims.tolist()):
                edges.append([i, j])
                edge_weights.append(sim)

        # Bonus edges: queries with shared relevant documents
        if query_doc_relevance is not None:
            self._add_relevance_edges(query_doc_relevance, edges, edge_weights)

        # Convert to tensors
        if len(edges) == 0:
            # No edges found, create empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weights_tensor = torch.zeros(0, dtype=torch.float)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weights_tensor = torch.tensor(edge_weights, dtype=torch.float)

        self.edges = edges
        self.edge_weights = edge_weights

        return edge_index, edge_weights_tensor, self.query_embeddings

    def _add_relevance_edges(
        self,
        query_doc_relevance: Dict[int, List[int]],
        edges: List[List[int]],
        edge_weights: List[float]
    ):
        """Add edges between queries that share relevant documents."""
        # Build doc -> queries mapping
        doc_to_queries = defaultdict(list)
        for query_idx, doc_indices in query_doc_relevance.items():
            for doc_idx in doc_indices:
                doc_to_queries[doc_idx].append(query_idx)

        # Add edges for queries sharing docs
        existing_edges = set((e[0], e[1]) for e in edges)

        for doc_idx, query_indices in doc_to_queries.items():
            if len(query_indices) < 2:
                continue

            # Connect all queries that share this document
            for i in range(len(query_indices)):
                for j in range(i + 1, len(query_indices)):
                    q1, q2 = query_indices[i], query_indices[j]

                    # Skip if edge already exists
                    if (q1, q2) in existing_edges or (q2, q1) in existing_edges:
                        continue

                    # Add bidirectional edges with high weight (shared relevance)
                    edges.append([q1, q2])
                    edge_weights.append(0.9)  # High weight for shared relevance
                    edges.append([q2, q1])
                    edge_weights.append(0.9)

                    existing_edges.add((q1, q2))
                    existing_edges.add((q2, q1))

    def get_neighbor_queries(self, query_idx: int, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Get top-k most similar queries for a given query.

        Returns:
            List of (neighbor_idx, similarity, neighbor_text) tuples
        """
        if self.query_embeddings is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")

        neighbors = []
        for edge, weight in zip(self.edges, self.edge_weights):
            if edge[0] == query_idx:
                neighbor_idx = edge[1]
                neighbors.append((neighbor_idx, weight, self.query_texts[neighbor_idx]))

        # Sort by similarity and return top-k
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]
