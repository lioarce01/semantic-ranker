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
        graph_batch_size: int = 200,
        device: Optional[str] = None,
        use_knn: bool = True,
        k_neighbors: int = 15
    ):
        """
        Args:
            embedding_model: SentenceTransformer model name
            similarity_threshold: Minimum cosine similarity for edge creation (threshold mode)
            max_neighbors: Maximum neighbors per query node (threshold mode)
            graph_batch_size: Batch size for graph construction
            device: Device for embeddings (cuda/cpu)
            use_knn: Use k-NN construction instead of threshold-based (DQGAN mode)
            k_neighbors: Number of neighbors for k-NN mode (DQGAN)
        """
        self.similarity_threshold = similarity_threshold
        self.max_neighbors = max_neighbors
        self.graph_batch_size = graph_batch_size
        self.use_knn = use_knn
        self.k_neighbors = k_neighbors

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

        # Choose graph construction method
        if self.use_knn:
            edges, edge_weights = self._build_knn_graph(self.query_embeddings, queries)
        else:
            edges, edge_weights = self._build_threshold_graph(self.query_embeddings, queries)

        # Store edges for compatibility
        self.edges = edges
        self.edge_weights = edge_weights

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

    def _build_knn_graph(
        self,
        query_embeddings: torch.Tensor,
        queries: List[str]
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Build k-NN graph using k nearest neighbors for each query.

        DQGAN Phase 2: Guarantees dense graph with exactly k neighbors per node.

        Args:
            query_embeddings: Query embeddings [num_queries, embedding_dim]
            queries: List of query texts

        Returns:
            edges: List of [src, dst] edge pairs
            edge_weights: List of edge weights (cosine similarities)
        """
        edges = []
        edge_weights = []
        num_queries = len(queries)

        # Compute full similarity matrix in chunks to save memory
        chunk_size = min(self.graph_batch_size, num_queries)

        for i in range(0, num_queries, chunk_size):
            end_i = min(i + chunk_size, num_queries)
            chunk_embeddings_i = query_embeddings[i:end_i]

            # Compute similarities for this chunk against all queries
            similarities = torch.nn.functional.cosine_similarity(
                chunk_embeddings_i.unsqueeze(1),  # [chunk_size, 1, dim]
                query_embeddings.unsqueeze(0),     # [1, num_queries, dim]
                dim=2
            )  # [chunk_size, num_queries]

            # For each query in chunk, find k nearest neighbors
            for local_i in range(similarities.shape[0]):
                global_i = i + local_i
                sim_scores = similarities[local_i]

                # Mask self-similarity
                sim_scores[global_i] = -1

                # Get top-k neighbors
                k = min(self.k_neighbors, num_queries - 1)
                top_k_sims, top_k_indices = torch.topk(sim_scores, k=k, largest=True)

                # Add bidirectional edges
                for neighbor_idx, sim in zip(top_k_indices.tolist(), top_k_sims.tolist()):
                    # Add edge from i to neighbor
                    edges.append([global_i, neighbor_idx])
                    edge_weights.append(sim)

            # Clear chunk tensors to free memory
            del similarities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return edges, edge_weights

    def _build_threshold_graph(
        self,
        query_embeddings: torch.Tensor,
        queries: List[str]
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Build graph using threshold-based edge creation (original method).

        Args:
            query_embeddings: Query embeddings [num_queries, embedding_dim]
            queries: List of query texts

        Returns:
            edges: List of [src, dst] edge pairs
            edge_weights: List of edge weights (cosine similarities)
        """
        edges = []
        edge_weights = []

        # Process in chunks to avoid OOM with large query sets
        chunk_size = min(self.graph_batch_size, len(queries))

        for i in range(0, len(queries), chunk_size):
            end_i = min(i + chunk_size, len(queries))
            chunk_embeddings_i = query_embeddings[i:end_i]

            for j in range(i, len(queries), chunk_size):  # Start from i to avoid duplicate work
                end_j = min(j + chunk_size, len(queries))
                chunk_embeddings_j = query_embeddings[j:end_j]

                # Compute similarities for this chunk pair
                similarities_chunk = torch.nn.functional.cosine_similarity(
                    chunk_embeddings_i.unsqueeze(1),  # [chunk_i, 1, dim]
                    chunk_embeddings_j.unsqueeze(0),  # [1, chunk_j, dim]
                    dim=2
                )  # [chunk_i, chunk_j]

                # Process similarities for each query in chunk_i
                for local_i in range(similarities_chunk.shape[0]):
                    global_i = i + local_i
                    sim_scores = similarities_chunk[local_i]

                    # For diagonal chunks, mask self-similarity
                    if i == j:
                        sim_scores[local_i] = -1

                    # Apply threshold
                    valid_neighbors = (sim_scores >= self.similarity_threshold).nonzero(as_tuple=True)[0]
                    valid_neighbors_global = valid_neighbors + j  # Convert to global indices

                    if len(valid_neighbors) == 0:
                        continue

                    # Get similarities for valid neighbors
                    neighbor_sims = sim_scores[valid_neighbors]

                    # Sort by similarity and take top-k
                    sorted_indices = torch.argsort(neighbor_sims, descending=True)
                    top_k = min(self.max_neighbors, len(sorted_indices))
                    top_neighbors_local = valid_neighbors[sorted_indices[:top_k]]
                    top_neighbors_global = valid_neighbors_global[sorted_indices[:top_k]]
                    top_sims = neighbor_sims[sorted_indices[:top_k]]

                    # Add edges (only i->j where j > i to avoid duplicates)
                    for local_j, global_j, sim in zip(top_neighbors_local.tolist(), top_neighbors_global.tolist(), top_sims.tolist()):
                        if global_j > global_i:
                            edges.append([global_i, global_j])
                            edge_weights.append(sim)

                # Clear chunk tensors to free memory
                del similarities_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return edges, edge_weights

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
