# Query Graph Neural Reranking (QG-Rerank)

## Overview

QG-Rerank is a novel reranking approach that combines cross-encoder relevance scoring with Graph Neural Networks (GNN) operating over query graphs, enabling cross-query knowledge transfer and improved generalization.

## Key Innovation

**First GNN-based reranker that builds graphs over queries instead of documents.**

Existing approaches (G-RAG, GNRR) build graphs over documents or corpus. QG-Rerank uniquely constructs semantic graphs where nodes are queries, enabling the model to learn from relationships between similar information needs.

## Algorithm Components

### 1. Query Graph Construction

Builds a semantic graph where:
- **Nodes**: Unique queries in training set
- **Edges**: Created between semantically similar queries using:
  - Cosine similarity of query embeddings (SentenceTransformer: all-mpnet-base-v2)
  - Shared relevant documents (queries that retrieve the same relevant docs)
- **Edge weights**: Cosine similarity scores

**Parameters:**
- `similarity_threshold: 0.7` - Minimum similarity for edge creation
- `max_neighbors: 10` - Maximum edges per query node

### 2. Graph Neural Network

2-layer Graph Convolutional Network (GCN) that performs message passing over the query graph:

```
Input: 768-dim query embeddings (from SentenceTransformer)
  ↓
GCN Layer 1: 768 → 256 (+ ReLU + LayerNorm + Dropout)
  ↓
GCN Layer 2: 256 → 128 (+ LayerNorm)
  ↓
Output: 128-dim refined query embeddings
```

Message passing enables each query to aggregate information from semantically related queries.

### 3. Cross-Encoder Integration

The GNN query embeddings are integrated with the cross-encoder via attention mechanism:

1. Cross-encoder processes query-document pairs → hidden states
2. GNN provides refined query representations
3. Attention layer combines both signals
4. Final linear layer predicts relevance score

### 4. Multi-Task Loss

Training optimizes three complementary objectives:

**L_total = L_BCE + λ_contrastive * L_contrastive + λ_rank * L_rank**

- **L_BCE**: Binary cross-entropy for relevance prediction (main task)
- **L_contrastive**: InfoNCE loss in query space - queries with shared relevant docs should have similar embeddings
- **L_rank**: Ranking loss using GNN embeddings - encourages queries with more relevant docs to have higher-norm embeddings

**Loss weights:**
- `λ_contrastive: 0.1`
- `λ_rank: 0.05`
- `temperature: 0.07` (for contrastive loss)

## Query Clustering Hypothesis

**Core idea**: If document D is relevant for query Q1, and query Q2 is semantically similar to Q1, then D is likely relevant for Q2.

This extends the traditional document clustering hypothesis to the query space, enabling the model to transfer relevance patterns across similar queries.

## Training Purpose

### Why Train with Query Graphs?

1. **Cross-Query Learning**: Model learns from relationships between queries, not just individual query-document pairs

2. **Better Generalization**: By understanding query similarity patterns, the model can better handle unseen queries similar to training queries

3. **Semantic Understanding**: GNN message passing captures deeper semantic relationships beyond lexical overlap

4. **Knowledge Transfer**: Relevance signals propagate through the query graph, allowing underrepresented queries to benefit from similar well-represented queries

5. **Robustness**: Multi-task learning with contrastive and ranking losses provides additional supervision signals beyond point-wise relevance

## Comparison to Existing Methods

| Approach | Graph Type | Graph Construction | Message Passing |
|----------|-----------|-------------------|-----------------|
| **Quantum FT** | Query-query | Jaccard similarity | None (penalty terms only) |
| **G-RAG** | Document-document | Document similarity | GNN over docs |
| **GNRR** | Corpus graph | BM25 retrieval | GNN over corpus |
| **QG-Rerank (Ours)** | **Query-query** | **Semantic embeddings** | **GNN over queries** |

## Expected Benefits

1. **Improved zero-shot performance** on queries dissimilar to training queries
2. **Better handling of sparse queries** through knowledge transfer from dense queries
3. **Enhanced semantic understanding** via learned query relationships
4. **Robust to domain shift** by capturing query-level patterns

## Training Configuration

```yaml
gnn:
  gnn_mode: true
  embedding_model: all-mpnet-base-v2
  similarity_threshold: 0.7
  max_neighbors: 10
  gnn_hidden_dim: 256
  gnn_output_dim: 128
  gnn_dropout: 0.1
  lambda_contrastive: 0.1
  lambda_rank: 0.05
  temperature: 0.07
```

Inherits proven hyperparameters from quantum approach:
- Learning rate: 2e-5
- Batch size: 16 (effective 64 with gradient accumulation)
- Epochs: 3
- LoRA fine-tuning (r=8, alpha=16)

## Usage

```bash
# Train QG-Rerank model
python -m cli.qg_train --config configs/qg_rerank.yaml --model-name qg_rerank_v1

# Evaluate
python -m cli.eval --model-path models/qg_rerank_v1/best --dataset superset_comprehensive
```

## Research Contribution

QG-Rerank represents the first application of query graph neural networks to document reranking, introducing:
- Query Clustering Hypothesis
- Cross-query knowledge transfer framework
- Multi-task training combining relevance, contrastive, and ranking objectives

Target venues: SIGIR, NeurIPS, WWW, ACL
