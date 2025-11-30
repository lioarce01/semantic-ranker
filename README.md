# Semantic Reranker for RAG

A state-of-the-art semantic document reranking model built from scratch for Retrieval-Augmented Generation (RAG) systems.

## Overview

This project implements a two-stage retrieval pipeline:
1. **Fast bi-encoder** for initial document retrieval (embedding-based)
2. **Cross-encoder reranker** for precise relevance scoring

The cross-encoder jointly encodes query-document pairs, providing more accurate relevance scores than independent embeddings.

## Features

- 🎯 **Cross-Encoder Training**: Fine-tune transformer models (BERT, RoBERTa, DeBERTa, MiniLM) on ranking datasets
- 📊 **Dataset Support**: MS MARCO, Quora, TREC DL, BEIR, and custom datasets
- ⚡ **Optimization**: ONNX export, INT8 quantization, mixed precision, LoRA fine-tuning
- 🔍 **Hard Negative Mining**: Automated mining for better training
- 📈 **Evaluation Metrics**: NDCG@k, MRR@k, MAP, Hit Rate
- 🚀 **RAG Integration**: Ready-to-use with FAISS, Chroma, Pinecone
- 🎨 **Flexible Architecture**: Support for multiple pretrained models

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Collection and Preprocessing

```python
from semantic_ranker.data import MSMARCODataLoader, DataPreprocessor

# Load MS MARCO dataset
loader = MSMARCODataLoader()
train_data, val_data, test_data = loader.load_and_split()

# Preprocess into (query, positive, negative) triples
preprocessor = DataPreprocessor()
train_triples = preprocessor.create_triples(train_data)
```

### 2. Train Cross-Encoder

```python
from semantic_ranker.training import CrossEncoderTrainer

trainer = CrossEncoderTrainer(
    model_name="bert-base-uncased",
    num_labels=1,
    loss_function="bce"
)

trainer.train(
    train_samples=train_triples,
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    output_dir="./models/reranker"
)
```

### 3. Evaluate

```python
from semantic_ranker.evaluation import RankerEvaluator

evaluator = RankerEvaluator(model_path="./models/reranker")
metrics = evaluator.evaluate(test_data, metrics=["ndcg@10", "mrr@10", "map"])
print(metrics)
```

### 4. Use in RAG Pipeline

```python
from semantic_ranker.rag import RAGPipeline

pipeline = RAGPipeline(
    retriever_model="sentence-transformers/all-MiniLM-L6-v2",
    reranker_model="./models/reranker",
    top_k_retrieval=50,
    top_k_rerank=5
)

# Index documents
pipeline.index_documents(documents)

# Query with reranking
query = "What is semantic search?"
results = pipeline.retrieve_and_rerank(query)
```

### 5. Optimize for Production

```python
from semantic_ranker.optimization import ModelOptimizer

optimizer = ModelOptimizer(model_path="./models/reranker")

# Export to ONNX with quantization
optimizer.export_to_onnx(
    output_path="./models/reranker.onnx",
    quantize=True,
    precision="int8"
)
```

## Project Structure

```
semantic-ranker/
├── semantic_ranker/       # Core package
│   ├── data/              # Data collection and preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training scripts and utilities
│   ├── evaluation/        # Evaluation metrics
│   ├── rag/              # RAG pipeline integration
│   └── optimization/      # ONNX, quantization, LoRA
├── scripts/              # Utility scripts
│   ├── check_dependencies.py    # Check required packages
│   ├── list_available_models.py # List trained models
│   ├── retrain_best_model.py    # Retrain best model
│   └── train_any_domain.py      # Train any domain
├── examples/              # Example scripts (01-09)
├── datasets/              # Training datasets by domain
├── models/                # Trained model checkpoints
├── docs/                  # Documentation
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
├── setup.py             # Package setup
├── README.md            # This file
└── LICENSE              # License
```

## Advanced Features

### Hard Negative Mining

```python
from semantic_ranker.training import HardNegativeMiner

miner = HardNegativeMiner(bi_encoder_model="all-MiniLM-L6-v2")
hard_negatives = miner.mine(queries, documents, top_k=10)
```

### LoRA Fine-tuning

```python
trainer = CrossEncoderTrainer(
    model_name="bert-base-uncased",
    use_lora=True,
    lora_r=8,
    lora_alpha=16
)
```

### Mixed Precision Training

```python
trainer.train(
    train_samples=train_triples,
    mixed_precision="fp16",
    gradient_accumulation_steps=4
)
```

## Evaluation Metrics

- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MRR@k**: Mean Reciprocal Rank
- **MAP**: Mean Average Precision
- **Hit Rate@k**: Percentage of queries with relevant docs in top-k

## 📚 Documentation

- **[📖 Glosario ML](docs/GLOSARIO_ML.md)**: Guía completa para entender métricas de entrenamiento y evaluar modelos
- **[🚀 Guía Paso a Paso](docs/GUIA_PASO_A_PASO.md)**: Tutorial detallado en español
- **[⚡ Quick Start](docs/QUICKSTART.md)**: Inicio rápido para usuarios avanzados
- **[📂 Estructura del Proyecto](.project_structure.md)**: Documentación completa de la organización del código

## 🔬 Research Implementation

Based on cutting-edge research:
- "Passage Re-ranking with BERT" (Nogueira et al., 2019)
- "ColBERT: Efficient and Effective Passage Search" (Khattab & Zaharia, 2020)
- "LoRA: Low-Rank Adaptation" (Hu et al., 2021)
- Sentence Transformers documentation and best practices

## References

- [Training Reranker Models - Hugging Face](https://huggingface.co/blog/train-reranker)
- [Rerankers and Two-Stage Retrieval - Pinecone](https://www.pinecone.io/learn/series/rag/rerankers/)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [Sentence Transformers Documentation](https://www.sbert.net/)

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
