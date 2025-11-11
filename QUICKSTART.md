# Quickstart Guide - Semantic Reranker for RAG

## 🚀 Quick Start (5 minutes)

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Test RAG Pipeline (No Training Required)
```bash
python examples/03_rag_pipeline.py
```

This demonstrates the complete RAG system with retrieval and reranking.

---

## 📖 Complete Workflow

### Step 1: Basic Training
```bash
python examples/01_basic_training.py
```
- Trains a cross-encoder reranker
- Uses ~1K samples for demo
- Saves model to `./models/basic_reranker/`
- **Time**: 5-10 min (CPU) / 2-3 min (GPU)

### Step 2: Hard Negative Mining
```bash
python examples/02_hard_negative_mining.py
```
- Mines challenging negative examples
- Improves model performance by 5-10%
- **Time**: ~2x basic training

### Step 3: Evaluation
```bash
python examples/04_evaluation.py
```
- Computes NDCG@k, MRR@k, MAP metrics
- Analyzes per-query performance
- Saves results to JSON

### Step 4: Optimization
```bash
python examples/05_optimization.py
```
- Exports to ONNX
- Applies INT8 quantization
- Reduces model size by 75%
- Improves inference speed 2-3x

### Step 5: Complete Workflow
```bash
python examples/06_complete_workflow.py
```
- Runs entire pipeline end-to-end
- **Time**: ~30 min (demo)

---

## 🎯 Using in Your Application

```python
from semantic_ranker.rag import RAGPipeline

# Initialize with trained reranker
pipeline = RAGPipeline(
    retriever_model="sentence-transformers/all-MiniLM-L6-v2",
    reranker_model="./models/basic_reranker/final",
    top_k_retrieval=50,
    top_k_rerank=5
)

# Index your documents
pipeline.index_documents(your_documents)

# Query
results = pipeline.retrieve_and_rerank("your query")

# Get context for LLM
context = pipeline.get_context_for_llm("your query", top_k=3)
```

---

## 📊 Expected Performance

| Metric | Demo (1K samples) | Production (100K samples) |
|--------|-------------------|---------------------------|
| NDCG@10 | 0.45-0.55 | 0.70-0.85 |
| MRR@10 | 0.50-0.60 | 0.75-0.90 |
| Inference | 50ms | 50ms |

---

## 🔧 Customization

### Use Your Own Data
```python
from semantic_ranker.data import CustomDataLoader

loader = CustomDataLoader(
    data_path="your_data.jsonl",
    format="jsonl"
)
data = loader.load()
```

**Required format:**
```json
{"query": "question", "positive": "relevant doc", "negative": "irrelevant doc"}
```

### Change Base Model
```python
trainer = CrossEncoderTrainer(
    model_name="roberta-base",  # or "bert-base-uncased", "microsoft/deberta-v3-base"
    num_labels=1
)
```

### Use LoRA for Efficient Training
```python
trainer = CrossEncoderTrainer(
    model_name="bert-base-uncased",
    use_lora=True,
    lora_r=8,
    lora_alpha=16
)
```

---

## ⚙️ Configuration Guide

### For Prototyping (Fast)
```python
trainer.train(
    epochs=2,
    batch_size=16,
    learning_rate=2e-5,
    max_samples=1000
)
```

### For Production (High Quality)
```python
trainer.train(
    epochs=5-10,
    batch_size=32,
    learning_rate=1e-5,
    max_samples=None,  # Use all data
    mixed_precision="fp16"
)
```

### For Limited Resources
```python
trainer = CrossEncoderTrainer(
    model_name="distilbert-base-uncased",  # Smaller model
    use_lora=True  # Parameter-efficient
)
trainer.train(
    batch_size=8,
    gradient_accumulation_steps=2
)
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` to 8 or 4
- Use `gradient_accumulation_steps=2`
- Try smaller model (DistilBERT)

### Slow Training
- Enable mixed precision: `mixed_precision="fp16"`
- Increase `batch_size` if GPU has capacity
- Use smaller `max_length` (e.g., 128 instead of 512)

### Poor Performance
- Increase training data (>10K samples)
- Use hard negative mining
- Train more epochs (5-10)
- Try different base model

---

## 📚 Additional Resources

- **Full Guide**: See `GUIA_PASO_A_PASO.md` (Spanish)
- **Examples**: Check `/examples/` directory
- **API Docs**: See docstrings in each module

---

## 🎓 Key Concepts

### Two-Stage Retrieval
1. **Bi-Encoder (Fast)**: Retrieves ~50 candidates using embeddings
2. **Cross-Encoder (Accurate)**: Reranks top candidates by scoring each pair

### Why Cross-Encoders?
- Process query + document together
- Capture fine-grained interactions
- More accurate than bi-encoders
- Essential for high-precision RAG

### When to Retrain?
- Performance drops >5%
- New data available (quarterly)
- Query distribution changes
- Domain shift detected

---

## 🚢 Production Checklist

- [ ] Trained on sufficient data (>10K examples)
- [ ] NDCG@10 > 0.5
- [ ] Model optimized (ONNX/quantized)
- [ ] Latency < 100ms
- [ ] Monitoring configured
- [ ] Fallback strategy ready
- [ ] Retraining schedule defined

---

**Need help?** Open an issue on GitHub or check the full documentation.
