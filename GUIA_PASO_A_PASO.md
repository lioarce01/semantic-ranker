# Guía Paso a Paso - Semantic Ranker para RAG

## 📋 Índice
1. [Instalación](#instalación)
2. [Inicio Rápido](#inicio-rápido)
3. [Entrenamiento Completo](#entrenamiento-completo)
4. [Evaluación](#evaluación)
5. [Optimización](#optimización)
6. [Integración con RAG](#integración-con-rag)
7. [Puntos Importantes](#puntos-importantes)
8. [Solución de Problemas](#solución-de-problemas)

---

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd semantic-ranker
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

**Nota**: La instalación puede tardar varios minutos debido a PyTorch y Transformers.

---

## ⚡ Inicio Rápido

### Probar el RAG Pipeline (Sin Entrenamiento)

```bash
python examples/03_rag_pipeline.py
```

Este ejemplo usa únicamente recuperación (sin reranker) y muestra cómo funciona el sistema completo.

**Lo que hace:**
- Indexa documentos de ejemplo
- Ejecuta consultas
- Muestra los documentos más relevantes
- Genera contexto para LLMs

**Tiempo estimado**: 1-2 minutos

---

## 🎓 Entrenamiento Completo

### Paso 1: Entrenamiento Básico

```bash
python examples/01_basic_training.py
```

**Parámetros importantes a modificar:**
- `max_samples`: Número de muestras (default: 1000 para demo, usa `None` para dataset completo)
- `epochs`: Épocas de entrenamiento (default: 3, recomendado: 5-10 para producción)
- `batch_size`: Tamaño del batch (default: 16, ajusta según tu GPU)
- `learning_rate`: Tasa de aprendizaje (default: 2e-5)

**Salida:**
- Modelo guardado en `./models/basic_reranker/`
- Historial de entrenamiento en `training_history.json`

**Tiempo estimado**:
- Demo (1K samples): ~5-10 minutos (CPU) / ~2-3 minutos (GPU)
- Full dataset: varias horas

### Paso 2: Entrenamiento con Hard Negatives

```bash
python examples/02_hard_negative_mining.py
```

**¿Por qué usar hard negatives?**
Los hard negatives son ejemplos difíciles que el modelo debe aprender a distinguir, mejorando significativamente el rendimiento.

**Mejoras esperadas:**
- +5-10% en NDCG@10
- Mejor precisión en casos ambiguos

**Tiempo estimado**:
- ~2x el tiempo del entrenamiento básico (incluye minería de negativos)

### Paso 3: Workflow Completo

```bash
python examples/06_complete_workflow.py
```

Este script ejecuta todo el pipeline:
1. ✅ Carga de datos
2. ✅ Preprocesamiento
3. ✅ Entrenamiento
4. ✅ Evaluación
5. ✅ Optimización
6. ✅ Despliegue en RAG

**Tiempo estimado**: ~30 minutos (demo) / varias horas (producción)

---

## 📊 Evaluación

### Evaluar Modelo Entrenado

```bash
python examples/04_evaluation.py
```

**Métricas calculadas:**
- **NDCG@k**: Calidad del ranking (0-1, mayor es mejor)
- **MRR@k**: Ranking del primer resultado relevante
- **MAP@k**: Precisión promedio
- **Hit Rate@k**: % de consultas con ≥1 resultado relevante

**Interpretación de resultados:**
- NDCG@10 > 0.7: Excelente
- NDCG@10 > 0.5: Bueno
- NDCG@10 < 0.3: Necesita mejoras

**Salida:**
- Resultados guardados en `evaluation_results.json`

---

## ⚙️ Optimización para Producción

```bash
python examples/05_optimization.py
```

**Optimizaciones aplicadas:**
1. **Exportación ONNX**: Formato universal, ~10-30% más rápido
2. **Cuantización INT8**: Reduce tamaño 4x, ~2-3x más rápido
3. **Precisión FP16**: GPU-friendly, ~2x más rápido

**Comparación de rendimiento:**
| Versión | Tamaño | Latencia | Throughput |
|---------|--------|----------|------------|
| Original | 250 MB | 50 ms | 20 q/s |
| ONNX | 250 MB | 35 ms | 28 q/s |
| INT8 Quantized | 65 MB | 20 ms | 50 q/s |

**Recomendaciones:**
- **CPU**: Usa INT8 quantized
- **GPU**: Usa FP16 o ONNX
- **Edge devices**: Usa INT8 quantized + ONNX

---

## 🔗 Integración con RAG

### Uso Básico

```python
from semantic_ranker.rag import RAGPipeline

# 1. Inicializar pipeline
pipeline = RAGPipeline(
    retriever_model="sentence-transformers/all-MiniLM-L6-v2",
    reranker_model="./models/basic_reranker/final",
    top_k_retrieval=50,  # Recuperar 50 candidatos
    top_k_rerank=5       # Reranking top-5
)

# 2. Indexar documentos
documents = ["doc1", "doc2", "doc3", ...]
pipeline.index_documents(documents)

# 3. Consultar
query = "¿Qué es machine learning?"
results = pipeline.retrieve_and_rerank(query)

# 4. Obtener contexto para LLM
context = pipeline.get_context_for_llm(query, top_k=3)
```

### Pipeline de Dos Etapas

```
Usuario Query
     ↓
[Bi-Encoder Retrieval] → top-50 candidatos (rápido, ~10ms)
     ↓
[Cross-Encoder Reranking] → top-5 mejores (preciso, ~40ms)
     ↓
[LLM con Contexto] → Respuesta final
```

**Ventajas:**
- ✅ Recuperación rápida con bi-encoder
- ✅ Precisión alta con cross-encoder
- ✅ Balance óptimo velocidad/calidad

---

## ⚠️ Puntos Importantes

### 1. **Requisitos de Hardware**

| Tarea | CPU | GPU | RAM | Disco |
|-------|-----|-----|-----|-------|
| Inferencia | ✅ | Opcional | 8 GB | 2 GB |
| Entrenamiento (demo) | ✅ | Recomendado | 16 GB | 5 GB |
| Entrenamiento (full) | ⚠️ | ✅ Requerido | 32 GB | 20 GB |

### 2. **Datos de Entrenamiento**

**Formato esperado:**
```python
{
    'query': "¿Qué es ML?",
    'positive': "Machine learning es...",
    'negative': "Documento irrelevante..."  # Opcional
}
```

**Tamaño mínimo recomendado:**
- **Prototipo**: 1,000 ejemplos
- **Producción**: 10,000+ ejemplos
- **Óptimo**: 100,000+ ejemplos

### 3. **Hiperparámetros Clave**

```python
# Recomendaciones por escenario
ESCENARIOS = {
    'prototipo_rapido': {
        'epochs': 2,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'max_samples': 1000
    },
    'produccion': {
        'epochs': 5-10,
        'batch_size': 32,
        'learning_rate': 1e-5,
        'max_samples': None
    },
    'fine_tuning': {
        'epochs': 3,
        'batch_size': 8,
        'learning_rate': 5e-6,
        'use_lora': True
    }
}
```

### 4. **Selección de Modelo Base**

| Modelo | Tamaño | Velocidad | Precisión | Uso Recomendado |
|--------|--------|-----------|-----------|-----------------|
| distilbert-base | 66M | ⚡⚡⚡ | ⭐⭐ | Desarrollo/CPU |
| bert-base | 110M | ⚡⚡ | ⭐⭐⭐ | Balanceado |
| roberta-base | 125M | ⚡⚡ | ⭐⭐⭐⭐ | Producción |
| deberta-v3-base | 184M | ⚡ | ⭐⭐⭐⭐⭐ | Máxima precisión |

### 5. **Monitoreo y Mantenimiento**

**Métricas a monitorear:**
- Latencia P50, P95, P99
- NDCG@10 en producción
- Tasa de caché hit
- Uso de memoria/CPU

**Cuándo reentrenar:**
- ⚠️ NDCG@10 cae >5%
- 🔄 Nuevos datos disponibles (cada 1-3 meses)
- 🆕 Cambio en distribución de queries

### 6. **Costos de Inferencia**

**Estimaciones (1M queries/mes):**
| Setup | Costo AWS | Latencia |
|-------|-----------|----------|
| CPU (t3.medium) | ~$30/mes | 50ms |
| GPU (g4dn.xlarge) | ~$250/mes | 10ms |
| Lambda + ONNX | ~$15/mes | 100ms |

---

## 🔧 Solución de Problemas

### Error: "CUDA out of memory"
**Solución:**
```python
# Reducir batch_size
trainer.train(batch_size=8)  # en vez de 16

# Usar gradient accumulation
trainer.train(
    batch_size=8,
    gradient_accumulation_steps=2  # Efectivo: 16
)
```

### Error: "Dataset not found"
**Solución:**
- Verificar conexión a internet
- Usar `cache_dir` personalizado
- O usar datos sintéticos para testing:
```python
loader = MSMARCODataLoader()
train, val, test = loader.load_and_split(max_samples=100)
```

### Rendimiento pobre en evaluación
**Diagnóstico:**
1. ✅ Verificar tamaño de datos (¿muy pocos ejemplos?)
2. ✅ Revisar calidad de negativos (¿muy fáciles?)
3. ✅ Ajustar learning rate (probar 5e-6, 1e-5, 2e-5)
4. ✅ Entrenar más épocas (5-10)
5. ✅ Usar hard negative mining

### Inferencia muy lenta
**Soluciones:**
1. Usar modelo cuantizado (INT8)
2. Exportar a ONNX
3. Reducir `top_k_retrieval`
4. Usar modelo más pequeño (DistilBERT)
5. Implementar batching

---

## 📚 Recursos Adicionales

### Documentación
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Sentence Transformers](https://www.sbert.net/)
- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)

### Papers Relevantes
- "Passage Re-ranking with BERT" (Nogueira et al., 2019)
- "ColBERT: Efficient and Effective Passage Search" (Khattab & Zaharia, 2020)
- "LoRA: Low-Rank Adaptation" (Hu et al., 2021)

### Comunidad
- GitHub Issues: Para reportar bugs
- Discussions: Para preguntas y mejores prácticas

---

## 🎯 Checklist de Producción

Antes de desplegar en producción, verifica:

- [ ] Modelo entrenado con datos suficientes (>10K ejemplos)
- [ ] Evaluación NDCG@10 > 0.5
- [ ] Modelo optimizado (ONNX o cuantizado)
- [ ] Benchmarks de latencia aceptables (<100ms)
- [ ] Tests de integración pasando
- [ ] Monitoreo configurado
- [ ] Estrategia de reentrenamiento definida
- [ ] Fallback configurado (si reranker falla)
- [ ] Documentación actualizada
- [ ] Plan de rollback listo

---

## 🤝 Contribuciones

¿Encontraste un bug o quieres añadir una feature?
1. Fork el repositorio
2. Crea una branch: `git checkout -b feature/nueva-feature`
3. Commit: `git commit -m 'Add nueva feature'`
4. Push: `git push origin feature/nueva-feature`
5. Abre un Pull Request

---

**¿Preguntas?** Abre un issue en GitHub o consulta la documentación completa en `/docs`.
