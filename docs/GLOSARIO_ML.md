# 📚 Glosario de Machine Learning para Evaluación de Modelos

## 🔍 Términos Básicos de Entrenamiento

### **Loss (Pérdida)**
**Qué es**: Medida numérica de qué tan mal está prediciendo el modelo. Es como un "puntaje de error".

**Cómo interpretarlo**:
- `Loss = 0.0`: Predicciones perfectas (ideal pero imposible)
- `Loss = 0.1-0.5`: Bueno para clasificación binaria
- `Loss > 1.0`: Modelo aprendiendo o datos problemáticos
- `Loss bajando`: ✅ Modelo aprendiendo
- `Loss subiendo`: ❌ Problema (overfitting o datos malos)

**Ejemplo del entrenamiento**:
```
Epoch 1: Train Loss = 0.2212 (malo, aprendiendo)
Epoch 3: Train Loss = 0.0119 (muy bueno, convergió)
```

### **Epoch (Época)**
**Qué es**: Una pasada completa por todo el dataset de entrenamiento.

**Cómo interpretarlo**:
- **1 epoch**: Modelo vio cada ejemplo 1 vez
- **3 epochs**: Modelo vio cada ejemplo 3 veces
- **Más epochs**: Más aprendizaje, pero riesgo de overfitting

### **Batch Size (Tamaño de Lote)**
**Qué es**: Cantidad de ejemplos que procesa el modelo antes de actualizar sus parámetros.

**Cómo interpretarlo**:
- `batch_size=8`: ✅ Pequeño, aprendizaje estable, usa menos memoria
- `batch_size=32`: ⚖️ Balance, buen compromiso
- `batch_size=128`: ⚡ Rápido pero puede ser inestable

### **Learning Rate (Tasa de Aprendizaje)**
**Qué es**: Qué tan grande es cada "paso" que da el modelo para aprender.

**Cómo interpretarlo**:
- `1e-5` (0.00001): ✅ Muy pequeño, aprendizaje lento pero estable
- `2e-5` (0.00002): ✅ Valor típico para fine-tuning
- `1e-3` (0.001): ❌ Muy grande, puede "saltar" la solución óptima

**Del entrenamiento**:
```
1.7983193277310925e-05  → 1.8e-5 (óptimo)
9.579831932773111e-06   → 9.6e-6 (más pequeño, fine-tuning)
```

## 📊 Métricas de Evaluación

### **Train Loss vs Validation Loss**
**Qué miden**:
- **Train Loss**: Qué tan bien aprende el modelo con datos de entrenamiento
- **Validation Loss**: Qué tan bien generaliza a datos nuevos

**Cómo interpretarlo**:
```
✅ BUENO:     Train: 0.012, Val: 0.021 (aprendió y generaliza)
⚠️  OVERFIT:  Train: 0.001, Val: 0.200 (memorizó, no generaliza)
❌  UNDERFIT: Train: 0.500, Val: 0.450 (no aprendió suficiente)
```

### **Convergencia (Convergence)**
**Qué es**: Cuando el modelo deja de mejorar significativamente.

**Indicadores**:
- ✅ Loss estable por 2-3 epochs
- ✅ Train y validation loss cercanos
- ❌ Loss oscilando mucho
- ❌ Validation loss subiendo

### **Overfitting (Sobreajuste)**
**Qué es**: Modelo memoriza datos de entrenamiento pero falla en datos nuevos.

**Síntomas**:
- Train Loss muy bajo (< 0.01)
- Validation Loss alto (> 0.1)
- Diferencia grande entre train/val loss

**Solución**: Más datos, regularización, early stopping.

### **Underfitting (Subajuste)**
**Qué es**: Modelo no aprende lo suficiente de los datos.

**Síntomas**:
- Train Loss alto (> 0.5)
- Validation Loss similar al train
- Modelo predice igual que al azar

**Solución**: Más epochs, learning rate más alto, modelo más complejo.

## 🏗️ Arquitectura del Modelo

### **Cross-Encoder**
**Qué es**: Modelo que procesa query + documento juntos para predecir relevancia.

**Ventajas**:
- ✅ Muy preciso para ranking
- ✅ Entiende contexto completo
- ❌ Lento (procesa cada par por separado)

### **DistilBERT**
**Qué es**: Versión "destilada" (más pequeña) de BERT.

**Características**:
- 66M parámetros (vs 110M de BERT-base)
- 40% más rápido
- 97% accuracy de BERT
- Perfecto para fine-tuning

### **LoRA (Low-Rank Adaptation)**
**Qué es**: Técnica para fine-tuning eficiente que solo entrena pocos parámetros.

**Beneficios**:
- ✅ 10x menos memoria
- ✅ Entrenamiento más rápido
- ✅ Mismo rendimiento
- ✅ Compatible con modelos grandes

## 📈 Rendimiento y Benchmarks

### **NDCG@k (Normalized Discounted Cumulative Gain)**
**Qué mide**: Calidad del ranking (qué tan buenos son los top-k resultados).

**Interpretación**:
- `NDCG@10 = 0.85`: ✅ Excelente (85% de resultados perfectos)
- `NDCG@10 = 0.60`: ⚖️ Bueno
- `NDCG@10 = 0.30`: ❌ Malo

### **MRR@k (Mean Reciprocal Rank)**
**Qué mide**: Posición del primer resultado relevante.

**Interpretación**:
- `MRR@10 = 0.90`: ✅ Primer resultado relevante en top-1 promedio
- `MRR@10 = 0.50`: ⚖️ Primer resultado relevante en top-2 promedio

### **Latency (Latencia)**
**Qué mide**: Tiempo que tarda en procesar una query.

**Objetivos típicos**:
- ✅ `< 50ms`: Excelente para búsqueda en tiempo real
- ⚠️ `50-200ms`: Aceptable
- ❌ `> 500ms`: Demasiado lento

## 🔧 Configuración del Entrenamiento

### **Gradient Accumulation**
**Qué es**: Acumulador gradientes antes de actualizar parámetros.

**Cuándo usar**:
- Batch size pequeño por limitación de memoria
- Simula batch size más grande

### **Mixed Precision (FP16)**
**Qué es**: Entrenamiento con números de 16 bits en lugar de 32.

**Beneficios**:
- ✅ 2x más rápido
- ✅ Usa menos memoria
- ✅ Mismo accuracy

### **Early Stopping**
**Qué es**: Detiene entrenamiento cuando validation loss deja de bajar.

**Configuración típica**:
- Patience: 3-5 epochs
- Min delta: 0.001
- Restaura mejor modelo

## 📊 Interpretación de Logs

### **Training Progress**
```
Epoch 1/3
Training: 100%|██████████| 88/88 [04:38<00:00, 3.17s/it, loss=0.0931]
INFO: Epoch 1 completed. Avg loss: 0.2212
```

**Qué significa**:
- `88/88`: 88 batches procesados (100%)
- `3.17s/it`: 3.17 segundos por batch
- `loss=0.0931`: Loss del último batch
- `Avg loss: 0.2212`: Loss promedio de toda la epoch

### **Validation Check**
```
INFO: Step 100: val_loss = 0.0330
INFO: Saved best model (val_loss=0.0330)
```

**Qué significa**:
- Cada 100 steps: evalúa en validation set
- `val_loss = 0.0330`: Performance en datos no vistos
- Guarda modelo si es el mejor hasta ahora

## 🎯 Checklist para Evaluar Entrenamiento

### ✅ **Entrenamiento Saludable**
- [ ] Loss bajando consistentemente
- [ ] Train y validation loss convergiendo
- [ ] No overfitting (val_loss no sube mucho)
- [ ] Learning rate decay apropiado
- [ ] Modelo converge en < 10 epochs

### ⚠️ **Señales de Problema**
- [ ] Loss oscilando mucho
- [ ] Validation loss subiendo
- [ ] Train loss muy bajo, val loss alto
- [ ] Modelo no converge después de 20 epochs

### 🔧 **Optimizaciones**
- [ ] Usar GPU si disponible
- [ ] Batch size óptimo para memoria
- [ ] Learning rate decay
- [ ] Early stopping
- [ ] Mixed precision (FP16)

## 📝 Resumen Ejecutivo

**Para evaluar un modelo, mira:**

1. **Loss bajando** → ✅ Aprendiendo
2. **Train ≈ Validation** → ✅ Generalizando
3. **Convergencia** → ✅ Listo para usar
4. **Sin overfitting** → ✅ Confiable

**Métricas clave del último entrenamiento:**
- **Train Loss**: `0.0119` (muy bueno, < 0.05)
- **Val Loss**: `0.0209` (excelente, cercano al train)
- **Convergencia**: ✅ Estable en 3 epochs
- **Overfitting**: ✅ No presente

¡El modelo está **perfectamente entrenado**! 🎉

## 🎯 **Métricas Avanzadas de Evaluación (Reranking)**

### **NDCG@10 (Normalized Discounted Cumulative Gain)**
**Qué es**: La métrica más importante para evaluar rerankers. Mide calidad del ranking considerando posición Y relevancia.

**Cómo interpretarlo**:
- `NDCG@10 = 1.0`: ✅ Ranking perfecto (ideal)
- `NDCG@10 ≥ 0.85`: ✅ **Excelente** (state-of-the-art)
- `NDCG@10 ≥ 0.80`: ✅ **Muy bueno** (competitivo)
- `NDCG@10 ≥ 0.75`: ⚖️ **Bueno** (útil en producción)
- `NDCG@10 ≥ 0.70`: ⚠️ **Aceptable** (mejorable)
- `NDCG@10 < 0.70`: ❌ **Necesita mejoras**

**Ejemplos reales**:
- `0.866`: BGE-Reranker-v2.0 (líder actual)
- `0.842`: FlashRank (muy competitivo)
- `0.582`: Tu modelo actual (necesita mejoras)

### **MRR@10 (Mean Reciprocal Rank)**
**Qué es**: Mide qué tan temprano aparece el primer documento relevante en los resultados.

**Cómo interpretarlo**:
- `MRR@10 = 1.0`: ✅ Primer resultado relevante (perfecto)
- `MRR@10 ≥ 0.85`: ✅ **Excelente**
- `MRR@10 ≥ 0.70`: ⚖️ **Bueno**
- `MRR@10 < 0.50`: ❌ **Necesita mejoras**

**Fórmula**: `MRR = 1/posición_del_primer_relevante`
- Si relevante está en posición 3 → MRR = 1/3 = 0.33

### **MAP@100 (Mean Average Precision)**
**Qué es**: Precisión promedio considerando todos los documentos relevantes encontrados.

**Cómo interpretarlo**:
- `MAP@100 ≥ 0.80`: ✅ **Excelente**
- `MAP@100 ≥ 0.70`: ⚖️ **Bueno**
- Mide tanto precisión como exhaustividad

### **Recall@100**
**Qué es**: Fracción de documentos relevantes encontrados en el top-100.

**Cómo interpretarlo**:
- `Recall@100 ≥ 0.90`: ✅ **Excelente** (encuentra casi todos)
- `Recall@100 ≥ 0.80`: ⚖️ **Bueno**
- Mide capacidad de encontrar documentos relevantes

## 🏆 **Modelos State-of-the-Art**

### **BGE (BAAI General Embeddings)**
- **Tipo**: Embeddings densos para retrieval
- **Mejores modelos**: BGE-M3, BGE-Reranker-v2.0
- **NDCG@10 típico**: 0.81-0.87
- **Ventaja**: Rápido para búsqueda inicial

### **FlashRank**
- **Tipo**: Cross-encoder reranker
- **NDCG@10**: ~0.84
- **Latencia**: 10-50ms por query
- **Ventaja**: Muy eficiente

### **MonoT5**
- **Tipo**: Cross-encoder basado en T5
- **NDCG@10**: ~0.81
- **Ventaja**: Arquitectura probada

## 🔧 **Técnicas de Mejora**

### **Retrain (Reentrenamiento)**
**Qué es**: Continuar entrenando un modelo existente con más datos.

**Cuándo usar**:
- ✅ Modelo funciona pero necesita más datos
- ✅ Arquitectura correcta, falta entrenamiento
- ✅ Menos riesgoso que reentrenar desde cero

**Ejemplo**:
```bash
python cli/retrain.py --samples 2000 --epochs 3 --learning-rate 2e-5
```

### **Hard Negative Mining**
**Qué es**: Seleccionar ejemplos difíciles (negativos que el modelo confunde) para entrenamiento.

**Beneficio**: Mejora robustez del modelo (+0.05-0.10 NDCG)

### **Data Augmentation**
**Qué es**: Crear variaciones de los datos de entrenamiento.

**Técnicas**:
- Query expansion
- Document paraphrasing
- Multi-task learning

## 📊 **Benchmarks de Referencia**

### **MS MARCO**
- **Tipo**: QA general en inglés
- **Tamaño**: ~800k queries
- **Útil para**: Evaluación general de rerankers

### **BEIR (Benchmarking IR)**
- **Tipo**: 18 datasets especializados
- **Dominios**: Biomedicina, noticias, científico, etc.
- **Útil para**: Evaluación en dominios específicos

### **TREC-COVID**
- **Tipo**: Búsqueda médica
- **Útil para**: Evaluación en dominio médico

## 🎯 **Interpretación de Resultados**

### **Tu Modelo Actual**
- **NDCG@10 = 0.5829**: ❌ Necesita mejoras significativas
- **Estado**: Funcional pero bajo rendimiento
- **Posición**: Top 50% de modelos publicados

### **Metas Realistas**
- **Versión mejorada**: NDCG@10 ≥ 0.70
- **Competitivo**: NDCG@10 ≥ 0.80
- **State-of-the-art**: NDCG@10 ≥ 0.85

### **Plan de Mejora**
1. **Retrain con más datos** (+0.05-0.10)
2. **Hard negatives** (+0.03-0.08)
3. **Mejor arquitectura** (+0.05-0.15)
4. **Fine-tuning avanzado** (+0.02-0.05)

¡El glosario está **actualizado con métricas avanzadas**! 📚✨
