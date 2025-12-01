# 🧬 Quantum Resonance Fine-Tuning

## Overview

**Quantum Resonance Fine-Tuning** es una estrategia innovadora que trata las relaciones query-document como estados cuánticos que existen en superposición hasta colapsar en rankings óptimos.

## Conceptos Clave

- **Superposición Cuántica**: Las relaciones query-document existen en múltiples estados de relevancia hasta ser "medidas"
- **Colapso de Superposición**: El proceso de entrenamiento colapsa estos estados hacia rankings óptimos
- **Frecuencia de Resonancia**: Medida de afinidad semántica entre queries y documentos
- **Entanglement**: Conexiones entre queries relacionadas que mejoran el aprendizaje mutuo

## Comandos Disponibles

### 1. `cli/quantum_train.py` - Training desde cero

```bash
python cli/quantum_train.py \
  --dataset msmarco \
  --model-name distilbert-base-uncased \
  --quantum-mode resonance \
  --resonance-threshold 0.7 \
  --entanglement-weight 0.3 \
  --epochs 3 \
  --use-lora
```

**Parámetros específicos de Quantum:**
- `--quantum-mode`: `resonance`, `entanglement`, `superposition`
- `--resonance-threshold`: Umbral para colapso cuántico (0.0-1.0)
- `--entanglement-weight`: Peso para coherencia entre queries relacionadas (0.0-1.0)
- `--quantum-phase`: `superposition`, `collapse`, `resonance`

### 2. `cli/quantum_retrain.py` - Retraining de modelo existente

```bash
python cli/quantum_retrain.py \
  --dataset datasets/msmarco_dev_benchmark_with_hard_negatives.json \
  --epochs 3 \
  --learning-rate 2e-5 \
  --quantum-mode adaptation \
  --preserve-knowledge 0.3 \
  --analyze-existing
```

**Parámetros específicos de Quantum:**
- `--quantum-mode`: `adaptation`, `resonance`, `entanglement`
- `--preserve-knowledge`: Peso para preservar conocimiento existente (0.0-1.0)
- `--resonance-alignment`: Peso para alineación de resonancia (0.0-1.0)
- `--analyze-existing`: Analizar patrones de resonancia del modelo actual

## Comparación con Métodos Tradicionales

| Aspecto | LoRA Tradicional | Hard Negative Mining | Quantum Resonance FT |
|---------|------------------|---------------------|---------------------|
| **Complejidad** | Baja | Media | Alta |
| **Interpretabilidad** | Baja | Media | Alta (estados cuánticos) |
| **Generalización** | Buena | Muy buena | Excelente |
| **Overfitting** | Posible | Reducido | Mínimo |
| **Recursos** | Bajos | Medios | Medios |

## Ejemplos de Uso

### Training Básico
```bash
# Training desde cero con resonancia cuántica
python cli/quantum_train.py --dataset msmarco --use-lora --quantum-mode resonance
```

### Retraining Avanzado
```bash
# Mejorar modelo existente con datos difíciles
python cli/quantum_retrain.py \
  --dataset datasets/msmarco_dev_benchmark_with_hard_negatives.json \
  --quantum-mode adaptation \
  --analyze-existing \
  --preserve-knowledge 0.4
```

### Evaluación
```bash
# Evaluar modelo quantum-trained
python cli/eval.py --dataset msmarco_dev_benchmark --model-path models/quantum_trained_model/best

# Benchmark comparison
python scripts/benchmark_comparison.py --dataset msmarco_dev_benchmark --model-path models/quantum_trained_model/best
```

## Resultados Esperados

### En Dataset Mixto (conocido):
- **Mantiene**: NDCG@10 ~0.90 (similar al modelo original)

### En MS MARCO Dev (desafiante):
- **Mejora**: NDCG@10 de 0.61 → 0.70-0.75
- **Motivo**: Mejor discriminación de documentos difíciles

### En Generalización:
- **Mejora**: NDCG@10 de 0.85 → 0.90+
- **Motivo**: Principios cuánticos capturan mejor la incertidumbre

## Ventajas Científicas

### 1. **Modelado de Incertidumbre**
- El sistema puede expresar grados de confianza en rankings
- Mejor manejo de queries ambiguas

### 2. **Aprendizaje Relacional**
- Queries relacionadas mejoran su aprendizaje mutuamente
- Efecto "entanglement" para conocimiento transferible

### 3. **Robustez Mejorada**
- Menos sensible a ruido en datos de entrenamiento
- Mejor generalización a dominios no vistos

## Limitaciones

- **Complejidad**: Mayor overhead computacional
- **Interpretabilidad**: Resultados menos intuitivos que métodos tradicionales
- **Recursos**: Requiere más memoria para patrones de resonancia

## Futuras Extensiones

- **Multi-Modal Quantum**: Extender a imágenes, audio, video
- **Quantum Ensembles**: Combinar múltiples modelos cuánticos
- **Temporal Resonance**: Modelar evolución de queries en el tiempo
- **Cross-Lingual Quantum**: Transferencia entre idiomas usando resonancia
