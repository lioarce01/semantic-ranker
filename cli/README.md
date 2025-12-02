# Command Line Interface

Interfaz de línea de comandos con responsabilidades separadas para semantic-ranker.

## 🎛️ Sistema de Configuración

Todos los comandos soportan configuración centralizada con perfiles YAML:

```bash
# Usar perfil de configuración
python cli/train.py --config-profile quick_test

# Cargar config personalizado
python cli/train.py --config configs/my_experiment.yaml

# Sobrescribir valores específicos
python cli/train.py --config-profile default --epochs 5 --learning-rate 1e-5
```

**Perfiles disponibles** (en `configs/`):
- `default.yaml` - Configuración estándar
- `quick_test.yaml` - Testing rápido (1 época, 100 muestras)
- `full_training.yaml` - Entrenamiento completo (10 épocas, negativos difíciles)
- `lora_training.yaml` - Configuración específica para LoRA
- `quantum_training.yaml` - Quantum fine-tuning
- `retrain.yaml` - Reentrenamiento (LR bajo, preservación de conocimiento)

## 📋 Comandos Disponibles

### 🚀 `train.py` - Entrenamiento
**Responsabilidad**: Solo entrenamiento de modelos.

```bash
# Entrenamiento básico
python cli/train.py --dataset msmarco --epochs 3 --batch-size 16

# Con perfil de configuración
python cli/train.py --config-profile full_training --dataset msmarco

# Con LoRA (eficiente en memoria)
python cli/train.py --config-profile lora_training --use-lora
```

**Parámetros principales:**
- `--config-profile` / `--config`: Cargar configuración desde YAML
- `--dataset`: Dataset de entrenamiento (msmarco o archivos en datasets/)
- `--model-name`: Modelo base (default: bert-base-uncased)
- `--epochs`: Número de epochs
- `--batch-size`: Tamaño del batch
- `--learning-rate`: Tasa de aprendizaje
- `--use-lora`: Usar LoRA para entrenamiento eficiente

### 📊 `eval.py` - Evaluación
**Responsabilidad**: Solo evaluación del mejor modelo entrenado.

```bash
# Evaluación básica
python cli/eval.py --dataset msmarco --samples 100

# Evaluación optimizada (3-5x más rápido)
python cli/eval.py --dataset msmarco --samples 1000 --query-batch-size 8

# Con perfil de configuración
python cli/eval.py --config-profile default --samples 500
```

**Parámetros principales:**
- `--config-profile` / `--config`: Cargar configuración desde YAML
- `--dataset`: Dataset para evaluación
- `--samples`: Número de muestras
- `--query-batch-size`: Procesar N queries simultáneamente (default: 8, optimización de velocidad)

**Características:**
- Busca automáticamente el mejor modelo en `./models/`
- Evalúa en dataset separado (nunca visto en entrenamiento)
- Calcula métricas IR: NDCG, MRR, MAP, Hit Rate
- **Nuevo**: Batch query processing para 3-5x speedup

### 🧪 `test.py` - Testing
**Responsabilidad**: Solo testing/inferencia del mejor modelo.

```bash
# Usar queries reales del dataset MS MARCO
python cli/test.py --domain msmarco

# Usar queries completamente frescas de MS MARCO
python cli/test.py --domain msmarco_fresh

# Usar queries de ejemplo por dominio
python cli/test.py --domain medical
python cli/test.py --domain legal
python cli/test.py --domain technical

# Usar tus propias queries
python cli/test.py --queries "¿Qué es machine learning?" "¿Cómo funciona Python?"

# Con perfil de configuración
python cli/test.py --config-profile default --domain technical
```

**Características:**
- **msmarco**: Queries reales del mismo 15% usado por eval (⚠️ no completamente fresco)
- **msmarco_fresh**: Queries completamente diferentes (recomendado para testing honesto)
- **Dominios**: Queries de ejemplo especializadas (medical, legal, technical)
- **Custom**: Tus propias queries específicas
- Muestra ranking de documentos por relevancia en tiempo real

### 🔄 `retrain.py` - Reentrenamiento
**Responsabilidad**: Solo reentrenamiento del mejor modelo.

```bash
# Reentrenamiento estándar
python cli/retrain.py --dataset legal_spanish --epochs 2 --learning-rate 1e-5

# Reentrenamiento con Quantum Resonance Fine-Tuning
python cli/retrain.py --dataset legal_spanish --epochs 2 --quantum-mode

# Con perfil de configuración (incluye quantum_mode)
python cli/retrain.py --config-profile retrain --dataset custom_data
```

**Parámetros principales:**
- `--config-profile` / `--config`: Cargar configuración desde YAML
- `--dataset`: Dataset adicional para reentrenamiento
- `--epochs`: Número de épocas adicionales (default: 2)
- `--learning-rate`: Tasa de aprendizaje (default: 1e-5, más bajo que training)
- `--samples`: Número de muestras adicionales
- `--quantum-mode`: **Nuevo** - Habilita quantum resonance fine-tuning (preservación de conocimiento)

**Características:**
- Carga automáticamente el mejor modelo
- Agrega datos adicionales para fine-tuning
- Learning rate más bajo para no "olvidar" lo aprendido
- **Quantum mode**: Preserva conocimiento existente con principios cuánticos (ver configs/retrain.yaml)

### 🧬 `quantum_train.py` - Quantum Training
**Responsabilidad**: Entrenamiento con Quantum Resonance.

```bash
# Quantum training desde cero
python cli/quantum_train.py --dataset msmarco --epochs 3

# Con configuración quantum
python cli/quantum_train.py --config-profile quantum_training
```

**Características:**
- Entrenamiento con principios de resonancia cuántica
- Mejor manejo de ejemplos difíciles
- Patrones de coherencia en predicciones

### 🌊 `quantum_retrain.py` - Quantum Retraining
**Responsabilidad**: Reentrenamiento especializado con quantum principles.

```bash
# Quantum retraining con análisis de resonancia
python cli/quantum_retrain.py --dataset new_domain --analyze-existing

# Configurar pesos de preservación
python cli/quantum_retrain.py --dataset new_domain \
  --preserve-knowledge 0.4 --resonance-alignment 0.3
```

**Parámetros quantum:**
- `--quantum-mode`: Modo quantum (adaptation, resonance, entanglement)
- `--preserve-knowledge`: Peso para preservación de conocimiento (0.0-1.0)
- `--resonance-alignment`: Peso para alineación de resonancia (0.0-1.0)
- `--analyze-existing`: Analizar patrones de resonancia antes de reentrenar

### 📊 `batch_eval.py` - Batch Evaluation
**Responsabilidad**: Evaluación en batch de múltiples datasets/configuraciones.

```bash
# Evaluar múltiples datasets
python cli/batch_eval.py --datasets msmarco legal medical

# Con configuración
python cli/batch_eval.py --config-profile default --datasets msmarco custom
```

**Características:**
- Evalúa múltiples datasets en una sola ejecución
- Compara rendimiento entre datasets
- Genera reporte comparativo

## 🎯 Principio de Responsabilidad Única

Cada comando tiene una sola responsabilidad clara:

| Comando | Entrada | Proceso | Salida | Nunca hace |
|---------|---------|---------|--------|------------|
| `train.py` | Dataset crudo | Entrenamiento | Modelo entrenado | Evaluación |
| `eval.py` | Modelo entrenado | Evaluación | Métricas IR | Entrenamiento |
| `test.py` | Modelo entrenado | Inference | Rankings | Entrenamiento |
| `retrain.py` | Modelo + datos | Fine-tuning | Modelo mejorado | Evaluación nueva |
| `quantum_train.py` | Dataset crudo | Quantum training | Modelo quantum | Evaluación |
| `quantum_retrain.py` | Modelo + datos | Quantum adaptation | Modelo adaptado | Evaluación nueva |
| `batch_eval.py` | Múltiples datasets | Batch evaluation | Reporte comparativo | Entrenamiento |

## 🔄 Flujos de Trabajo

### Workflow Básico
```bash
# 1. Entrenar modelo
python cli/train.py --config-profile default --dataset msmarco

# 2. Evaluar rendimiento (optimizado)
python cli/eval.py --dataset msmarco --query-batch-size 8

# 3. Probar con queries reales
python cli/test.py --domain technical

# 4. Reentrenar si es necesario
python cli/retrain.py --dataset legal_spanish --epochs 2
```

### Workflow con Quantum Mode
```bash
# 1. Entrenar con quantum resonance
python cli/quantum_train.py --config-profile quantum_training

# 2. Evaluar
python cli/eval.py --dataset msmarco

# 3. Reentrenar preservando conocimiento
python cli/retrain.py --dataset new_domain --quantum-mode --epochs 2
```

### Workflow de Testing Rápido
```bash
# Usar perfil quick_test para iteración rápida
python cli/train.py --config-profile quick_test
python cli/eval.py --config-profile quick_test
```

## 📂 Estructura de Modelos

Los comandos crean esta estructura automáticamente:

```
models/
├── trained_model/              # De train.py
│   ├── best/                   # Mejor checkpoint
│   │   ├── model.safetensors  # O adapter_model.safetensors (LoRA)
│   │   ├── config.json
│   │   └── model_config.json
│   ├── final/                  # Último checkpoint
│   ├── epoch_1/, epoch_2/      # Checkpoints intermedios
│   └── training_history.json
├── trained_model_retrained/    # De retrain.py
│   ├── best/
│   └── ...
└── trained_model_quantum_retrained/  # De quantum_retrain.py
    ├── best/
    └── ...
```

## ⚙️ Configuración por Defecto

- **Modelo**: `bert-base-uncased` (balance rendimiento/velocidad)
- **Batch size**: 16 (equilibrio memoria/velocidad)
- **Learning rate**: 2e-5 (óptimo para fine-tuning)
- **Dataset**: msmarco (puedes cambiar a cualquier .json en datasets/)
- **Evaluación**: Siempre busca el mejor modelo automáticamente
- **Query batch size**: 8 (evaluación optimizada)

## 🚀 Optimizaciones de Performance

### Evaluación Rápida
```bash
# 3-5x más rápido con batch processing
python cli/eval.py --samples 1000 --query-batch-size 8
```

### Entrenamiento Eficiente con LoRA
```bash
# Reduce uso de memoria ~3x
python cli/train.py --config-profile lora_training --use-lora
```

### Testing Rápido
```bash
# 1 época, 100 muestras para validación rápida
python cli/train.py --config-profile quick_test
```

## 🚨 Notas Importantes

- **Eval y test siempre usan el mejor modelo** disponible
- **Los datasets deben estar en la carpeta `datasets/`**
- **Los modelos se guardan en la carpeta `models/`**
- **Eval nunca usa datos de entrenamiento** (siempre datos separados)
- **Config profiles** permiten reproducibilidad completa
- **Quantum mode** en `retrain.py` preserva conocimiento existente (ver configs/retrain.yaml)
- **Query batch size** acelera evaluación procesando múltiples queries simultáneamente

## 🆘 Troubleshooting

### "No models directory found"
```bash
python cli/train.py --dataset msmarco  # Entrena un modelo primero
```

### "No trained models found"
```bash
python cli/train.py --dataset [dataset]  # Debes tener al menos un modelo entrenado
```

### Dataset no encontrado
```bash
ls datasets/  # Verifica que el archivo existe
python cli/train.py --dataset [nombre_sin_.json]
```

### Config profile no encontrado
```bash
ls configs/  # Verifica perfiles disponibles
python cli/train.py --config-profile default  # Usa perfil válido
```

### Data leakage en testing
```bash
# ❌ MAL: Usa mismo conjunto que eval
python cli/test.py --domain msmarco

# ✅ BUENO: Usa datos completamente frescos
python cli/test.py --domain msmarco_fresh

# ✅ BUENO: Usa queries custom nunca vistas
python cli/test.py --queries "tu query completamente nueva"
```

### Evaluación lenta
```bash
# ✅ Usa optimización de batch queries
python cli/eval.py --dataset msmarco --query-batch-size 8
```

### MS MARCO no disponible para test
```bash
# Si hay problemas con MS MARCO, usa dominios predefinidos
python cli/test.py --domain general  # Queries genéricas
python cli/test.py --domain medical  # Queries médicas
```

## 📚 Ejemplos de Configuración

### Crear config personalizado
```yaml
# configs/my_experiment.yaml
model:
  model_name: bert-base-uncased
  max_length: 256
  use_lora: true

training:
  epochs: 5
  batch_size: 16
  learning_rate: 0.00002

data:
  dataset: msmarco
  max_samples: 10000
```

```bash
# Usar config personalizado
python cli/train.py --config configs/my_experiment.yaml
```

### Sobrescribir config desde CLI
```bash
# Config dice epochs=5, pero queremos 10
python cli/train.py --config configs/my_experiment.yaml --epochs 10
```

## 🎓 Jerarquía de Configuración

Prioridad (mayor a menor):
1. **CLI arguments** (--epochs 10)
2. **Config file** (--config configs/my.yaml)
3. **Config profile** (--config-profile default)
4. **Defaults** (hardcoded en código)

Ejemplo:
```bash
# retrain.yaml tiene quantum_mode: false
# CLI override activa quantum mode
python cli/retrain.py --config-profile retrain --quantum-mode
```
