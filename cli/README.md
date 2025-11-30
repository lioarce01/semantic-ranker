# Command Line Interface

Interfaz de línea de comandos con responsabilidades separadas para semantic-ranker.

## 📋 Comandos Disponibles

### 🚀 `train.py` - Entrenamiento
**Responsabilidad**: Solo entrenamiento de modelos.

```bash
python cli/train.py --dataset msmarco --epochs 3 --batch-size 16
```

**Parámetros:**
- `--dataset`: Dataset de entrenamiento (msmarco o archivos en datasets/)
- `--model-name`: Modelo base (default: distilbert-base-uncased)
- `--epochs`: Número de epochs
- `--batch-size`: Tamaño del batch
- `--learning-rate`: Tasa de aprendizaje
- `--use-lora`: Usar LoRA para entrenamiento eficiente

### 📊 `eval.py` - Evaluación
**Responsabilidad**: Solo evaluación del mejor modelo entrenado.

```bash
python cli/eval.py --dataset msmarco --samples 100
```

**Características:**
- Busca automáticamente el mejor modelo en `./models/`
- Evalúa en dataset separado (nunca visto en entrenamiento)
- Calcula métricas IR: NDCG, MRR, MAP, Hit Rate

### 🧪 `test.py` - Testing
**Responsabilidad**: Solo testing/inferencia del mejor modelo.

```bash
# Usar queries reales del dataset MS MARCO (mismo 15% que eval)
python cli/test.py --domain msmarco

# Usar queries completamente frescas de MS MARCO (diferente seed)
python cli/test.py --domain msmarco_fresh

# Usar queries de ejemplo por dominio
python cli/test.py --domain medical
python cli/test.py --domain legal
python cli/test.py --domain technical

# Usar tus propias queries
python cli/test.py --queries "¿Qué es machine learning?" "¿Cómo funciona Python?"
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
python cli/retrain.py --dataset legal_spanish --epochs 2 --learning-rate 1e-5
```

**Características:**
- Carga automáticamente el mejor modelo
- Agrega datos adicionales para fine-tuning
- Learning rate más bajo para no "olvidar" lo aprendido

## 🎯 Principio de Responsabilidad Única

Cada comando tiene una sola responsabilidad clara:

| Comando | Entrada | Proceso | Salida | Nunca hace |
|---------|---------|---------|--------|------------|
| `train.py` | Dataset crudo | Entrenamiento | Modelo entrenado | Evaluación |
| `eval.py` | Modelo entrenado | Evaluación | Métricas IR | Entrenamiento |
| `test.py` | Modelo entrenado | Inference | Rankings | Entrenamiento |
| `retrain.py` | Modelo + datos | Fine-tuning | Modelo mejorado | Evaluación nueva |

## 🔄 Flujo de Trabajo Típico

```bash
# 1. Entrenar modelo
python cli/train.py --dataset msmarco --epochs 3

# 2. Evaluar rendimiento
python cli/eval.py --dataset msmarco

# 3. Probar con queries reales
python cli/test.py --domain technical

# 4. Reentrenar si es necesario
python cli/retrain.py --dataset legal_spanish --epochs 2
```

## 📂 Estructura de Modelos

Los comandos crean esta estructura automáticamente:

```
models/
├── trained_model/           # De train.py
│   ├── best/               # Mejor checkpoint
│   ├── final/              # Último checkpoint
│   ├── epoch_1/, epoch_2/  # Checkpoints intermedios
│   └── training_history.json
└── trained_model_retrained/ # De retrain.py
    ├── best/
    └── ...
```

## ⚙️ Configuración por Defecto

- **Modelo**: `distilbert-base-uncased` (rápido y eficiente)
- **Batch size**: 16 (equilibrio memoria/velocidad)
- **Learning rate**: 2e-5 (óptimo para fine-tuning)
- **Dataset**: msmarco (puedes cambiar a cualquier .json en datasets/)
- **Evaluación**: Siempre busca el mejor modelo automáticamente

## 🚨 Notas Importantes

- **Eval y test siempre usan el mejor modelo** disponible
- **Los datasets deben estar en la carpeta `datasets/`**
- **Los modelos se guardan en la carpeta `models/`**
- **Eval nunca usa datos de entrenamiento** (siempre datos separados)

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

### Data leakage en testing
```bash
# ❌ MAL: Usa mismo conjunto que eval
python cli/test.py --domain msmarco

# ✅ BUENO: Usa datos completamente frescos
python cli/test.py --domain msmarco_fresh

# ✅ BUENO: Usa queries custom nunca vistas
python cli/test.py --queries "tu query completamente nueva"
```

### MS MARCO no disponible para test
```bash
# Si hay problemas con MS MARCO, usa dominios predefinidos
python cli/test.py --domain general  # Queries genéricas
python cli/test.py --domain medical  # Queries médicas
```
