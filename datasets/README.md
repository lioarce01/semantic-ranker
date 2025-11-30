# Datasets para Fine-tuning del Reranker

Esta carpeta contiene datasets de diferentes dominios para entrenar el modelo de reranking.

## 🎯 Estrategias de Entrenamiento

### Opción 1: Modelo por Dominio (Recomendado)
- Entrena un modelo específico para cada dominio
- Mejor precisión y especialización
- Más fácil de mantener y actualizar

### Opción 2: Modelo Multi-Dominio
- Entrena un solo modelo con datos de múltiples dominios
- Mayor generalización pero posible confusión
- Útil cuando dominios están relacionados

### Opción 3: Modelo Base + Fine-tuning por Dominio
- Entrena base con datos generales
- Fine-tuning específico por dominio
- Mejor de ambos mundos

## 📊 Datasets Disponibles

| Dominio | Archivo | Muestras | Características |
|---------|---------|----------|----------------|
| **Legal** | `legal_spanish.json` | 150 | Contratos, leyes, casos judiciales |
| **Médico** | `medical_spanish.json` | 120 | Síntomas, tratamientos, diagnósticos |
| **Técnico** | `technical_dev.json` | 100 | Programación, DevOps, arquitectura |
| **E-commerce** | `ecommerce_products.json` | 80 | Productos, reseñas, categorías |
| **Educativo** | `education_academic.json` | 90 | Contenido académico, cursos, tutoriales |
| **Multi-dominio** | `multidomain_balanced.json` | 200 | Mezcla balanceada de todos |

## 🔍 Análisis: ¿Múltiples dominios confunden al modelo?

### ✅ Ventajas del Multi-dominio:
- **Generalización**: Modelo aprende patrones transversales
- **Transferencia**: Conocimiento útil entre dominios relacionados
- **Eficiencia**: Un solo modelo para múltiples tareas

### ⚠️ Riesgos del Multi-dominio:
- **Confusión**: Dominios muy diferentes pueden interferir
- **Sub-optimización**: Rendimiento inferior en dominios específicos
- **Datos desbalanceados**: Dominio con más datos domina el aprendizaje

### 🎯 Recomendación:
- **Usa multi-dominio** cuando dominios están relacionados (ej: legal → contratos, médico → salud)
- **Usa modelos separados** cuando dominios son muy diferentes
- **Balancea los datos** si combinas dominios
- **Evalúa rendimiento** por dominio individual

## 📋 Formato de Datos

Cada dataset sigue el formato:
```json
[
  {
    "query": "¿Pregunta del usuario?",
    "positive": "Respuesta correcta y detallada...",
    "negatives": ["Respuesta irrelevante 1", "Respuesta irrelevante 2"]
  }
]
```

## 🚀 Scripts de Entrenamiento

### Entrenar modelo específico por dominio:
```bash
# Legal
python examples/01_custom_domain_training.py  # Modificar para usar datasets/legal_spanish.json

# Médico
python examples/01_custom_domain_training.py  # Modificar para usar datasets/medical_spanish.json

# Técnico
python examples/01_custom_domain_training.py  # Modificar para usar datasets/technical_dev.json
```

### Entrenar modelo multi-dominio:
```bash
# Comparar automáticamente modelos específicos vs multi-dominio
python examples/08_multidomain_comparison.py
```

### Demo de uso:
```bash
# Probar modelo entrenado
python examples/07_domain_reranking_demo.py
```

## 🔬 Experimento: ¿Confunden múltiples dominios?

El script `08_multidomain_comparison.py` entrena automáticamente:

1. **Modelos específicos**: Uno por cada dominio (legal, médico, técnico)
2. **Modelo multi-dominio**: Un solo modelo con datos de todos los dominios

**Mide**: Precisión en cada dominio para ambos enfoques

### Resultados Esperados:

- **Si dominios son similares**: Multi-dominio funciona igual o mejor
- **Si dominios son diferentes**: Modelos específicos funcionan mejor
- **Trade-off**: Especialización vs simplicidad de mantenimiento

## 📊 Estadísticas de Datasets

| Dataset | Muestras | Tokens Query (avg) | Tokens Doc (avg) | Relación P/N |
|---------|----------|-------------------|------------------|-------------|
| Legal | 150 | 12 | 85 | 1:2 |
| Médico | 120 | 11 | 92 | 1:2 |
| Técnico | 100 | 10 | 78 | 1:2 |
| E-commerce | 80 | 9 | 65 | 1:2 |
| Educativo | 90 | 13 | 88 | 1:2 |
| Multi-dominio | 200 | 11 | 82 | 1:2 |

## 🎯 Recomendaciones de Uso

### Para Producción:
- **Dominios relacionados**: Usar multi-dominio
- **Dominios diferentes**: Modelos específicos
- **Datasets pequeños**: Multi-dominio para más datos
- **Latencia crítica**: Modelos específicos (más pequeños)

### Para Experimentación:
- Empezar con modelo específico por dominio
- Medir rendimiento con `08_multidomain_comparison.py`
- Iterar basado en resultados

## 🔧 Personalización

Para crear tu propio dataset:

1. **Recopilar datos**: 50-200 ejemplos por dominio
2. **Formato JSON**: Seguir estructura mostrada
3. **Balance**: Relación 1:1 positivos:negativos
4. **Entrenar**: Usar `01_custom_domain_training.py`
5. **Evaluar**: Medir NDCG@10, precisión en dominio
