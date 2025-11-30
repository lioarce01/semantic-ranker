# Scripts Utilitarios

Esta carpeta contiene scripts utilitarios para facilitar el uso de semantic-ranker.

## Scripts Disponibles

### 🔍 `check_dependencies.py`
Verifica que todas las dependencias requeridas estén instaladas.

```bash
python scripts/check_dependencies.py
```

### 📋 `list_available_models.py`
Lista todos los modelos entrenados disponibles para reentrenamiento.

```bash
python scripts/list_available_models.py
```

### 🔄 `retrain_best_model.py`
Reentrena automáticamente el modelo best con datos adicionales.

```bash
python scripts/retrain_best_model.py
```

### 🎯 `train_any_domain.py`
Entrena un modelo para cualquier dominio disponible en `datasets/`.

```bash
python scripts/train_any_domain.py --list          # Ver dominios disponibles
python scripts/train_any_domain.py legal           # Entrenar legal
python scripts/train_any_domain.py medical         # Entrenar médico
python scripts/train_any_domain.py technical       # Entrenar técnico
```

### 🆕 `create_fresh_test_set.py` - Crear Test Set Fresco
**Propósito**: Crear un conjunto de test completamente separado para evitar data leakage.

```bash
python scripts/create_fresh_test_set.py --samples 200 --output datasets/my_fresh_test.json
```

**Por qué es importante:**
- **Data Leakage Prevention**: Asegura que el test use datos nunca vistos
- **Evaluación Honesta**: Métricas más realistas del rendimiento del modelo
- **Reproducibilidad**: Seed diferente garantiza separación completa

### 🚀 `run.py` (Script Maestro)
Ejecuta cualquiera de los scripts anteriores desde un solo comando.

```bash
python scripts/run.py check-deps     # Verificar dependencias
python scripts/run.py list-models    # Listar modelos
python scripts/run.py retrain-best   # Reentrenar best
python scripts/run.py train-domain legal  # Entrenar dominio
```

## Estructura de Ejecución

Todos los scripts siguen el mismo patrón:

1. **Verificación**: Chequean que el entorno esté listo
2. **Ejecución**: Realizan la tarea principal
3. **Resultado**: Muestran resultados y próximos pasos

## Integración con CLI

Los scripts se integran perfectamente con los comandos CLI en `cli/`:

- Los scripts generan salida compatible con CLI
- Se pueden combinar para flujos de trabajo complejos
- Complementan la funcionalidad de los comandos principales

## Desarrollo

Para agregar un nuevo script utilitario:

1. Crear el script en `scripts/nombre_script.py`
2. Seguir el patrón de argumentos y logging
3. Probar que funcione correctamente
4. Documentar en este README

## Troubleshooting

### "No models directory found"
```bash
python cli/train.py  # Entrena un modelo primero
```

### Dataset no encontrado
```bash
ls datasets/  # Verifica que el archivo existe
```

### Data leakage en testing
```bash
# Crear test set completamente fresco
python scripts/create_fresh_test_set.py --samples 100 --output datasets/fresh_test.json

# Usar para testing honesto
python cli/test.py --queries @ datasets/fresh_test.json
```
