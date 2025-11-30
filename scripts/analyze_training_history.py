#!/usr/bin/env python3
"""
Analyze training history from trained models
"""

import json
import os
from pathlib import Path

def analyze_training_history(model_path):
    """Analyze training history for a model"""
    history_file = Path(model_path) / 'training_history.json'
    if not history_file.exists():
        return None

    with open(history_file, 'r') as f:
        data = json.load(f)

    train_losses = data.get('train_loss', [])
    val_losses = data.get('val_loss', [])

    result = {
        'model': model_path,
        'epochs': len(train_losses),
        'avg_train_loss': sum(train_losses) / len(train_losses) if train_losses else 0,
        'avg_val_loss': sum(val_losses) / len(val_losses) if val_losses else 0,
        'final_train_loss': train_losses[-1] if train_losses else 0,
        'final_val_loss': val_losses[-1] if val_losses else 0
    }
    return result

def main():
    # Analizar modelos disponibles
    models_dir = Path('models')
    results = []

    if models_dir.exists():
        # Buscar en subdirectorios de models/
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                result = analyze_training_history(str(model_dir))
                if result:
                    results.append(result)

    # También buscar en el root (por si hay models_retrained)
    root_models = ['models_retrained']
    for model_name in root_models:
        model_path = Path(model_name)
        if model_path.exists():
            result = analyze_training_history(str(model_path))
            if result:
                results.append(result)

    print('=== ANÁLISIS DE LOSS EN MODELOS ENTRENADOS ===')
    print()

    if not results:
        print('❌ No se encontraron archivos training_history.json')
        return

    for result in results:
        print(f'📊 Modelo: {result["model"]}')
        print(f'   Epochs: {result["epochs"]}')
        print(f'   Avg Train Loss: {result["avg_train_loss"]:.6f}')
        print(f'   Avg Val Loss: {result["avg_val_loss"]:.6f}')
        print(f'   Final Train Loss: {result["final_train_loss"]:.6f}')
        print(f'   Final Val Loss: {result["final_val_loss"]:.6f}')
        print()

    # Comparación si hay múltiples modelos
    if len(results) > 1:
        print('🏆 COMPARACIÓN:')
        best_model = min(results, key=lambda x: x['final_val_loss'])
        print(f'   Mejor modelo: {best_model["model"]}')
        print('.6f')
        print()

if __name__ == "__main__":
    main()
