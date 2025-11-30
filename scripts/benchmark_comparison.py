#!/usr/bin/env python3
"""
Benchmark Comparison Script

Compara el rendimiento de tu modelo con benchmarks state-of-the-art.
Ejecuta evaluaciones formales y muestra comparación con modelos líderes.
"""

import sys
import os
import json
from pathlib import Path
import argparse

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from semantic_ranker.evaluation.evaluator import RankerEvaluator
from semantic_ranker.data.data_loader import MSMARCODataLoader
from cli.eval import find_best_model

def load_benchmark_results():
    """Carga resultados de benchmarks conocidos"""
    benchmarks = {
        "bge-reranker-v2.0": {
            "ndcg@10": 0.866,
            "mrr@10": 0.879,
            "map@100": 0.812,
            "recall@100": 0.923
        },
        "flashrank": {
            "ndcg@10": 0.842,
            "mrr@10": 0.855,
            "map@100": 0.783,
            "recall@100": 0.915
        },
        "monot5": {
            "ndcg@10": 0.814,
            "mrr@10": 0.831,
            "map@100": 0.756,
            "recall@100": 0.908
        },
        "bge-large-en": {
            "ndcg@10": 0.812,
            "mrr@10": 0.829,
            "map@100": 0.742,
            "recall@100": 0.918
        }
    }
    return benchmarks

def convert_data_format(data):
    """Convierte datos del formato MSMARCO al formato esperado por evaluator"""
    converted = []
    for item in data:
        if 'documents' in item and 'labels' in item:
            # Ya está en el formato correcto
            converted.append(item)
        elif 'positive' in item and 'negatives' in item:
            # Convertir del formato antiguo
            positive = item['positive']
            negatives = item.get('negatives', [])

            documents = [positive] + negatives
            labels = [1] + [0] * len(negatives)  # 1 para positivo, 0 para negativos

            converted_item = {
                'query': item['query'],
                'documents': documents,
                'labels': labels
            }
            converted.append(converted_item)
        else:
            # Formato desconocido, intentar usar como está
            converted.append(item)

    return converted

def evaluate_model(model_path=None):
    """Evalúa un modelo específico o el mejor disponible en el test set"""
    if model_path:
        print(f"🔍 Evaluando modelo específico: {model_path}")
        if not Path(model_path).exists():
            print(f"❌ Model path does not exist: {model_path}")
            return None
    else:
        print("🔍 Evaluando el mejor modelo disponible...")
        # Encontrar el mejor modelo automáticamente
        model_path = find_best_model()
        if not model_path:
            print("❌ No se encontró ningún modelo entrenado")
            return None

    print(f"📍 Usando modelo: {model_path}")

    # Cargar datos de test
    loader = MSMARCODataLoader()
    _, _, test_data = loader.load_and_split(max_samples=1000)  # Usar muestra para evaluación rápida

    # Convertir formato si es necesario
    test_data = convert_data_format(test_data)

    print(f"📊 Datos convertidos: {len(test_data)} queries")
    if test_data:
        sample = test_data[0]
        print(f"📊 Formato: query={sample.get('query', '')[:50]}...")
        print(f"📊 Documents: {len(sample.get('documents', []))}")
        print(f"📊 Labels: {sample.get('labels', [])}")

    # Crear evaluador
    evaluator = RankerEvaluator(model_path=model_path)

    # Evaluar
    results = evaluator.evaluate(test_data, metrics=["ndcg@10", "mrr@10", "map@100", "recall@100"])

    return results

def print_comparison(your_results, benchmarks):
    """Imprime comparación detallada"""
    print("\n" + "="*80)
    print("🏆 COMPARACIÓN CON MODELOS STATE-OF-THE-ART")
    print("="*80)

    print("\n📊 TU MODELO:")

    def safe_format(value, default="N/A"):
        """Formatea valores de manera segura"""
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
            return f"{float(value):.4f}"
        else:
            return str(value) if value is not None else default

    print(f"   NDCG@10: {safe_format(your_results.get('ndcg@10'))}")
    print(f"   MRR@10:  {safe_format(your_results.get('mrr@10'))}")
    print(f"   MAP@100: {safe_format(your_results.get('map@100'))}")
    print(f"   Recall@100: {safe_format(your_results.get('recall@100'))}")

    print("\n" + "-"*80)

    for model_name, metrics in benchmarks.items():
        print(f"\n🔄 {model_name.upper()}:")
        print(f"   NDCG@10: {metrics['ndcg@10']:.4f}")
        print(f"   MRR@10:  {metrics['mrr@10']:.4f}")
        print(f"   MAP@100: {metrics['map@100']:.4f}")
        print(f"   Recall@100: {metrics['recall@100']:.4f}")

        # Comparación
        if 'ndcg@10' in your_results:
            your_ndcg_raw = your_results['ndcg@10']
            if isinstance(your_ndcg_raw, str):
                your_ndcg = float(your_ndcg_raw)
            else:
                your_ndcg = your_ndcg_raw

            diff = your_ndcg - metrics['ndcg@10']
            if diff > 0:
                print(f"   📈 Diferencia NDCG: +{diff:.4f} (¡Mejor!)")
            else:
                print(f"   📉 Diferencia NDCG: {diff:.4f}")
        else:
            print("   ⚠️ No hay datos para comparar NDCG")

def print_interpretation(your_results):
    """Interpreta los resultados"""
    print("\n" + "="*80)
    print("🎯 INTERPRETACIÓN DE RESULTADOS")
    print("="*80)

    if 'ndcg@10' not in your_results:
        print("❌ No se pudieron calcular métricas. Verificar evaluación.")
        return

    ndcg_raw = your_results['ndcg@10']

    # Convertir a float si es necesario
    if isinstance(ndcg_raw, str):
        ndcg = float(ndcg_raw)
    else:
        ndcg = ndcg_raw

    if ndcg >= 0.90:
        rating = "🏆 STATE-OF-THE-ART"
        desc = "¡Extraordinario! Pocos modelos alcanzan este nivel."
    elif ndcg >= 0.85:
        rating = "🥇 EXCELENTE"
        desc = "Top tier. Compite con los mejores modelos comerciales."
    elif ndcg >= 0.80:
        rating = "🥈 MUY BUENO"
        desc = "Competitivo. Útil para aplicaciones reales."
    elif ndcg >= 0.75:
        rating = "🥉 BUENO"
        desc = "Aceptable para producción. Puede mejorarse."
    elif ndcg >= 0.70:
        rating = "⚠️ ACEPTABLE"
        desc = "Funcional pero necesita mejoras significativas."
    else:
        rating = "❌ NECESITA MEJORAS"
        desc = "Requiere optimización importante."

    print(f"\n{rating}")
    print(f"NDCG@10: {ndcg:.4f}")
    print(f"Descripción: {desc}")

    # Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    if ndcg < 0.75:
        print("• Entrenar con más datos (hard negatives)")
        print("• Usar mejor arquitectura base")
        print("• Ajustar hiperparámetros")
        print("• Implementar técnicas de regularización")
    elif ndcg < 0.80:
        print("• Fine-tuning adicional")
        print("• Más epochs de entrenamiento")
        print("• Mejor selección de datos")
    else:
        print("• ¡Excelente trabajo!")
        print("• Considera publicar resultados")
        print("• Evaluar en más benchmarks")

def main():
    parser = argparse.ArgumentParser(description='Compare your model with state-of-the-art benchmarks')
    parser.add_argument('--model-path', help='Path to specific model directory (if not provided, uses best model)')

    args = parser.parse_args()

    print("🚀 Benchmark Comparison Tool")
    print("Compara tu modelo con state-of-the-art")
    print()

    # Cargar benchmarks conocidos
    benchmarks = load_benchmark_results()

    try:
        # Evaluar modelo actual
        your_results = evaluate_model(args.model_path)
        if your_results is None:
            return
        print("✅ Evaluación completada")

        # Mostrar comparación
        print_comparison(your_results, benchmarks)

        # Interpretar resultados
        print_interpretation(your_results)

    except Exception as e:
        print(f"❌ Error durante evaluación: {e}")
        print("💡 Asegúrate de que el modelo esté entrenado y disponible")

        # Mostrar solo benchmarks conocidos
        print("\n📊 Benchmarks de referencia (para comparación futura):")
        for model_name, metrics in benchmarks.items():
            print(f"\n{model_name.upper()}:")
            print(f"   NDCG@10: {metrics['ndcg@10']:.4f}")

if __name__ == "__main__":
    main()
