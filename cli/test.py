#!/usr/bin/env python3
"""
Test the best trained model with sample queries.
Responsibility: Only testing/inference, finds best model automatically.
"""

import argparse
from pathlib import Path

# Import shared utilities
from cli.utils import (
    setup_project_path,
    setup_logging,
    find_best_model
)

# Setup project imports
setup_project_path()

# Setup logging
logger = setup_logging()

# Now import semantic_ranker modules
from semantic_ranker.models import CrossEncoderModel


def get_sample_queries(domain="general"):
    """Get sample queries for testing."""

    if domain == "msmarco":
        # Load real queries from MS MARCO test set (same as eval uses)
        try:
            from semantic_ranker.data import MSMARCODataLoader
            loader = MSMARCODataLoader()
            _, _, test_data = loader.load_and_split(max_samples=100)  # Small sample for testing
            queries = [item.get('query', '') for item in test_data[10:20]]  # Different 10 queries (11-20)
            queries = [q for q in queries if q]  # Filter empty queries
            if queries:
                logger.warning(f"⚠️ Using same test set as eval (queries 11-20). For completely fresh data, use --domain general")
                logger.info(f"📚 Loaded {len(queries)} real queries from MS MARCO test set")
                return queries
        except Exception as e:
            logger.warning(f"Could not load MS MARCO queries: {e}")

    elif domain == "msmarco_fresh":
        # Try to get completely fresh data with different seed
        try:
            from semantic_ranker.data import MSMARCODataLoader
            loader = MSMARCODataLoader()
            _, _, fresh_test_data = loader.load_and_split(max_samples=100, seed=12345)  # Different seed
            queries = [item.get('query', '') for item in fresh_test_data[:10]]
            queries = [q for q in queries if q]
            if queries:
                logger.info(f"🆕 Loaded {len(queries)} completely fresh queries from MS MARCO (different seed)")
                return queries
        except Exception as e:
            logger.warning(f"Could not load fresh MS MARCO queries: {e}")

    # Fallback to hardcoded queries
    queries_by_domain = {
        "general": [
            "¿Cómo funciona el machine learning?",
            "¿Qué es la inteligencia artificial?",
            "¿Cómo programar en Python?",
        ],
        "medical": [
            "¿Cuáles son los síntomas de la diabetes?",
            "¿Cómo se trata la hipertensión?",
            "¿Qué es la insuficiencia cardíaca?",
        ],
        "legal": [
            "¿Qué es un contrato de compraventa?",
            "¿Cuáles son los derechos del consumidor?",
            "¿Cómo funciona el derecho de desistimiento?",
        ],
        "technical": [
            "¿Cómo implementar autenticación JWT?",
            "¿Qué es Docker y para qué sirve?",
            "¿Cómo optimizar una base de datos SQL?",
        ]
    }

    return queries_by_domain.get(domain, queries_by_domain["general"])


def main():
    parser = argparse.ArgumentParser(description='Test a trained model')
    parser.add_argument('--model-path', help='Path to specific model directory (if not provided, uses best model)')
    parser.add_argument('--queries', nargs='+',
                       help='Custom queries to test (optional)')
    parser.add_argument('--domain', choices=['general', 'medical', 'legal', 'technical', 'msmarco', 'msmarco_fresh'],
                       default='msmarco', help='Domain for sample queries. "msmarco" uses same test set as eval, "msmarco_fresh" uses completely different data')
    parser.add_argument('--documents', type=int, default=4,
                       help='Number of documents to rank per query')

    args = parser.parse_args()

    logger.info("=== Model Testing ===")
    print()

    # 1. Get model path
    if args.model_path:
        model_path = args.model_path
        if not Path(model_path).exists():
            logger.error(f"❌ Model path does not exist: {model_path}")
            return
        logger.info(f"📍 Using specified model: {model_path}")
    else:
        model_path = find_best_model()
        if not model_path:
            return

    logger.info("Loading model...")
    model = CrossEncoderModel.load(model_path)
    logger.info("✅ Model loaded successfully")

    # 2. Get queries
    if args.queries:
        queries = args.queries
        logger.info(f"Using {len(queries)} custom queries")
    else:
        queries = get_sample_queries(args.domain)
        logger.info(f"Using {len(queries)} sample queries from {args.domain} domain")

    # 3. Sample documents for each query (mock documents for testing)
    sample_docs = [
        "Esta es una respuesta detallada y completa que explica el tema solicitado.",
        "Información adicional relevante sobre el tema consultado.",
        "Contenido complementario que puede ser útil para entender mejor.",
        "Otra perspectiva o ejemplo relacionado con la pregunta.",
        "Información adicional que proporciona contexto importante.",
        "Respuesta alternativa que aborda el mismo tema desde otro ángulo.",
        "Explicación técnica más detallada del concepto solicitado.",
        "Ejemplos prácticos y casos de uso reales."
    ]

    # 4. Test each query
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 60)

        # Select random documents for this query
        import random
        random.seed(42 + i)  # Deterministic but different per query
        selected_docs = random.sample(sample_docs, min(args.documents, len(sample_docs)))

        # Rank documents
        scores = model.predict([query] * len(selected_docs), selected_docs)

        # Sort by score
        ranked_results = sorted(zip(selected_docs, scores), key=lambda x: x[1], reverse=True)

        # Display results
        for rank, (doc, score) in enumerate(ranked_results, 1):
            status = "⭐" if rank == 1 else "👍" if rank <= 3 else "ℹ️"
            preview = doc[:80] + "..." if len(doc) > 80 else doc
            print(f"{status} [{score:.4f}] {preview}")

    print("\n✅ Testing completed!")
    print("💡 The model ranks documents by relevance to each query.")
    print("   Higher scores = More relevant to the query")


if __name__ == "__main__":
    main()
