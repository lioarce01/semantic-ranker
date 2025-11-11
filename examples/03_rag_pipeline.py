"""
Example: Use trained reranker in a complete RAG pipeline.
"""

import logging
from semantic_ranker.rag import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate RAG pipeline with reranking."""

    logger.info("=== RAG Pipeline Example ===\n")

    # Sample document corpus
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning is a type of machine learning based on artificial neural networks with multiple layers.",
        "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "Natural language processing (NLP) is a branch of AI focused on interaction between computers and human language.",
        "Computer vision enables computers to extract meaningful information from digital images and videos.",
        "Reinforcement learning is a type of machine learning where agents learn by interacting with an environment.",
        "Supervised learning uses labeled data to train models to make predictions.",
        "Unsupervised learning finds patterns in unlabeled data without explicit instruction.",
        "Transfer learning is a technique where a model trained on one task is adapted for a different task.",
        "The Transformer architecture revolutionized NLP with its attention mechanism."
    ]

    # 1. Initialize pipeline
    logger.info("Step 1: Initializing RAG pipeline...")

    # Option A: Without reranker (retrieval only)
    pipeline_no_rerank = RAGPipeline(
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model=None,  # No reranker
        top_k_retrieval=10,
        top_k_rerank=3
    )

    # Option B: With reranker (two-stage retrieval)
    # pipeline = RAGPipeline(
    #     retriever_model="sentence-transformers/all-MiniLM-L6-v2",
    #     reranker_model="./models/basic_reranker/final",  # Use trained reranker
    #     top_k_retrieval=10,
    #     top_k_rerank=3
    # )

    pipeline = pipeline_no_rerank  # Use no-rerank version for demo

    # 2. Index documents
    logger.info("\nStep 2: Indexing documents...")
    pipeline.index_documents(documents)

    # 3. Query and retrieve
    queries = [
        "What is machine learning?",
        "Explain deep learning and neural networks",
        "How does NLP work?"
    ]

    logger.info("\nStep 3: Retrieving and reranking...\n")

    for query in queries:
        logger.info(f"Query: {query}")

        # Retrieve and rerank
        results = pipeline.retrieve_and_rerank(
            query,
            top_k_retrieval=5,
            top_k_rerank=3,
            return_scores=True
        )

        # Display results
        logger.info("Top results:")
        for i, result in enumerate(results, 1):
            doc = result['document']
            score = result.get('score', 0)
            logger.info(f"  {i}. (score: {score:.4f}) {doc[:80]}...")

        logger.info("")

    # 4. Get context for LLM
    logger.info("Step 4: Getting formatted context for LLM...")
    query = "What is the difference between supervised and unsupervised learning?"
    context = pipeline.get_context_for_llm(query, top_k=3)

    logger.info(f"\nQuery: {query}\n")
    logger.info("Context for LLM:")
    logger.info(context)

    # 5. Augment prompt
    logger.info("\n\nStep 5: Creating augmented prompt...")
    augmented_prompt = pipeline.augment_prompt(query, top_k=3)

    logger.info("Augmented Prompt:")
    logger.info(augmented_prompt)

    # 6. Benchmark performance
    logger.info("\n\nStep 6: Benchmarking pipeline...")
    benchmark_queries = queries * 10  # Repeat for better stats
    stats = pipeline.benchmark(benchmark_queries, num_runs=3)

    logger.info("\n=== Pipeline Demo Complete ===")


if __name__ == "__main__":
    main()
