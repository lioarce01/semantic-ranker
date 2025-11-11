"""
Example: Optimize a trained model for production deployment.
"""

import logging
from semantic_ranker.optimization import ModelOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Optimize model for production."""

    logger.info("=== Model Optimization Example ===\n")

    # Note: This assumes you have a trained model at ./models/basic_reranker/final
    # If not, run 01_basic_training.py first, or use a pretrained model

    model_path = "./models/basic_reranker/final"

    # Check if model exists
    from pathlib import Path
    if not Path(model_path).exists():
        logger.warning(f"Model not found at {model_path}")
        logger.info("Please train a model first using 01_basic_training.py")
        logger.info("Or use a pretrained model like 'cross-encoder/ms-marco-MiniLM-L6-v2'")
        return

    # 1. Initialize optimizer
    logger.info("Step 1: Initializing optimizer...")
    optimizer = ModelOptimizer(model_path=model_path)

    # 2. Get original model size
    logger.info("\nStep 2: Analyzing original model...")
    original_size = optimizer.get_model_size(model_path)
    logger.info(f"Original model size: {original_size['size_mb']:.2f} MB")

    # 3. Export to ONNX
    logger.info("\nStep 3: Exporting to ONNX...")
    try:
        onnx_path = optimizer.export_to_onnx(
            output_path="./models/optimized/model.onnx",
            quantize=False,
            precision="fp32"
        )
        logger.info(f"ONNX model saved to: {onnx_path}")
    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
        logger.info("Continuing with other optimizations...")

    # 4. Quantize to INT8
    logger.info("\nStep 4: Quantizing to INT8...")
    try:
        quantized_path = optimizer.quantize(
            output_path="./models/optimized/quantized",
            quantization_type="dynamic",
            precision="int8"
        )
        logger.info(f"Quantized model saved to: {quantized_path}")
    except Exception as e:
        logger.warning(f"Quantization failed: {e}")

    # 5. Full optimization pipeline
    logger.info("\nStep 5: Running full optimization pipeline...")
    try:
        optimized_models = optimizer.optimize_for_inference(
            output_path="./models/optimized",
            enable_onnx=True,
            enable_quantization=True,
            target_device="cpu"
        )

        logger.info("\nOptimized models created:")
        for name, path in optimized_models.items():
            logger.info(f"  {name}: {path}")

    except Exception as e:
        logger.warning(f"Full optimization failed: {e}")

    # 6. Compare model sizes
    logger.info("\nStep 6: Comparing model sizes...")
    try:
        comparison = optimizer.compare_models(
            original_path=model_path,
            optimized_paths=optimized_models
        )
    except:
        logger.info("Skipping comparison due to optimization errors")

    # 7. Benchmark performance (optional)
    logger.info("\nStep 7: Benchmarking performance...")
    try:
        benchmark_results = optimizer.benchmark(
            model_paths={
                'original': model_path,
                **optimized_models
            },
            num_samples=50
        )

        logger.info("\nBenchmark results:")
        for model_name, stats in benchmark_results.items():
            if 'error' not in stats:
                logger.info(f"\n{model_name}:")
                logger.info(f"  Avg latency: {stats['avg_latency_ms']:.2f} ms")
                logger.info(f"  P95 latency: {stats['p95_latency_ms']:.2f} ms")
                logger.info(f"  Throughput: {stats['throughput_qps']:.2f} queries/sec")

    except Exception as e:
        logger.warning(f"Benchmarking failed: {e}")

    logger.info("\n=== Optimization Complete ===")
    logger.info("Optimized models saved to: ./models/optimized/")


if __name__ == "__main__":
    main()
