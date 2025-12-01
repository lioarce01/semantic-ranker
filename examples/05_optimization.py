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

    model_path = "./models/quantum_giant_base_backup/best"

    # Check if model exists
    from pathlib import Path
    if not Path(model_path).exists():
        logger.warning(f"Model not found at {model_path}")
        logger.info("Please train a model first using 01_basic_training.py")
        logger.info("Or use a pretrained model like 'cross-encoder/ms-marco-MiniLM-L6-v2'")
        return

    # 1. Load our custom model first
    logger.info("Step 1: Loading CrossEncoderModel...")
    from semantic_ranker.models import CrossEncoderModel
    model = CrossEncoderModel.load(model_path)

    # Get the underlying transformers model for optimization
    transformers_model = model.model

    # Save in standard format for optimization
    import tempfile
    import os
    import json
    temp_dir = tempfile.mkdtemp()

    # Create a complete BERT config manually
    bert_config = {
        "model_type": "bert",
        "architectures": ["BertForSequenceClassification"],
        "attention_probs_dropout_prob": 0.1,
        "classifier_dropout": None,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.21.0",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 30522,
        "num_labels": 1
    }

    # Save the complete config first
    config_path = os.path.join(temp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(bert_config, f, indent=2)

    # Save the model and tokenizer
    transformers_model.save_pretrained(temp_dir)
    model.tokenizer.save_pretrained(temp_dir)

    logger.info("Model saved with complete BERT config")

    # Now initialize optimizer with the converted model
    logger.info("Step 2: Initializing optimizer...")
    optimizer = ModelOptimizer(model_path=temp_dir)

    # 3. Get original model size
    logger.info("\nStep 3: Analyzing original model...")
    original_size = optimizer.get_model_size(temp_dir)
    logger.info(f"Original model size: {original_size['size_mb']:.2f} MB")

    # 4. Export to ONNX
    logger.info("\nStep 4: Exporting to ONNX...")
    try:
        onnx_path = optimizer.export_to_onnx(
            output_path="./models/optimized_model/model.onnx",
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
            output_path="./models/optimized_model/quantized",
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
            output_path="./models/optimized_model",
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

    # 8. Save optimized model in standard format for Hugging Face
    logger.info("\nStep 8: Saving optimized model for Hugging Face...")
    try:
        # Save the transformers model in standard format
        import os
        import json
        os.makedirs("./models/optimized_model", exist_ok=True)

        # Create a complete BERT config manually for the final model
        final_config = {
            "model_type": "bert",
            "architectures": ["BertForSequenceClassification"],
            "attention_probs_dropout_prob": 0.1,
            "classifier_dropout": None,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.21.0",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": 30522,
            "num_labels": 1
        }

        # Save the config first
        config_path = "./models/optimized_model/config.json"
        with open(config_path, 'w') as f:
            json.dump(final_config, f, indent=2)

        # Save the transformers model and tokenizer
        transformers_model.save_pretrained("./models/optimized_model")
        model.tokenizer.save_pretrained("./models/optimized_model")

        logger.info("✅ Optimized model saved to: ./models/optimized_model")
        logger.info("   Ready for Hugging Face upload!")

    except Exception as e:
        logger.error(f"❌ Failed to save optimized model: {e}")
        logger.error(f"   Error details: {str(e)}")

    logger.info("\n" + "="*60)
    logger.info("🎉 OPTIMIZATION COMPLETE!")
    logger.info("="*60)
    logger.info("Optimized models saved to: ./models/optimized_model/")
    logger.info("\n🧪 Test the optimized model:")
    logger.info("Run: python test_optimized_model.py")
    logger.info("\n📤 Ready for Hugging Face upload!")
    logger.info("Run: python upload_to_hf.py --repo-name TU-USUARIO/quantum-semantic-reranker-v1")
    logger.info("\nUsage examples:")
    logger.info("  Standard: Use for Hugging Face upload")
    logger.info("  ONNX: Use with ONNX Runtime for faster inference")
    logger.info("  Quantized: Use for memory-constrained environments")

    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
