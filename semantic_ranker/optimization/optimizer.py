"""
Model optimization for production deployment.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from transformers import AutoConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Optimizes models for production deployment.

    Supports:
    - ONNX export
    - Quantization (INT8, FP16)
    - Model pruning
    - Knowledge distillation
    """

    def __init__(self, model_path: str):
        """
        Initialize optimizer.

        Args:
            model_path: Path to model to optimize
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.config = AutoConfig.from_pretrained(str(self.model_path))

        logger.info(f"Initialized optimizer for {model_path}")

    def export_to_onnx(
        self,
        output_path: str,
        quantize: bool = False,
        precision: str = "fp32",
        opset_version: int = 14
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output_path: Output path for ONNX model
            quantize: Whether to quantize model
            precision: Precision ('fp32', 'fp16', 'int8')
            opset_version: ONNX opset version

        Returns:
            Path to exported model
        """
        try:
            from .onnx_exporter import export_to_onnx

            output_path = export_to_onnx(
                model_path=str(self.model_path),
                output_path=output_path,
                quantize=quantize,
                precision=precision,
                opset_version=opset_version
            )

            logger.info(f"Model exported to ONNX: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

    def quantize(
        self,
        output_path: str,
        quantization_type: str = "dynamic",
        precision: str = "int8"
    ) -> str:
        """
        Quantize model for faster inference.

        Args:
            output_path: Output path for quantized model
            quantization_type: Type of quantization ('dynamic', 'static')
            precision: Target precision ('int8', 'fp16')

        Returns:
            Path to quantized model
        """
        try:
            from .quantization import quantize_model

            output_path = quantize_model(
                model_path=str(self.model_path),
                output_path=output_path,
                quantization_type=quantization_type,
                precision=precision
            )

            logger.info(f"Model quantized: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise

    def optimize_for_inference(
        self,
        output_path: str,
        enable_onnx: bool = True,
        enable_quantization: bool = True,
        target_device: str = "cpu"
    ) -> Dict[str, str]:
        """
        Apply all optimizations for inference.

        Args:
            output_path: Output directory
            enable_onnx: Whether to export to ONNX
            enable_quantization: Whether to quantize
            target_device: Target device ('cpu', 'cuda')

        Returns:
            Dictionary with paths to optimized models
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        logger.info("Starting full optimization pipeline")

        # Export to ONNX
        if enable_onnx:
            try:
                onnx_path = self.export_to_onnx(
                    output_path=str(output_dir / "model.onnx"),
                    quantize=False,
                    precision="fp32"
                )
                results['onnx'] = onnx_path

                # ONNX with quantization
                if enable_quantization:
                    onnx_quantized_path = self.export_to_onnx(
                        output_path=str(output_dir / "model_quantized.onnx"),
                        quantize=True,
                        precision="int8"
                    )
                    results['onnx_quantized'] = onnx_quantized_path

            except Exception as e:
                logger.warning(f"ONNX optimization failed: {e}")

        # PyTorch quantization
        if enable_quantization:
            try:
                quantized_path = self.quantize(
                    output_path=str(output_dir / "model_quantized"),
                    quantization_type="dynamic",
                    precision="int8"
                )
                results['pytorch_quantized'] = quantized_path

            except Exception as e:
                logger.warning(f"PyTorch quantization failed: {e}")

        logger.info("Optimization pipeline complete")
        return results

    def benchmark(
        self,
        model_paths: Dict[str, str],
        num_samples: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark different model versions.

        Args:
            model_paths: Dictionary of model name -> path
            num_samples: Number of samples to test

        Returns:
            Benchmark results
        """
        import time
        import numpy as np

        logger.info(f"Benchmarking {len(model_paths)} models")

        results = {}

        # Generate test data
        test_queries = ["What is machine learning?"] * num_samples
        test_docs = ["Machine learning is a subset of AI."] * num_samples

        for model_name, model_path in model_paths.items():
            logger.info(f"Benchmarking {model_name}...")

            try:
                # Load model
                if model_path.endswith('.onnx'):
                    # Load ONNX model
                    from .onnx_exporter import ONNXInferenceSession
                    model = ONNXInferenceSession(model_path)
                else:
                    # Load PyTorch model
                    from ..models.cross_encoder import CrossEncoderModel
                    model = CrossEncoderModel.load(model_path)

                # Warmup
                for _ in range(10):
                    if hasattr(model, 'predict'):
                        model.predict(test_queries[:1], test_docs[:1])
                    else:
                        model.run([test_queries[0]], [test_docs[0]])

                # Benchmark
                times = []
                for i in range(num_samples):
                    start = time.time()

                    if hasattr(model, 'predict'):
                        model.predict([test_queries[i]], [test_docs[i]])
                    else:
                        model.run([test_queries[i]], [test_docs[i]])

                    times.append(time.time() - start)

                # Calculate stats
                results[model_name] = {
                    'avg_latency_ms': np.mean(times) * 1000,
                    'p50_latency_ms': np.percentile(times, 50) * 1000,
                    'p95_latency_ms': np.percentile(times, 95) * 1000,
                    'p99_latency_ms': np.percentile(times, 99) * 1000,
                    'throughput_qps': 1.0 / np.mean(times)
                }

                logger.info(f"{model_name} - Avg latency: {results[model_name]['avg_latency_ms']:.2f}ms")

            except Exception as e:
                logger.error(f"Benchmarking {model_name} failed: {e}")
                results[model_name] = {'error': str(e)}

        return results

    def get_model_size(self, model_path: str) -> Dict[str, Any]:
        """
        Get model size information.

        Args:
            model_path: Path to model

        Returns:
            Size information
        """
        import os

        model_path = Path(model_path)

        if model_path.is_file():
            size_bytes = os.path.getsize(model_path)
        else:
            # Directory - sum all files
            size_bytes = sum(
                f.stat().st_size
                for f in model_path.rglob('*')
                if f.is_file()
            )

        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_bytes / (1024 * 1024 * 1024)

        return {
            'size_bytes': size_bytes,
            'size_mb': size_mb,
            'size_gb': size_gb
        }

    def compare_models(
        self,
        original_path: str,
        optimized_paths: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Compare original and optimized models.

        Args:
            original_path: Path to original model
            optimized_paths: Dictionary of optimized model paths

        Returns:
            Comparison results
        """
        logger.info("Comparing models...")

        comparison = {
            'original': self.get_model_size(original_path)
        }

        for name, path in optimized_paths.items():
            comparison[name] = self.get_model_size(path)

            # Calculate compression ratio
            original_size = comparison['original']['size_bytes']
            optimized_size = comparison[name]['size_bytes']
            compression_ratio = original_size / optimized_size if optimized_size > 0 else 0

            comparison[name]['compression_ratio'] = compression_ratio
            comparison[name]['size_reduction_pct'] = (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0

        logger.info("Model comparison:")
        logger.info(f"Original: {comparison['original']['size_mb']:.2f} MB")

        for name in optimized_paths.keys():
            if name in comparison:
                logger.info(f"{name}: {comparison[name]['size_mb']:.2f} MB "
                          f"({comparison[name]['size_reduction_pct']:.1f}% reduction)")

        return comparison
