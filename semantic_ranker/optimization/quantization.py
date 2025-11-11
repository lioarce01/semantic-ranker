"""
Model quantization utilities.
"""

import logging
from typing import Optional
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quantize_model(
    model_path: str,
    output_path: str,
    quantization_type: str = "dynamic",
    precision: str = "int8"
) -> str:
    """
    Quantize a PyTorch model.

    Args:
        model_path: Path to original model
        output_path: Output path for quantized model
        quantization_type: Type of quantization ('dynamic', 'static')
        precision: Target precision ('int8', 'fp16')

    Returns:
        Path to quantized model
    """
    logger.info(f"Quantizing model: {model_path}")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()

    if precision == "int8":
        if quantization_type == "dynamic":
            quantized_model = _dynamic_quantization_int8(model)
        else:
            quantized_model = _static_quantization_int8(model, tokenizer)

    elif precision == "fp16":
        quantized_model = _fp16_conversion(model)

    else:
        raise ValueError(f"Unsupported precision: {precision}")

    # Save quantized model
    quantized_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info(f"Quantized model saved to {output_path}")
    return str(output_path)


def _dynamic_quantization_int8(model):
    """
    Apply dynamic INT8 quantization.

    Args:
        model: Model to quantize

    Returns:
        Quantized model
    """
    logger.info("Applying dynamic INT8 quantization")

    # Quantize linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    return quantized_model


def _static_quantization_int8(model, tokenizer):
    """
    Apply static INT8 quantization.

    Requires calibration data.

    Args:
        model: Model to quantize
        tokenizer: Tokenizer

    Returns:
        Quantized model
    """
    logger.info("Applying static INT8 quantization")

    # Set quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare for quantization
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with sample data
    calibration_data = [
        ("What is machine learning?", "ML is a subset of AI."),
        ("Explain deep learning", "Deep learning uses neural networks."),
        ("Define NLP", "NLP is natural language processing.")
    ]

    with torch.no_grad():
        for query, doc in calibration_data:
            inputs = tokenizer(
                query,
                doc,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            model(**inputs)

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    return model


def _fp16_conversion(model):
    """
    Convert model to FP16.

    Args:
        model: Model to convert

    Returns:
        FP16 model
    """
    logger.info("Converting to FP16")

    # Convert to half precision
    model = model.half()

    return model


def quantize_with_optimum(
    model_path: str,
    output_path: str,
    quantization_approach: str = "dynamic"
) -> str:
    """
    Quantize using Optimum library.

    Args:
        model_path: Path to original model
        output_path: Output path
        quantization_approach: Quantization approach

    Returns:
        Path to quantized model
    """
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        logger.info("Quantizing with Optimum")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create quantizer
        quantizer = ORTQuantizer.from_pretrained(model_path)

        # Configure quantization
        if quantization_approach == "dynamic":
            qconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=False,
                per_channel=False
            )
        else:
            qconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=True,
                per_channel=True
            )

        # Quantize
        quantizer.quantize(
            save_dir=str(output_path),
            quantization_config=qconfig
        )

        logger.info(f"Quantization complete: {output_path}")
        return str(output_path)

    except ImportError:
        logger.warning("optimum not installed, falling back to PyTorch quantization")
        return quantize_model(model_path, output_path, quantization_approach, "int8")


def compare_model_sizes(
    original_path: str,
    quantized_path: str
) -> dict:
    """
    Compare sizes of original and quantized models.

    Args:
        original_path: Path to original model
        quantized_path: Path to quantized model

    Returns:
        Dictionary with size comparison
    """
    import os

    def get_size(path):
        path = Path(path)
        if path.is_file():
            return os.path.getsize(path)
        else:
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())

    original_size = get_size(original_path)
    quantized_size = get_size(quantized_path)

    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    size_reduction_pct = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0

    comparison = {
        'original_size_mb': original_size / (1024 * 1024),
        'quantized_size_mb': quantized_size / (1024 * 1024),
        'compression_ratio': compression_ratio,
        'size_reduction_pct': size_reduction_pct
    }

    logger.info(f"Original: {comparison['original_size_mb']:.2f} MB")
    logger.info(f"Quantized: {comparison['quantized_size_mb']:.2f} MB")
    logger.info(f"Reduction: {size_reduction_pct:.1f}%")

    return comparison
