"""
ONNX export utilities for cross-encoder models.
"""

import logging
from typing import Optional, List
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str,
    quantize: bool = False,
    precision: str = "fp32",
    opset_version: int = 14
) -> str:
    """
    Export PyTorch model to ONNX format.

    Args:
        model_path: Path to PyTorch model
        output_path: Output path for ONNX model
        quantize: Whether to quantize
        precision: Precision ('fp32', 'fp16', 'int8')
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX model
    """
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
        from optimum.onnxruntime.configuration import OptimizationConfig

        logger.info(f"Exporting model from {model_path} to ONNX")

        # Load model
        model = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            export=True
        )

        # Save ONNX model
        output_path = Path(output_path)
        if output_path.suffix != '.onnx':
            output_path = output_path.parent / (output_path.stem + '.onnx')

        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(output_dir))

        # Apply optimization
        optimizer = ORTOptimizer.from_pretrained(model_path)

        optimization_config = OptimizationConfig(
            optimization_level=2,  # Enable extended optimizations
            optimize_for_gpu=(precision in ["fp16"])
        )

        optimizer.optimize(
            save_dir=str(output_dir),
            optimization_config=optimization_config
        )

        # Quantize if requested
        if quantize and precision == "int8":
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            quantizer = ORTQuantizer.from_pretrained(str(output_dir))

            qconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=False,
                per_channel=False
            )

            quantizer.quantize(
                save_dir=str(output_dir),
                quantization_config=qconfig
            )

            logger.info("Model quantized to INT8")

        logger.info(f"ONNX export complete: {output_path}")
        return str(output_path)

    except ImportError:
        logger.warning("optimum not installed, using alternative export method")
        return _export_to_onnx_alternative(
            model_path,
            output_path,
            opset_version
        )


def _export_to_onnx_alternative(
    model_path: str,
    output_path: str,
    opset_version: int = 14
) -> str:
    """
    Alternative ONNX export using torch.onnx.

    Args:
        model_path: Path to PyTorch model
        output_path: Output path for ONNX model
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX model
    """
    logger.info("Using torch.onnx for export")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()

    # Create dummy input
    dummy_text = ["What is machine learning?", "Machine learning is AI."]
    dummy_input = tokenizer(
        dummy_text[0],
        dummy_text[1],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Export
    output_path = Path(output_path)
    if output_path.suffix != '.onnx':
        output_path = output_path.parent / (output_path.stem + '.onnx')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ['input_ids', 'attention_mask']
    output_names = ['logits']

    # Add token_type_ids if present
    if 'token_type_ids' in dummy_input:
        input_names.append('token_type_ids')

    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size'}
    }

    if 'token_type_ids' in dummy_input:
        dynamic_axes['token_type_ids'] = {0: 'batch_size', 1: 'sequence'}

    # Prepare input tuple
    input_tuple = (
        dummy_input['input_ids'],
        dummy_input['attention_mask'],
    )

    if 'token_type_ids' in dummy_input:
        input_tuple += (dummy_input['token_type_ids'],)

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_tuple,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True
        )

    logger.info(f"ONNX export complete: {output_path}")

    # Save tokenizer
    tokenizer.save_pretrained(output_path.parent)

    return str(output_path)


class ONNXInferenceSession:
    """
    ONNX inference session for cross-encoder models.
    """

    def __init__(self, onnx_model_path: str):
        """
        Initialize ONNX inference session.

        Args:
            onnx_model_path: Path to ONNX model
        """
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                onnx_model_path,
                providers=['CPUExecutionProvider']
            )

            # Load tokenizer
            model_dir = Path(onnx_model_path).parent
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

            logger.info(f"ONNX session initialized: {onnx_model_path}")

        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

    def run(
        self,
        queries: List[str],
        documents: List[str]
    ) -> np.ndarray:
        """
        Run inference on query-document pairs.

        Args:
            queries: List of queries
            documents: List of documents

        Returns:
            Array of scores
        """
        # Tokenize
        encoded = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        # Prepare inputs
        ort_inputs = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

        if 'token_type_ids' in encoded:
            ort_inputs['token_type_ids'] = encoded['token_type_ids']

        # Run inference
        outputs = self.session.run(None, ort_inputs)
        logits = outputs[0]

        # Convert to scores
        if logits.shape[-1] == 1:
            scores = logits.squeeze(-1)
        else:
            # Softmax for multi-class
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            scores = probs[:, 1]  # Positive class probability

        return scores

    def predict(
        self,
        queries: List[str],
        documents: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """
        Predict scores with batching.

        Args:
            queries: List of queries
            documents: List of documents
            batch_size: Batch size

        Returns:
            List of scores
        """
        all_scores = []

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]

            scores = self.run(batch_queries, batch_docs)
            all_scores.extend(scores.tolist())

        return all_scores
