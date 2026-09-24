"""
Converts trained Keras model (.h5) to TensorFlow Lite (.tflite) format.
Supports float16 and dynamic-range quantization, and outputs file size statistics.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ConvertTFLite")

# Guarded TensorFlow import
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    load_model = None

DEFAULT_MODEL_PATH = os.path.join("models", "plant_disease_model_final.h5")
DEFAULT_OUTPUT_DIR = "models_tflite"


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for TFLite conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Keras .h5 model to TensorFlow Lite (.tflite)."
    )
    parser.add_argument(
        "--model-path", "--model", "-m",
        dest="model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to input .h5 model file (default: {DEFAULT_MODEL_PATH})."
    )
    parser.add_argument(
        "--output-dir", "--output", "-o",
        dest="output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save .tflite model (default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional custom filename for the .tflite model."
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        choices=["float16", "dynamic", "none"],
        default="float16",
        help="Quantization type: 'float16' (default), 'dynamic' (dynamic-range), or 'none'."
    )
    return parser.parse_args(args)


def format_bytes(size_bytes: int) -> str:
    """Formats bytes into human-readable MB and raw bytes."""
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f} MB ({size_bytes:,} bytes)"


def convert_model(
    model_path: str,
    output_dir: str,
    quantization: str = "float16",
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Converts a Keras .h5 model to .tflite format with the specified quantization strategy.
    Returns conversion metadata and size statistics.
    """
    if tf is None or load_model is None:
        raise ImportError(
            "TensorFlow is required for TFLite conversion. "
            "Please install requirements-dev.txt or tensorflow."
        )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Input model file not found: {model_path}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename
    if output_name:
        filename = output_name if output_name.endswith(".tflite") else f"{output_name}.tflite"
    else:
        base_name = Path(model_path).stem
        filename = f"{base_name}_{quantization}.tflite"

    output_path = os.path.join(output_dir, filename)

    logger.info(f"Loading Keras model from: {model_path}")
    keras_model = load_model(model_path)

    logger.info(f"Configuring TFLiteConverter with '{quantization}' quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    quant_lower = quantization.lower()
    if quant_lower == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant_lower == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant_lower == "none":
        converter.optimizations = []
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")

    logger.info("Converting model to TFLite format...")
    tflite_bytes = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_bytes)

    orig_size = os.path.getsize(model_path)
    tflite_size = os.path.getsize(output_path)
    reduction_pct = (1.0 - (tflite_size / orig_size)) * 100.0 if orig_size > 0 else 0.0

    stats = {
        "source_model": model_path,
        "output_path": output_path,
        "quantization": quantization,
        "original_size_bytes": orig_size,
        "tflite_size_bytes": tflite_size,
        "reduction_percentage": reduction_pct
    }

    print_conversion_summary(stats)
    return stats


def print_conversion_summary(stats: Dict[str, Any]) -> None:
    """Prints a structured summary of model conversion and file sizes."""
    orig_str = format_bytes(stats["original_size_bytes"])
    tflite_str = format_bytes(stats["tflite_size_bytes"])
    red_pct = stats["reduction_percentage"]

    print("\n" + "=" * 60)
    print("      TFLite Model Conversion Completed Successfully")
    print("=" * 60)
    print(f"Source Model:       {stats['source_model']}")
    print(f"Target TFLite:      {stats['output_path']}")
    print(f"Quantization:       {stats['quantization']}")
    print("-" * 60)
    print(f"Original Size:      {orig_str}")
    print(f"TFLite Size:        {tflite_str}")
    print(f"Size Reduction:     {red_pct:.2f}%")
    print("=" * 60 + "\n")


def main(args: Optional[List[str]] = None) -> None:
    parsed = parse_args(args)
    try:
        convert_model(
            model_path=parsed.model_path,
            output_dir=parsed.output_dir,
            quantization=parsed.quantization,
            output_name=parsed.output_name
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
