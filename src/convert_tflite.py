"""Offline export utility to convert Keras .h5 models to TensorFlow Lite format.

This module loads a trained TensorFlow/Keras neural network checkpoint, configures
the TensorFlow Lite Converter with user-selected weight quantization (Float16,
8-bit dynamic range, or unquantized Float32), serializes the model to a .tflite
binary, and computes storage reduction statistics.

Usage:
    Run from command line:
        python src/convert_tflite.py --quantization float16
        python src/convert_tflite.py --model models/plant_disease_model_final.h5 --output models_tflite/

Inputs:
    Trained Keras model file (.h5 or SavedModel format).

Outputs:
    Optimized .tflite binary file saved to destination directory (default: models_tflite/).

Pipeline Context:
    Offline optimization bridge between workstation research (training / research_and_training.ipynb)
    and low-power edge deployment on the Raspberry Pi 4.
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

DEFAULT_MODEL_PATH: str = os.path.join("models", "plant_disease_model_final.h5")
DEFAULT_OUTPUT_DIR: str = "models_tflite"
BYTES_PER_MEGABYTE: int = 1024 * 1024
SUMMARY_LINE_WIDTH: int = 60


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for TFLite model conversion.

    Args:
        args: Optional list of argument strings (defaults to sys.argv[1:]).

    Returns:
        Populated argparse.Namespace with model path, output directory, and quantization mode.
    """
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
    """Formats a raw byte count into a human-readable megabyte string.

    Args:
        size_bytes: The number of bytes to format.

    Returns:
        Formatted string presenting size in megabytes and exact byte count.
    """
    size_mb = size_bytes / BYTES_PER_MEGABYTE
    return f"{size_mb:.2f} MB ({size_bytes:,} bytes)"


def convert_model(
    model_path: str,
    output_dir: str,
    quantization: str = "float16",
    output_name: Optional[str] = None
) -> Dict[str, Any]:
    """Converts a Keras .h5 model to .tflite format with the specified quantization.

    Args:
        model_path: Absolute or relative filesystem path to the source .h5 model.
        output_dir: Destination directory where the .tflite file will be written.
        quantization: Optimization strategy: 'float16', 'dynamic', or 'none'.
        output_name: Optional custom filename for the output artifact.

    Returns:
        Dictionary containing metadata, file paths, and size reduction percentage.

    Raises:
        ImportError: If TensorFlow is not installed in the execution environment.
        FileNotFoundError: If the source model path does not exist on disk.
        ValueError: If an unsupported quantization mode is specified.
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
    """Prints a structured summary of model conversion and file sizes.

    Args:
        stats: Dictionary containing conversion metrics generated by convert_model().
    """
    orig_str = format_bytes(stats["original_size_bytes"])
    tflite_str = format_bytes(stats["tflite_size_bytes"])
    red_pct = stats["reduction_percentage"]

    print("\n" + "=" * SUMMARY_LINE_WIDTH)
    print("      TFLite Model Conversion Completed Successfully")
    print("=" * SUMMARY_LINE_WIDTH)
    print(f"Source Model:       {stats['source_model']}")
    print(f"Target TFLite:      {stats['output_path']}")
    print(f"Quantization:       {stats['quantization']}")
    print("-" * SUMMARY_LINE_WIDTH)
    print(f"Original Size:      {orig_str}")
    print(f"TFLite Size:        {tflite_str}")
    print(f"Size Reduction:     {red_pct:.2f}%")
    print("=" * SUMMARY_LINE_WIDTH + "\n")


def main(args: Optional[List[str]] = None) -> None:
    """CLI entry point for running standalone model conversion.

    Args:
        args: Optional list of CLI argument strings.
    """
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
