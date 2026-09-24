"""Offline model evaluation script for the Farmer Eye plant disease classification system.

Purpose:
    Evaluates trained convolutional neural network models (Keras .h5, SavedModel, or
    quantized TensorFlow Lite .tflite formats) against a partitioned test dataset.
    Computes rigorous classification metrics (accuracy, macro/weighted precision,
    recall, F1-score, per-class breakdowns), renders a confusion matrix heatmap,
    and exports artifacts for documentation and benchmarking.

How It Is Run:
    Executed standalone from the command line:
        python src/evaluate.py --model models/plant_disease_model_final.h5 --data path/to/test_dataset
        python src/evaluate.py --model models_tflite/plant_disease_model_quantized.tflite --data path/to/test_dataset

Inputs:
    - Model file (.h5, directory, or .tflite).
    - Image directory structured into class subdirectories named after plant disease categories.
Outputs:
    - docs/assets/results/metrics.json: Structured metrics and per-class stats.
    - docs/assets/results/confusion_matrix.png: Visual confusion matrix heatmap.
    - docs/assets/results/classification_report.txt: Formatted text classification report.

Pipeline Placement:
    Offline evaluation and validation stage. Sits between model training/conversion and
    production edge deployment to verify inference accuracy against ground truth test sets.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Canonical class names and normalization helper from the Farmer Eye pipeline
try:
    from src.class_names import CLASS_NAMES, normalize_disease_name
except ImportError:
    from class_names import CLASS_NAMES, normalize_disease_name

# Lazy/guarded TensorFlow import to allow module discovery or test execution without crashing
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger: logging.Logger = logging.getLogger("Evaluate")

# Supported image file extensions for dataset scanning
SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default configuration constants
DEFAULT_OUTPUT_DIR: str = "docs/assets/results"
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_IMG_SIZE: int = 224

# Evaluation and visualization formatting constants
REPORT_DIGITS: int = 4
PROGRESS_INTERVAL_BATCHES: int = 10
CONFUSION_MATRIX_FIGSIZE: tuple[int, int] = (14, 12)
CONFUSION_MATRIX_DPI: int = 300
TITLE_FONT_SIZE: int = 16
LABEL_FONT_SIZE: int = 12
TITLE_PAD: int = 20
TICK_ROTATION_X: int = 90
TICK_ROTATION_Y: int = 0


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for offline model evaluation.

    Args:
        args: Optional sequence of command-line argument strings. If None,
            sys.argv[1:] is parsed.

    Returns:
        Populated namespace containing model_path, data_dir, output_dir,
        batch_size, and img_size.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Farmer Eye disease classification model on a test dataset."
    )
    parser.add_argument(
        "--model-path", "--model", "-m",
        dest="model_path",
        type=str,
        required=True,
        help="Path to the trained model file (.h5, SavedModel, or .tflite format)."
    )
    parser.add_argument(
        "--data-dir", "--data", "-d",
        dest="data_dir",
        type=str,
        required=True,
        help="Path to the test data directory containing class subfolders."
    )
    parser.add_argument(
        "--output-dir", "--output", "-o",
        dest="output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save evaluation artifacts (default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for model inference (default: {DEFAULT_BATCH_SIZE})."
    )
    parser.add_argument(
        "--img-size", "-s",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Input image resolution expected by model (default: {DEFAULT_IMG_SIZE})."
    )
    return parser.parse_args(args)


def build_class_mapping() -> dict[str, int]:
    """Builds a normalized name to canonical index lookup mapping for all classes.

    Returns:
        Dictionary mapping normalized class directory names to their canonical
        integer index in CLASS_NAMES.
    """
    mapping: dict[str, int] = {}
    for idx, name in enumerate(CLASS_NAMES):
        norm = normalize_disease_name(name)
        mapping[norm] = idx
    return mapping


def discover_test_samples(data_dir: str) -> tuple[list[str], list[int], list[str]]:
    """Scans the test data directory for class subfolders and collects image paths.

    Args:
        data_dir: Filesystem path to the root test data directory containing
            subdirectories named after plant disease classes.

    Returns:
        A tuple of (image_paths, labels, present_class_names) where:
            - image_paths: List of absolute or relative file paths to images.
            - labels: List of canonical integer class indices matching CLASS_NAMES.
            - present_class_names: Alphabetically sorted class names discovered in data_dir.

    Raises:
        FileNotFoundError: If data_dir does not exist or is not a directory.
        ValueError: If no class subdirectories or no valid images are discovered.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")

    norm_to_idx = build_class_mapping()
    image_paths: list[str] = []
    labels: list[int] = []
    present_indices: set[int] = set()

    subdirs = [p for p in data_path.iterdir() if p.is_dir()]
    if not subdirs:
        raise ValueError(
            f"No class subdirectories found in '{data_dir}'. Expected one folder per class."
        )

    for subdir in subdirs:
        norm_name = normalize_disease_name(subdir.name)
        if norm_name not in norm_to_idx:
            logger.warning(
                f"Directory '{subdir.name}' does not match any known class in CLASS_NAMES. Skipping."
            )
            continue

        class_idx = norm_to_idx[norm_name]
        files = [
            str(f) for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            logger.warning(f"No supported images found in '{subdir.name}'.")
            continue

        image_paths.extend(files)
        labels.extend([class_idx] * len(files))
        present_indices.add(class_idx)

    if not image_paths:
        raise ValueError(f"No valid images found across class subfolders in '{data_dir}'.")

    logger.info(
        f"Found {len(image_paths)} images across {len(present_indices)} recognized classes."
    )
    present_class_names = [CLASS_NAMES[i] for i in sorted(present_indices)]
    return image_paths, labels, present_class_names


def preprocess_image(image_path: str, img_size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """Loads and preprocesses an image matching the training and inference pipeline.

    Preprocessing replicates training: resize to (img_size, img_size) and scale
    pixel values to [0.0, 1.0].

    Args:
        image_path: Filesystem path to the image file to read.
        img_size: Target square pixel dimension for model input.

    Returns:
        Preprocessed image array of shape (img_size, img_size, 3) and dtype float32.

    Raises:
        ValueError: If OpenCV fails to decode the image file.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at '{image_path}' via OpenCV.")
    img = cv2.resize(img, (img_size, img_size))
    # Preprocessing must match training: scale float32 pixel values to [0.0, 1.0]
    img = img.astype(np.float32) / 255.0
    return img


def run_inference_in_batches(
    model: Any,
    image_paths: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMG_SIZE,
) -> np.ndarray:
    """Executes batched inference over all supplied images using the model runner.

    Args:
        model: ModelRunner instance capable of predicting on image batches.
        image_paths: Sequence of file paths to images to evaluate.
        batch_size: Number of images per inference batch.
        img_size: Target image resolution expected by the model.

    Returns:
        Probability matrix of shape (N, num_classes) where N is len(image_paths).
    """
    num_samples = len(image_paths)
    all_predictions: list[np.ndarray] = []

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_paths = image_paths[start_idx:end_idx]

        batch_imgs: list[np.ndarray] = []
        for p in batch_paths:
            try:
                batch_imgs.append(preprocess_image(p, img_size=img_size))
            except Exception as e:
                logger.error(f"Failed preprocessing {p}: {e}")
                # Zero array fallback to maintain batch dimension alignment on corrupt sample
                batch_imgs.append(np.zeros((img_size, img_size, 3), dtype=np.float32))

        batch_tensor = np.array(batch_imgs, dtype=np.float32)
        preds = model.predict(batch_tensor, verbose=0)
        all_predictions.append(preds)

        if (start_idx // batch_size) % PROGRESS_INTERVAL_BATCHES == 0 or end_idx == num_samples:
            logger.info(f"Evaluated {end_idx}/{num_samples} samples...")

    return np.vstack(all_predictions)


def generate_and_save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    present_class_names: list[str],
    present_indices: list[int],
    output_dir: str,
) -> dict[str, Any]:
    """Calculates classification metrics, confusion matrix plot, and reports.

    Generates three artifacts in output_dir:
        1. metrics.json: Overall accuracy, macro/weighted averages, and per-class stats.
        2. confusion_matrix.png: Visual confusion matrix heatmap.
        3. classification_report.txt: Text summary of precision, recall, and F1 scores.

    Args:
        y_true: 1D array of ground truth class integer indices.
        y_pred: 1D array of predicted class integer indices.
        present_class_names: Names of classes present in the test evaluation set.
        present_indices: Integer class indices matching present_class_names.
        output_dir: Destination directory path for generated metric files.

    Returns:
        Dictionary containing overall accuracy, macro/weighted averages, and
        per-class metrics.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    # Set headless non-interactive backend to prevent GUI display requirements
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Compute classification report dictionary and formatted text representation
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=present_indices,
        target_names=present_class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=present_indices,
        target_names=present_class_names,
        digits=REPORT_DIGITS,
        zero_division=0,
    )

    # Save classification_report.txt
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Saved classification report to: {report_path}")

    # Build structured metrics dictionary
    metrics_data: dict[str, Any] = {
        "accuracy": float(report_dict.get("accuracy", 0.0)),
        "macro_avg": {
            "precision": float(report_dict.get("macro avg", {}).get("precision", 0.0)),
            "recall": float(report_dict.get("macro avg", {}).get("recall", 0.0)),
            "f1-score": float(report_dict.get("macro avg", {}).get("f1-score", 0.0)),
            "support": int(report_dict.get("macro avg", {}).get("support", len(y_true))),
        },
        "weighted_avg": {
            "precision": float(report_dict.get("weighted avg", {}).get("precision", 0.0)),
            "recall": float(report_dict.get("weighted avg", {}).get("recall", 0.0)),
            "f1-score": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
            "support": int(report_dict.get("weighted avg", {}).get("support", len(y_true))),
        },
        "per_class": {},
    }

    for name in present_class_names:
        if name in report_dict:
            metrics_data["per_class"][name] = {
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1-score": float(report_dict[name]["f1-score"]),
                "support": int(report_dict[name]["support"]),
            }

    # Save metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved structured metrics to: {metrics_path}")

    # Render confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred, labels=present_indices)

    fig, ax = plt.subplots(figsize=CONFUSION_MATRIX_FIGSIZE)
    try:
        import seaborn as sns
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=present_class_names,
            yticklabels=present_class_names,
            ax=ax,
            cbar=True,
        )
    except ImportError:
        # Fallback to standard matplotlib if seaborn is not installed
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_xticks(range(len(present_class_names)))
        ax.set_yticks(range(len(present_class_names)))
        ax.set_xticklabels(present_class_names, rotation=TICK_ROTATION_X)
        ax.set_yticklabels(present_class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    ax.set_title("Test Set Confusion Matrix", fontsize=TITLE_FONT_SIZE, pad=TITLE_PAD)
    ax.set_xlabel("Predicted Label", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("True Label", fontsize=LABEL_FONT_SIZE)
    plt.xticks(rotation=TICK_ROTATION_X)
    plt.yticks(rotation=TICK_ROTATION_Y)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=CONFUSION_MATRIX_DPI)
    plt.close(fig)
    logger.info(f"Saved confusion matrix figure to: {cm_path}")

    return metrics_data


class ModelRunner:
    """Unified inference runner supporting Keras and TensorFlow Lite formats.

    Provides a consistent batched predict() interface whether the underlying
    model is a standard Keras model or a quantized .tflite flatbuffer.
    """

    def __init__(self, model_path: str):
        """Initializes the model runner with the specified model artifact.

        Args:
            model_path: Filesystem path to the model file (.h5, SavedModel,
                or .tflite).

        Raises:
            ImportError: If the required runtime (TensorFlow or tflite_runtime)
                is not installed.
        """
        self.model_path: str = model_path
        self.is_tflite: bool = model_path.lower().endswith(".tflite")

        if self.is_tflite:
            logger.info("Initializing TensorFlow Lite Interpreter...")
            try:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
            except (ImportError, AttributeError):
                try:
                    from tflite_runtime.interpreter import Interpreter
                    self.interpreter = Interpreter(model_path=model_path)
                except ImportError:
                    raise ImportError(
                        "Neither TensorFlow nor tflite_runtime is installed. "
                        "Cannot run inference on .tflite models."
                    )

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_index = self.input_details[0]["index"]
            self.output_index = self.output_details[0]["index"]
            self.expected_dtype = self.input_details[0]["dtype"]
            logger.info(
                f"TFLite model loaded. Input shape: {self.input_details[0]['shape']}, "
                f"dtype: {self.expected_dtype}"
            )
        else:
            logger.info("Loading Keras model...")
            if load_model is None:
                raise ImportError(
                    "TensorFlow is not installed in the current environment. "
                    "Please install requirements.txt or tensorflow to run model evaluation."
                )
            self.model = load_model(model_path)

    def predict(self, batch_tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Executes inference on a batch of preprocessed images.

        Args:
            batch_tensor: 4D float32 tensor of shape (batch_size, height, width, channels).
            verbose: Verbosity level passed to Keras predict (ignored for TFLite).

        Returns:
            Probability matrix of shape (batch_size, num_classes) with float32 values.
        """
        if self.is_tflite:
            batch_preds: list[np.ndarray] = []
            for i in range(batch_tensor.shape[0]):
                sample = np.expand_dims(batch_tensor[i], axis=0)
                if sample.dtype != self.expected_dtype:
                    sample = sample.astype(self.expected_dtype)
                self.interpreter.set_tensor(self.input_index, sample)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_index)
                batch_preds.append(out[0])
            return np.array(batch_preds, dtype=np.float32)
        else:
            return self.model.predict(batch_tensor, verbose=verbose)


def main(args: Optional[list[str]] = None) -> None:
    """Coordinates model loading, sample discovery, inference, and metric export.

    Args:
        args: Optional command-line argument list passed to parse_args.
    """
    parsed = parse_args(args)

    if not os.path.exists(parsed.model_path):
        logger.error(f"Model path does not exist: {parsed.model_path}")
        sys.exit(1)

    logger.info(f"Loading trained model from '{parsed.model_path}'...")
    try:
        model = ModelRunner(parsed.model_path)
    except Exception as e:
        logger.error(f"Failed loading model: {e}")
        sys.exit(1)

    logger.info(f"Scanning test data in '{parsed.data_dir}'...")
    try:
        image_paths, labels, present_classes = discover_test_samples(parsed.data_dir)
    except Exception as e:
        logger.error(f"Failed reading test data: {e}")
        sys.exit(1)

    norm_to_idx = build_class_mapping()
    present_indices = [norm_to_idx[normalize_disease_name(c)] for c in present_classes]

    logger.info(f"Running inference on {len(image_paths)} images...")
    predictions = run_inference_in_batches(
        model=model,
        image_paths=image_paths,
        batch_size=parsed.batch_size,
        img_size=parsed.img_size,
    )

    y_true = np.array(labels)
    y_pred = np.argmax(predictions, axis=1)

    logger.info("Computing metrics and generating figures...")
    try:
        metrics = generate_and_save_metrics(
            y_true=y_true,
            y_pred=y_pred,
            present_class_names=present_classes,
            present_indices=present_indices,
            output_dir=parsed.output_dir,
        )
        logger.info(
            f"Evaluation complete! Overall Accuracy: {metrics['accuracy'] * 100:.2f}%"
        )
    except Exception as e:
        logger.error(f"Failed generating metrics/plots: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
