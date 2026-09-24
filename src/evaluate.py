"""
Offline model evaluation script for Farmer Eye plant disease classification.
Loads a trained Keras/TensorFlow model and a directory of test images structured
into subfolders per class, runs batched inference using the canonical preprocessing
pipeline and class indexing, and saves metrics to the specified output directory.
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

# Import canonical class names and normalization
try:
    from src.class_names import CLASS_NAMES, normalize_disease_name
except ImportError:
    from class_names import CLASS_NAMES, normalize_disease_name

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Evaluate")

# Lazy/guarded TensorFlow import to allow importing evaluate.py without crashing
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(args: List[str] = None) -> argparse.Namespace:
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Farmer Eye disease classification model on a test dataset."
    )
    parser.add_argument(
        "--model-path", "--model", "-m",
        dest="model_path",
        type=str,
        required=True,
        help="Path to the trained Keras model file (.h5 or SavedModel format)."
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
        default="docs/assets/results",
        help="Directory to save evaluation artifacts (default: docs/assets/results)."
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for model inference (default: 32)."
    )
    parser.add_argument(
        "--img-size", "-s",
        type=int,
        default=224,
        help="Input image resolution expected by model (default: 224)."
    )
    return parser.parse_args(args)


def build_class_mapping() -> Dict[str, int]:
    """
    Builds a normalized name -> canonical index mapping for all 25 classes.
    """
    mapping = {}
    for idx, name in enumerate(CLASS_NAMES):
        norm = normalize_disease_name(name)
        mapping[norm] = idx
    return mapping


def discover_test_samples(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Scans data_dir for class subdirectories and resolves image paths and labels.
    Returns:
        image_paths: List of absolute file paths to images.
        labels: List of integer class indices matching CLASS_NAMES.
        present_class_names: List of class names actually present in the dataset.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Test data directory not found: {data_dir}")

    norm_to_idx = build_class_mapping()
    image_paths: List[str] = []
    labels: List[int] = []
    present_indices = set()

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


def preprocess_image(image_path: str, img_size: int = 224) -> np.ndarray:
    """
    Loads and preprocesses a single image exactly as real_time_detection.py:
    Resizes to (img_size, img_size) and scales float32 pixels to [0.0, 1.0].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at '{image_path}' via OpenCV.")
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    return img


def run_inference_in_batches(
    model: Any,
    image_paths: List[str],
    batch_size: int = 32,
    img_size: int = 224
) -> np.ndarray:
    """
    Runs batched inference and returns predictions probability matrix.
    """
    num_samples = len(image_paths)
    all_predictions = []

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_paths = image_paths[start_idx:end_idx]

        batch_imgs = []
        for p in batch_paths:
            try:
                batch_imgs.append(preprocess_image(p, img_size=img_size))
            except Exception as e:
                logger.error(f"Failed preprocessing {p}: {e}")
                # Zero fallback if single image is corrupted
                batch_imgs.append(np.zeros((img_size, img_size, 3), dtype=np.float32))

        batch_tensor = np.array(batch_imgs, dtype=np.float32)
        preds = model.predict(batch_tensor, verbose=0)
        all_predictions.append(preds)

        if (start_idx // batch_size) % 10 == 0 or end_idx == num_samples:
            logger.info(f"Evaluated {end_idx}/{num_samples} samples...")

    return np.vstack(all_predictions)


def generate_and_save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    present_class_names: List[str],
    present_indices: List[int],
    output_dir: str
) -> Dict[str, Any]:
    """
    Calculates classification metrics, confusion matrix plot, and classification report,
    saving them to output_dir:
    1. metrics.json
    2. confusion_matrix.png
    3. classification_report.txt
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # 1. Classification report dictionary and formatted text
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=present_indices,
        target_names=present_class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=present_indices,
        target_names=present_class_names,
        digits=4,
        zero_division=0
    )

    # Save classification_report.txt
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Saved classification report to: {report_path}")

    # Build structured metrics.json
    metrics_data = {
        "accuracy": float(report_dict.get("accuracy", 0.0)),
        "macro_avg": {
            "precision": float(report_dict.get("macro avg", {}).get("precision", 0.0)),
            "recall": float(report_dict.get("macro avg", {}).get("recall", 0.0)),
            "f1-score": float(report_dict.get("macro avg", {}).get("f1-score", 0.0)),
            "support": int(report_dict.get("macro avg", {}).get("support", len(y_true)))
        },
        "weighted_avg": {
            "precision": float(report_dict.get("weighted avg", {}).get("precision", 0.0)),
            "recall": float(report_dict.get("weighted avg", {}).get("recall", 0.0)),
            "f1-score": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
            "support": int(report_dict.get("weighted avg", {}).get("support", len(y_true)))
        },
        "per_class": {}
    }

    for name in present_class_names:
        if name in report_dict:
            metrics_data["per_class"][name] = {
                "precision": float(report_dict[name]["precision"]),
                "recall": float(report_dict[name]["recall"]),
                "f1-score": float(report_dict[name]["f1-score"]),
                "support": int(report_dict[name]["support"])
            }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved structured metrics to: {metrics_path}")

    # 2. Confusion matrix heatmap plot
    cm = confusion_matrix(y_true, y_pred, labels=present_indices)

    fig, ax = plt.subplots(figsize=(14, 12))
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
            cbar=True
        )
    except ImportError:
        # Fallback to pure matplotlib if seaborn is unavailable
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_xticks(range(len(present_class_names)))
        ax.set_yticks(range(len(present_class_names)))
        ax.set_xticklabels(present_class_names, rotation=90)
        ax.set_yticklabels(present_class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    ax.set_title("Test Set Confusion Matrix", fontsize=16, pad=20)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrix figure to: {cm_path}")

    return metrics_data


def main(args: List[str] = None):
    parsed = parse_args(args)

    if load_model is None:
        logger.error(
            "TensorFlow is not installed in the current environment. "
            "Please install requirements.txt or tensorflow to run model evaluation."
        )
        sys.exit(1)

    if not os.path.exists(parsed.model_path):
        logger.error(f"Model path does not exist: {parsed.model_path}")
        sys.exit(1)

    logger.info(f"Loading trained model from '{parsed.model_path}'...")
    try:
        model = load_model(parsed.model_path)
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
        img_size=parsed.img_size
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
            output_dir=parsed.output_dir
        )
        logger.info(
            f"Evaluation complete! Overall Accuracy: {metrics['accuracy'] * 100:.2f}%"
        )
    except Exception as e:
        logger.error(f"Failed generating metrics/plots: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
