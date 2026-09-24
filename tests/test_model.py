"""
Tests for model loading and inference output vector.
Skips gracefully if TensorFlow/Keras is not installed.
"""
import os
import pytest
import numpy as np
from class_names import CLASS_NAMES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "plant_disease_model_final.h5")


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    assert file_size_mb > 1.0, f"Model file seems incomplete or empty: {file_size_mb:.2f} MB"


def test_model_load_and_prediction():
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        pytest.skip("TensorFlow is not installed in the current environment; skipping model inference test.")

    model = load_model(MODEL_PATH)
    assert model is not None, "Model failed to load"

    # Generate dummy input matching (1, 224, 224, 3)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    predictions = model.predict(dummy_input, verbose=0)

    # 1. Output shape must match (1, number_of_classes)
    expected_classes = len(CLASS_NAMES)
    assert predictions.shape == (1, expected_classes), (
        f"Expected output shape (1, {expected_classes}), got {predictions.shape}"
    )

    # 2. Values should be valid probabilities summing to approx 1.0
    prob_sum = float(np.sum(predictions[0]))
    assert np.isclose(prob_sum, 1.0, atol=1e-3), (
        f"Expected output probabilities to sum to ~1.0, got {prob_sum}"
    )
    assert np.all(predictions >= 0.0) and np.all(predictions <= 1.0), (
        "Predicted probabilities must fall in range [0.0, 1.0]"
    )
