"""
Tests for src/evaluate.py CLI arguments and dataset discovery logic.
"""
import os
import pytest
from pathlib import Path
from evaluate import parse_args, build_class_mapping, discover_test_samples
from class_names import CLASS_NAMES


def test_evaluate_parse_args():
    args = parse_args([
        "--model-path", "models/test_model.h5",
        "--data-dir", "data/test",
        "--output-dir", "custom/output",
        "--batch-size", "16",
        "--img-size", "128"
    ])
    assert args.model_path == "models/test_model.h5"
    assert args.data_dir == "data/test"
    assert args.output_dir == "custom/output"
    assert args.batch_size == 16
    assert args.img_size == 128


def test_build_class_mapping():
    mapping = build_class_mapping()
    assert len(mapping) == len(CLASS_NAMES)
    assert "tomato early blight" in mapping
    assert mapping["tomato early blight"] == CLASS_NAMES.index("Tomato_Early_blight")


def test_discover_test_samples_nonexistent_dir():
    with pytest.raises(FileNotFoundError):
        discover_test_samples("non_existent_directory_xyz")


def test_discover_test_samples_with_synthetic_data(tmp_path):
    import cv2
    import numpy as np

    # Create dummy class folders matching CLASS_NAMES
    folder1 = tmp_path / "Tomato___Early_blight"
    folder1.mkdir()
    dummy_img1 = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(str(folder1 / "img1.jpg"), dummy_img1)

    folder2 = tmp_path / "Healthy_cotton"
    folder2.mkdir()
    dummy_img2 = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(str(folder2 / "img2.png"), dummy_img2)

    image_paths, labels, present_classes = discover_test_samples(str(tmp_path))

    assert len(image_paths) == 2
    assert len(labels) == 2
    assert "Tomato_Early_blight" in present_classes
    assert "Healthy_cotton" in present_classes
