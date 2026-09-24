"""
Tests for image preprocessing pipeline (input shape, dtype, and normalization range).
"""
import numpy as np
import pytest
from combined_detection_stream import preprocess_frame_for_model, encode_frame


def test_preprocessing_shape_dtype_and_range():
    # Simulate a raw BGR/RGB frame captured from camera (e.g. 640x480)
    raw_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    processed = preprocess_frame_for_model(raw_frame)
    
    assert processed is not None, "Preprocessing returned None"
    # 1. Output shape must be (1, 224, 224, 3)
    assert processed.shape == (1, 224, 224, 3), f"Expected shape (1, 224, 224, 3), got {processed.shape}"
    # 2. Output dtype must be float32
    assert processed.dtype == np.float32, f"Expected dtype float32, got {processed.dtype}"
    # 3. Normalized value range must be [0.0, 1.0]
    assert np.min(processed) >= 0.0, "Processed pixel values contain negative numbers"
    assert np.max(processed) <= 1.0, "Processed pixel values exceed 1.0"


def test_preprocessing_handles_different_resolutions():
    # Test unusual aspect ratios
    square_frame = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    processed = preprocess_frame_for_model(square_frame)
    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32


def test_frame_encoding():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    encoded = encode_frame(frame)
    assert isinstance(encoded, str), "Encoded frame should be a base64 string"
    assert len(encoded) > 0, "Encoded string should not be empty"
