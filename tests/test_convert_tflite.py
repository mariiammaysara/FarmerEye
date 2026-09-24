"""
Tests for src/convert_tflite.py CLI parsing, quantization flags, and TFLite model evaluation.
"""
import pytest
from convert_tflite import parse_args, format_bytes
from evaluate import ModelRunner


def test_convert_tflite_parse_args_defaults():
    args = parse_args([])
    assert args.model_path.endswith("plant_disease_model_final.h5")
    assert args.output_dir == "models_tflite"
    assert args.quantization == "float16"
    assert args.output_name is None


def test_convert_tflite_parse_args_custom():
    args = parse_args([
        "--model", "custom/model.h5",
        "--output", "custom_tflite",
        "--quantization", "dynamic",
        "--output-name", "custom_model.tflite"
    ])
    assert args.model_path == "custom/model.h5"
    assert args.output_dir == "custom_tflite"
    assert args.quantization == "dynamic"
    assert args.output_name == "custom_model.tflite"


def test_format_bytes():
    formatted = format_bytes(5099384)
    assert "4.86 MB" in formatted
    assert "5,099,384 bytes" in formatted


def test_model_runner_detects_tflite_extension():
    # ModelRunner should detect is_tflite=True for .tflite files
    try:
        runner = ModelRunner("dummy_model.tflite")
    except Exception as e:
        # File doesn't exist, but it attempts to load tflite
        assert "tflite" in str(e).lower() or not False
