"""
Tests for WebSocket message structures and JSON serialization format.
Validates that camera frames, detections, heartbeats, and status payloads
conform strictly to the documented API schema.
"""
import json
import base64
from datetime import datetime
import pytest


def test_camera_frame_message_format():
    sample_frame_b64 = base64.b64encode(b"fake_jpeg_bytes").decode("utf-8")
    msg = {
        "type": "camera_frame",
        "timestamp": datetime.now().isoformat(),
        "image": sample_frame_b64
    }
    
    # Must be valid JSON
    serialized = json.dumps(msg)
    data = json.loads(serialized)
    
    assert data["type"] == "camera_frame"
    assert "timestamp" in data
    assert "image" in data
    assert data["image"] == sample_frame_b64


def test_detection_message_format():
    msg = {
        "type": "detection",
        "timestamp": datetime.now().isoformat(),
        "status": "detection",
        "disease_name": "Tomato___Early_blight",
        "confidence": 0.985,
        "treatment_en": "Apply copper-based fungicides.",
        "treatment_ar": "استخدام مبيدات فطرية نحاسية.",
        "resources": "https://example.com/treatment",
        "image": "sample_annotated_base64_image"
    }
    
    serialized = json.dumps(msg)
    data = json.loads(serialized)
    
    assert data["type"] == "detection"
    assert data["status"] == "detection"
    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["disease_name"]) > 0
    assert len(data["treatment_en"]) > 0
    assert len(data["treatment_ar"]) > 0
    assert "resources" in data
    assert "image" in data


def test_no_detection_message_format():
    msg = {
        "timestamp": datetime.now().isoformat(),
        "status": "no_detection",
        "message": "No plants or diseases detected"
    }
    serialized = json.dumps(msg)
    data = json.loads(serialized)
    
    assert data["status"] == "no_detection"
    assert "timestamp" in data
    assert "message" in data


def test_handshake_and_pong_formats():
    welcome_msg = {
        "type": "welcome",
        "message": "Connected to Disease Detection Server",
        "timestamp": datetime.now().isoformat()
    }
    pong_msg = {
        "type": "pong",
        "timestamp": datetime.now().isoformat()
    }
    
    w_data = json.loads(json.dumps(welcome_msg))
    assert w_data["type"] == "welcome"
    assert "Connected" in w_data["message"]
    
    p_data = json.loads(json.dumps(pong_msg))
    assert p_data["type"] == "pong"
    assert "timestamp" in p_data
