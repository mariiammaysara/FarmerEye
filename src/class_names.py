"""Canonical class labels and disease normalization utilities for Farmer Eye.

This module serves as the single source of truth for the 25 plant condition
classes across the project pipeline. It provides deterministic indexing matching
the neural network output nodes and a normalization function to bridge naming
discrepancies between model predictions and the treatment database.

Usage:
    Imported as a library module across inference, evaluation, and training:
        from src.class_names import CLASS_NAMES, normalize_disease_name

Pipeline Context:
    1. Model output vector (argmax index 0..24) -> CLASS_NAMES[index]
    2. Predicted class name -> normalize_disease_name() -> Excel lookup
"""
import re

# Canonical 25-class list matching the neural network's final softmax output layer.
CLASS_NAMES: list[str] = [
    'Aphids_cotton',
    'Army worm_cotton',
    'Bacterial blight_cotton',
    'Healthy_cotton',
    'Pepper_bell_bacterial_spot',
    'Pepper_bell_healthy',
    'Potato__Early_blight',
    'Potato_Late_blight',
    'Potato_healthy',
    'Powdery mildew_cotton',
    'Strawberry_Leaf_scorch',
    'Strawberry_healthy',
    'Target spot_cotton',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites Two-spotted_spider_mite',
    'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato___healthy',
    'cotton_curl_virus',
    'cotton_fussarium_wilt'
]


def normalize_disease_name(name: str) -> str:
    """Normalizes a disease label for tolerant lookup against the treatment database.

    Different sources format disease names with varying numbers of underscores
    (e.g., 'Potato__Early_blight' vs 'Potato_Early_blight'), irregular spaces,
    or casing. This function maps all representations to a canonical normalized
    string to ensure consistent database joins without modifying the raw Excel file.

    Args:
        name: The raw predicted disease name or database entry string.

    Returns:
        The normalized lowercase string with single spaces, or an empty string
        if the input is None or not a string.
    """
    if not isinstance(name, str):
        return ""
    # Lowercase and replace runs of whitespace and underscores with a single space.
    s = name.strip().lower()
    s = re.sub(r'[\s_]+', ' ', s)
    # Correct known transcription anomaly present in the treatment reference sheet.
    s = re.sub(r'\bbellhealthy\b', 'bell healthy', s)
    return s.strip()
