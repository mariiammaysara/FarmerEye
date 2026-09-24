"""
Single source of truth for plant disease detection class names and normalization.
"""
import re

CLASS_NAMES = [
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
    """
    Normalizes a disease name for tolerant lookup against the treatment database:
    - Lowercases text
    - Strips leading and trailing whitespace
    - Collapses multiple underscores and whitespace into a single space
    - Handles the common transcription typo 'bellhealthy' -> 'bell healthy'
    """
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(r'[\s_]+', ' ', s)
    s = re.sub(r'\bbellhealthy\b', 'bell healthy', s)
    return s.strip()
