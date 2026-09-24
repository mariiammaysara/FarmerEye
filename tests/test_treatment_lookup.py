"""
Tests for treatment database lookup and tolerant disease normalization.
Validates that known classes return bilingual treatments, unknown classes fail gracefully,
and all 25 classes in CLASS_NAMES resolve successfully against plant_disease_data.xlsx.
"""
import os
import pytest
from class_names import CLASS_NAMES, normalize_disease_name

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREATMENT_FILE = os.path.join(REPO_ROOT, "data", "plant_disease_data.xlsx")


def test_treatment_file_exists():
    assert os.path.exists(TREATMENT_FILE), f"Database file missing at {TREATMENT_FILE}"


def test_normalize_disease_name():
    # Test collapse of multiple underscores
    assert normalize_disease_name("Tomato___Early_blight") == "tomato early blight"
    assert normalize_disease_name("Potato__Early_blight") == "potato early blight"
    assert normalize_disease_name("Pepper_bell__bacterial_spot") == "pepper bell bacterial spot"

    # Test whitespace and case normalization
    assert normalize_disease_name("  Healthy_Cotton  ") == "healthy cotton"
    assert normalize_disease_name("COTTON_CURL_VIRUS") == "cotton curl virus"

    # Test bellhealthy typo tolerance
    assert normalize_disease_name("Pepper_bellhealthy") == "pepper bell healthy"
    assert normalize_disease_name("Pepper_bell_healthy") == "pepper bell healthy"

    # Test edge cases
    assert normalize_disease_name(None) == ""
    assert normalize_disease_name("") == ""


def test_treatment_lookup_known_class():
    try:
        import pandas as pd
        import openpyxl
    except ImportError:
        pytest.skip("pandas/openpyxl not installed; skipping live Excel lookup test.")

    from combined_detection_stream import load_resources, get_treatment_info

    assert load_resources() is True or os.path.exists(TREATMENT_FILE)
    
    info = get_treatment_info("Tomato_Early_blight")
    assert info is not None, "Failed to resolve known disease 'Tomato_Early_blight'"
    assert "treatment_en" in info and len(info["treatment_en"]) > 10, "Missing or empty English treatment"
    assert "treatment_ar" in info and len(info["treatment_ar"]) > 10, "Missing or empty Arabic treatment"


def test_treatment_lookup_unknown_class_fails_gracefully():
    try:
        import pandas as pd
        import openpyxl
    except ImportError:
        pytest.skip("pandas/openpyxl not installed; skipping Excel lookup test.")

    from combined_detection_stream import load_resources, get_treatment_info
    load_resources()

    info = get_treatment_info("NonExistent_Alien_Disease_123")
    assert info is None, "Unknown disease should return None without raising an exception"


def test_all_25_classes_resolve_bilingual():
    """Regression test: every class in CLASS_NAMES must resolve to English and Arabic text."""
    try:
        import pandas as pd
        import openpyxl
    except ImportError:
        pytest.skip("pandas/openpyxl not installed; skipping 25-class regression test.")

    from combined_detection_stream import load_resources, get_treatment_info
    assert load_resources() is True, "Failed to load treatment dataframe"

    missing_classes = []
    missing_bilingual = []

    for name in CLASS_NAMES:
        info = get_treatment_info(name)
        if not info:
            missing_classes.append(name)
        else:
            en = info.get("treatment_en", "")
            ar = info.get("treatment_ar", "")
            if not en or en == "N/A" or not ar or ar == "غير متوفر":
                missing_bilingual.append(name)

    assert not missing_classes, f"Classes failed to resolve: {missing_classes}"
    assert not missing_bilingual, f"Classes missing bilingual treatments: {missing_bilingual}"
