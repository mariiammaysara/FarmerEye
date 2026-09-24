# Developer & Contributor Guide

Welcome to the **Farmer Eye** developer guide! This document explains how to set up your local development environment, run hardware-free automated tests, contribute new features, and use repository utility tools.

---

## 1. Local Environment Setup

You can develop, test, and evaluate models on any desktop or laptop (Windows, macOS, or Linux). Physical Raspberry Pi hardware is **not** required.

### Prerequisites
- Python 3.10 or 3.11
- Git
- Recommended IDE: Visual Studio Code, Cursor, or PyCharm

### Step-by-Step Setup:
```bash
# 1. Clone the repository
git clone https://github.com/mariiammaysara/FarmerEye.git
cd FarmerEye

# 2. Create and activate a Python virtual environment
python -m venv venv

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# 3. Upgrade pip and install development dependencies
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

---

## 2. Dependency Philosophy: Runtime vs. Development

We strictly separate our dependencies into two tiers:

1. **[`requirements.txt`](../requirements.txt) (Edge Runtime)**:
   - Contains only the minimal libraries needed to run inference and WebSocket streaming on the Raspberry Pi: `numpy`, `pandas`, `openpyxl`, `websockets`, `opencv-python`, and `tensorflow`.
   - *Note*: `picamera2` is installed via the system package manager (`apt`) on Raspberry Pi OS and must **not** be in `requirements.txt`.
2. **[`requirements-dev.txt`](../requirements-dev.txt) (Development & CI)**:
   - Includes everything in `requirements.txt` plus testing and visualization tools: `pytest`, `matplotlib`, `seaborn`, `pillow`, `scikit-learn`, `notebook`, and `ipykernel`.

---

## 3. Running Hardware-Free Unit Tests

The repository features an automated test suite configured to run seamlessly on machines without GPIO pins or physical CSI cameras.

### How Hardware Mocking Works
In [tests/conftest.py](../tests/conftest.py), pytest injects virtual stubs into `sys.modules`:
- `RPi.GPIO`: Emulates pin modes (`BCM`), states (`HIGH`/`LOW`), and records function calls.
- `gpiozero`: Stubs motor and sensor abstractions.
- `picamera2`: Custom `MockPicamera2` that mimics preview configuration and returns synthetic RGB NumPy arrays for frame capture.

### Executing the Tests:
```bash
# Run the complete test suite with verbose output
pytest -v

# Run with concise output
pytest -q

# Run a specific test module
pytest tests/test_preprocessing.py -v
```

### Test Coverage Overview:
- `tests/test_preprocessing.py`: Validates input dimensions `(1, 224, 224, 3)`, float32 normalization `[0.0, 1.0]`, and base64 encoding.
- `tests/test_model.py`: Checks model file presence and inference probability outputs (skips gracefully if TensorFlow is absent).
- `tests/test_treatment_lookup.py`: Tests tolerant string normalization and asserts that all 25 classes resolve bilingual treatment protocols.
- `tests/test_websocket.py`: Validates message schemas (`camera_frame`, `detection`, `no_detection`, `welcome`, `pong`).
- `tests/test_motor.py`: Tests 4-pin L298N directional commands (`forward`, `backward`, `left`, `right`, `stop`).
- `tests/test_evaluate.py`: Verifies CLI arguments and dataset discovery logic.
- `tests/test_convert_tflite.py`: Verifies TFLite quantization flags and byte formatting.

---

## 4. Continuous Integration (GitHub Actions)

Every `push` and `pull_request` triggers the automated CI workflow defined in [`.github/workflows/tests.yml`](../.github/workflows/tests.yml):
- Target Environment: `ubuntu-latest`
- Python Version: `3.10`
- Caching: Automated pip cache (`actions/setup-python@v5` with `cache: "pip"`)
- Action: Automatically runs `pytest -v` to ensure zero regressions before merging.

---

## 5. Developer Utilities & CLI Reference

### A. Central Source of Truth: `src/class_names.py`
All class names, canonical indexing, and tolerant name matching logic live in [src/class_names.py](../src/class_names.py).
- Never hardcode the 25 class names in scripts. Always import `CLASS_NAMES` or `normalize_disease_name`.

### B. Offline Model Evaluation: `src/evaluate.py`
Evaluate any trained `.h5` or `.tflite` model against a folder of test images:
```bash
python src/evaluate.py --model models/plant_disease_model_final.h5 --data path/to/test_data --output docs/assets/results/
```

### C. TFLite Conversion: `src/convert_tflite.py`
Convert Keras models to optimized edge formats:
```bash
python src/convert_tflite.py --quantization float16
```

---

## 6. Contribution Standards & Conventions

1. **Clean Code & Small Diffs**:
   Keep edits focused and minimal. Do not rewrite working modules.
2. **Graceful Hardware Guards**:
   Any new hardware or edge imports must be wrapped in `try ... except ImportError: ... = None` guards so desktop testing remains frictionless.
3. **Commit Messages**:
   Use Conventional Commits:
   - `feat:` New functionality (e.g. `feat: add TFLite conversion script`)
   - `fix:` Bug fixes (e.g. `fix: resolve disease normalization typo`)
   - `test:` Test additions or updates
   - `docs:` Documentation improvements
