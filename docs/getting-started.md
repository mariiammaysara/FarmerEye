# Getting Started

What this page covers:
This page provides installation and setup instructions for Farmer Eye.
It covers environment setup on development laptops and Raspberry Pi hardware, shows how to run each script, and lists solutions to common errors.

---

## System Requirements

### Development Laptop (Testing and Evaluation)
- **Operating System**: Linux, macOS, or Windows 10/11.
- **Python**: Version 3.10 or 3.11.
- **Hardware**: Standard x86_64 or ARM64 computer. No camera or GPIO hardware is required because automated test suites use software mocks.

### Raspberry Pi (Edge Deployment)
- **Computer**: Raspberry Pi 4 Model B (4 GB or 8 GB RAM recommended).
- **Operating System**: Raspberry Pi OS (64-bit, Debian Bookworm or Bullseye).
- **Camera**: Raspberry Pi Camera Module (v2 or v3) connected through the CSI ribbon port.
- **Network**: Local Wi-Fi router connecting the Raspberry Pi and client phone to the same subnet.

---

## Installation Guide

### Option A: Setup on a Development Laptop

1. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/mariiammaysara/FarmerEye.git
   cd FarmerEye
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows (PowerShell):
   venv\Scripts\Activate.ps1
   
   # On Linux or macOS:
   source venv/bin/activate
   ```

3. Install development and testing dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Verify the installation by running the test suite:
   ```bash
   pytest -v
   ```
   *(Verified on this machine: all 27 unit tests pass, 1 hardware model test skipped gracefully when TensorFlow is absent).*

---

### Option B: Setup on a Raspberry Pi

1. Update system packages and install native camera libraries:
   ```bash
   sudo apt update
   sudo apt install -y python3-picamera2 python3-pip python3-venv
   ```
   *Note: `picamera2` must be installed using the system package manager (`apt`), not through `pip`, because it requires underlying Raspberry Pi OS video drivers.*

2. Clone the repository:
   ```bash
   git clone https://github.com/mariiammaysara/FarmerEye.git
   cd FarmerEye
   ```

3. Create a virtual environment that can access system packages:
   ```bash
   python3 -m venv --system-site-packages venv
   source venv/bin/activate
   ```

4. Install Python runtime dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run Each Script

### 1. Run the Automated Tests (Laptop or Pi)
Runs the complete test suite using mock objects for hardware components:
```bash
pytest -v
```
*(Verified on this machine).*

### 2. Run the Combined Edge Server (Raspberry Pi)
Starts the unified camera streaming, detection, and WebSocket server on port 8765:
```bash
python src/combined_detection_stream.py
```
*(Not tested on this machine: requires a physical Raspberry Pi and Camera Module).*

### 3. Run the Dual-Port Detection Server (Raspberry Pi)
Starts the streaming service on port 8765 and motor command listener on port 8766:
```bash
python src/real_time_detection.py
```
*(Not tested on this machine: requires a physical Raspberry Pi and Camera Module).*

### 4. Run the Standalone Camera Streamer (Raspberry Pi)
Starts only the camera feed on port 8765 without deep learning inference:
```bash
python src/raspberry_pi_camera_stream.py
```
*(Not tested on this machine: requires a physical Raspberry Pi and Camera Module).*

### 5. Convert Model to TensorFlow Lite (Laptop or Pi)
Converts a trained Keras model into an optimized `.tflite` file:
```bash
# Convert with default float16 quantization
python src/convert_tflite.py --model models/plant_disease_model_final.h5 --output-dir models_tflite/

# Convert with 8-bit dynamic range quantization
python src/convert_tflite.py --model models/plant_disease_model_final.h5 --quantization dynamic --output-dir models_tflite/
```
*(Verified by checking script argument parser flags).*

### 6. Run Offline Model Evaluation (Laptop or Pi)
Evaluates a model checkpoint against a directory of labeled test images:
```bash
python src/evaluate.py --model models/plant_disease_model_final.h5 --data path/to/test_data --output docs/assets/results/
```
*(Verified by checking script argument parser flags).*

### 7. Run Model Training (Workstation with GPU)
Executes the research training pipeline exported from the project notebook:
```bash
python src/app.py
```
*(Requires downloading the raw dataset to `data/plantvillage dataset/color/`).*

---

## Environment Variables and Configuration

The scripts use the following default configurations:

| Parameter | Default Value | Configured In | Description |
|---|---|---|---|
| Stream Port | `8765` | `src/combined_detection_stream.py` | TCP port for WebSocket video and alerts. |
| Control Port | `8766` | `src/real_time_detection.py` | TCP port for driving commands. |
| Model Path | `models/plant_disease_model_final.h5` | `src/combined_detection_stream.py` | Path to trained weights. |
| Database Path | `data/plant_disease_data.xlsx` | `src/combined_detection_stream.py` | Path to bilingual Excel treatments. |

---

## Common Errors and Solutions

### Error: `ModuleNotFoundError: No module named 'picamera2'`
- **Cause**: Trying to run edge streaming scripts on a standard laptop, or installing `picamera2` with pip instead of apt.
- **Solution**: On a laptop, run unit tests using `pytest -v`, which provides mock stubs in `tests/conftest.py`. On a Raspberry Pi, install the system package: `sudo apt install python3-picamera2`.

### Error: `FileNotFoundError: No such file or directory: 'plant_disease_data.xlsx'`
- **Cause**: The treatment database is missing or named differently.
- **Solution**: Confirm that `plant_disease_data.xlsx` is located in the `data/` directory.

### Error: `OSError: [Errno 98] Address already in use` (Port 8765)
- **Cause**: A previous server instance is still running in the background and holding the network port.
- **Solution**: Find and stop the running process:
  - On Linux / Raspberry Pi: `fuser -k 8765/tcp`
  - On Windows: `netstat -ano | findstr :8765` and terminate the PID with `taskkill /PID <PID> /F`.

### Error: Model load fails due to missing TensorFlow
- **Cause**: `tensorflow` is not installed in the active environment.
- **Solution**: For inference on edge devices, install `tflite-runtime` or convert the model to `.tflite` format using `src/convert_tflite.py`.

---

## Next Steps

- Review the component layout in [System Architecture](architecture.md).
- Understand how data flows through the system in [How It Works](how-it-works.md).
- Explore model structure and metrics in [Model and Training](model.md).
