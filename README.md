<div align="center">
  <h1>Farmer Eye Robotic Car</h1>
  <h3>Graduation Project</h3>
  <img src="docs/assets/robotic_car_image.jpg" width="350" height="350" alt="Farmer Eye Robotic Car">
</div>

<br>

Farmer Eye is a professional-grade AI-IoT solution designed to identify plant diseases in real-time. By combining deep learning (CNNs) with IoT hardware (Raspberry Pi) integrated into a robotic car, it provides farmers with instant diagnostic feedback and treatment recommendations to improve crop yield and health.

## Project Overview

This project features a remote-controlled robotic car equipped with a Raspberry Pi and high-definition camera. The system patrols agricultural fields, captures real-time video streams, and uses a Convolutional Neural Network (CNN) to detect diseases in crops like Cotton, Tomato, Potato, Pepper, and Strawberry.

## Mobile Application

A cross-platform mobile application was developed using **Flutter** to provide a seamless user interface for the system.
- **Real-Time Monitoring**: View the live camera feed directly from the robotic car.
- **Instant Notifications**: Receive immediate alerts when a disease is detected, including the disease type and confidence level.
- **Treatment Database**: Access a comprehensive guide of treatment protocols in both English and Arabic.
- **Remote Control**: Interface for monitoring the car's status and telemetry.

## Features

- Real-Time Detection: Continuous monitoring of plant health using Raspberry Pi camera streams.
- Deep Learning Accuracy: Fine-tuned CNN models optimized for identifying various plant diseases from the PlantVillage dataset.
- IoT Integration: Seamless communication between hardware (Raspberry Pi) and a WebSocket-based server for instant delivery of results.
- Multi-Crop Support: Detects diseases across various plants including Cotton, Tomato, Potato, Pepper, and Strawberry.
- Automated Treatment Guidance: Provides both English and Arabic treatment protocols directly from an integrated database.
- Production-Ready Structure: Professionally organized codebase for scalability and maintainability.

## Tech Stack

- Machine Learning: TensorFlow, Keras, Scikit-learn
- Data Processing: Pandas, NumPy
- Computer Vision: OpenCV, PIL (Pillow)
- Mobile App: Flutter (Dart)
- IoT/Hardware: Raspberry Pi, Picamera2, Robotic Car Chassis
- Connectivity: WebSockets (Asyncio)
- Database: Excel-based treatment reference (Openpyxl)

## Project Structure

```text
FarmerEye/
├── data/           # Plant disease database and datasets
├── models/         # Pre-trained and fine-tuned .h5 models
├── src/            # Core source code (.py files)
├── notebooks/      # Research and development Jupyter notebooks
├── docs/           # Project documentation and specifications
├── tests/          # Unit and integration tests
├── requirements.txt # Project dependencies
├── .gitignore      # Python-specific git configuration
└── README.md       # Project overview and documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mariiammaysara/FarmerEye.git
   cd FarmerEye
   ```

2. Set up a virtual environment (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Hardware Setup (for Raspberry Pi):
   - Ensure the Raspberry Pi camera is enabled.
   - Install Picamera2 according to the official Raspberry Pi documentation.

## Usage

### Training the Model
To re-train or fine-tune the model, use the provided notebook in the `notebooks/` directory or run the training script:
```bash
python src/app.py
```

### Starting the Real-Time Detection Server
To start the WebSocket server and the monitoring system on a Raspberry Pi:
```bash
python src/real_time_detection.py
```

### Unified Stream Analysis
To run the combined detection and streaming service:
```bash
python src/combined_detection_stream.py
```

<br><br>

---

<p align="center">
  <br>
  <b>Developed and Designed by</b><br>
  Mariam Maysara • Fatma Zayed • Mohamed Magdy • Mohamed Hesham
  <br><br>
  <b>FarmerEye Team</b>
  <br><br>
