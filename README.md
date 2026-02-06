<div align="center">
  <h1>🚜 Farmer Eye Robotic Car – Graduation Project</h1>
  <h3>Smart AI & IoT System for Real-Time Plant Disease Detection</h3>
  <p align="center">
    <img src="docs/assets/robotic_car_image.jpg" width="350" height="350" alt="Farmer Eye Robotic Car">
  </p>
</div>

---

## 📖 Project Overview

**Farmer Eye** is a professional-grade, end-to-end AI-IoT ecosystem designed to modernize agriculture by automating plant disease diagnostics. The system integrates a **remote-controlled robotic car**, high-performance **Deep Learning models**, and a **Raspberry Pi-powered edge device** to patrol fields and identify crop diseases in real-time.

By bridging the gap between hardware and software, Farmer Eye provides farmers with instant diagnostic feedback and localized treatment recommendations (available in English and Arabic) to prevent crop loss and optimize harvest health.

## 📱 Mobile Application

The system includes a dedicated cross-platform mobile application built with **Flutter**, serving as the central hub for monitoring and control:

- 🎥 **Real-Time Live Feed**: Low-latency video streaming from the robotic car's onboard camera.
- 🔔 **Instant Alerts**: Push notifications sent the moment a plant disease is detected, including classification and confidence metrics.
- 💊 **Treatment Intelligence**: Integrated pharmaceutical database providing clinical diagnostics and treatment protocols.
- 🕹️ **Remote Telemetry**: Real-time status monitoring for hardware health and connectivity.

## ✨ System Features

- **Real-Time Edge Inference**: Continuous monitoring and detection powered by localized processing on Raspberry Pi.
- **High-Accuracy CNN**: Fine-tuned Convolutional Neural Networks optimized for high-precision identification across various plant classes.
- **Multi-Crop Support**: Robust detection for Cotton, Tomato, Potato, Pepper, and Strawberry.
- **Bi-Lingual Diagnostics**: Comprehensive treatment guidance in both English and Arabic.
- **Production-Ready Architecture**: Decoupled, modular codebase designed for scalability and maintainability.
- **IoT-Cloud Synchronization**: WebSocket-based communication ensuring instant data delivery between edge and mobile.

## 🛠️ Tech Stack

### 🧠 Artificial Intelligence & Data
- **Frameworks**: TensorFlow, Keras, Scikit-learn
- **Libraries**: NumPy, Pandas, OpenCV, PIL (Pillow)
- **Deep Learning**: Convolutional Neural Networks (CNN)

### ⚙️ Hardware & IoT
- **Compute**: Raspberry Pi
- **Camera**: PiCamera2 / HD Modules
- **Mechanics**: Robotic Car Chassis, L298N Motor Drivers
- **Connectivity**: WebSockets (Asyncio)

### 📱 Mobile & Frontend
- **Framework**: Flutter (Dart)
- **State Management**: Provider / BLoC
- **Communication**: WebSocket Client

### 📂 Backend & Infrastructure
- **Server**: Python-based WebSocket Server
- **Database**: Excel/CSV-based treatment reference (Openpyxl)

## 🏗️ Project Structure

```text
FarmerEye/
├── data/
│   └── plant_disease_data.xlsx      # Database for treatments and diagnostics
├── docs/
│   └── assets/
│       └── robotic_car_image.jpg    # Project visual assets
├── models/
│   ├── fine_tuned_model.h5          # Optimized CNN model
│   └── plant_disease_model_final.h5 # Final production-ready model
├── notebooks/
│   └── research_and_training.ipynb  # ML development and training pipeline
├── src/
│   ├── app.py                       # Main application entry point
│   ├── combined_detection_stream.py # Combined UI and streaming logic
│   ├── real_time_detection.py       # Core inference and hardware logic
│   └── raspberry_pi_camera_stream.py# Low-level camera streaming service
├── tests/                           # System validation and testing
├── requirements.txt                 # Dependency manifest
└── README.md                        # Project documentation
```

## 🚀 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mariiammaysara/FarmerEye.git
   cd FarmerEye
   ```

2. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🔌 Hardware Setup

1. **Camera Configuration**:
   - Enable the camera interface on Raspberry Pi.
   - Install `Picamera2` according to the official Raspberry Pi documentation.
2. **Motor Driver**:
   - Connect the motor driver to the GPIO pins as configured in the source code.
3. **Power Management**:
   - Ensure stable power supply for both the Raspberry Pi and the motor chassis.

## 🖥️ Usage

### 1. Training & Research
Explore the model development phase via Jupyter:
```bash
jupyter notebook notebooks/
```

### 2. Real-Time Detection
Start the monitoring system on the Raspberry Pi:
```bash
python src/real_time_detection.py
```

### 3. Unified Stream Analysis
Run the combined detection and streaming service:
```bash
python src/combined_detection_stream.py
```

## 📊 Output
Upon detection, the system provides:
- **Disease Classification**: Accurate identification of the plant condition.
- **Confidence Score**: Statistical probability of the detection.
- **Treatment Protocol**: Actionable advice in English/Arabic fetched from the database.

## 👥 Development Team

<p align="center">
  <b>Developed and Designed by</b><br>
  Mariam Maysara • Fatma Zayed • Mohamed Magdy • Mohamed Hesham
  <br><br>
  <b>FarmerEye Team</b>
</p>

## 📄 License

This project is licensed under the **MIT License**.

---
<p align="center">
  <i>Developed as part of an advanced AI-IoT initiative for modern agriculture.</i>
</p>
