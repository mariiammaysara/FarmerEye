<div align="center">
  <h1>Smart Vehicle for Plant Diseases Detection and Classification Using AI and IoT (Farmer Eye Robotic Car)</h1>
  <p>An edge-AI precision agriculture system combining mobile robotics, computer vision, and IoT to detect crop diseases in real time. | <a href="https://lnkd.in/p/epwcGhRv"><b>Watch Demo</b></a></p>

  <p>
    <a href="https://github.com/mariiammaysara/FarmerEye/actions/workflows/tests.yml"><img src="https://github.com/mariiammaysara/FarmerEye/actions/workflows/tests.yml/badge.svg" alt="Tests Status"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  </p>

  <p>
    <img src="./docs/assets/robotic_car.svg" alt="Farmer Eye Robotic Car" width="220">
  </p>

  <p>
    <a href="#what-it-does"><b>Overview</b></a> •
    <a href="#system-architecture"><b>Architecture</b></a> •
    <a href="#quick-start"><b>Quick Start</b></a> •
    <a href="#results-snapshot"><b>Results</b></a> •
    <a href="#documentation"><b>Documentation</b></a> •
    <a href="#credits--acknowledgments"><b>Credits</b></a>
  </p>
</div>

---

## What It Does

- **Real-Time Edge Video**: Streams live camera images over WebSockets from a Raspberry Pi 4 to client devices.
- **Deep Learning Classification**: Evaluates crop leaves against 25 health categories across Cotton, Tomato, Potato, Pepper, and Strawberry.
- **Bilingual Actionable Advice**: Automatically matches diagnosed diseases with practical treatment instructions in English and Arabic.
- **Hardware-Free Testing**: Includes a mock test suite allowing development and verification on standard laptops without physical cameras or GPIO pins.

---

## System Architecture

```mermaid
graph LR
    Cam[PiCamera2 Sensor] --> EdgeServer[Edge Server: combined_detection_stream.py]
    EdgeServer --> Preproc[Preprocessing: 224x224, /255.0]
    Preproc --> CNN[CNN Model: plant_disease_model_final.h5]
    CNN --> DB[(Treatment DB: plant_disease_data.xlsx)]
    DB --> EdgeServer
    EdgeServer <-->|"WebSocket: ws://<IP>:8765"| App[Mobile Client]
```

---

## Tech Stack

| Area | Tools | Used for |
|---|---|---|
| Language | Python | Core backend logic, model training, and test suites |
| Deep learning | TensorFlow (`>=2.15.0,<2.18.0`), Keras | CNN classification model architecture and inference |
| Image processing | OpenCV (`opencv-python>=4.8.0,<5.0.0`), NumPy (`>=1.24.0,<2.0.0`) | Frame capture, resizing (224x224), normalization, and encoding |
| Camera (Raspberry Pi) | Picamera2 | Hardware camera frame acquisition on Raspberry Pi 4 |
| Real-time communication | websockets (`>=12.0,<14.0`), asyncio | Async WebSocket server for live video and payload delivery |
| Data (treatment database) | pandas (`>=2.0.0,<3.0.0`), openpyxl (`>=3.1.0,<4.0.0`) | Loading and querying bilingual treatment advice from Excel |
| Training and evaluation | scikit-learn (`>=1.3.0,<1.6.0`), matplotlib (`>=3.7.0,<4.0.0`), seaborn (`>=0.12.0,<0.14.0`) | Metrics calculation, confusion matrix evaluation, and plots |
| Testing and CI | pytest (`>=7.4.0,<9.0.0`), GitHub Actions | Automated unit/integration test suite and CI workflow |

> **Note**: Mobile app (Flutter) and vehicle hardware: developed separately, not included in this repository.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mariiammaysara/FarmerEye.git && cd FarmerEye

# 2. Create and activate a virtual environment
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1

# 3. Install testing dependencies
pip install -r requirements-dev.txt

# 4. Verify installation with automated tests
pytest -v

# 5. Launch edge streaming server (requires Raspberry Pi and camera)
python src/combined_detection_stream.py
```

---

## Results Snapshot

Evaluated on an independent holdout test set of 7,955 images:
- **Test Accuracy**: 97.95%
- **Macro Average F1-Score**: 0.98

*For complete 25-class precision, recall, confusion matrix heatmaps, and training curves, see [docs/model.md](docs/model.md).*

---

## Documentation

Full project guides and specifications are located in the [docs/](docs/README.md) directory:

| Guide | Description |
|---|---|
| [How It Works](docs/how-it-works.md) | Follows one camera image from capture to mobile treatment alert. |
| [Getting Started](docs/getting-started.md) | Setup, dependency installation, running scripts, and troubleshooting. |
| [System Architecture](docs/architecture.md) | Component layouts, data flow, server options, and threading models. |
| [Model and Training](docs/model.md) | Neural network structure, training parameters, and benchmark tables. |
| [WebSocket API](docs/websocket-api.md) | JSON message formats, client registration handshake, and event types. |
| [Treatment Database](docs/treatment-database.md) | Excel data schema, disease name normalization, and class addition steps. |
| [Hardware Setup](docs/hardware.md) | Camera module connection, physical boundaries, and testing limits. |
| [Development and Testing](docs/development.md) | Running pytest, hardware mocking architecture, CI, and code style. |
| [Limitations and Future Work](docs/limitations.md) | Technical boundaries, domain gap factors, and future roadmap. |
| [Project Context and Credits](docs/project-context.md) | Academic context, development team, supervisors, and funding recognition. |
| [Glossary](docs/glossary.md) | Definitions and explanations of all machine learning and engineering terms. |

---

## Project Structure

```text
FarmerEye/
├── .github/workflows/tests.yml   # CI pipeline: Python 3.10 and pytest
├── data/
│   ├── plant_disease_data.xlsx   # Bilingual treatment database
│   └── README.md                 # Dataset provenance card and splits
├── docs/                         # Technical documentation and visual assets
├── models/
│   └── plant_disease_model_final.h5 # Trained Keras CNN model weights
├── notebooks/
│   └── research_and_training.ipynb  # Exploratory training notebook
├── src/                          # Inference, streaming, and conversion code
├── tests/                        # Hardware-mocked pytest suite
├── requirements.txt              # Raspberry Pi runtime dependencies
├── requirements-dev.txt          # Development and evaluation dependencies
├── LICENSE                       # MIT License
└── README.md                     # Project entry point
```

---

## Limitations

The model was trained primarily on laboratory leaf images with uniform backgrounds; performance under harsh outdoor sunlight or soil clutter may vary. WebSockets are unencrypted (`ws://`), and treatments are informational guidelines that require verification by an agronomist. See [docs/limitations.md](docs/limitations.md).

---

## Dataset and Citations

The training dataset incorporates images from the **PlantVillage** dataset:
- Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. [arXiv:1511.08060](https://arxiv.org/abs/1511.08060).
- Detailed dataset breakdown and splits: [data/README.md](data/README.md).

---

## Credits & Acknowledgments

### Team
- Mohamed Magdy
- Mohamed Hesham Shawky
- Fatma Zayed
- Mariam Maysara

### Supervisors
- Dr. Saeed Mohsen
- Dr. Ahmed Farouk

Supervisor's announcement: [LinkedIn post](https://lnkd.in/p/e2M9fTMc)

### Recognition
- ASRT "My Project is My Beginning" graduation projects funding program, academic year 2024-2025: [Announcement](https://www.linkedin.com/posts/mmagdyx_academyabrofabrscientificabrresearchabrandabrtechnology-ugcPost-7290718864219242497-3t-A)
- ITIDA "ITAC University Student Projects" funding: [Announcement](https://www.linkedin.com/posts/mariam-maysara_itac-graduationsupportedprogram-innovation-activity-7320806920112488449-Zfre)
- 3rd International Youth AI Forum: reached the final stage with this project: [Announcement](https://www.linkedin.com/posts/mmagdyx_%D8%B3%D8%B9%D9%8A%D8%AF-%D8%A8%D8%A7%D9%84%D9%85%D8%B4%D8%A7%D8%B1%D9%83%D8%A9-%D9%81%D9%8A-%D8%A7%D9%84%D9%85%D9%82%D8%A7%D8%A8%D9%84%D8%A7%D8%AA-%D8%A7%D9%84%D8%B4%D8%AE%D8%B5%D9%8A%D8%A9-%D8%A7%D9%84%D8%AE%D8%A7%D8%B5%D8%A9-ugcPost-7289640037082656768-KyxY)

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p>
    <b>Graduation Project (2024–2025)</b><br>
    Awarded Grade A+ (Highest Honors) to All Team Members<br>
    Faculty of Computer Science and Engineering, King Salman International University (KSIU)
  </p>
</div>

