# Hardware Setup, Wiring & Assembly Guide

This guide details the physical hardware components, power regulation, wiring schematics, and operating system configuration required to assemble the **Farmer Eye** robotic vehicle.

---

## 1. Bill of Materials (BOM)

| Component | Specification / Model | Purpose | Quantity |
|---|---|---|:---:|
| **Single Board Computer** | Raspberry Pi 4 Model B (4GB or 8GB RAM) | Edge processing, camera streaming, inference server | 1 |
| **Camera Module** | Raspberry Pi Camera v2 / v3 or HQ Camera | Video stream capture and leaf diagnostics | 1 |
| **Ribbon Cable** | 15-pin 1.0mm pitch FFC cable (CSI) | Connects camera module to Pi CSI port | 1 |
| **Motor Driver** | L298N Dual H-Bridge Controller | Drives high-current DC geared motors | 1 |
| **Chassis Kit** | 4WD / 2WD Smart Robot Car Chassis with TT Motors | Physical mobility platform | 1 |
| **Motor Power** | 2x 18650 Li-ion rechargeable batteries (7.4V–8.4V) | Powers L298N and drive motors | 1 pack |
| **Logic Power** | 5V / 3A USB-C Power Bank or DC-DC Buck Converter | Clean, isolated power for Raspberry Pi | 1 |
| **Jumper Wires** | Male-to-Female, Female-to-Female DuPont wires | Logic and GPIO connections | ~10 |

---

## 2. Power Architecture & Grounding

> [!CAUTION]
> **Do not power the Raspberry Pi directly from the motor battery without a voltage regulator.**
> DC motors draw significant inductive current spikes and cause voltage brownouts that will reboot or permanently damage the Raspberry Pi's processor.

```
 [ 7.4V - 8.4V Li-ion Battery ]
          │
          ├───> [ L298N 12V Terminal ] ───> Powers TT Motors
          │
          └───> [ Common Ground (GND) ] ──┬──> [ L298N GND ]
                                          │
                                          └──> [ Raspberry Pi GND (Pin 6) ]

 [ Dedicated 5V 3A Power Bank ]
          │
          └───> [ USB-C Port ] ───────────> Powers Raspberry Pi 4
```

### Essential Power Rules:
1. **Common Ground (Shared GND)**:
   The Raspberry Pi's GND and the L298N's GND **must be tied together**. Without a common ground reference, logic signals (`HIGH`/`LOW`) sent over GPIO cannot be interpreted correctly by the motor driver.
2. **Dedicated Logic Power**:
   Use a separate 5V / 3A supply (e.g. mobile power bank) connected to the Raspberry Pi's USB-C port to prevent motor noise from interfering with computer vision inference.

---

## 3. Wiring & Pinout Reference Table

> [!IMPORTANT]
> **Codebase Implementation Status**: The current `src/` directory implements video capture, AI inference, and WebSockets, but **does not include active motor control code**. The pin assignments below represent the project's tested H-bridge abstraction implemented in [tests/test_motor.py](../tests/test_motor.py). Pin connections marked **verify on hardware** should be validated against the physical car chassis before deployment.

| Header / Port | BCM GPIO | Physical Header Pin | Target Component & Pin | Functional Role | Source Reference | Verification Status |
|---|:---:|:---:|---|---|---|:---:|
| **CSI Camera Port** | — | 15-pin Ribbon | PiCamera2 / Camera Module | Real-time video ingestion | [src/combined_detection_stream.py](../src/combined_detection_stream.py#L13) | **Confirmed in code** |
| **GPIO Header** | `GPIO 17` | Pin 11 | L298N `IN1` | Left Motor Forward | [tests/test_motor.py](../tests/test_motor.py#L12) | *Verify on hardware* |
| **GPIO Header** | `GPIO 27` | Pin 13 | L298N `IN2` | Left Motor Backward | [tests/test_motor.py](../tests/test_motor.py#L13) | *Verify on hardware* |
| **GPIO Header** | `GPIO 22` | Pin 15 | L298N `IN3` | Right Motor Forward | [tests/test_motor.py](../tests/test_motor.py#L14) | *Verify on hardware* |
| **GPIO Header** | `GPIO 23` | Pin 16 | L298N `IN4` | Right Motor Backward | [tests/test_motor.py](../tests/test_motor.py#L15) | *Verify on hardware* |
| **Power Header** | `5V` | Pin 2 or 4 | L298N `5V Logic` | Logic rail for L298N driver | — | *Verify on hardware* |
| **Ground Header** | `GND` | Pin 6 (or 9, 14, 20) | L298N `GND` | Common ground reference | — | *Verify on hardware* |
| **Chassis Battery** | — | Terminal Block | L298N `12V / VCC` | High-voltage motor drive power | — | *Verify on hardware* |

---

## 4. Raspberry Pi OS Configuration

### Step 1: Install 64-bit Raspberry Pi OS
We recommend **Raspberry Pi OS (64-bit Debian Bookworm)** flashed using the official [Raspberry Pi Imager](https://www.raspberrypi.com/software/).

### Step 2: Enable the Camera Interface
Open the Raspberry Pi configuration utility in the terminal:
```bash
sudo raspi-config
```
Navigate to:
`Interface Options` ➔ `Camera / Legacy Camera` ➔ `Enable` ➔ `Finish` ➔ `Reboot`.

### Step 3: Install Picamera2 via APT
> [!WARNING]
> Do **NOT** install `picamera2` using `pip install`. The Python package requires underlying native C bindings and video drivers provided by Raspberry Pi OS:
```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-pip python3-venv
```

### Step 4: Verify Camera Functionality
Test the camera hardware using the native command-line tool:
```bash
# Capture a test JPEG image to verify CSI communication
rpicam-jpeg -o test_camera.jpg
```
If the image captures successfully, the hardware interface is functioning properly.

---

## 5. Laptop Development & Testing (No Hardware Required)

You do **not** need physical Raspberry Pi hardware or a camera to develop, run unit tests, or evaluate models in this repository.

The repository includes a comprehensive mock configuration in [tests/conftest.py](../tests/conftest.py) that injects virtual stubs for:
- `picamera2` (generates virtual test frames)
- `RPi.GPIO` (records and verifies pin high/low states)
- `gpiozero`

To verify the software stack on any PC or laptop:
```bash
pytest -v
```
