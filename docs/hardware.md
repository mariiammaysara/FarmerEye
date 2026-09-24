# Hardware Setup

What this page covers:
This page describes the physical camera hardware interface for the Raspberry Pi.
It specifies connection steps, documents which features are verified versus mocked, and clarifies hardware boundaries.

---

## Camera Interface and Specifications

Farmer Eye uses the official Raspberry Pi Camera Module (Version 2 or Version 3) connected via a flexible ribbon cable to the Camera Serial Interface (CSI) port on the Raspberry Pi 4.

| Specification | Configuration |
|---|---|
| **Physical Interface** | 15-pin CSI ribbon cable directly to Raspberry Pi camera bus. |
| **Driver Framework** | `libcamera` and `python3-picamera2` native Linux drivers. |
| **Stream Resolution** | 640 pixels wide by 480 pixels high (4:3 aspect ratio). |
| **Color Space** | 3-channel RGB image format. |
| **Target Capture Rate** | ~20 frames per second. |

---

## Physical Installation Steps

1. **Power Down**: Disconnect power from the Raspberry Pi before attaching or detaching ribbon cables to avoid electrical damage.
2. **Open CSI Connector**: Gently pull up on the plastic edges of the camera connector labeled `CAMERA` on the Raspberry Pi board.
3. **Insert Cable**: Insert the ribbon cable with the metal contacts facing the HDMI ports (away from the Ethernet and USB ports).
4. **Lock Connector**: Push the plastic clip down firmly to clamp the cable securely in place.
5. **Verify Installation**: Power on the Raspberry Pi and run:
   ```bash
   rpicam-hello -t 3000
   ```
   *(Requires physical Raspberry Pi hardware; not tested on development laptops).*

---

## What Is Tested vs. What Is Not

To support rapid development on standard workstations without requiring physical electronics, this repository separates hardware logic from test automation:

| Component / Function | Laptop Status | Raspberry Pi Status | Notes |
|---|:---:|:---:|---|
| **Camera Capture** | Software Mock | Real Hardware | Handled by `picamera2` stub in `tests/conftest.py` during automated tests. |
| **Image Preprocessing** | Verified | Verified | Resizing, normalization, and RGB transformations execute in pure software. |
| **Model Inference** | Verified | Verified | Executes via standard TensorFlow or TensorFlow Lite runtime. |
| **WebSocket Networking** | Verified | Verified | TCP communication tested on local loopback interface (`127.0.0.1`). |
| **Physical Camera Ribbon** | Not Tested | Tested on Pi | Verified on physical hardware by the deployment team. |

---

## Motor Hardware Boundary

> [!IMPORTANT]
> **Motor Controller Code Is Not Included**:
> This repository contains software for computer vision inference, image streaming, and treatment lookup.
> It does not include the physical motor driving implementation or rover chassis locomotion controllers.
> The pin assignments referenced in earlier documentation and `tests/test_motor.py` represent mock unit tests for an L298N H-Bridge interface, not active runtime drivers in this repository.

---

## Next Steps

- Learn about local laptop setup in [Getting Started](getting-started.md).
- Follow a camera frame through the system in [How It Works](how-it-works.md).
- Review known hardware and software limits in [Limitations and Future Work](limitations.md).
