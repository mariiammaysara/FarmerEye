"""
Pytest configuration and hardware stubs.
Provides sys.modules mocks for RPi.GPIO, gpiozero, and picamera2
so tests can execute on any desktop/laptop/CI without hardware.
"""
import sys
import os
from unittest.mock import MagicMock

# Ensure repo root and src/ are in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 1. Mock RPi and RPi.GPIO
rpi_mock = MagicMock()
rpi_gpio_mock = MagicMock()
rpi_gpio_mock.BCM = "BCM"
rpi_gpio_mock.BOARD = "BOARD"
rpi_gpio_mock.OUT = "OUT"
rpi_gpio_mock.IN = "IN"
rpi_gpio_mock.HIGH = 1
rpi_gpio_mock.LOW = 0
rpi_gpio_mock.setmode = MagicMock()
rpi_gpio_mock.setup = MagicMock()
rpi_gpio_mock.output = MagicMock()
rpi_gpio_mock.cleanup = MagicMock()
rpi_mock.GPIO = rpi_gpio_mock

sys.modules.setdefault("RPi", rpi_mock)
sys.modules.setdefault("RPi.GPIO", rpi_gpio_mock)

# 2. Mock gpiozero
gpiozero_mock = MagicMock()
sys.modules.setdefault("gpiozero", gpiozero_mock)

# 3. Mock picamera2
picamera2_mock = MagicMock()


class MockPicamera2:
    def __init__(self, *args, **kwargs):
        self.is_running = False

    def create_preview_configuration(self, *args, **kwargs):
        return {"main": {"size": (640, 480)}}

    def configure(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        self.is_running = True

    def stop(self, *args, **kwargs):
        self.is_running = False

    def close(self, *args, **kwargs):
        self.is_running = False

    def capture_array(self, *args, **kwargs):
        import numpy as np
        # Return dummy RGB 640x480 frame
        return np.zeros((480, 640, 3), dtype=np.uint8)


picamera2_mock.Picamera2 = MockPicamera2
sys.modules.setdefault("picamera2", picamera2_mock)
