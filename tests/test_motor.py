"""
Tests for motor command mapping with mocked GPIO.
Tests that directional commands map correctly to H-bridge (L298N) pin logic
without requiring physical Raspberry Pi hardware.
"""
import pytest
import RPi.GPIO as GPIO

# Standard L298N 4-pin mapping definition
# IN1, IN2 control left motor; IN3, IN4 control right motor
DEFAULT_MOTOR_PINS = {
    "IN1": 17,
    "IN2": 27,
    "IN3": 22,
    "IN4": 23
}

COMMAND_PIN_STATES = {
    "forward":  {"IN1": GPIO.HIGH, "IN2": GPIO.LOW,  "IN3": GPIO.HIGH, "IN4": GPIO.LOW},
    "backward": {"IN1": GPIO.LOW,  "IN2": GPIO.HIGH, "IN3": GPIO.LOW,  "IN4": GPIO.HIGH},
    "left":     {"IN1": GPIO.LOW,  "IN2": GPIO.HIGH, "IN3": GPIO.HIGH, "IN4": GPIO.LOW},
    "right":    {"IN1": GPIO.HIGH, "IN2": GPIO.LOW,  "IN3": GPIO.LOW,  "IN4": GPIO.HIGH},
    "stop":     {"IN1": GPIO.LOW,  "IN2": GPIO.LOW,  "IN3": GPIO.LOW,  "IN4": GPIO.LOW},
}


class L298NMotorController:
    """Motor controller abstraction for L298N driver."""
    def __init__(self, pins=None):
        self.pins = pins or DEFAULT_MOTOR_PINS
        GPIO.setmode(GPIO.BCM)
        for pin in self.pins.values():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        self.current_command = "stop"

    def execute_command(self, command: str):
        cmd = command.strip().lower()
        if cmd not in COMMAND_PIN_STATES:
            raise ValueError(f"Unknown motor command: {command}")
        
        target_states = COMMAND_PIN_STATES[cmd]
        for pin_name, state in target_states.items():
            GPIO.output(self.pins[pin_name], state)
        self.current_command = cmd
        return target_states

    def cleanup(self):
        self.execute_command("stop")
        GPIO.cleanup()


def test_gpio_mock_is_active():
    assert hasattr(GPIO, "setmode")
    assert hasattr(GPIO, "setup")
    assert hasattr(GPIO, "output")
    assert GPIO.HIGH == 1
    assert GPIO.LOW == 0


def test_motor_controller_initialization():
    controller = L298NMotorController()
    assert controller.current_command == "stop"
    GPIO.setmode.assert_called_with(GPIO.BCM)


def test_motor_directional_commands():
    controller = L298NMotorController()

    for cmd, expected_states in COMMAND_PIN_STATES.items():
        states = controller.execute_command(cmd)
        assert states == expected_states
        assert controller.current_command == cmd


def test_motor_invalid_command_raises():
    controller = L298NMotorController()
    with pytest.raises(ValueError):
        controller.execute_command("fly_to_moon")


def test_motor_cleanup():
    controller = L298NMotorController()
    controller.cleanup()
    GPIO.cleanup.assert_called()
