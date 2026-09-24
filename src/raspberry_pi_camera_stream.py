"""Standalone video streaming service for Raspberry Pi camera over WebSockets.

This module captures live video frames from the CSI camera using Picamera2,
compresses them to JPEG format, encodes them to base64 text, and broadcasts
them to connected WebSocket clients. It runs independently of model inference
to provide a dedicated low-latency video feed.

Usage:
    Run directly on Raspberry Pi:
        python src/raspberry_pi_camera_stream.py

Network Configuration:
    Port: 8766 (configurable via WEBSOCKET_PORT environment variable)
    Host: 0.0.0.0 (configurable via WEBSOCKET_HOST environment variable)

Pipeline Context:
    CSI Camera Module -> Picamera2.capture_array() -> cv2.imencode() -> WebSocket broadcast
"""
import os
import cv2
import asyncio
import websockets
import json
import base64
from typing import Dict, Any

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

# Network configuration
WEBSOCKET_HOST: str = os.environ.get("WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT: int = int(os.environ.get("WEBSOCKET_PORT", 8766))

# Camera acquisition and streaming constants
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
JPEG_QUALITY: int = 60
FRAME_INTERVAL_SECONDS: float = 0.033  # Interval for target rate of ~30 FPS (1 / 30 = 0.0333)

connected_clients: set = set()


async def handle_client(websocket, path: str = "/") -> None:
    """Handles lifecycle of an incoming WebSocket client connection.

    Registers new client connections, awaits client closure, and guarantees
    cleanup from the connected client pool upon disconnection.

    Args:
        websocket: The WebSocket connection instance.
        path: The requested URI path (defaults to '/').
    """
    print(f"🔌 New client: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    except Exception as e:
        print(f"⚠️ Client error: {str(e)}")
    finally:
        connected_clients.remove(websocket)
        print(f"❌ Client disconnected: {websocket.remote_address}")


async def broadcast_frame(frame_data: Dict[str, Any]) -> None:
    """Broadcasts a JSON-serialized frame payload to all active clients.

    Args:
        frame_data: Dictionary containing frame metadata and base64 image data.
    """
    if connected_clients:
        message = json.dumps(frame_data)
        await asyncio.gather(
            *[client.send(message) for client in connected_clients],
            return_exceptions=True
        )


def encode_frame(frame) -> str:
    """Compresses a BGR/RGB image frame to JPEG and encodes it as base64 string.

    Args:
        frame: A NumPy ndarray image frame in RGB or BGR format.

    Returns:
        A UTF-8 base64 string representing the compressed JPEG image.
    """
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buffer).decode('utf-8')


async def stream_camera() -> None:
    """Initializes camera hardware and continuously broadcasts video frames.

    Configures the Picamera2 device with the designated resolution, starts video
    capture, and broadcasts compressed frames in an asynchronous loop until interrupted.
    """
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    print(f"✅ Camera stream started on port {WEBSOCKET_PORT}")

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_data = {
                'type': 'camera_frame',
                'image': encode_frame(frame)
            }
            await broadcast_frame(frame_data)
            await asyncio.sleep(FRAME_INTERVAL_SECONDS)
    except Exception as e:
        print(f"❌ Camera stream error: {str(e)}")
    finally:
        picam2.stop()


async def main() -> None:
    """Starts the WebSocket listener and begins streaming camera frames."""
    server = await websockets.serve(handle_client, WEBSOCKET_HOST, WEBSOCKET_PORT)
    print(f"✅ WebSocket server started on port {WEBSOCKET_PORT}")
    await stream_camera()


if __name__ == "__main__":
    asyncio.run(main())