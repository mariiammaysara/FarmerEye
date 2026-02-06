import cv2
import asyncio
import websockets
import json
import base64
from picamera2 import Picamera2

# WebSocket port
WEBSOCKET_PORT = 8766
connected_clients = set()

async def handle_client(websocket, path="/"):
    """Handle incoming WebSocket clients."""
    print(f"🔌 New client: {websocket.remote_address}")
    connected_clients.add(websocket)
    try:
        await websocket.wait_closed()
    except Exception as e:
        print(f"⚠️ Client error: {str(e)}")
    finally:
        connected_clients.remove(websocket)
        print(f"❌ Client disconnected: {websocket.remote_address}")

async def broadcast_frame(frame_data):
    """Send frame data to all connected WebSocket clients."""
    if connected_clients:
        message = json.dumps(frame_data)
        await asyncio.gather(
            *[client.send(message) for client in connected_clients],
            return_exceptions=True  # Prevent single failure from crashing
        )

def encode_frame(frame):
    """Encode image to base64 string."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buffer).decode('utf-8')

async def stream_camera():
    """Continuously capture and stream frames."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
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
            await asyncio.sleep(0.033)  # ~30 FPS
    except Exception as e:
        print(f"❌ Camera stream error: {str(e)}")
    finally:
        picam2.stop()

async def main():
    """Start WebSocket server and camera stream."""
    server = await websockets.serve(handle_client, "0.0.0.0", WEBSOCKET_PORT)
    print(f"✅ WebSocket server started on port {WEBSOCKET_PORT}")
    await stream_camera()

if _name_ == "_main_":
    asyncio.run(main())