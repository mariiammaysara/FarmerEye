"""Unified real-time camera streaming and plant disease detection service.

This module provides the primary edge service for Farmer Eye on Raspberry Pi.
It concurrently captures live camera frames, streams compressed JPEG video over
WebSockets to connected client apps, and periodically runs deep learning inference
with disease treatment lookup in an asynchronous, non-blocking pipeline.

Usage:
    Run directly on Raspberry Pi:
        python src/combined_detection_stream.py

Network Configuration:
    Port: 8765 (configurable via WEBSOCKET_PORT environment variable)
    Host: 0.0.0.0 (configurable via WEBSOCKET_HOST environment variable)

Pipeline Context:
    Picamera2 (CSI) -> Frame Capture -> cv2.imencode() -> WebSocket (Video Feed)
                     -> Preprocess (224x224, /255) -> CNN Inference (Cooldown 2.0s)
                     -> Database Lookup -> WebSocket (Diagnostic Alerts)
"""
import os
import cv2
import numpy as np
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None
import asyncio
import websockets
import json
import base64
import time
import pandas as pd
from datetime import datetime              
from typing import Dict, Any, Optional, Tuple
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None
import logging
import sys

# Filesystem and model paths
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH: str = os.path.join(BASE_DIR, 'models', 'plant_disease_model_final.h5')
TREATMENT_FILE_PATH: str = os.path.join(BASE_DIR, 'data', 'plant_disease_data.xlsx')

# Computer vision and preprocessing constants (must match model training resolution: 224x224)
IMG_SIZE: int = 224

# Network interface configuration
WEBSOCKET_HOST: str = os.environ.get('WEBSOCKET_HOST', '0.0.0.0')
WEBSOCKET_PORT: int = int(os.environ.get('WEBSOCKET_PORT', 8765))

# Inference and streaming threshold parameters
DETECTION_THRESHOLD: float = 0.98  # Minimum softmax confidence required to trigger alert
DETECTION_COOLDOWN: float = 2.0  # Seconds between consecutive inference scans to prevent thermal throttling
CAMERA_RESOLUTION: Tuple[int, int] = (640, 480)
LORES_RESOLUTION: Tuple[int, int] = (320, 240)
CAMERA_FRAMERATE: int = 20  # Acquisition rate in frames per second
STREAM_QUALITY: int = 70  # JPEG quality (0-100) balancing visual clarity and network bandwidth
DETECTION_IMAGE_QUALITY: int = 80  # Slightly higher JPEG quality for saved/sent alert images
PING_INTERVAL: float = 20.0  # Heartbeat ping check interval in seconds
PING_TIMEOUT: float = 35.0  # Seconds of silence before disconnecting an unresponsive client
CLIENT_REGISTRATION_TIMEOUT_SECONDS: float = 10.0
TIMEOUT_RETRY_INTERVAL_SECONDS: float = 10.0
CAMERA_WARMUP_SECONDS: float = 2.0
IDLE_CLIENT_CHECK_INTERVAL_SECONDS: float = 1.0
CAPTURE_RETRY_BACKOFF_SECONDS: float = 0.5
FPS_REPORT_INTERVAL_SECONDS: float = 10.0
MIN_TREATMENT_COLUMNS: int = 4  # Expected schema: [disease, treat_en, treat_ar, resources]

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]      
)
logger = logging.getLogger('UnifiedServer')

# Global runtime state
connected_clients: Dict[Any, Dict[str, Any]] = {}
model = None
treatment_df = None
picam2 = None

# Class Names
try:
    from src.class_names import CLASS_NAMES, normalize_disease_name
except ImportError:
    from class_names import CLASS_NAMES, normalize_disease_name
logger.info(f"Defined {len(CLASS_NAMES)} class names.")


def load_resources() -> bool:
    """Loads the Keras neural network model and pharmaceutical treatment spreadsheet.

    Returns:
        True if all required resources loaded successfully, or False if critical
        data files are missing or malformed.
    """
    global model, treatment_df
    try:
        if load_model is not None and os.path.exists(MODEL_PATH):
            logger.info(f"Loading disease detection model from: {MODEL_PATH}")
            model = load_model(MODEL_PATH)
            logger.info("Model loaded successfully.")
        else:
            model = None
            if load_model is None:
                logger.warning("TensorFlow is not installed. Model not loaded.")
            else:
                logger.warning(f"Model file not found at: {MODEL_PATH}")

        logger.info(f"Loading treatment data from: {TREATMENT_FILE_PATH}")
        treatment_df = pd.read_excel(TREATMENT_FILE_PATH)
        # Validate that the sheet contains at least the four required diagnostic columns
        if treatment_df.shape[1] < MIN_TREATMENT_COLUMNS:
            logger.error(f"Treatment file seems malformed. Expected at least {MIN_TREATMENT_COLUMNS} columns, found {treatment_df.shape[1]}.")
            treatment_df = None
            return False
        logger.info(f"Treatment data loaded successfully ({treatment_df.shape[0]} rows).")
        logger.info(f"   Columns: {list(treatment_df.columns)}")
        return True

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}. Cannot load resources.")
        return False
    except Exception as e:
        logger.error(f"Error loading resources: {str(e)}")
        return False


def get_treatment_info(disease_name: str) -> Optional[Dict[str, str]]:
    """Fetches bilingual treatment details for a given disease name.

    Applies tolerant string normalization to both the query label and database
    entries to ensure consistent matches despite formatting differences.

    Args:
        disease_name: The predicted disease class string.

    Returns:
        Dictionary containing disease name, English treatment, Arabic treatment,
        and reference URL, or None if no match is found.
    """
    if treatment_df is None:
        logger.warning("Treatment data not loaded. Cannot fetch info.")
        return None

    try:
        disease_col = treatment_df.columns[0]
        treat_en_col = treatment_df.columns[1]
        treat_ar_col = treatment_df.columns[2]
        resources_col = treatment_df.columns[3]

        norm_target = normalize_disease_name(disease_name)
        match = treatment_df[treatment_df[disease_col].apply(normalize_disease_name) == norm_target]

        if not match.empty:
            row = match.iloc[0]
            return {
                'disease': row[disease_col].strip(),
                'treatment_en': str(row[treat_en_col]).strip() if pd.notna(row[treat_en_col]) else 'N/A',
                'treatment_ar': str(row[treat_ar_col]).strip() if pd.notna(row[treat_ar_col]) else 'غير متوفر',
                'resources': str(row[resources_col]).strip() if pd.notna(row[resources_col]) else ''
            }
        else:
            logger.warning(f"Treatment info not found for disease: '{disease_name}'")
            return None
    except Exception as e:
        logger.error(f"Error fetching treatment info for '{disease_name}': {str(e)}")
        return None


def encode_frame(frame: np.ndarray, quality: int = STREAM_QUALITY) -> Optional[str]:
    """Compresses a NumPy image frame to JPEG and encodes it as base64 string.

    Args:
        frame: Image pixel matrix (NumPy ndarray).
        quality: JPEG compression quality factor between 0 and 100.

    Returns:
        Base64 UTF-8 string on success, or None if encoding fails.
    """
    try:
        is_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not is_success:
            logger.error("Failed to encode frame to JPEG.")
            return None
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error during frame encoding: {e}")
        return None


def preprocess_frame_for_model(frame: np.ndarray) -> Optional[np.ndarray]:
    """Resizes and scales a camera frame to match the input specification of the model.

    The model expects images scaled to 224x224 pixels with float32 values normalized
    to the interval [0.0, 1.0], matching training preprocessing.

    Args:
        frame: Captured camera frame as NumPy ndarray.

    Returns:
        Preprocessed 4D batch tensor of shape (1, 224, 224, 3), or None on error.
    """
    try:
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error during frame preprocessing: {e}")
        return None


async def register_client(websocket) -> bool:
    """Registers a newly connected client after receiving its identification handshake.

    Args:
        websocket: The connecting client's WebSocket instance.

    Returns:
        True if the client successfully identified and registered; False otherwise.
    """
    client_ip = websocket.remote_address
    logger.info(f"New client connecting from: {client_ip}")
    try:
        message = await asyncio.wait_for(websocket.recv(), timeout=CLIENT_REGISTRATION_TIMEOUT_SECONDS)
        data = json.loads(message)
        client_type = data.get('client_type', 'unknown_client')

        connected_clients[websocket] = {
            'last_ping': time.time(),
            'type': client_type,
            'address': client_ip
        }
        logger.info(f"Client {client_ip} registered as type: '{client_type}'")

        await websocket.send(json.dumps({
            'type': 'welcome',
            'message': 'Connected to Plant Disease Detection Server (Unified)',
            'timestamp': datetime.now().isoformat()
        }))
        return True

    except asyncio.TimeoutError:
        logger.warning(f"Client {client_ip} did not identify itself in time. Closing connection.")
        await websocket.close(reason='Identification timeout')
        return False
    except websockets.exceptions.ConnectionClosed:
        logger.warning(f"Connection closed by {client_ip} during registration.")
        return False
    except json.JSONDecodeError:
        logger.error(f"Invalid identification message from {client_ip}. Closing connection.")
        await websocket.close(reason='Invalid identification message')
        return False
    except Exception as e:
        logger.error(f"Error during client registration ({client_ip}): {e}")
        await websocket.close(reason='Registration error')
        return False


async def unregister_client(websocket) -> None:
    """Removes a disconnected client from the active client registry.

    Args:
        websocket: The WebSocket instance to remove.
    """
    client_info = connected_clients.pop(websocket, None)
    if client_info:
        logger.info(f"Client disconnected: {client_info.get('address', 'Unknown IP')} (Type: {client_info.get('type', 'N/A')})")
    else:
        logger.info(f"Client disconnected (already removed or registration failed): {websocket.remote_address}")


async def handle_client_messages(websocket) -> None:
    """Listens for inbound messages such as keepalive pings from a client session.

    Args:
        websocket: The active WebSocket client connection.
    """
    client_ip = connected_clients.get(websocket, {}).get('address', websocket.remote_address)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get('type')

                if msg_type == 'ping':
                    if websocket in connected_clients:
                        connected_clients[websocket]['last_ping'] = time.time()
                        await websocket.send(json.dumps({
                            'type': 'pong',
                            'timestamp': datetime.now().isoformat()
                        }))
                    else:
                        logger.warning(f"Received ping from unregistered client? {client_ip}")

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from {client_ip}: {message[:100]}...")
            except Exception as e:
                logger.error(f"Error processing message from {client_ip}: {e}")

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_ip} closed connection gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_ip} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in message handler for {client_ip}: {e}")
    finally:
        await unregister_client(websocket)


async def client_handler(websocket, path: str = "/") -> None:
    """Top-level connection handler dispatching registration and message listening.

    Args:
        websocket: The newly established WebSocket instance.
        path: Requested connection path.
    """
    if await register_client(websocket):
        await handle_client_messages(websocket)


async def broadcast_message(message_data: Dict[str, Any]) -> None:
    """Sends a JSON-serialized message payload concurrently to all registered clients.

    Args:
        message_data: Dictionary structure to serialize and transmit.
    """
    if not connected_clients:
        return

    message_json = json.dumps(message_data)
    tasks = [client.send(message_json) for client in connected_clients.keys()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    disconnected_clients = []
    for client, result in zip(list(connected_clients.keys()), results):
        if isinstance(result, Exception):
            client_ip = connected_clients.get(client, {}).get('address', client.remote_address)
            logger.error(f"Failed to send message to client {client_ip}: {result}")
            disconnected_clients.append(client)
            try:
                await client.close(reason="Send failed")
            except:
                pass

    for client in disconnected_clients:
        await unregister_client(client)


async def check_client_timeouts() -> None:
    """Periodically verifies client keepalives and terminates timed-out connections."""
    logger.info("Starting client timeout checker...")
    while True:
        try:
            await asyncio.sleep(PING_INTERVAL)
            current_time = time.time()
            timed_out_clients = []

            for client, info in list(connected_clients.items()):
                if current_time - info['last_ping'] > PING_TIMEOUT:
                    client_ip = info.get('address', client.remote_address)
                    logger.warning(f"Client timed out: {client_ip} (Type: {info.get('type', 'N/A')}). Last ping: {info['last_ping']:.2f}")
                    timed_out_clients.append(client)
                    try:
                        await client.close(reason='Ping timeout')
                    except Exception as e:
                        logger.error(f"Error closing timed out client {client_ip}: {e}")

            for client in timed_out_clients:
                await unregister_client(client)

        except asyncio.CancelledError:
            logger.info("Client timeout checker cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in client timeout checker: {e}")
            await asyncio.sleep(TIMEOUT_RETRY_INTERVAL_SECONDS)


async def process_frame_for_detection(frame: np.ndarray) -> bool:
    """Runs deep learning inference on a frame and broadcasts alerts if disease is found.

    Args:
        frame: RGB image array from camera.

    Returns:
        True if a disease above threshold was detected and broadcast; False otherwise.
    """
    if model is None:
        return False

    try:
        processed_img = preprocess_frame_for_model(frame)
        if processed_img is None:
            return False

        predictions = model.predict(processed_img, verbose=0)
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])

        if confidence >= DETECTION_THRESHOLD:
            disease_name = CLASS_NAMES[class_index]
            
            # Healthy foliage does not warrant pharmaceutical treatment alert
            if "healthy" in disease_name.lower():
                return False
                
            logger.info(f"Potential Detection: {disease_name} ({confidence*100:.1f}%)")

            treatment_info = get_treatment_info(disease_name)

            if treatment_info:
                annotated_frame = frame.copy()
                text = f"{treatment_info['disease']} ({confidence*100:.1f}%)"
                cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)

                encoded_annotated_frame = encode_frame(annotated_frame, quality=DETECTION_IMAGE_QUALITY)

                if encoded_annotated_frame:
                    detection_data = {
                        'type': 'detection',
                        'timestamp': datetime.now().isoformat(),
                        'status': 'detection',
                        'disease_name': treatment_info['disease'],
                        'confidence': confidence,
                        'treatment_en': treatment_info['treatment_en'],
                        'treatment_ar': treatment_info['treatment_ar'],
                        'resources': treatment_info['resources'],
                        'image': encoded_annotated_frame
                    }
                    await broadcast_message(detection_data)
                    logger.info(f"Sent detection data for: {treatment_info['disease']}")
                    return True
            else:
                logger.warning(f"Detected '{disease_name}' but no treatment info available.")

    except Exception as e:
        logger.error(f"Error during disease detection processing: {e}")

    return False


async def run_camera_and_stream() -> None:
    """Manages continuous camera acquisition, video streaming, and detection cooldown."""
    global picam2
    logger.info("Initializing Camera...")
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": CAMERA_RESOLUTION},
            lores={"size": LORES_RESOLUTION},
            encode="main",
            controls={"FrameRate": CAMERA_FRAMERATE}
        )
        picam2.configure(config)
        picam2.start()
        logger.info(f"Camera started with resolution {CAMERA_RESOLUTION} @ {CAMERA_FRAMERATE}fps.")
        await asyncio.sleep(CAMERA_WARMUP_SECONDS)
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        picam2 = None
        return

    last_detection_attempt_time = 0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            if not connected_clients:
                await asyncio.sleep(IDLE_CLIENT_CHECK_INTERVAL_SECONDS)
                frame_count = 0
                start_time = time.time()
                continue

            try:
                frame = picam2.capture_array("main")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Error capturing frame: {e}")
                await asyncio.sleep(CAPTURE_RETRY_BACKOFF_SECONDS)
                continue

            # Stream current RGB frame
            encoded_frame = encode_frame(frame_rgb, quality=STREAM_QUALITY)
            if encoded_frame:
                stream_data = {
                    'type': 'camera_frame',
                    'timestamp': datetime.now().isoformat(),
                    'image': encoded_frame
                }
                await broadcast_message(stream_data)
            else:
                logger.warning("Failed to encode frame for streaming.")

            # Run detection if cooldown elapsed
            current_time = time.time()
            if current_time - last_detection_attempt_time >= DETECTION_COOLDOWN:
                last_detection_attempt_time = current_time
                await process_frame_for_detection(frame_rgb)

            # Performance monitoring
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= FPS_REPORT_INTERVAL_SECONDS:
                fps = frame_count / elapsed_time
                logger.info(f"Streaming FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

            await asyncio.sleep(1.0 / CAMERA_FRAMERATE)

    except asyncio.CancelledError:
        logger.info("Camera streaming task cancelled.")
    except Exception as e:
        logger.error(f"Unexpected error in camera/stream loop: {e}", exc_info=True)
    finally:
        if picam2:
            logger.info("Stopping camera...")
            picam2.stop()
            logger.info("Camera stopped.")


async def main() -> None:
    """Bootstraps background services and runs the unified WebSocket server."""
    logger.info("--- Starting Unified WebSocket Server ---")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"WebSocket Host: {WEBSOCKET_HOST}")
    logger.info(f"WebSocket Port: {WEBSOCKET_PORT}")

    if not load_resources():
        logger.error("Failed to load critical resources. Server cannot start.")
        return

    server = await websockets.serve(
        client_handler,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT,
        ping_interval=None,
        ping_timeout=None
    )
    logger.info(f"WebSocket server listening on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")

    timeout_task = asyncio.create_task(check_client_timeouts())
    camera_task = asyncio.create_task(run_camera_and_stream())

    try:
        await asyncio.gather(timeout_task, camera_task)
    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
    finally:
        logger.info("Shutting down server...")
        if not timeout_task.done():
            timeout_task.cancel()
        if not camera_task.done():
            camera_task.cancel()
        await asyncio.gather(timeout_task, camera_task, return_exceptions=True)

        server.close()
        await server.wait_closed()
        logger.info("Server shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, initiating shutdown...")
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {e}", exc_info=True)