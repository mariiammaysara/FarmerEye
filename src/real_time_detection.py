"""Real-time plant disease detection and streaming service for Farmer Eye.

This module captures camera frames, segments green plant foliage using HSV color
thresholding, crops potential plant regions, runs inference through the trained
convolutional neural network, looks up localized treatments, and broadcasts
structured alerts and heartbeat telemetry over a WebSocket connection.

Usage:
    Run on Raspberry Pi with local display or headless:
        python src/real_time_detection.py

Network Configuration:
    Port: 8765 (configurable via WEBSOCKET_PORT environment variable)
    Host: 0.0.0.0 (configurable via WEBSOCKET_HOST environment variable)

Pipeline Context:
    Picamera2 -> detect_plants() [HSV Mask] -> CNN Inference [224x224, /255]
    -> get_treatment_info() -> broadcast_detection() [WebSockets]
"""
import os
import cv2
import numpy as np
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None
import time
import pandas as pd
import websockets
import asyncio
import json
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DiseaseDetector')

# Configuration and filesystem paths
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH: str = os.path.join(BASE_DIR, 'models', 'plant_disease_model_final.h5')
TREATMENT_FILE_PATH: str = os.path.join(BASE_DIR, 'data', 'plant_disease_data.xlsx')

# Model input dimensions (must match training resolution: 224x224x3)
IMG_SIZE: int = 224

# Network connection settings
WEBSOCKET_HOST: str = os.environ.get('WEBSOCKET_HOST', '0.0.0.0')
WEBSOCKET_PORT: int = int(os.environ.get('WEBSOCKET_PORT', 8765))
NO_DETECTION_INTERVAL: float = 2.0  # Seconds between status heartbeat broadcasts
PING_TIMEOUT: int = 35  # Client timeout threshold in seconds

# Plant segmentation and detection thresholds
CONFIDENCE_THRESHOLD: float = 0.7  # Minimum softmax probability required to trigger treatment alert
MIN_PLANT_CONTOUR_AREA: float = 1000.0  # Pixel area cutoff to filter out minor green noise
HSV_LOWER_GREEN: np.ndarray = np.array([35, 40, 40])  # Lower bound for foliage green hue
HSV_UPPER_GREEN: np.ndarray = np.array([85, 255, 255])  # Upper bound for foliage green hue

# Camera acquisition parameters
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
CAMERA_WARMUP_SECONDS: float = 1.0
TIMEOUT_CHECK_INTERVAL_SECONDS: float = 5.0
MAX_WEBSOCKET_MESSAGE_BYTES: int = 10 * 1024 * 1024  # 10 MB payload limit for base64 imagery
WEBSOCKET_PING_INTERVAL: int = 20
WEBSOCKET_PING_TIMEOUT: int = 30
WEBSOCKET_MAX_QUEUE: int = 32

connected_clients: Dict[Any, float] = {}  # Stores client websocket instance mapped to last ping timestamp
last_detection_time: float = 0
last_no_detection_message_time: float = 0

try:
    from src.class_names import CLASS_NAMES, normalize_disease_name
except ImportError:
    from class_names import CLASS_NAMES, normalize_disease_name

# Load disease classification model
logger.info("Loading model...")
try:
    if load_model is not None and os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully.")
    else:
        model = None
        if load_model is None:
            logger.warning("TensorFlow is not installed. Model not loaded.")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None


async def handle_client(websocket) -> None:
    """Handles an active WebSocket client session and dispatches ping/pong messages.

    Args:
        websocket: The connected WebSocket client instance.
    """
    client_info = f"Client connected from {websocket.remote_address}"
    logger.info(f"New client connected: {client_info}")
    connected_clients[websocket] = time.time()
    
    try:
        # Initial greeting to confirm bidirectional handshake
        await websocket.send(json.dumps({
            'type': 'welcome',
            'message': 'Connected to Disease Detection Server',
            'timestamp': datetime.now().isoformat()
        }))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get('type') == 'ping':
                    connected_clients[websocket] = time.time()
                    await websocket.send(json.dumps({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }))
                elif data.get('type') == 'hello':
                    logger.info(f"Received hello from client: {data.get('client')}")
                    await websocket.send(json.dumps({
                        'type': 'hello_response',
                        'message': 'Hello received',
                        'timestamp': datetime.now().isoformat()
                    }))
            except json.JSONDecodeError:
                logger.error(f"Invalid message received from {client_info}")
            except Exception as e:
                logger.error(f"Error processing message from {client_info}: {str(e)}")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Connection closed for {client_info}")
    except Exception as e:
        logger.error(f"Unexpected error for {client_info}: {str(e)}")
    finally:
        logger.info(f"Client disconnected: {client_info}")
        connected_clients.pop(websocket, None)


async def check_client_timeouts() -> None:
    """Monitors connected clients and disconnects un-responsive sessions."""
    while True:
        current_time = time.time()
        disconnected_clients = []
        
        for client, last_ping in connected_clients.items():
            if current_time - last_ping > PING_TIMEOUT:
                disconnected_clients.append(client)
        
        for client in disconnected_clients:
            logger.warning(f"Client timed out: {client.remote_address}")
            await client.close()
            connected_clients.pop(client, None)
        
        await asyncio.sleep(TIMEOUT_CHECK_INTERVAL_SECONDS)


async def broadcast_detection(detection_data: Dict[str, Any]) -> None:
    """Broadcasts a disease detection payload to all active WebSocket clients.

    Args:
        detection_data: Structured dictionary containing disease name, confidence,
            bilingual treatments, and annotated image data.
    """
    if not connected_clients:
        logger.info("No connected clients to broadcast to")
        return
        
    logger.info(f"Broadcasting to {len(connected_clients)} clients")
    message = json.dumps(detection_data)
    disconnected = []
    
    for client in connected_clients:
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.error(f"Failed to send to client {client.remote_address}")
            disconnected.append(client)
        except Exception as e:
            logger.error(f"Error sending to client {client.remote_address}: {str(e)}")
            disconnected.append(client)
    
    for client in disconnected:
        connected_clients.pop(client, None)
        
    if len(disconnected) == 0:
        logger.info("Broadcast successful")


def encode_image(frame: np.ndarray) -> str:
    """Encodes an OpenCV image array to a JPEG base64 string.

    Args:
        frame: NumPy ndarray containing image pixels.

    Returns:
        Base64-encoded UTF-8 string of the JPEG-compressed image.
    """
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def detect_plants(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Segments potential plant leaves in a frame using HSV green thresholding.

    Transforms image to HSV color space, isolates green vegetation, filters noise
    via morphological opening and closing, and extracts bounding rectangles.

    Args:
        frame: BGR image from the camera.

    Returns:
        List of (x, y, w, h) bounding boxes corresponding to segmented plant areas.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_PLANT_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
    return boxes


def get_treatment_info(disease_name: str) -> Optional[Dict[str, str]]:
    """Retrieves bilingual treatment guidance from the Excel reference workbook.

    Matches the predicted disease label against the database using tolerant
    normalization to handle whitespace or underscore inconsistencies.

    Args:
        disease_name: Predicted plant condition label.

    Returns:
        Dictionary containing disease name, English advice, Arabic advice,
        and reference URL, or None if no match is found or an error occurs.
    """
    try:
        treatment_df = pd.read_excel(TREATMENT_FILE_PATH)
        disease_column_name = treatment_df.columns[0]
        norm_target = normalize_disease_name(disease_name)
        treatment_info = treatment_df[treatment_df[disease_column_name].apply(normalize_disease_name) == norm_target]
        
        if not treatment_info.empty:
            return {
                'disease': disease_name,
                'treatment_en': treatment_info.iloc[0, 1],
                'treatment_ar': treatment_info.iloc[0, 2],
                'resources': treatment_info.iloc[0, 3]
            }
        return None
    except Exception as e:
        logger.error(f"Error reading treatment file: {str(e)}")
        return None


async def send_no_detection() -> None:
    """Broadcasts a periodic heartbeat message indicating no disease detected."""
    if connected_clients:
        no_detection_data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'no_detection',
            'message': 'No plants or diseases detected'
        }
        message = json.dumps(no_detection_data)
        await asyncio.gather(
            *[client.send(message) for client in connected_clients]
        )


async def process_frame(frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Processes candidate plant regions, executes inference, and triggers alerts.

    Args:
        frame: Full BGR camera frame.
        boxes: List of candidate plant bounding boxes (x, y, w, h).

    Returns:
        Annotated BGR image frame with bounding boxes and classification labels.
    """
    global last_detection_time
    current_time = time.time()
    detection_made = False

    if boxes:
        max_confidence = 0
        best_prediction = None
        best_box = None

        for (x, y, w, h) in boxes:
            plant_img = frame[y:y+h, x:x+w]
            # Preprocessing matches training: resize to 224x224 and scale to [0, 1]
            img = cv2.resize(plant_img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            predictions = model.predict(img)
            class_index = np.argmax(predictions)
            confidence = predictions[0][class_index]

            if confidence > max_confidence:
                max_confidence = confidence
                best_prediction = (CLASS_NAMES[class_index], confidence)
                best_box = (x, y, w, h)

        if best_prediction and best_box:
            x, y, w, h = best_box
            disease_name, confidence = best_prediction
            label = f"{disease_name} ({confidence*100:.1f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if confidence >= CONFIDENCE_THRESHOLD:
                detection_made = True
                last_detection_time = current_time
                treatment_info = get_treatment_info(disease_name)
                if treatment_info:
                    detection_data = {
                        'timestamp': datetime.now().isoformat(),
                        'status': 'detection',
                        'disease_name': treatment_info['disease'],
                        'confidence': float(confidence),
                        'treatment_en': treatment_info['treatment_en'],
                        'treatment_ar': treatment_info['treatment_ar'],
                        'resources': treatment_info['resources'],
                        'image': encode_image(frame[y:y+h, x:x+w])
                    }
                    await broadcast_detection(detection_data)

                    treatment_text = f"Treatment (EN): {treatment_info['treatment_en'][:50]}..."
                    cv2.putText(frame, treatment_text, (10, frame.shape[0]-60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Send heartbeat if no detection occurred within interval
    global last_no_detection_message_time
    if not detection_made and (current_time - last_no_detection_message_time) >= NO_DETECTION_INTERVAL:
        await send_no_detection()
        last_no_detection_message_time = current_time

    return frame


async def main() -> None:
    """Initializes the WebSocket server, camera capture, and real-time processing loop.

    Raises:
        Exception: Any critical network or hardware failure encountered during runtime.
    """
    try:
        server = await websockets.serve(
            handle_client,
            WEBSOCKET_HOST,
            WEBSOCKET_PORT,
            ping_interval=WEBSOCKET_PING_INTERVAL,
            ping_timeout=WEBSOCKET_PING_TIMEOUT,
            max_size=MAX_WEBSOCKET_MESSAGE_BYTES,
            compression=None,
            max_queue=WEBSOCKET_MAX_QUEUE
        )
        
        logger.info(f"WebSocket server running on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        logger.info("Connect to this server from your Flutter app using:")
        logger.info(f"   ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")

        timeout_checker = asyncio.create_task(check_client_timeouts())

        logger.info("Real-time detection started. Press 'q' to quit.")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (CAMERA_WIDTH, CAMERA_HEIGHT)})
        picam2.configure(config)
        picam2.start()
        await asyncio.sleep(CAMERA_WARMUP_SECONDS)

        try:
            while True:
                if not connected_clients:
                    await asyncio.sleep(0.1)
                    continue

                start_time = time.time()
                frame = picam2.capture_array()
                boxes = detect_plants(frame)
                frame = await process_frame(frame, boxes)

                fps = 1.0 / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow("Plant Disease Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                await asyncio.sleep(0)
        finally:
            timeout_checker.cancel()
            picam2.close()
            cv2.destroyAllWindows()
            
        await server.wait_closed()

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise
    finally:
        logger.info("Application exited cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise