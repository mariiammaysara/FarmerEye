import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import pandas as pd
import websockets
import asyncio
import json
import base64
from datetime import datetime
from picamera2 import Picamera2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DiseaseDetector')

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'plant_disease_model_final.h5')
TREATMENT_FILE_PATH = os.path.join(BASE_DIR, 'data', 'plant_disease_data.xlsx')
IMG_SIZE = 224
WEBSOCKET_PORT = 8765
NO_DETECTION_INTERVAL = 2.0  # Send no detection message every 2 seconds
PING_TIMEOUT = 35  # Timeout for ping in seconds
connected_clients = {}  # Changed to dict to store last ping time
last_detection_time = 0
last_no_detection_message_time = 0

CLASS_NAMES = [
    'Aphids_cotton', 'Army worm_cotton', 'Bacterial blight_cotton', 'Healthy_cotton',
    'Pepper_bell_bacterial_spot', 'Pepper_bellhealthy', 'Potato__Early_blight',
    'Potato_Late_blight', 'Potato_healthy', 'Powdery mildew_cotton',
    'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Target spot_cotton',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus',
    'Tomato___healthy', 'cotton_curl_virus', 'cotton_fussarium_wilt'
]

# === LOAD MODEL ===
logger.info("📦 Loading model...")
model = load_model(MODEL_PATH)
logger.info("✅ Model loaded successfully.")

async def handle_client(websocket):
    """Handle a connection and dispatch it to the shared server."""
    client_info = f"Client connected from {websocket.remote_address}"
    logger.info(f"✅ New client connected: {client_info}")
    connected_clients[websocket] = time.time()
    
    try:
        # Send welcome message
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
                logger.error(f"❌ Invalid message received from {client_info}")
            except Exception as e:
                logger.error(f"❌ Error processing message from {client_info}: {str(e)}")
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"❌ Connection closed for {client_info}")
    except Exception as e:
        logger.error(f"❌ Unexpected error for {client_info}: {str(e)}")
    finally:
        logger.info(f"❌ Client disconnected: {client_info}")
        connected_clients.pop(websocket, None)

async def check_client_timeouts():
    while True:
        current_time = time.time()
        disconnected_clients = []
        
        for client, last_ping in connected_clients.items():
            if current_time - last_ping > PING_TIMEOUT:
                disconnected_clients.append(client)
        
        for client in disconnected_clients:
            logger.warning(f"⚠️ Client timed out: {client.remote_address}")
            await client.close()
            connected_clients.pop(client, None)
        
        await asyncio.sleep(5)  # Check every 5 seconds

async def broadcast_detection(detection_data):
    if not connected_clients:
        logger.info("⚠️ No connected clients to broadcast to")
        return
        
    logger.info(f"📤 Broadcasting to {len(connected_clients)} clients")
    message = json.dumps(detection_data)
    disconnected = []
    
    for client in connected_clients:
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.error(f"❌ Failed to send to client {client.remote_address}")
            disconnected.append(client)
        except Exception as e:
            logger.error(f"❌ Error sending to client {client.remote_address}: {str(e)}")
            disconnected.append(client)
    
    # Remove disconnected clients
    for client in disconnected:
        connected_clients.pop(client, None)
        
    if len(disconnected) == 0:
        logger.info("✅ Broadcast successful")

def encode_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def detect_plants(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
    return boxes

def get_treatment_info(disease_name):
    try:
        treatment_df = pd.read_excel(TREATMENT_FILE_PATH)
        disease_column_name = treatment_df.columns[0]
        treatment_info = treatment_df[treatment_df[disease_column_name].str.strip().str.lower() == disease_name.strip().lower()]
        
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

async def send_no_detection():
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

async def process_frame(frame, boxes):
    global last_detection_time
    current_time = time.time()
    detection_made = False

    if boxes:
        max_confidence = 0
        best_prediction = None
        best_box = None

        for (x, y, w, h) in boxes:
            plant_img = frame[y:y+h, x:x+w]
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

            if confidence >= 0.7:
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

    # Send no detection message if no detection was made for a while
    global last_no_detection_message_time
    if not detection_made and (current_time - last_no_detection_message_time) >= NO_DETECTION_INTERVAL:
        await send_no_detection()
        last_no_detection_message_time = current_time

    return frame

async def main():
    try:
        # Create SSL context for secure WebSocket (wss://)
        server = await websockets.serve(
            handle_client,
            "0.0.0.0",  # Listen on all network interfaces
            WEBSOCKET_PORT,
            ping_interval=20,  # Send ping every 20 seconds
            ping_timeout=30,   # Wait 30 seconds for pong response
            max_size=10 * 1024 * 1024,  # 10MB max message size
            compression=None,  # Disable compression for better compatibility
            max_queue=32  # Limit message queue size
        )
        
        logger.info(f"✅ WebSocket server running on ws://10.220.90.215:{WEBSOCKET_PORT}")
        logger.info("💡 Connect to this server from your Flutter app using:")
        logger.info(f"   ws://10.220.90.215:{WEBSOCKET_PORT}")

        # Start the timeout checker
        timeout_checker = asyncio.create_task(check_client_timeouts())

        logger.info("🎥 Real-time detection started. Press 'q' to quit.")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        await asyncio.sleep(1)  # Use asyncio.sleep instead of time.sleep

        try:
            while True:
                if not connected_clients:
                    await asyncio.sleep(0.1)  # Don't process if no clients
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

                # Allow other tasks to run
                await asyncio.sleep(0)
        finally:
            timeout_checker.cancel()
            picam2.close()
            cv2.destroyAllWindows()
            
        await server.wait_closed()

    except Exception as e:
        logger.error(f"❌ Server error: {str(e)}")
        raise
    finally:
        logger.info("🛑 Application exited cleanly.")

if _name_ == "_main_":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        raise