import cv2
import numpy as np
from tensorflow.keras.models import load_model
import asyncio
import websockets
import json
import base64
import time
import pandas as pd
from datetime import datetime              
from picamera2 import Picamera2 # Use Picamera2
import logging
import sys # For checking Python version

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'plant_disease_model_final.h5')
TREATMENT_FILE_PATH = os.path.join(BASE_DIR, 'data', 'plant_disease_data.xlsx')
IMG_SIZE = 224 # Should match the model's expected input size
WEBSOCKET_PORT = 8765
WEBSOCKET_HOST = "0.0.0.0" # Listen on all available network interfaces
DETECTION_THRESHOLD = 0.98 # Minimum confidence for detection
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE = 20 # Target framerate
STREAM_QUALITY = 70 # JPEG quality for streaming (0-100)
DETECTION_COOLDOWN = 2.0 # Seconds between detection attempts
PING_INTERVAL = 20.0 # Seconds for server to expect ping from client
PING_TIMEOUT = 35.0 # Seconds before disconnecting unresponsive client

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to stdout
    ]      
)
logger = logging.getLogger('UnifiedServer')

# --- Global Variables ---
connected_clients = {} # {websocket: {'last_ping': timestamp, 'type': 'client_type_string'}}
model = None
treatment_df = None
picam2 = None # PiCamera2 instance

# --- Class Names (Ensure this matches your model's output classes exactly) ---
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
logger.info(f"Defined {len(CLASS_NAMES)} class names.")

# --- Helper Functions ---

def load_resources():
    """Loads the Keras model and treatment data."""
    global model, treatment_df
    try:
        logger.info(f"📦 Loading disease detection model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        logger.info("✅ Model loaded successfully.")

        logger.info(f"📖 Loading treatment data from: {TREATMENT_FILE_PATH}")
        treatment_df = pd.read_excel(TREATMENT_FILE_PATH)
        # Basic validation of the excel file structure
        if treatment_df.shape[1] < 4:
             logger.error(f"❌ Treatment file seems malformed. Expected at least 4 columns, found {treatment_df.shape[1]}.")
             treatment_df = None # Prevent usage if malformed
             return False
        logger.info(f"✅ Treatment data loaded successfully ({treatment_df.shape[0]} rows).")
        logger.info(f"   Columns: {list(treatment_df.columns)}")
        return True

    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}. Cannot load resources.")
        return False
    except Exception as e:
        logger.error(f"❌ Error loading resources: {str(e)}")
        return False

def get_treatment_info(disease_name):
    """Fetches treatment details for a given disease name."""
    if treatment_df is None:
        logger.warning("Treatment data not loaded. Cannot fetch info.")
        return None

    try:
        # Assuming the first column is disease name, 2nd is Treat_EN, 3rd is Treat_AR, 4th is Resources
        disease_col = treatment_df.columns[0]
        treat_en_col = treatment_df.columns[1]
        treat_ar_col = treatment_df.columns[2]
        resources_col = treatment_df.columns[3]

        # Case-insensitive and whitespace-insensitive matching
        match = treatment_df[treatment_df[disease_col].str.strip().str.lower() == disease_name.strip().lower()]

        if not match.empty:
            row = match.iloc[0]
            return {
                'disease': row[disease_col].strip(), # Return consistent name
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

def encode_frame(frame, quality=STREAM_QUALITY):
    """Encodes numpy array (image) to base64 string."""
    try:
        is_success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not is_success:
            logger.error("❌ Failed to encode frame to JPEG.")
            return None
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"❌ Error during frame encoding: {e}")
        return None

def preprocess_frame_for_model(frame):
    """Resizes and normalizes frame for model prediction."""
    try:
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0 # Normalize
        img = np.expand_dims(img, axis=0) # Add batch dimension
        return img
    except Exception as e:
        logger.error(f"❌ Error during frame preprocessing: {e}")
        return None

# --- WebSocket Handling ---

async def register_client(websocket):
    """Registers a new client connection."""
    client_ip = websocket.remote_address
    logger.info(f"✅ New client connecting from: {client_ip}")
    try:
        # Wait for the client to identify itself
        message = await asyncio.wait_for(websocket.recv(), timeout=10.0) # 10 sec timeout
        data = json.loads(message)
        client_type = data.get('client_type', 'unknown_client')

        connected_clients[websocket] = {
            'last_ping': time.time(),
            'type': client_type,
            'address': client_ip
        }
        logger.info(f"✅ Client {client_ip} registered as type: '{client_type}'")

        # Send welcome message
        await websocket.send(json.dumps({
            'type': 'welcome',
            'message': f'Connected to Plant Disease Detection Server (Unified)',
            'timestamp': datetime.now().isoformat()
        }))
        return True

    except asyncio.TimeoutError:
         logger.warning(f"⚠️ Client {client_ip} did not identify itself in time. Closing connection.")
         await websocket.close(reason='Identification timeout')
         return False
    except websockets.exceptions.ConnectionClosed:
         logger.warning(f"⚠️ Connection closed by {client_ip} during registration.")
         return False
    except json.JSONDecodeError:
         logger.error(f"❌ Invalid identification message from {client_ip}. Closing connection.")
         await websocket.close(reason='Invalid identification message')
         return False
    except Exception as e:
         logger.error(f"❌ Error during client registration ({client_ip}): {e}")
         await websocket.close(reason='Registration error')
         return False


async def unregister_client(websocket):
    """Removes a client from the connected list."""
    client_info = connected_clients.pop(websocket, None)
    if client_info:
        logger.info(f"❌ Client disconnected: {client_info.get('address', 'Unknown IP')} (Type: {client_info.get('type', 'N/A')})")
    else:
        # This might happen if deregistered twice or connection closed before registration
        logger.info(f"❌ Client disconnected (already removed or registration failed): {websocket.remote_address}")


async def handle_client_messages(websocket):
    """Listens for messages (like ping) from a connected client."""
    client_ip = connected_clients.get(websocket, {}).get('address', websocket.remote_address)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get('type')

                if msg_type == 'ping':
                    if websocket in connected_clients:
                        connected_clients[websocket]['last_ping'] = time.time()
                        # logger.debug(f"Received ping from {client_ip}") # Too verbose for INFO
                        await websocket.send(json.dumps({
                            'type': 'pong',
                            'timestamp': datetime.now().isoformat()
                        }))
                    else:
                         logger.warning(f"⚠️ Received ping from unregistered client? {client_ip}")

                # Add handlers for other client message types if needed
                # elif msg_type == 'some_other_command':
                #    handle_other_command(data)

            except json.JSONDecodeError:
                logger.error(f"❌ Invalid JSON received from {client_ip}: {message[:100]}...") # Log first 100 chars
            except Exception as e:
                logger.error(f"❌ Error processing message from {client_ip}: {e}")

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_ip} closed connection gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_ip} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in message handler for {client_ip}: {e}")
    finally:
        await unregister_client(websocket)


async def client_handler(websocket, path="/"):
    """Main entry point for handling a new WebSocket connection."""
    if await register_client(websocket):
        await handle_client_messages(websocket)
    # unregister_client is called within handle_client_messages or register_client on failure


async def broadcast_message(message_data):
    """Sends a JSON message to all currently connected clients."""
    if not connected_clients:
        return # No clients to send to

    message_json = json.dumps(message_data)
    # Create a list of tasks for sending messages concurrently
    tasks = [client.send(message_json) for client in connected_clients.keys()]

    # Execute tasks and gather results (including potential exceptions)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results to find failed sends and log/remove clients
    disconnected_clients = []
    for client, result in zip(list(connected_clients.keys()), results):
        if isinstance(result, Exception):
            client_ip = connected_clients.get(client, {}).get('address', client.remote_address)
            logger.error(f"❌ Failed to send message to client {client_ip}: {result}")
            disconnected_clients.append(client)
            # Force close the connection on the server side
            try:
                 await client.close(reason="Send failed")
            except:
                 pass # Ignore errors during close, client might already be gone

    # Remove clients that failed to receive the message
    for client in disconnected_clients:
        await unregister_client(client) # Use the unregister function


async def check_client_timeouts():
    """Periodically checks for clients that haven't pinged recently."""
    logger.info("⏲️ Starting client timeout checker...")
    while True:
        try:
             await asyncio.sleep(PING_INTERVAL) # Check every PING_INTERVAL seconds
             current_time = time.time()
             timed_out_clients = []

             # Iterate safely over a copy of the keys
             for client, info in list(connected_clients.items()):
                 if current_time - info['last_ping'] > PING_TIMEOUT:
                     client_ip = info.get('address', client.remote_address)
                     logger.warning(f"⚠️ Client timed out: {client_ip} (Type: {info.get('type', 'N/A')}). Last ping: {info['last_ping']:.2f}")
                     timed_out_clients.append(client)
                     # Force close the connection
                     try:
                          await client.close(reason='Ping timeout')
                     except Exception as e:
                          logger.error(f"Error closing timed out client {client_ip}: {e}")


             # Remove timed out clients from the main dictionary
             for client in timed_out_clients:
                 await unregister_client(client) # Use the unregister function

        except asyncio.CancelledError:
             logger.info("Client timeout checker cancelled.")
             break
        except Exception as e:
             logger.error(f"❌ Error in client timeout checker: {e}")
             await asyncio.sleep(10) # Wait a bit before retrying after an error


async def process_frame_for_detection(frame):
    """Processes a single frame for disease detection."""
    if model is None:
        # logger.warning("Model not loaded, skipping detection.")
        return # Silently skip if model isn't ready

    try:
        processed_img = preprocess_frame_for_model(frame)
        if processed_img is None:
            return # Preprocessing failed

        # Make prediction
        predictions = model.predict(processed_img, verbose=0) # verbose=0 prevents Keras logs per prediction
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index]) # Ensure it's a standard float

        if confidence >= DETECTION_THRESHOLD:
            disease_name = CLASS_NAMES[class_index]
            
            # Skip if the detected class is a healthy plant
            if "healthy" in disease_name.lower():
                # logger.info(f"🌿 Healthy plant detected ({confidence*100:.1f}%), skipping notification")
                return False
                
            logger.info(f"🔍 Potential Detection: {disease_name} ({confidence*100:.1f}%)")

            # Fetch treatment info
            treatment_info = get_treatment_info(disease_name)

            if treatment_info:
                # Prepare annotated frame (optional, but good for debugging/confirmation)
                annotated_frame = frame.copy()
                text = f"{treatment_info['disease']} ({confidence*100:.1f}%)"
                cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)

                # Encode the annotated frame for the detection message
                encoded_annotated_frame = encode_frame(annotated_frame, quality=80) # Use slightly higher quality for detection frame

                if encoded_annotated_frame:
                    detection_data = {
                        'type': 'detection',
                        'timestamp': datetime.now().isoformat(),
                        'status': 'detection', # Consistent status field
                        'disease_name': treatment_info['disease'],
                        'confidence': confidence, # Send as float
                        'treatment_en': treatment_info['treatment_en'],
                        'treatment_ar': treatment_info['treatment_ar'],
                        'resources': treatment_info['resources'],
                        'image': encoded_annotated_frame # Base64 encoded annotated image
                    }
                    await broadcast_message(detection_data)
                    logger.info(f"✅ Sent detection data for: {treatment_info['disease']}")
                    return True # Indicate detection occurred
            else:
                # Disease detected but no treatment info found
                 logger.warning(f"Detected '{disease_name}' but no treatment info available.")
                 # Optionally send a message without treatment details if needed
                 # Or just log it as done here.
        # else: # Confidence below threshold
            # logger.debug(f"Confidence below threshold ({confidence*100:.1f}%) for class {CLASS_NAMES[class_index]}")

    except Exception as e:
        logger.error(f"❌ Error during disease detection processing: {e}")

    return False # Indicate no detection was sent


async def run_camera_and_stream():
    """Main loop for capturing frames, streaming, and detecting."""
    global picam2
    logger.info("📷 Initializing Camera...")
    try:
        picam2 = Picamera2()
        # Configure for preview (lower res faster processing) and still (higher res maybe?)
        # We'll use the preview configuration's main stream for processing
        config = picam2.create_preview_configuration(
            main={"size": CAMERA_RESOLUTION},
            lores={"size": (320, 240)}, # Optional low-res stream if needed later
            encode="main", # Use main stream for encoding output if saving video
            controls={"FrameRate": CAMERA_FRAMERATE} # Set framerate
        )
        picam2.configure(config)
        picam2.start()
        logger.info(f"✅ Camera started with resolution {CAMERA_RESOLUTION} @ {CAMERA_FRAMERATE}fps.")
        await asyncio.sleep(2) # Allow camera to warm up
    except Exception as e:
        logger.error(f"❌ Failed to initialize camera: {e}")
        picam2 = None # Ensure picam2 is None if initialization failed
        return # Cannot proceed without camera

    last_detection_attempt_time = 0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            if not connected_clients:
                # No clients connected, pause camera processing to save resources
                # logger.info("No clients connected, pausing camera stream...")
                await asyncio.sleep(1.0) # Check for clients every second
                # Reset frame rate calculation when resuming
                frame_count = 0
                start_time = time.time()
                continue # Skip the rest of the loop

            # Capture frame
            try:
                frame = picam2.capture_array("main") # Capture from main stream
                # Picamera2 captures in BGR order by default with capture_array,
                # but TensorFlow models usually expect RGB. Convert if necessary.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                 logger.error(f"❌ Error capturing frame: {e}")
                 await asyncio.sleep(0.5) # Wait before retrying capture
                 continue


            # 1. Stream the current frame (RGB)
            encoded_frame = encode_frame(frame_rgb, quality=STREAM_QUALITY)
            if encoded_frame:
                stream_data = {
                    'type': 'camera_frame',
                    'timestamp': datetime.now().isoformat(),
                    'image': encoded_frame # Base64 encoded RGB frame
                }
                await broadcast_message(stream_data)
            else:
                logger.warning("⚠️ Failed to encode frame for streaming.")


            # 2. Check for disease detection (with cooldown)
            current_time = time.time()
            if current_time - last_detection_attempt_time >= DETECTION_COOLDOWN:
                last_detection_attempt_time = current_time
                # Run detection in a separate task to avoid blocking the stream loop?
                # For simplicity now, run it directly. If it's slow, consider asyncio.create_task
                detection_occured = await process_frame_for_detection(frame_rgb) # Use RGB frame for detection
                # Optional: Can add a 'no_detection' message broadcast here if needed


            # Calculate and log FPS occasionally
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 10.0: # Log FPS every 10 seconds
                fps = frame_count / elapsed_time
                logger.info(f"Streaming FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()


            # Control loop speed - aim for target framerate
            # This simple sleep might not be precise, more advanced timing could be used.
            await asyncio.sleep(1.0 / CAMERA_FRAMERATE)


    except asyncio.CancelledError:
        logger.info("Camera streaming task cancelled.")
    except Exception as e:
        logger.error(f"❌ Unexpected error in camera/stream loop: {e}", exc_info=True) # Log traceback
    finally:
        if picam2:
            logger.info("Stopping camera...")
            picam2.stop()
            logger.info("Camera stopped.")


async def main():
    """Initializes resources and starts server tasks."""
    logger.info("--- Starting Unified WebSocket Server ---")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"WebSocket Host: {WEBSOCKET_HOST}")
    logger.info(f"WebSocket Port: {WEBSOCKET_PORT}")

    # Load model and treatment data first
    if not load_resources():
        logger.error("❌ Failed to load critical resources. Server cannot start.")
        return # Exit if resources failed to load

    # Start the WebSocket server
    server = await websockets.serve(
        client_handler,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT,
        ping_interval=None, # Disable automatic pings from server (we handle manually)
        ping_timeout=None   # Disable automatic timeouts from server (we handle manually)
    )
    logger.info(f"✅ WebSocket server listening on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")

    # Start background tasks
    timeout_task = asyncio.create_task(check_client_timeouts())
    camera_task = asyncio.create_task(run_camera_and_stream())

    # Keep the main task running until tasks are done or interrupted
    try:
        await asyncio.gather(timeout_task, camera_task)
    except asyncio.CancelledError:
         logger.info("Main task cancelled.")
    finally:
        logger.info("Shutting down server...")
        # Cancel background tasks explicitly
        if not timeout_task.done(): timeout_task.cancel()
        if not camera_task.done(): camera_task.cancel()
        # Wait for tasks to finish cancelling
        await asyncio.gather(timeout_task, camera_task, return_exceptions=True)

        server.close()
        await server.wait_closed()
        logger.info("✅ Server shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Received keyboard interrupt, initiating shutdown...")
    except Exception as e:
        logger.critical(f"💥 Unhandled exception in main execution: {e}", exc_info=True)