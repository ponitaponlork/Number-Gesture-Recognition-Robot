# file: run.py
"""
AI-Controlled Robot with Live Web Interface
Detects numbers 0-5 using a trained model and controls robot movements accordingly.
"""

import cv2
import numpy as np
import threading
import time
import os
# import serial.tools.list_ports
from flask import Flask, Response, render_template_string

# MediaPipe for hand landmark detection
try:
    import mediapipe as mp
except ImportError:
    mp = None
    print("WARNING: mediapipe not installed. Install with: pip install mediapipe")

# TensorFlow Lite compatibility - try full tensorflow first, fallback to tflite-runtime
try:
    import tensorflow as tf
except ImportError:
    # Use tflite-runtime for Raspberry Pi
    import tflite_runtime.interpreter as tflite_interpreter
    # Create compatibility shim
    class tf:
        class lite:
            Interpreter = tflite_interpreter.Interpreter
        # Note: keras models won't work with tflite-runtime
        keras = None

# ONNX Runtime for ONNX models
try:
    import onnxruntime as ort
except ImportError:
    ort = None

from auppbot import AUPPBot

# ============================================================================
# CONFIGURATION - Modify these values as needed
# ============================================================================

# Model Configuration
MODEL_PATH = "gesture_model.onnx"  # Path to your trained model
MODEL_TYPE = "onnx"  # Options: "onnx", "tflite", "h5", "keras"
INPUT_SIZE = (224, 224)  # Model input size (width, height)

# Camera Configuration
CAMERA_INDEX = 0  # Default camera (0 for built-in, 1+ for external)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Performance Configuration
INFERENCE_SKIP_FRAMES = 2  # Skip N frames between inferences (0 = no skip, 2 = every 3rd frame)

# Robot Configuration
ROBOT_PORT = "/dev/ttyUSB0"   # Serial port for robot
ROBOT_BAUD = 115200

# Motor Speed Settings (0-99)
SPEED_SLOW = 15
SPEED_MEDIUM = 20
SPEED_FAST = 30
TURN_SPEED = 20

# Action Duration Settings - DISABLED FOR CONTINUOUS CONTROL
# ACTION_DURATION = 4.0  # Duration in seconds for each action (3-5 seconds range)
# COOLDOWN_AFTER_ACTION = 1.0  # Cooldown period after action completes before detecting next gesture

# Web Server Configuration
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000

# ============================================================================
# ACTION MAPPING
# ============================================================================

ACTION_NAMES = {
    0: "Stop / Idle",
    1: "Move Forward (Slow)",
    2: "Move Forward (Medium)",
    3: "Move Forward (Fast)",
    4: "Turn Left",
    5: "Turn Right"
}

# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """Handles loading and inference for different model formats"""
    
    def __init__(self, model_path, model_type="tflite"):
        self.model_path = model_path
        self.model_type = model_type.lower()
        self.model = None
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.onnx_session = None
        self.load_model()
    
    def load_model(self):
        """Load the model based on its type"""
        try:
            if self.model_type == "onnx":
                self._load_onnx()
            elif self.model_type == "tflite":
                self._load_tflite()
            elif self.model_type in ["h5", "keras"]:
                self._load_keras()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            print(f"✓ Model loaded successfully: {self.model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def _load_onnx(self):
        """Load ONNX model"""
        if ort is None:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        
        self.onnx_session = ort.InferenceSession(self.model_path)
        
        # Get input/output details
        self.input_details = self.onnx_session.get_inputs()[0]
        self.output_details = self.onnx_session.get_outputs()[0]
        
        # Print model input/output details for debugging
        print(f"  Model input name: {self.input_details.name}")
        print(f"  Model input shape: {self.input_details.shape}")
        print(f"  Model input dtype: {self.input_details.type}")
        print(f"  Model output name: {self.output_details.name}")
        print(f"  Model output shape: {self.output_details.shape}")
    
    def _load_tflite(self):
        """Load TensorFlow Lite model"""
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Print model input/output details for debugging
        print(f"  Model input shape: {self.input_details[0]['shape']}")
        print(f"  Model input dtype: {self.input_details[0]['dtype']}")
        print(f"  Model output shape: {self.output_details[0]['shape']}")
    
    def _load_keras(self):
        """Load Keras/H5 model"""
        self.model = tf.keras.models.load_model(self.model_path)
    
    def predict(self, image):
        """
        Run inference on preprocessed image
        Args:
            image: Preprocessed numpy array ready for model input
        Returns:
            Predicted class (0-5)
        """
        try:
            if self.model_type == "onnx":
                return self._predict_onnx(image)
            elif self.model_type == "tflite":
                return self._predict_tflite(image)
            else:
                return self._predict_keras(image)
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0  # Default to stop on error
    
    def _predict_tflite(self, image):
        """TFLite inference"""
        # Get expected input shape from model
        expected_shape = self.input_details[0]['shape']
        
        # Reshape to match expected input
        if len(expected_shape) == 2:
            # Flattened input [1, N]
            if len(image.shape) == 1:
                image = np.expand_dims(image, axis=0)
            else:
                image = image.reshape(1, -1)
        elif len(expected_shape) == 4:
            # Image input [1, H, W, C]
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
        
        # Convert to correct dtype
        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.uint8:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = np.argmax(output[0])
        return int(predicted_class)
    
    def _predict_onnx(self, image):
        """ONNX model inference"""
        # Get expected input shape
        expected_shape = self.input_details.shape
        
        # Handle dynamic dimensions (None or -1)
        if expected_shape[0] in [None, -1, 'batch_size']:
            expected_shape = [1] + list(expected_shape[1:])
        
        # Reshape based on expected input
        if len(expected_shape) == 2:
            # Flattened input [1, N]
            if len(image.shape) == 1:
                image = np.expand_dims(image, axis=0)
            else:
                image = image.reshape(1, -1)
        elif len(expected_shape) == 4:
            # Image input [1, H, W, C]
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
        
        # Ensure float32 dtype
        image = image.astype(np.float32)
        
        # Run inference
        input_name = self.input_details.name
        output_name = self.output_details.name
        outputs = self.onnx_session.run([output_name], {input_name: image})
        
        # Debug: Print prediction probabilities occasionally (reduced frequency for better FPS)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        if self._debug_counter % 300 == 0:  # Every 300 frames (10x less frequent)
            print(f"Debug - Predictions: {outputs[0][0]}")
            print(f"Debug - Input range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Get predicted class
        predicted_class = np.argmax(outputs[0][0])
        return int(predicted_class)
    
    def _predict_keras(self, image):
        """Keras model inference"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        return int(predicted_class)

# ============================================================================
# ROBOT CONTROLLER
# ============================================================================

class RobotController:
    """Handles robot actions based on detected numbers"""
    
    def __init__(self, robot):
        self.robot = robot
        self.current_action = 0
        self.last_executed_action = 0
        self.lock = threading.Lock()
    
    def execute_action(self, action_number):
        """
        Execute robot action based on detected number - IMMEDIATE response mode
        Args:
            action_number: Integer 0-5 representing the action
        """
        with self.lock:
            # Only update if action changed
            if self.current_action == action_number:
                return
            
            # Update current action
            self.current_action = action_number
            
            # Execute the action immediately
            if action_number == 0:
                self._action_stop()
            elif action_number == 1:
                self._action_forward_slow()
            elif action_number == 2:
                self._action_forward_medium()
            elif action_number == 3:
                self._action_forward_fast()
            elif action_number == 4:
                self._action_turn_left()
            elif action_number == 5:
                self._action_turn_right()
            else:
                print(f"Unknown action: {action_number}")
                self._action_stop()
            
            # Log the action change
            if action_number != 0:
                print(f"▶ Action {action_number}: {ACTION_NAMES.get(action_number, 'Unknown')}")
                self.last_executed_action = action_number
    
    def get_status(self):
        """Get current execution status"""
        with self.lock:
            return {
                'is_executing': self.current_action != 0,
                'current_action': self.current_action,
                'last_executed_action': self.last_executed_action,
                'in_cooldown': False
            }
    
    # -------- Action Methods (Modify these to change robot behavior) --------
    
    def _action_stop(self):
        """Action 0: Stop all motors"""
        self.robot.stop_all()
    
    def _action_forward_slow(self):
        """Action 1: Move forward at slow speed"""
        self.robot.motor1.forward(SPEED_SLOW)
        self.robot.motor2.forward(SPEED_SLOW)
        self.robot.motor3.forward(SPEED_SLOW)
        self.robot.motor4.forward(SPEED_SLOW)
    
    def _action_forward_medium(self):
        """Action 2: Move forward at medium speed"""
        self.robot.motor1.forward(SPEED_MEDIUM)
        self.robot.motor2.forward(SPEED_MEDIUM)
        self.robot.motor3.forward(SPEED_MEDIUM)
        self.robot.motor4.forward(SPEED_MEDIUM)
    
    def _action_forward_fast(self):
        """Action 3: Move forward at fast speed"""
        self.robot.motor1.forward(SPEED_FAST)
        self.robot.motor2.forward(SPEED_FAST)
        self.robot.motor3.forward(SPEED_FAST)
        self.robot.motor4.forward(SPEED_FAST)
    
    def _action_turn_left(self):
        """Action 4: Turn left (left motors backward, right motors forward)"""
        self.robot.motor1.speed(-TURN_SPEED)  # motors 1,2 backward (left side)
        self.robot.motor2.speed(-TURN_SPEED)
        self.robot.motor3.speed(TURN_SPEED)   # motors 3,4 forward (right side)
        self.robot.motor4.speed(TURN_SPEED)
    
    def _action_turn_right(self):
        """Action 5: Turn right (left motors forward, right motors backward)"""
        self.robot.motor1.speed(TURN_SPEED)    # motors 1,2 forward (left side)
        self.robot.motor2.speed(TURN_SPEED)
        self.robot.motor3.speed(-TURN_SPEED)   # motors 3,4 backward (right side)
        self.robot.motor4.speed(-TURN_SPEED)

# ============================================================================
# VIDEO PROCESSOR
# ============================================================================

class VideoProcessor:
    """Handles camera capture, model inference, and frame processing"""
    
    def __init__(self, model_loader, robot_controller):
        self.model = model_loader
        self.controller = robot_controller
        self.camera = None
        self.current_frame = None
        self.detected_number = 0
        self.running = False
        self.lock = threading.Lock()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Cache for MediaPipe results to avoid duplicate processing
        self.cached_hand_landmarks = None
        self.cached_hand_detected = False
        
        # Frame skipping for inference optimization
        self.inference_frame_counter = 0
        self.last_detected_number = 0
        
        # Initialize MediaPipe Hands
        if mp:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
        else:
            self.hands = None
            print("WARNING: MediaPipe not available, hand detection will not work")
    
    def start(self):
        """Start camera capture"""
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Reduce buffer size for lower latency (better real-time performance)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Enable hardware acceleration if available
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
        
        print(f"✓ Camera opened successfully (Index: {CAMERA_INDEX})")
        self.running = True
        
        # Start processing thread
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.camera:
            self.camera.release()
    
    def _process_loop(self):
        """Main processing loop - runs in separate thread"""
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Rotate frame 180 degrees (fix upside-down camera)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Frame skipping logic for inference optimization
            should_run_inference = (self.inference_frame_counter % (INFERENCE_SKIP_FRAMES + 1)) == 0
            self.inference_frame_counter += 1
            
            if should_run_inference:
                # Preprocess frame for model (extract hand landmarks)
                preprocessed = self._preprocess_frame(frame)
                
                # Only run inference if hand is detected
                if preprocessed is not None:
                    # Run inference
                    detected_num = self.model.predict(preprocessed)
                    self.last_detected_number = detected_num
                    
                    # Execute robot action immediately
                    self.controller.execute_action(detected_num)
                else:
                    # No hand detected - stop robot
                    detected_num = 0
                    self.last_detected_number = 0
                    self.controller.execute_action(0)
            else:
                # Skipped frame - still process hand detection but use last prediction
                preprocessed = self._preprocess_frame(frame)
                detected_num = self.last_detected_number
            
            # Annotate frame for display (includes hand landmarks visualization)
            annotated_frame = self._annotate_frame(frame, detected_num, preprocessed is not None)
            
            # Update shared state
            with self.lock:
                self.current_frame = annotated_frame
                self.detected_number = detected_num
                self.frame_count += 1
                
                # Calculate FPS every 30 frames
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = 30 / elapsed if elapsed > 0 else 0
                    self.start_time = time.time()
    
    def _preprocess_frame(self, frame):
        """
        Preprocess frame for model input using MediaPipe hand landmarks
        Returns: 63-element array of hand landmarks (x, y, z for 21 points)
                 or None if no hand detected
        """
        if not self.hands:
            return None
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        result = self.hands.process(rgb_frame)
        
        # Cache the result to avoid reprocessing in _annotate_frame
        self.cached_hand_landmarks = result.multi_hand_landmarks
        self.cached_hand_detected = result.multi_hand_landmarks is not None
        
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            
            # Extract landmarks as input for model (21 landmarks × 3 coordinates = 63 values)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def _annotate_frame(self, frame, detected_num, hand_detected):
        """Add visual annotations to frame"""
        h, w = frame.shape[:2]
        
        # Draw hand landmarks using cached results (avoid reprocessing)
        if hand_detected and self.hands and self.cached_hand_landmarks:
            self.mp_draw.draw_landmarks(
                frame, 
                self.cached_hand_landmarks[0], 
                self.mp_hands.HAND_CONNECTIONS
            )
        
        # Get robot status
        robot_status = self.controller.get_status()
        
        # Simplified overlay - smaller and faster
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
        
        # Display detected number
        if hand_detected:
            text = f"Detected: {detected_num}"
            color = (0, 255, 0)
        else:
            text = "No hand"
            color = (0, 165, 255)
        
        cv2.putText(frame, text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display status
        if robot_status['is_executing']:
            status_text = f"EXEC: {ACTION_NAMES.get(robot_status['current_action'], '?')}"
            status_color = (0, 255, 255)
        elif robot_status['in_cooldown']:
            status_text = "COOLDOWN"
            status_color = (0, 165, 255)
        else:
            status_text = "READY"
            status_color = (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Display FPS in corner
        fps_text = f"FPS: {self.fps:.0f}"
        cv2.putText(frame, fps_text, (w - 90, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def get_current_frame(self):
        """Get the current annotated frame (thread-safe)"""
        with self.lock:
            # Return reference instead of copy - frame encoding will handle it
            return self.current_frame if self.current_frame is not None else None
    
    def get_detected_number(self):
        """Get the current detected number (thread-safe)"""
        with self.lock:
            return self.detected_number

# ============================================================================
# WEB INTERFACE
# ============================================================================

# HTML Template for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Robot Control</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 10px;
            background: #1a1a1a;
            color: #fff;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin: 10px 0;
            font-size: 1.8em;
        }
        .video-container {
            background: #000;
            text-align: center;
            margin-bottom: 15px;
            border-radius: 5px;
        }
        .video-container img {
            max-width: 100%;
            height: auto;
            display: block;
        }
        .status {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
        }
        .status.ready { color: #2ecc71; }
        .status.executing { color: #f1c40f; }
        .status.cooldown { color: #3498db; }
        .info {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .info-box {
            flex: 1;
            background: #2a2a2a;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .info-label {
            font-size: 0.8em;
            color: #888;
            margin-bottom: 5px;
        }
        .info-value {
            font-size: 1.5em;
            font-weight: bold;
        }
        table {
            width: 100%;
            background: #2a2a2a;
            border-radius: 5px;
            border-collapse: collapse;
        }
        td {
            padding: 8px;
            border-bottom: 1px solid #444;
        }
        td:first-child {
            font-weight: bold;
            color: #3498db;
            width: 40px;
            text-align: center;
        }
        tr:last-child td {
            border-bottom: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Robot Control</h1>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Live Feed">
        </div>
        
        <div class="status" id="status">INITIALIZING...</div>
        
        <div class="info">
            <div class="info-box">
                <div class="info-label">Detected</div>
                <div class="info-value" id="detected">-</div>
            </div>
            <div class="info-box">
                <div class="info-label">Last Action</div>
                <div class="info-value" id="last-action">-</div>
            </div>
        </div>
        
        <table>
            <tr><td>0</td><td>Stop/Idle</td></tr>
            <tr><td>1</td><td>Forward (Slow)</td></tr>
            <tr><td>2</td><td>Forward (Medium)</td></tr>
            <tr><td>3</td><td>Forward (Fast)</td></tr>
            <tr><td>4</td><td>Turn Left</td></tr>
            <tr><td>5</td><td>Turn Right</td></tr>
        </table>
    </div>
    
    <script>
        function update() {
            fetch('/status')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('detected').textContent = d.detected_number;
                    document.getElementById('last-action').textContent = 
                        d.last_executed_action > 0 ? d.last_executed_action : '-';
                    
                    const status = document.getElementById('status');
                    status.className = 'status';
                    
                    if (d.state === 'EXECUTING') {
                        status.className += ' executing';
                        status.textContent = '▶ ' + d.current_action;
                    } else if (d.state === 'COOLDOWN') {
                        status.className += ' cooldown';
                        status.textContent = '⏸ COOLDOWN';
                    } else {
                        status.className += ' ready';
                        status.textContent = '✓ READY';
                    }
                });
        }
        
        setInterval(update, 200);
        update();
    </script>
</body>
</html>
"""

# Flask app
app = Flask(__name__)

# Global variables for sharing state
video_processor = None

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def status():
    """API endpoint for current detection status"""
    detected_num = video_processor.get_detected_number()
    robot_status = video_processor.controller.get_status()
    
    # Determine current state
    if robot_status['is_executing']:
        state = "EXECUTING"
        current_action = ACTION_NAMES.get(robot_status['current_action'], "Unknown")
    elif robot_status['in_cooldown']:
        state = "COOLDOWN"
        current_action = "Waiting..."
    else:
        state = "READY"
        current_action = "Waiting for gesture"
    
    return {
        'detected_number': detected_num,
        'action_name': ACTION_NAMES.get(detected_num, "Unknown"),
        'state': state,
        'current_action': current_action,
        'last_executed_action': robot_status['last_executed_action'],
        'last_action_name': ACTION_NAMES.get(robot_status['last_executed_action'], "None")
    }

@app.route('/video_feed')
def video_feed():
    """Stream video frames as MJPEG"""
    def generate():
        while True:
            frame = video_processor.get_current_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Encode frame as JPEG with optimized quality for faster encoding
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.02)  # ~50 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================================
# PORT DETECTION
# ============================================================================

def find_robot_port():
    """Auto-detect robot serial port."""
    # Disabled - serial.tools.list_ports commented out
    # ports = serial.tools.list_ports.comports()
    # for port in ports:
    #     if "USB" in port.device or "ACM" in port.device:
    #         return port.device
    return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    global video_processor
    
    print("=" * 70)
    print("AI-CONTROLLED ROBOT - Starting System")
    print("=" * 70)
    
    robot = None
    
    try:
        # 1. Initialize Robot
        print("\n[1/4] Initializing robot...")
        
        # Try to find robot port if hardcoded port doesn't exist
        actual_port = ROBOT_PORT
        if not os.path.exists(ROBOT_PORT):
            print(f"⚠ Port {ROBOT_PORT} not found, attempting auto-detection...")
            detected_port = find_robot_port()
            if detected_port:
                actual_port = detected_port
                print(f"✓ Found robot on {detected_port}")
            else:
                print("✗ Error: Could not find robot serial port")
                print("   Please check:")
                print("   - Is the robot connected via USB?")
                print("   - Is the USB device recognized? (try: ls -l /dev/tty*)")
                print(f"   - Or set ROBOT_PORT in run.py to the correct port")
                raise RuntimeError(f"Robot port not found. Expected: {ROBOT_PORT}")
        
        robot = AUPPBot(port=actual_port, baud=ROBOT_BAUD, auto_safe=True)
        print(f"✓ Robot connected on {actual_port}")
        
        # 2. Load AI Model
        print("\n[2/4] Loading AI model...")
        model_loader = ModelLoader(MODEL_PATH, MODEL_TYPE)
        
        # 3. Initialize Robot Controller
        print("\n[3/4] Initializing robot controller...")
        robot_controller = RobotController(robot)
        print("✓ Robot controller ready")
        
        # 4. Start Video Processing
        print("\n[4/4] Starting video processor...")
        video_processor = VideoProcessor(model_loader, robot_controller)
        video_processor.start()
        print("✓ Video processor started")
        
        # Start Web Server
        print("\n" + "=" * 70)
        print(f"🌐 Web Interface: http://localhost:{WEB_PORT}")
        print(f"🎥 Camera: Index {CAMERA_INDEX} ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
        print(f"🤖 Robot: {actual_port} @ {ROBOT_BAUD} baud")
        print(f"🧠 Model: {MODEL_PATH} ({MODEL_TYPE})")
        print("=" * 70)
        print("\nPress Ctrl+C to stop\n")
        
        # Run Flask app (blocking)
        app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        if video_processor:
            video_processor.stop()
        if robot:
            robot.safe()
            robot.close()
        print("✓ Shutdown complete")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

