import base64
import numpy as np
import cv2
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import os
import mediapipe as mp # REQUIRED FOR KEYPOINT EXTRACTION

# Import necessary layers from the current Keras installation
from keras.layers import InputLayer, LSTM, Dense, Dropout

app = Flask(__name__)
CORS(app) 

# --- Configuration ---
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'deployables', 'my_model.h5')

SEQUENCE_LENGTH = 30 # CRITICAL: Model expects 30 frames of sequential data

# --- Global State for Sequence Buffer ---
GLOBAL_SEQUENCE = [] # Buffer to store the last 30 frames of extracted keypoints

# --- MediaPipe Setup ---
mp_holistic = mp.solutions.holistic 
HOLISTIC = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# --- Feature Extraction Function ---
def extract_keypoints(results):
    """
    Extracts concatenated keypoints (Pose + Left Hand + Right Hand).
    Total features: 258 features.
    """
    # Pose (132 features)
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Left Hand (63 features)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right Hand (63 features)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

# --- Model Loading with Debugging (KEY CHANGE HERE) ---
model = None
try:
    absolute_model_path = os.path.abspath(MODEL_PATH)
    print(f"🎯 Checking for model at absolute path: {absolute_model_path}")
    
    if not os.path.exists(absolute_model_path):
        raise FileNotFoundError(f"Model file not found at: {absolute_model_path}")
        
    # === KEY FIX: Force Keras to use its internal layers for deserialization ===
    # This bypasses the 'Unrecognized keyword arguments' error by providing a mapping
    # to the layers it needs to deserialize.
    custom_objects = {
        'InputLayer': InputLayer,
        'LSTM': LSTM,
        'Dense': Dense,
        'Dropout': Dropout
        # Add any other layers used in your model (e.g., Bidirectional) here.
    }
    
    model = tf.keras.models.load_model(absolute_model_path, custom_objects=custom_objects)
    print("✅ Model loaded successfully using custom_objects fix.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load model.")
    print(f"   Reason: {e}")
    print("\n\n🔥 FINAL ACTION REQUIRED: Model Re-Saving 🔥")
    print("   If this failure persists, the ONLY remaining solution is to open your training environment (e.g., 'Sample Model.ipynb')")
    print("   and re-save the model using the SavedModel format (TF native format).")
    model = None

# --- Image Processing Utility ---

def base64_to_cv2_image(base64_string):
    """Decodes a Base64 string into an OpenCV image and converts to RGB for MediaPipe."""
    try:
        image_bytes = base64.b64decode(base64_string)
        np_array = np.frombuffer(image_bytes, np.uint8)
        cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR) 
        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        app.logger.error(f"Image decoding error: {e}")
        return None

# --- Flask API Endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict():
    global GLOBAL_SEQUENCE

    # 1. Error Guard Check
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    if not request.json or 'image' not in request.json:
        return jsonify({"error": "Missing image data"}), 400
        
    cv_image = base64_to_cv2_image(request.json['image'])
    
    if cv_image is None:
        return jsonify({"error": "Could not decode image"}), 400
        
    # --- 2. Extract Keypoints using MediaPipe ---
    cv_image.flags.writeable = False
    results = HOLISTIC.process(cv_image)
    cv_image.flags.writeable = True

    keypoints = extract_keypoints(results)
    
    # --- 3. Update Sequence Buffer ---
    GLOBAL_SEQUENCE.append(keypoints)
    
    if len(GLOBAL_SEQUENCE) > SEQUENCE_LENGTH:
        GLOBAL_SEQUENCE = GLOBAL_SEQUENCE[-SEQUENCE_LENGTH:]

    # --- 4. Check Sequence Readiness ---
    if len(GLOBAL_SEQUENCE) < SEQUENCE_LENGTH:
        current_count = len(GLOBAL_SEQUENCE)
        print(f"COLLECTING DATA: {current_count}/{SEQUENCE_LENGTH} frames")
        return jsonify({
            "result": "COLLECTING DATA",
            "status": f"Collecting {current_count}/{SEQUENCE_LENGTH}"
        })

    # --- 5. Prepare Data for Prediction (Sequence is ready) ---
    input_data = np.expand_dims(np.array(GLOBAL_SEQUENCE), axis=0)
    
    # --- 6. Prediction ---
    prediction = model.predict(input_data, verbose=0)
    
    # 7. Post-process (No confidence filtering)
    predicted_index = np.argmax(prediction[0])
    predicted_sign = CLASS_NAMES[predicted_index]

    # === Server-side Terminal Output ===
    print(f"Translation: {predicted_sign}")
    # ===================================

    return jsonify({
        "result": predicted_sign,
        "status": "Prediction Successful"
    })

if __name__ == '__main__':
    print("Starting Flask API on http://127.0.0.1:5000")
    app.run(port=5000)