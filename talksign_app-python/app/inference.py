import os
import time
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = '../model/asl_alphabet_model.keras'  # .keras file or SavedModel folder

ACTIONS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
    'O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'SPACE','DELETE','NOTHING'
]
SEQUENCE_LENGTH = 30
LANDMARK_DIM = 63

class InferenceModel:
    """Loads a gesture recognition model using tf.keras or SavedModel, with fallback MOCK."""

    def __init__(self, log_callback=print):
        self.log = log_callback
        self.model = None
        self.is_saved_model = False
        self.load_model()

    def load_model(self):
        """Detects and loads either a .keras/.h5 file or a SavedModel folder."""
        if not os.path.exists(MODEL_PATH):
            self.log(f"ERROR: Model path not found at {MODEL_PATH}. Using MOCK.")
            self._set_mock_model()
            return

        try:
            if os.path.isdir(MODEL_PATH):
                # SavedModel folder
                self.model = tf.saved_model.load(MODEL_PATH)
                self.model = self.model.signatures['serving_default']
                self.is_saved_model = True
                self.log(f"Loaded SavedModel from {MODEL_PATH}")
            elif MODEL_PATH.endswith(('.keras', '.h5')):
                # .keras or .h5 file
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy']
                )
                self.is_saved_model = False
                self.log(f"Loaded .keras model from {MODEL_PATH}")
            else:
                raise ValueError("Unknown model format. Must be .keras, .h5, or SavedModel folder.")
        except Exception as e:
            self.log(f"CRITICAL ERROR loading model: {e}. Falling back to MOCK.")
            self._set_mock_model()

    def _set_mock_model(self):
        """Sets a mock model for testing without a real model."""
        time.sleep(0.5)
        self.model = True  # MOCK flag
        self.log("Mock model ready.")

    def predict_action(self, sequence):
        """Runs prediction on a single input sequence."""
        if self.model is True:
            # MOCK prediction
            idx = np.random.randint(0, len(ACTIONS))
            confidence = 90 + np.random.rand() * 10
            return ACTIONS[idx], confidence

        if self.model is None:
            return "No Model Loaded", 0.0

        try:
            # 1. Standardize Sequence Input
            seq_array = np.array(sequence, dtype=np.float32)
            
            # **FIX:** Check if input is accidentally batched (e.g., shape (1, 30, 63) or (1, 1, 63))
            if seq_array.ndim > 2 and seq_array.shape[0] == 1:
                seq_array = seq_array[0]
            
            # 2. Extract the Single Frame (63 features)
            # Take the LAST frame if it's a sequence, or the array itself if it's already (63,)
            if seq_array.ndim == 2:
                single_frame_array = seq_array[-1, :] # Shape (63,)
            else: # Must be (63,)
                single_frame_array = seq_array
                

            # 3. Add the Batch Dimension for the model (1, 63)
            input_array = np.expand_dims(single_frame_array, axis=0) # Shape is (1, 63)
            
            if self.is_saved_model:
                # SavedModel expects tf.Tensor input
                input_tensor = tf.constant(input_array, dtype=tf.float32)
                predictions = list(self.model(input_tensor).values())[0].numpy()
                res = predictions[0]
            else:
                # .keras model expects numpy array input
                res = self.model.predict(input_array)[0]

            idx = np.argmax(res)
            confidence = res[idx] * 100
            return ACTIONS[idx], confidence
        except Exception as e:
            self.log(f"Prediction ERROR: {e}")
            return "Prediction Failed", 0.0

# -------------------------
# Example usage:

if __name__ == "__main__":
    def logger(msg):
        print("[LOG]", msg)

    model = InferenceModel(log_callback=logger)

    # Example mock input
    dummy_input = np.random.rand(1, 63).astype(np.float32)

    action, conf = model.predict_action(dummy_input)
    print(f"Predicted action: {action} ({conf:.2f}%)")