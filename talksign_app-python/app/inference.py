import os
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
# Note: Ensure this path is correct relative to where you run main.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'asl_alphabet_model.keras')

# Centralized list of supported gestures/actions
ACTIONS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
    'O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'SPACE','DELETE','NOTHING'
]

class InferenceModel:
    """
    Handles only the Loading and Execution of the Neural Network.
    Feature extraction and geometry are now handled by the Engine layer.
    """
    def __init__(self, log_callback=print):
        self.log = log_callback
        self.model = None
        self.is_saved_model = False
        # The model is usually loaded explicitly by main.py, 
        # but calling it here ensures it's ready if used standalone.
        self.load_model()

    def load_model(self, custom_path=None):
            """Loads the model from disk or falls back to Mock mode if not found."""
            target_path = custom_path if custom_path else MODEL_PATH
            
            if not os.path.exists(target_path):
                self.log(f"Model not found at {target_path}. Using mock mode.")
                self._set_mock_model()
                return

            try:
                # Check if it's a folder (SavedModel) or a file (.keras/.h5)
                if os.path.isdir(target_path):
                    self.model = tf.saved_model.load(target_path)
                    self.is_saved_model = True
                else:
                    self.model = tf.keras.models.load_model(target_path, compile=False)
                    self.is_saved_model = False
                
                self.log(f"Model loaded successfully from {target_path}")
            except Exception as e:
                self.log(f"Load Error: {e}. Falling back to mock.")
                self._set_mock_model()

    def _set_mock_model(self):
        """Enables development without a GPU or model file present."""
        self.model = True 
        self.log("Backend: Running in MOCK mode (Random Predictions).")

    def get_raw_prediction(self, input_array):
        """
        Takes a processed numpy array (shape 1, 80) and returns the softmax probabilities.
        """
        if self.model is True:
            res = np.random.dirichlet(np.ones(len(ACTIONS)), size=1)[0]
            return res

        try:
            if self.is_saved_model:
                input_tensor = tf.constant(input_array, dtype=tf.float32)
                predictions = list(self.model(input_tensor).values())[0].numpy()
                return predictions[0]
            else:
                return self.model.predict(input_array, verbose=0)[0]
        except Exception as e:
            self.log(f"Model Execution Error: {e}")
            return None

# -------------------------
# Example usage:
if __name__ == "__main__":
    def logger(msg):
        print("[LOG]", msg)

    # 1. Initialize
    model_wrapper = InferenceModel(log_callback=logger)

    # 2. Create mock input of 80 features (as expected by the Alphabet model)
    # Shape: (batch_size=1, features=80)
    dummy_input = np.random.rand(1, 80).astype(np.float32)

    # 3. Use the new method name
    raw_res = model_wrapper.get_raw_prediction(dummy_input)
    
    if raw_res is not None:
        idx = np.argmax(raw_res)
        print(f"Predicted action: {ACTIONS[idx]} ({raw_res[idx]*100:.2f}%)")