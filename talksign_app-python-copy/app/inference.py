import os
import numpy as np
import tensorflow as tf

# Default fallback
DEFAULT_ALPHA_MODEL = 'asl_alphabet_model.keras'

# Centralized list of supported gestures/actions
ACTIONS = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
    'O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'SPACE','DELETE','NOTHING'
]

class InferenceModel:
    def __init__(self, log_callback=print):
        self.log = log_callback
        self.model = None
        self.is_saved_model = False
        
        # Load default on startup
        self.load_model(DEFAULT_ALPHA_MODEL)

    def _find_model_file(self, filename):
        """
        Searches for the model file in the correct folders.
        """
        # We strip any path (like "../model/") and just use the filename "word_model.keras"
        clean_name = os.path.basename(filename)

        possible_paths = [
            # 1. Check inside "model" folder (Most likely for you)
            os.path.join('model', clean_name),
            
            # 2. Check current folder
            clean_name, 
            
            # 3. Check specific relative path for app structure
            os.path.join(os.path.dirname(__file__), '..', 'model', clean_name),
            
            # 4. Check absolute path if provided
            filename
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        return None

    def load_model(self, model_identifier=None):
        target_name = model_identifier if model_identifier else DEFAULT_ALPHA_MODEL
        
        # FIND THE FILE
        found_path = self._find_model_file(target_name)
        
        if not found_path:
            self.log(f"CRITICAL: Could not find '{target_name}' in 'model/' folder.")
            self.log("Falling back to MOCK MODE.")
            self._set_mock_model()
            return

        try:
            self.log(f"Loading model from: {found_path}")
            if os.path.isdir(found_path):
                self.model = tf.saved_model.load(found_path)
                self.is_saved_model = True
            else:
                self.model = tf.keras.models.load_model(found_path, compile=False)
                self.is_saved_model = False
            self.log("SUCCESS: Model loaded.")
        except Exception as e:
            self.log(f"Load Error: {e}")
            self._set_mock_model()

    def _set_mock_model(self):
        self.model = True 
        self.log("Backend: Running in MOCK mode.")

    def get_raw_prediction(self, input_array):
        if self.model is True:
            # Random guessing
            return np.random.dirichlet(np.ones(len(ACTIONS)), size=1)[0]

        try:
            if self.is_saved_model:
                input_tensor = tf.constant(input_array, dtype=tf.float32)
                predictions = list(self.model(input_tensor).values())[0].numpy()
                return predictions[0]
            else:
                return self.model.predict(input_array, verbose=0)[0]
        except Exception as e:
            self.log(f"Prediction Error: {e}")
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