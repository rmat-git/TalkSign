import os
import numpy as np
import tensorflow as tf
import keras

# --- CONFIGURATION ---
DEFAULT_ALPHA_MODEL = 'asl_alphabet_model.keras'
DEFAULT_WORD_MODEL = 'robust_word_model.keras'

# Centralized list of supported gestures/actions
ACTIONS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# --- CUSTOM LOSS FUNCTION FOR WORD MODEL ---
# Required to load the robust_word_model.keras correctly
@keras.saving.register_keras_serializable()
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for multi-class classification.
    Matches the deployment code's focal loss implementation.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.math.pow((1 - y_pred), gamma)
        return tf.math.reduce_sum(weight * cross_entropy, axis=-1)
    
    return focal_loss_fixed

# Create the loss function instance globally
focal_loss_fixed = categorical_focal_loss()


class InferenceModel:
    def __init__(self, log_callback=print):
        self.log = log_callback
        self.model = None
        self.is_saved_model = False
        
        # --- STORAGE FOR PRE-LOADED MODELS ---
        # Holds the loaded models in RAM to allow instant switching
        self.loaded_models = {
            'alphabet': None,
            'word': None
        }
        
        # --- PRE-LOAD ALL MODELS ON STARTUP ---
        self.log("System: Pre-loading models into RAM... (This may take a moment)")
        
        # Load both models immediately
        self._preload_model('alphabet', DEFAULT_ALPHA_MODEL)
        self._preload_model('word', DEFAULT_WORD_MODEL)
        
        # Set the initial active model
        self.load_model(DEFAULT_ALPHA_MODEL)

    def _find_model_file(self, filename):
        """
        Searches for the model file in the correct folders.
        """
        clean_name = os.path.basename(filename)
        
        # We define the specific absolute path based on your folder structure
        # NOTE: The 'r' before the string handles the backslashes correctly
        user_desktop_path = r"C:\Users\User\Desktop\TalkSign-main\TalkSign-main\talksign_app-python-copy\model"

        possible_paths = [
            # 1. YOUR SPECIFIC PATH (Highest Priority)
            os.path.join(user_desktop_path, clean_name),

            # 2. Check inside "model" folder relative to this script
            # This handles cases where you run the app from different folders
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', clean_name)),
            
            # 3. Check current folder
            clean_name, 
            
            # 4. Check "model" folder in current directory
            os.path.join('model', clean_name),
            
            # 5. Check absolute path if provided
            filename
        ]

        self.log(f"Searching for {clean_name}...") 

        for path in possible_paths:
            # Check if this path exists
            if os.path.exists(path):
                self.log(f" FOUND at: {path}")
                return os.path.abspath(path)
            else:
                # Optional: Uncomment this if you want to see where it failed
                # self.log(f" Not found at: {path}")
                pass
        
        self.log(f" CRITICAL: Could not find {clean_name} in any expected location.")
        return None

    def _preload_model(self, key, filename):
        """
        Internal helper to load a model from disk into the dictionary.
        """
        found_path = self._find_model_file(filename)
        
        if not found_path:
            self.log(f"WARNING: Could not find '{filename}'. Mock mode will be used for {key}.")
            self.loaded_models[key] = "MOCK"
            return

        try:
            self.log(f"Loading {key} model from: {found_path}")
            
            # Check if it's a directory (SavedModel format)
            if os.path.isdir(found_path):
                model = tf.saved_model.load(found_path)
                self.loaded_models[key] = (model, True) # (Model, IsSavedModel)
            else:
                # Try loading Keras model with custom loss support
                try:
                    model = tf.keras.models.load_model(
                        found_path,
                        custom_objects={'focal_loss_fixed': focal_loss_fixed},
                        compile=False
                    )
                except Exception:
                    try:
                        # Fallback to keras.saving (Newer Keras 3 format)
                        model = keras.saving.load_model(
                            found_path,
                            custom_objects={'focal_loss_fixed': focal_loss_fixed}
                        )
                    except Exception:
                        # Fallback to standard load
                        model = tf.keras.models.load_model(found_path, compile=False)
                
                self.loaded_models[key] = (model, False) # (Model, IsKeras)
            
            self.log(f"SUCCESS: {key} model loaded.")
            
        except Exception as e:
            self.log(f"ERROR loading {filename}: {e}")
            self.loaded_models[key] = "MOCK"

    def load_model(self, model_identifier=None):
        """
        Switches the active model pointer instantly using pre-loaded models.
        
        Args:
            model_identifier (str, optional): Filename or key ('alphabet'/'word'). 
                                              Defaults to Alphabet if None.
        """
        # CRITICAL FIX: Handle None argument (triggered by main.py)
        if model_identifier is None:
            model_identifier = DEFAULT_ALPHA_MODEL

        # Determine which key to use
        if 'word' in model_identifier.lower():
            key = 'word'
        else:
            key = 'alphabet'

        # Retrieve the pre-loaded model
        stored_data = self.loaded_models.get(key)
        
        if stored_data == "MOCK" or stored_data is None:
            self._set_mock_model()
        else:
            # Unpack the tuple (Model, IsSavedModel)
            self.model, self.is_saved_model = stored_data

    def _set_mock_model(self):
        self.model = True 
        self.log("Backend: Running in MOCK mode.")

    def get_raw_prediction(self, input_array):
        """
        Gets raw prediction probabilities from the active model.
        """
        if self.model is True:
            # Random guessing for mock mode
            return np.random.dirichlet(np.ones(len(ACTIONS)), size=1)[0]

        try:
            if self.is_saved_model:
                input_tensor = tf.constant(input_array, dtype=tf.float32)
                predictions = list(self.model(input_tensor).values())[0].numpy()
                return predictions[0]
            else:
                # Standard Keras prediction
                return self.model.predict(input_array, verbose=0)[0]
        except Exception as e:
            self.log(f"Prediction Error: {e}")
            return None

# -------------------------
# Example usage and testing
if __name__ == "__main__":
    def logger(msg):
        print("[LOG]", msg)

    print("\n=== Initializing Inference System ===")
    wrapper = InferenceModel(log_callback=logger)
    
    # Test 1: Alphabet Mode
    print("\n--- Testing Alphabet Mode ---")
    wrapper.load_model('asl_alphabet_model.keras')
    dummy_alpha = np.random.rand(1, 89).astype(np.float32)
    res = wrapper.get_raw_prediction(dummy_alpha)
    if res is not None:
        print(f"Alphabet prediction shape: {res.shape}")

    # Test 2: Word Mode (Instant Switch)
    print("\n--- Testing Word Mode ---")
    wrapper.load_model('robust_word_model.keras')
    dummy_word = np.random.rand(1, 30, 225).astype(np.float32)
    res = wrapper.get_raw_prediction(dummy_word)
    if res is not None:
        print(f"Word prediction shape: {res.shape}")