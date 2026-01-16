# app/engine/alphabet.py
import numpy as np

# Matches deployment code ALPHABET_CLASSES
ALPHABET_CLASSES = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 letters (no J, Z in static)

class AlphabetEngine:
    def __init__(self, inference_model):
        self.inference_model = inference_model
        # Use the alphabet-specific class list
        self.actions = ALPHABET_CLASSES

    def process_landmarks(self, landmarks):
        """
        Transforms 63 landmarks to 89 features.
        MATCHES DEPLOYMENT CODE: extract_hand_for_alphabet()
        
        Feature breakdown:
        - 63: Raw coordinates (21 landmarks × 3)
        - 13: Distances (inter-finger + tips)
        - 5:  Finger ratios (extension measurement)
        - 4:  Angles (adjacent finger relationships)
        - 4:  Z-depth (thumb vs other fingers)
        TOTAL: 89 features
        """
        data = landmarks.reshape(21, 3)
        
        # 1. Normalization (same as before)
        wrist = data[0]
        data = data - wrist
        
        # Use palm_size for normalization (deployment uses data[9], which is middle finger MCP)
        palm_size = np.linalg.norm(data[9]) + 1e-6
        data = data / palm_size
        
        flat_coords = data.flatten()  # 63 features
        
        # Define key landmarks
        tips = [4, 8, 12, 16, 20]        # Fingertips
        knuckles = [2, 5, 9, 13, 17]     # Knuckles (MCP joints)
        finger_bases = [2, 5, 9, 13, 17] # Base of fingers
        
        # 2. Distances (13 features) - SAME AS DEPLOYMENT
        distances = []
        
        # Thumb to other fingertips (4 distances)
        for i in range(1, 5):
            distances.append(np.linalg.norm(data[tips[0]] - data[tips[i]]))
        
        # Each fingertip to wrist (5 distances)
        for i in range(5):
            distances.append(np.linalg.norm(data[tips[i]]))
        
        # Adjacent fingertip distances (4 distances)
        for i in range(4):
            distances.append(np.linalg.norm(data[tips[i]] - data[tips[i+1]]))
        
        # 3. 🆕 Finger Ratios (5 features) - NEW FROM DEPLOYMENT
        finger_ratios = []
        for i in range(5):
            dist_tip = np.linalg.norm(data[tips[i]])
            dist_knuckle = np.linalg.norm(data[knuckles[i]]) + 1e-6
            finger_ratios.append(dist_tip / dist_knuckle)
        
        # 4. Angles (4 features) - SAME AS DEPLOYMENT
        vectors = []
        for i in range(5):
            vectors.append(data[tips[i]] - data[finger_bases[i]])
        
        angles = []
        for i in range(4):
            v1, v2 = vectors[i], vectors[i+1]
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angles.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
        
        # 5. 🆕 Z-Depth Features (4 features) - NEW FROM DEPLOYMENT
        # Thumb tip Z compared to other fingertips
        z_feats = [data[4][2] - data[i][2] for i in [8, 12, 16, 20]]
        
        # Concatenate all features: 63 + 13 + 5 + 4 + 4 = 89
        return np.concatenate([
            flat_coords,           # 63
            np.array(distances),   # 13
            np.array(finger_ratios), # 5
            np.array(angles),      # 4
            np.array(z_feats)      # 4
        ])

    def predict(self, landmarks):
        """Interface for camera.py to get a prediction."""
        try:
            features = self.process_landmarks(landmarks)
            # Verify feature count
            if len(features) != 89:
                print(f"WARNING: Expected 89 features, got {len(features)}")
                return "Feature Error", 0.0
            
            input_array = np.expand_dims(features, axis=0)  # Shape: (1, 89)

            # Get raw probability array from the InferenceModel wrapper
            res = self.inference_model.get_raw_prediction(input_array)
            
            if res is None:
                return "Model Error", 0.0

            idx = np.argmax(res)
            return self.actions[idx], res[idx] * 100
        except Exception as e:
            print(f"AlphabetEngine Error: {e}")
            return "Engine Error", 0.0