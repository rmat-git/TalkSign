# app/engine/alphabet.py
import numpy as np

class AlphabetEngine:
    def __init__(self, inference_model):
        self.inference_model = inference_model
        # Import ACTIONS here to ensure we use the same list as the rest of the app
        from app.inference import ACTIONS
        self.actions = ACTIONS

    def process_landmarks(self, landmarks):
        """Logic moved from inference.py: Transforms 63 landmarks to 80 features."""
        data = landmarks.reshape(21, 3)
        
        # 1. Normalization
        wrist = data[0]
        data = data - wrist
        scale_ref = np.linalg.norm(data[12]) + 1e-6
        data = data / scale_ref
        flat_coords = data.flatten()
        
        # 2. Distances (13 features)
        tips = [4, 8, 12, 16, 20]
        distances = []
        for i in range(1, 5):
            distances.append(np.linalg.norm(data[tips[0]] - data[tips[i]]))
        for i in range(4):
            distances.append(np.linalg.norm(data[tips[i]] - data[tips[i+1]]))
        for i in range(5):
            distances.append(np.linalg.norm(data[tips[i]]))
            
        # 3. Angles (4 features)
        finger_bases = [2, 5, 9, 13, 17]
        vectors = []
        for i in range(5):
            vectors.append(data[tips[i]] - data[finger_bases[i]])
        angles = []
        for i in range(4):
            v1, v2 = vectors[i], vectors[i+1]
            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angles.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
        
        return np.concatenate([flat_coords, np.array(distances), np.array(angles)])

    def predict(self, landmarks):
        """Interface for camera.py to get a prediction."""
        try:
            features = self.process_landmarks(landmarks)
            input_array = np.expand_dims(features, axis=0) 

            # Get raw probability array from the InferenceModel wrapper
            res = self.inference_model.get_raw_prediction(input_array)
            
            if res is None:
                return "Model Error", 0.0

            idx = np.argmax(res)
            return self.actions[idx], res[idx] * 100
        except Exception as e:
            print(f"AlphabetEngine Error: {e}")
            return "Engine Error", 0.0