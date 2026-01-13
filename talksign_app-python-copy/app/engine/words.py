# app/engine/words.py
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# The Model's Vocabulary (matches deployment code exactly)
# ---------------------------------------------------------
WORD_CLASSES = [
    "me", "you", "we", "they", "hello", 
    "house", "friend", "yes", "no", "fine", "help", "meet", "good",
    "want", "have", "like", "need", "go", "walk", 
    "play", "work", "learn", "eat", "drink", "finish",
    "book", "family", "school", "computer", "deaf", "j", "z"
]

class WordEngine:
    def __init__(self, inference_model):
        self.inference_model = inference_model
        self.sequence_buffer = []
        self.sequence_length = 30  # Must have 30 frames minimum (deployment standard)
        self.threshold = 0.75  # 85% confidence threshold
        self.num_landmarks = 75  # 33 pose + 21 left hand + 21 right hand

    def extract_body_landmarks(self, results):
        """
        Extracts raw landmarks.
        NO FLIPPING / MIRRORING applied (Raw Input).
        Matches deployment code extract_body_landmarks().
        """
        vec = []
        
        # Helper to get [x, y, z] without flipping
        def get_coords(landmark_list):
            temp = []
            if landmark_list:
                for lm in landmark_list.landmark:
                    # ORIGINAL INPUT: Use lm.x directly (no mirroring)
                    temp.extend([lm.x, lm.y, lm.z]) 
            else:
                temp.extend([0.0] * 63)
            return temp

        # 1. Pose (33 points * 3 = 99 features)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                # ORIGINAL INPUT: Use lm.x directly
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 99)
            
        # 2. Left Hand (21 points * 3 = 63 features)
        vec.extend(get_coords(results.left_hand_landmarks))
            
        # 3. Right Hand (21 points * 3 = 63 features)
        vec.extend(get_coords(results.right_hand_landmarks))
        
        # Total: 99 + 63 + 63 = 225 features
        return vec

    def pre_process_sequence(self, sequence):
        """
        Standard Normalization (matches deployment code pre_process_word_input).
        """
        try:
            data = np.array(sequence, dtype=np.float32)

            # 1. Interpolation - fill missing values
            df = pd.DataFrame(data)
            df = df.replace(0.0, np.nan)
            df = df.interpolate(method='linear', axis=0, limit_direction='both')
            df = df.fillna(0.0)
            data = df.values.astype(np.float32)

            # Reshape to (frames, landmarks, coords)
            frames = data.reshape(-1, self.num_landmarks, 3)

            # 2. Center around Shoulders (normalize position and scale)
            for i in range(frames.shape[0]):
                frame = frames[i]
                left_shoulder = frame[11]   # Pose landmark 11
                right_shoulder = frame[12]  # Pose landmark 12
                
                shoulder_center = (left_shoulder + right_shoulder) / 2.0
                width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
                
                # Normalize: center at origin, scale by shoulder width
                frame = (frame - shoulder_center) / (width / 2.0)
                frames[i] = frame

            # 3. Time Resizing (Force exactly 30 frames via interpolation)
            norm_seq = frames.reshape(-1, self.num_landmarks * 3)
            target_len = self.sequence_length
            res = np.zeros((target_len, norm_seq.shape[1]))
            
            # Interpolate each feature dimension to target length
            for j in range(norm_seq.shape[1]):
                res[:, j] = np.interp(
                    np.linspace(0, len(norm_seq)-1, target_len),
                    np.arange(len(norm_seq)),
                    norm_seq[:, j]
                )
            
            # Return shape: (1, 30, 225)    
            return np.expand_dims(res, axis=0)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def predict(self, results):
        """
        DEPLOYMENT CODE LOGIC:
        - Always return a prediction (word + confidence)
        - Let camera.py handle the stability/threshold checks
        - Never return "..." unless insufficient frames
        - Return actual prediction even if confidence is low
        """
        
        # 1. Safety Check - hands required for word signs
        has_hands = (results.left_hand_landmarks or results.right_hand_landmarks)
        if not has_hands:
            # Clear buffer when no hands present
            if self.sequence_buffer:
                self.sequence_buffer = []
            return "NO HAND/BODY", 0.0

        # 2. Buffer Management - Add current frame
        self.sequence_buffer.append(self.extract_body_landmarks(results))
        
        # Sliding Window: Keep max 45 frames (same as deployment)
        if len(self.sequence_buffer) > 45: 
            self.sequence_buffer.pop(0)

        # 3. Prediction - Run when we have enough frames
        # DEPLOYMENT: Requires 30 frames minimum (not 15)
        if len(self.sequence_buffer) >= 30:
            try:
                input_data = self.pre_process_sequence(self.sequence_buffer)
                if input_data is None:
                    return "...", 0.0

                res = self.inference_model.get_raw_prediction(input_data)
                
                if res is not None:
                    idx = np.argmax(res)
                    conf = res[idx]
                    
                    # CRITICAL CHANGE: Always return the prediction
                    # Don't filter by threshold here - let camera.py's stability system decide
                    # This ensures the stability system gets consistent predictions every frame
                    return WORD_CLASSES[idx], conf * 100
                    
            except Exception as e:
                # Log error but still return something
                print(f"WordEngine prediction error: {e}")
                return "...", 0.0
        
        # Not enough frames yet - still building buffer
        return "...", 0.0