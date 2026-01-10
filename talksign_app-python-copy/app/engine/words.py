# app/engine/words.py
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# The Model's Vocabulary
# ---------------------------------------------------------
WORD_CLASSES = [
    "me", "you", "we", "they", "hello", 
    "who", "what", "yes", "no", "fine", "help", "meet", "good",
    "want", "have", "like", "need", "go", "walk", 
    "play", "work", "learn", "eat", "drink", "finish",
    "book", "family", "school", "computer", "deaf", "j", "z"
]

class WordEngine:
    def __init__(self, inference_model):
        self.inference_model = inference_model
        self.sequence_buffer = []
        self.sequence_length = 30
        self.threshold = 0.85 
        self.num_landmarks = 75 

    def extract_body_landmarks(self, results):
        """
        Extracts raw landmarks.
        NO FLIPPING / MIRRORING applied (Raw Input).
        """
        vec = []
        
        # Helper to get [x, y, z] without flipping
        def get_coords(landmark_list):
            temp = []
            if landmark_list:
                for lm in landmark_list.landmark:
                    # ORIGINAL INPUT: Use lm.x directly
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
            
        # 2. Left Hand
        vec.extend(get_coords(results.left_hand_landmarks))
            
        # 3. Right Hand
        vec.extend(get_coords(results.right_hand_landmarks))
            
        return vec

    def pre_process_sequence(self, sequence):
        """Standard Normalization."""
        try:
            data = np.array(sequence, dtype=np.float32)

            # 1. Interpolation
            df = pd.DataFrame(data)
            df = df.replace(0.0, np.nan)
            df = df.interpolate(method='linear', axis=0, limit_direction='both')
            df = df.fillna(0.0)
            data = df.values.astype(np.float32)

            # Reshape
            frames = data.reshape(-1, self.num_landmarks, 3)

            # 2. Center around Shoulders
            for i in range(frames.shape[0]):
                frame = frames[i]
                left_shoulder = frame[11] 
                right_shoulder = frame[12]
                
                shoulder_center = (left_shoulder + right_shoulder) / 2.0
                width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
                
                frame = (frame - shoulder_center) / (width / 2.0)
                frames[i] = frame

            # 3. Time Resizing (Force 30 frames)
            norm_seq = frames.reshape(-1, self.num_landmarks * 3)
            target_len = self.sequence_length
            res = np.zeros((target_len, norm_seq.shape[1]))
            
            for j in range(norm_seq.shape[1]):
                res[:, j] = np.interp(
                    np.linspace(0, len(norm_seq)-1, target_len),
                    np.arange(len(norm_seq)),
                    norm_seq[:, j]
                )
                
            return np.expand_dims(res, axis=0)
        except:
            return None

    def predict(self, results):
        # 1. Safety Check
        has_hands = (results.left_hand_landmarks or results.right_hand_landmarks)
        if not has_hands:
            if self.sequence_buffer: self.sequence_buffer = [] 
            return "...", 0.0

        # 2. Buffer Management
        self.sequence_buffer.append(self.extract_body_landmarks(results))
        
        # Sliding Window: Keep exactly 45 frames
        if len(self.sequence_buffer) > 45: 
            self.sequence_buffer.pop(0)

        # 3. Prediction (Run EVERY frame for Stability)
        if len(self.sequence_buffer) >= 15:
            try:
                input_data = self.pre_process_sequence(self.sequence_buffer)
                if input_data is None: return "...", 0.0

                res = self.inference_model.get_raw_prediction(input_data)
                
                if res is not None:
                    idx = np.argmax(res)
                    conf = res[idx]
                    
                    # Debug print to verify stability
                    # print(f"DEBUG: {WORD_CLASSES[idx]} ({conf*100:.1f}%)")

                    if conf > self.threshold:
                        # FIX: DO NOT CLEAR BUFFER HERE.
                        # Let the Main App see this result for 5 frames in a row.
                        return WORD_CLASSES[idx], conf * 100
                    
            except Exception:
                pass
                
        return "...", 0.0