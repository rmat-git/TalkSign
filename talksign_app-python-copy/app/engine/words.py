# app/engine/words.py
import numpy as np
import pandas as pd

# Ensure this matches your Word_model.ipynb exactly
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
        self.threshold = 0.75 # Lower this to 0.5 if it's too strict
        self.num_landmarks = 75 

    def extract_body_landmarks(self, results):
        """
        Extracts features AND un-flips the X-coordinates to match training data.
        """
        vec = []
        
        # Helper to flip X: (1.0 - x) transforms 0.8 -> 0.2
        def get_coords(landmark_list, count):
            temp = []
            if landmark_list:
                for lm in landmark_list.landmark:
                    # --- CRITICAL FIX: UN-FLIP X COORDINATE ---
                    # Camera is mirrored (Right is Right), but Model expects 
                    # standard video (Right hand is on Left side).
                    temp.extend([1.0 - lm.x, lm.y, lm.z]) 
            else:
                temp.extend([0.0] * (count * 3))
            return temp

        # 1. Pose (33 points)
        vec.extend(get_coords(results.pose_landmarks, 33))
            
        # 2. Left Hand (21 points)
        vec.extend(get_coords(results.left_hand_landmarks, 21))
            
        # 3. Right Hand (21 points)
        vec.extend(get_coords(results.right_hand_landmarks, 21))
            
        return vec

    def pre_process_sequence(self, sequence):
        """
        Normalization logic matching the training exactly.
        """
        data = np.array(sequence, dtype=np.float32)

        # 1. Interpolation
        df = pd.DataFrame(data)
        df = df.replace(0.0, np.nan)
        df = df.interpolate(method='linear', axis=0, limit_direction='both')
        df = df.fillna(0.0)
        data = df.values.astype(np.float32)

        # Reshape to (Frames, Landmarks, 3)
        frames = data.reshape(-1, self.num_landmarks, 3)

        # 2. Body-Centric Normalization
        for i in range(frames.shape[0]):
            frame = frames[i]
            # Landmarks 11 & 12 are shoulders
            left_shoulder = frame[11] 
            right_shoulder = frame[12]
            
            shoulder_center = (left_shoulder + right_shoulder) / 2.0
            width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
            
            frame = (frame - shoulder_center) / (width / 2.0)
            frames[i] = frame

        norm_seq = frames.reshape(-1, self.num_landmarks * 3)

        # 3. Time Resizing
        target_len = self.sequence_length
        res = np.zeros((target_len, norm_seq.shape[1]))
        
        for j in range(norm_seq.shape[1]):
            res[:, j] = np.interp(
                np.linspace(0, len(norm_seq)-1, target_len),
                np.arange(len(norm_seq)),
                norm_seq[:, j]
            )
            
        return np.expand_dims(res, axis=0)

    def predict(self, results):
        # 1. VISIBILITY CHECK
        has_pose = results.pose_landmarks is not None
        has_hands = (results.left_hand_landmarks is not None or 
                     results.right_hand_landmarks is not None)

        if not has_pose or not has_hands:
            if len(self.sequence_buffer) > 0:
                self.sequence_buffer = [] 
            return "NO HAND/BODY", 0.0

        # 2. ADD TO BUFFER
        current_landmarks = self.extract_body_landmarks(results)
        self.sequence_buffer.append(current_landmarks)
        
        if len(self.sequence_buffer) > 45:
            self.sequence_buffer.pop(0)

        # 3. RUN PREDICTION
        if len(self.sequence_buffer) >= self.sequence_length:
            try:
                input_data = self.pre_process_sequence(self.sequence_buffer)
                
                res = self.inference_model.get_raw_prediction(input_data)
                
                if res is not None:
                    idx = np.argmax(res)
                    confidence = res[idx]
                    
                    if confidence > self.threshold:
                        word = WORD_CLASSES[idx]
                        # Optional: Clear buffer after success to prevent double-fire
                        # self.sequence_buffer = [] 
                        return word, confidence * 100
                    else:
                        return f"({WORD_CLASSES[idx]}?)", confidence * 100
                        
            except Exception as e:
                print(f"Word Engine Error: {e}")
                
        return "Thinking...", 0.0