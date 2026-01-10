# app/engine/words.py
import numpy as np
import pandas as pd

# --- CRITICAL CHECK ---
# This list MUST match the order of folders in your dataset exactly.
# If your dataset folders were: "book", "computer", "deaf"... (Alphabetical)
# But this list is: "me", "you", "we"... 
# Then the model will be 100% wrong.
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
        self.threshold = 0.60 # Lowered slightly for testing
        self.num_landmarks = 75 
        
        # Performance Optimization: Only predict every N frames
        self.frame_skip = 4 
        self.skip_counter = 0

    def extract_body_landmarks(self, results):
        """Extracts features and un-flips coordinates."""
        vec = []
        # Helper to un-flip X (transform 0.8 -> 0.2)
        def get_coords(landmark_list):
            temp = []
            if landmark_list:
                for lm in landmark_list.landmark:
                    temp.extend([1.0 - lm.x, lm.y, lm.z]) 
            else:
                # If missing, use 0.0 (will be filled by interpolation later)
                temp.extend([0.0] * 63) # 21 points * 3
            return temp

        # 1. Pose (33 points * 3 = 99)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                vec.extend([1.0 - lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 99)
            
        # 2. Left Hand (21 points * 3 = 63)
        vec.extend(get_coords(results.left_hand_landmarks))
            
        # 3. Right Hand (21 points * 3 = 63)
        vec.extend(get_coords(results.right_hand_landmarks))
            
        return vec

    def pre_process_sequence(self, sequence):
        """Normalizes data to match training."""
        try:
            data = np.array(sequence, dtype=np.float32)

            # 1. Interpolation (Fill missing frames)
            df = pd.DataFrame(data)
            df = df.replace(0.0, np.nan)
            df = df.interpolate(method='linear', axis=0, limit_direction='both')
            df = df.fillna(0.0)
            data = df.values.astype(np.float32)

            # Reshape: (Frames, Landmarks, 3)
            frames = data.reshape(-1, self.num_landmarks, 3)

            # 2. Body-Centric Normalization
            for i in range(frames.shape[0]):
                frame = frames[i]
                # Indices: 11=Left Shoulder, 12=Right Shoulder
                left_shoulder = frame[11] 
                right_shoulder = frame[12]
                
                shoulder_center = (left_shoulder + right_shoulder) / 2.0
                width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
                
                frame = (frame - shoulder_center) / (width / 2.0)
                frames[i] = frame

            # Flatten
            norm_seq = frames.reshape(-1, self.num_landmarks * 3)

            # 3. Time Resizing (Force to 30 frames)
            target_len = self.sequence_length
            res = np.zeros((target_len, norm_seq.shape[1]))
            
            for j in range(norm_seq.shape[1]):
                res[:, j] = np.interp(
                    np.linspace(0, len(norm_seq)-1, target_len),
                    np.arange(len(norm_seq)),
                    norm_seq[:, j]
                )
                
            return np.expand_dims(res, axis=0)
        except Exception as e:
            print(f"Preprocessing Error: {e}")
            return None

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

        # 3. OPTIMIZATION: Only run prediction every 4th frame
        self.skip_counter += 1
        if self.skip_counter < self.frame_skip:
            return "...", 0.0
        
        self.skip_counter = 0 # Reset counter

        # 4. RUN PREDICTION
        if len(self.sequence_buffer) >= 15:
            try:
                input_data = self.pre_process_sequence(self.sequence_buffer)
                if input_data is None: return "Error", 0.0

                res = self.inference_model.get_raw_prediction(input_data)
                
                if res is not None:
                    idx = np.argmax(res)
                    confidence = res[idx]
                    
                    # --- DEBUGGING OUTPUT ---
                    # Prints the top 3 guesses to the terminal
                    top_3 = np.argsort(res)[-3:][::-1]
                    debug_str = " | ".join([f"{WORD_CLASSES[i]}: {res[i]*100:.1f}%" for i in top_3])
                    print(f"DEBUG: {debug_str}")
                    # ------------------------

                    word = WORD_CLASSES[idx]
                    
                    if confidence > self.threshold:
                        self.sequence_buffer = [] # Reset after success
                        return word, confidence * 100
                    else:
                        return f"({word}?)", confidence * 100
                        
            except Exception as e:
                print(f"Word Engine Error: {e}")
                
        return "...", 0.0