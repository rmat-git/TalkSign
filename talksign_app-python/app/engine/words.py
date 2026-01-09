# app/engine/words.py
import numpy as np
import pandas as pd
import os

# Ensure this list matches the TARGET_GLOSSES in your Word_model.ipynb exactly
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
        self.threshold = 0.75  # Matches your test code WORD_THRESHOLD

    def extract_body_landmarks(self, results):
        """
        Extracts Pose (99), Left Hand (63), and Right Hand (63) features.
        Total: 225 features per frame.
        """
        vec = []
        
        # 1. Pose Landmarks (33 points * 3 = 99 features)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 99)

        # 2. Left Hand Landmarks (21 points * 3 = 63 features)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 63)

        # 3. Right Hand Landmarks (21 points * 3 = 63 features)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * 63)
            
        return vec

    def pre_process_sequence(self, sequence):
        """
        Matches training logic: Interpolation -> Normalization -> Resampling.
        """
        # Convert buffer to DataFrame for interpolation
        df = pd.DataFrame(sequence).replace(0.0, np.nan).interpolate(limit_direction='both').fillna(0.0)
        
        # Reshape to (Frames, Landmarks, XYZ) -> (30+, 75, 3)
        data = df.values.astype(np.float32).reshape(-1, 75, 3)
        
        # Shoulder-Center Normalization
        # Based on index 11 (Left Shoulder) and 12 (Right Shoulder)
        for i in range(data.shape[0]):
            left_shoulder = data[i, 11]
            right_shoulder = data[i, 12]
            
            center = (left_shoulder + right_shoulder) / 2.0
            width = np.linalg.norm(left_shoulder - right_shoulder) + 1e-6
            
            # Center the data and scale by shoulder width
            data[i] = (data[i] - center) / (width / 2.0)
        
        # Flatten back to (Frames, 225)
        norm_seq = data.reshape(-1, 225)
        
        # Resample/Interpolate to exactly 30 frames (Temporal normalization)
        resampled_data = np.zeros((30, 225))
        for j in range(225):
            resampled_data[:, j] = np.interp(
                np.linspace(0, len(norm_seq) - 1, 30),
                np.arange(len(norm_seq)),
                norm_seq[:, j]
            )
            
        return np.expand_dims(resampled_data, axis=0)

    def predict(self, results):
        """
        Main entry point called by camera.py.
        Returns (predicted_word, confidence_percent)
        """
        # Add current frame to buffer
        current_landmarks = self.extract_body_landmarks(results)
        self.sequence_buffer.append(current_landmarks)
        
        # Keep a sliding window slightly larger than needed for better interpolation
        if len(self.sequence_buffer) > 45:
            self.sequence_buffer.pop(0)

        # We need at least 30 frames to make a prediction
        if len(self.sequence_buffer) >= self.sequence_length:
            try:
                # 1. Process the buffer
                input_data = self.pre_process_sequence(self.sequence_buffer)
                
                # 2. Get prediction from the shared inference model
                # The word model expects shape (1, 30, 225)
                res = self.inference_model.get_raw_prediction(input_data)
                
                if res is not None:
                    idx = np.argmax(res)
                    confidence = res[idx]
                    
                    if confidence > self.threshold:
                        word = WORD_CLASSES[idx]
                        # Clear buffer after successful prediction to prevent double-triggering
                        # self.sequence_buffer = [] 
                        return word, confidence * 100
            except Exception as e:
                print(f"Word Engine Error: {e}")
                
        return "Thinking...", 0.0