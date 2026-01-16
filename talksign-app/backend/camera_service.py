import cv2
import numpy as np
import threading
import time
import os
import mediapipe as mp
from mediapipe import solutions as mp_solutions
from queue import Queue, Empty, Full
import pyvirtualcam 
from PIL import Image, ImageDraw, ImageFont

# --- AUDIO IMPORTS ---
import pyttsx3
from pygame import mixer

try:
    from pygrabber.dshow_graph import FilterGraph
except ImportError:
    print("WARNING: 'pygrabber' not installed.")
    FilterGraph = None

from app.engine import AlphabetEngine, WordEngine
try:
    from app.llm import GeminiHandler
except ImportError:
    print("WARNING: app.llm not found. Gemini features will be disabled.")
    GeminiHandler = None

# --- CONFIGURATION ---
GEMINI_API_KEY = "AIzaSyDyFzKLeUBpkl3Wmj7TPTh9ufV-c5SaBoI" 
SEQUENCE_LENGTH = 30
LANDMARK_DIM = 63
PREDICTION_THRESHOLD = 0.80
COOLDOWN_FRAMES = 20
MAX_QUEUE_SIZE = 1  # Keep small: We only want the AI working on the *latest* frame
HANDS_LOST_THRESHOLD = 2.0 

class CameraService:
    def __init__(self, inference_model):
        self.model_wrapper = inference_model
        
        self.status = {
            "mode": "alphabet",
            "prediction": "...",
            "confidence": 0.0,
            "sentence": "",
            "hands_detected": False,
            "is_cooldown": False,
            "llm_processing": False
        }

        # --- Visual Settings ---
        self.text_enabled = True
        self.text_size = 40
        self.text_color = "#FFFFFF"
        self.text_position = 10 
        
        # --- Font Setup ---
        self.font_path_custom = os.path.join(os.getcwd(), "presets", "fonts", "Rubik-Regular.ttf")
        self.font_path_system = "arial.ttf"
        
        # --- Camera & Threading ---
        self.cap = None
        self.camera_id = 0
        self.is_running = False
        self.camera_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.data_lock = threading.Lock()
        
        self.stop_event = threading.Event()
        # Queue now holds raw frames for the AI to process asynchronously
        self.frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self.processing_thread = None
        
        # --- Virtual Camera ---
        self.vcam = None
        self.vcam_enabled = False
        
        # --- Internal Counters ---
        self.frame_count = 0
        self.raw_tokens = [] 
        self.hands_lost_time = None
        self.force_new_token = False
        self.showing_result = False  
        
        # --- Settings ---
        self.tts_enabled = True
        self.voice_type = "F"

        # --- AI Handlers ---
        self.engine = AlphabetEngine(self.model_wrapper)
        if GeminiHandler:
            self.llm_handler = GeminiHandler(GEMINI_API_KEY, print)
        else:
            self.llm_handler = None

        # Helper for keypoint extraction logic
        self.mp_holistic_utils = mp_solutions.holistic

    # --- INPUT MANAGEMENT ---
    def clear_input(self):
        print("System: Clearing Input Memory")
        with self.data_lock:
            self.raw_tokens = []
            self.status["sentence"] = ""
            self.showing_result = False
            self.status["llm_processing"] = False
            self.force_new_token = True

    def process_backspace(self):
        with self.data_lock:
            if not self.raw_tokens: return
            if self.status["mode"] == "alphabet":
                last_token = self.raw_tokens[-1]
                if len(last_token) > 1: self.raw_tokens[-1] = last_token[:-1]
                else: self.raw_tokens.pop()
            else:
                self.raw_tokens.pop()
            self.status["sentence"] = " ".join(self.raw_tokens)

    # --- CAMERA DISCOVERY ---
    def get_available_cameras(self):
        devices = []
        if FilterGraph:
            try:
                graph = FilterGraph()
                names = graph.get_input_devices()
                for i, name in enumerate(names): devices.append({"id": i, "name": name})
            except Exception: pass
        if not devices:
            for i in range(3): devices.append({"id": i, "name": f"Camera {i}"})
        return devices

    # --- CAMERA CONTROL ---
    def start_camera(self, camera_id=0):
        with self.camera_lock:
            if self.is_running: return
            self.camera_id = camera_id
            print(f"Attempting to open Camera {self.camera_id}...")
            
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_MSMF)
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                print(f"CRITICAL ERROR: Could not open camera {self.camera_id}")
                self.cap = None
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.is_running = True
            self.stop_event.clear()
            
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()

            self.processing_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
            self.processing_thread.start()
            print(f"CameraService started on ID {camera_id}")

    def stop_camera(self):
        self.is_running = False 
        self.stop_event.set()
        if self.processing_thread: self.processing_thread.join(timeout=0.5)
        with self.camera_lock:
            if self.cap: self.cap.release(); self.cap = None
        if self.vcam: self.vcam.close(); self.vcam = None
        print("CameraService stopped.")

    def trigger_camera_switch(self, new_id=None):
        self.stop_camera()
        time.sleep(0.1) 
        if new_id is not None: target_id = int(new_id)
        else: target_id = 1 if self.camera_id == 0 else 0
        self.start_camera(target_id)
        if not self.is_running: self.stop_camera(); self.start_camera(0); return 0
        return target_id

    def toggle_vcam(self):
        self.vcam_enabled = not self.vcam_enabled
        if not self.vcam_enabled and self.vcam:
            self.vcam.close()
            self.vcam = None
        return self.vcam_enabled

    def set_mode(self, mode):
        print(f"System: Switching mode to {mode.upper()}...")
        ALPHA_FILE = 'asl_alphabet_model.keras'
        WORD_FILE = 'robust_word_model.keras' 
        with self.data_lock:
            try:
                if mode == "alphabet":
                    self.model_wrapper.load_model(ALPHA_FILE)
                    self.engine = AlphabetEngine(self.model_wrapper)
                    self.status["mode"] = "alphabet"
                elif mode == "word":
                    self.model_wrapper.load_model(WORD_FILE)
                    self.engine = WordEngine(self.model_wrapper)
                    self.status["mode"] = "word"
                self.frame_count = 0
                self.force_new_token = True
                print(f"System: Mode switched successfully to {mode}")
            except Exception as e:
                print(f"Error switching modes: {e}")

    # --- TEXT RENDERING ---
    def _draw_overlay(self, frame_rgb):
        if not self.text_enabled: return frame_rgb
        text = self.status["sentence"]
        if not text: text = "WAITING FOR INPUT..."
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        try:
            font_size = int(self.text_size * 2.0)
            try: font = ImageFont.truetype(self.font_path_custom, font_size)
            except OSError: font = ImageFont.truetype(self.font_path_system, font_size)
        except OSError: font = ImageFont.load_default()
        W, H = pil_img.size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (W - text_w) // 2, H - (H * (self.text_position / 100)) - text_h - 40 
        try: hex_c = self.text_color.lstrip('#'); rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        except: rgb = (255, 255, 255)
        draw.text((x, y), text, font=font, fill=rgb, stroke_width=3, stroke_fill=(0, 0, 0))
        return np.array(pil_img)

    # --- MAIN LOOP (NOW DECOUPLED & FAST) ---
    def generate_frames(self):
        while True:
            if not self.is_running: break
            with self.camera_lock:
                if self.cap is None or not self.cap.isOpened(): break
                success, frame = self.cap.read()
            if not success: break
            try:
                # 1. Base Frame
                frame_rgb_true = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb_mirror = cv2.flip(frame_rgb_true, 1)
                
                # 2. SEND TO AI (Non-Blocking)
                # We try to push to the queue. If it's full (AI is busy), 
                # we just skip this frame for AI processing but keep streaming video.
                try:
                    self.frame_queue.put_nowait(frame_rgb_mirror)
                except Full:
                    pass # Drop frame for AI, keep video smooth
                
                # 3. Draw Overlay (Fast)
                # Uses self.status which is updated in the background
                final_browser_frame = self._draw_overlay(frame_rgb_mirror)
                final_vcam_frame_true = self._draw_overlay(frame_rgb_true)
                
                # 4. Virtual Camera
                if self.vcam_enabled:
                    if self.vcam is None:
                        try:
                            self.vcam = pyvirtualcam.Camera(width=final_vcam_frame_true.shape[1], height=final_vcam_frame_true.shape[0], fps=30)
                        except Exception: self.vcam_enabled = False
                    if self.vcam:
                        self.vcam.send(cv2.flip(final_vcam_frame_true, 1))

                # 5. Browser Output
                final_bgr = cv2.cvtColor(final_browser_frame, cv2.COLOR_RGB2BGR)
                ret, buffer = cv2.imencode('.jpg', final_bgr)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception: break

    # --- AI WORKER LOOP (HANDLES HEAVY PROCESSING) ---
    def _ai_worker_loop(self):
        # Initialize MediaPipe INSIDE the thread for safety
        holistic = mp_solutions.holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        while not self.stop_event.is_set():
            try:
                # Wait for a frame (Blocking is fine here, it's a background thread)
                frame_rgb = self.frame_queue.get(timeout=0.1)
                
                # 1. Heavy Inference
                results = holistic.process(frame_rgb)
                
                # 2. Update Logic
                self._process_logic(results)
                
            except Empty: continue
            except Exception: pass
        
        holistic.close()

    # --- AI LOGIC ---
    def _process_logic(self, results):
        if self.showing_result or self.status["llm_processing"]: return
        if results is None: 
            self._handle_missing_hands(True)
            return

        if isinstance(self.engine, AlphabetEngine):
            keypoints, hand_detected = self.extract_keypoints(results)
            self.status["hands_detected"] = hand_detected
            if hand_detected:
                self._handle_missing_hands(False)
                action, confidence = self.engine.predict(keypoints)
            else:
                self._handle_missing_hands(True)
                action, confidence = "...", 0.0
        else:
            has_hands = bool(results.left_hand_landmarks or results.right_hand_landmarks)
            self.status["hands_detected"] = has_hands
            if has_hands:
                self._handle_missing_hands(False)
                action, confidence = self.engine.predict(results)
            else:
                self._handle_missing_hands(True)
                action, confidence = "...", 0.0

        self.frame_count += 1
        self.status["is_cooldown"] = (self.frame_count < COOLDOWN_FRAMES)
        
        is_confident = False
        if isinstance(self.engine, AlphabetEngine):
            is_confident = (confidence >= PREDICTION_THRESHOLD * 100)
        else:
            is_confident = (confidence >= 75)

        ignore_list = ['NOTHING', 'Error', 'Thinking...', '...', 'Prediction Failed', 'NO HAND']
        
        if (not self.status["is_cooldown"] and is_confident and action not in ignore_list):
            self._commit_to_sentence(action)
            self.frame_count = 0 

        self.status["prediction"] = action
        self.status["confidence"] = float(confidence)

    def _handle_missing_hands(self, is_missing):
        if is_missing:
            if self.hands_lost_time is None: self.hands_lost_time = time.time()
        else: self.hands_lost_time = None

    def _commit_to_sentence(self, action):
        with self.data_lock:
            char_to_add = action
            if action == 'SPACE': char_to_add = ' '
            elif action == 'DELETE':
                if self.raw_tokens: self.raw_tokens.pop()
                char_to_add = None
            
            if char_to_add:
                if isinstance(self.engine, WordEngine):
                    if not self.raw_tokens or self.raw_tokens[-1] != char_to_add:
                        self.raw_tokens.append(char_to_add)
                else:
                    if self.force_new_token or not self.raw_tokens:
                        self.raw_tokens.append(char_to_add)
                        self.force_new_token = False
                    else:
                        self.raw_tokens[-1] += char_to_add

            if not self.showing_result:
                self.status["sentence"] = " ".join(self.raw_tokens)

    def extract_keypoints(self, results):
        hand_keypoints = np.zeros(LANDMARK_DIM, dtype=np.float32)
        hand_landmarks_detected = False
        if results is None: return hand_keypoints, hand_landmarks_detected
        if getattr(results, 'right_hand_landmarks', None):
            hand_landmarks_detected = True
            kp = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32)
            hand_keypoints = kp.flatten()
        elif getattr(results, 'left_hand_landmarks', None):
            hand_landmarks_detected = True
            kp = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32)
            kp[:, 0] = -kp[:, 0]
            hand_keypoints = kp.flatten()
        return hand_keypoints, hand_landmarks_detected

    def trigger_gemini(self):
        with self.data_lock:
            if not self.raw_tokens: return False
            self.showing_result = True
            self.status["llm_processing"] = True
            text_to_process = " ".join(self.raw_tokens)
            self.raw_tokens = [] 
            self.status["sentence"] = "" 
        threading.Thread(target=self._process_llm_and_speak, args=(text_to_process,)).start()
        return True

    def _process_llm_and_speak(self, raw_text):
        if not self.llm_handler or not self.tts_enabled: 
            with self.data_lock: self.showing_result = False; self.status["llm_processing"] = False
            return
        try:
            print(f"Gemini Input: {raw_text}")
            final_text = self.llm_handler.translate(raw_text)
            print(f"Gemini Output: {final_text}")
            with self.data_lock: self.status["sentence"] = final_text
            self._speak(final_text)
            time.sleep(1.0)
            with self.data_lock: self.status["sentence"] = "" 
        except Exception as e:
            print(f"LLM/TTS Error: {e}")
            with self.data_lock: self.status["sentence"] = "Error processing request."
            time.sleep(2.0)
            with self.data_lock: self.status["sentence"] = ""
        finally:
            with self.data_lock: self.status["llm_processing"] = False; self.showing_result = False 

    def _speak(self, text):
        with self.audio_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 145) 
                voices = engine.getProperty('voices')
                if len(voices) > 1 and self.voice_type == "F": engine.setProperty('voice', voices[1].id)
                else: engine.setProperty('voice', voices[0].id)
                temp_wav = "tts_output.wav"
                engine.save_to_file(text, temp_wav)
                engine.runAndWait()
                del engine
                virtual_device = 'CABLE Input (VB-Audio Virtual Cable)'
                try:
                    if mixer.get_init(): mixer.quit()
                    mixer.init(devicename=virtual_device)
                    mixer.Sound(temp_wav).play()
                    while mixer.get_busy(): time.sleep(0.1)
                    mixer.quit()
                except Exception:
                    if mixer.get_init(): mixer.quit()
                    mixer.init()
                    mixer.Sound(temp_wav).play()
                    while mixer.get_busy(): time.sleep(0.1)
                    mixer.quit()
                if os.path.exists(temp_wav): os.remove(temp_wav)
            except Exception as e: print(f"Speaker Error: {e}")