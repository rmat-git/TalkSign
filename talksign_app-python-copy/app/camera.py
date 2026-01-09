import cv2
import numpy as np
from mediapipe import solutions as mp_solutions
from PIL import Image, ImageTk, ImageDraw, ImageFont
import time
import threading
from queue import Queue, Empty
import pygame
import pyttsx3
import os
from pygame import mixer
import pygame._sdl2.audio as sdl2_audio
#skeleton
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Import the new Engine
from app.engine import AlphabetEngine, WordEngine

mp_holistic = mp_solutions.holistic

SEQUENCE_LENGTH = 30
LANDMARK_DIM = 63
PREDICTION_THRESHOLD = 0.80
COOLDOWN_FRAMES = 20
MAX_QUEUE_SIZE = 2

class CameraProcessor:
    def __init__(self, canvas, inference_model, log_callback,
                 width=480, height=320, camera_id=0):
        self.canvas = canvas
        self.model_wrapper = inference_model # renamed for clarity
        self.log = log_callback
        self.width = width
        self.height = height
        self.camera_id = camera_id
        
        # --- NEW ENGINE INITIALIZATION ---
        # Instead of calling the model directly, we use the engine
        self.engine = AlphabetEngine(self.model_wrapper)
        # ---------------------------------

        self.vcam_manager = None
        self.tts_enabled = True
        self.last_spoken_sentence = ""

        # Camera + thread state
        self.cap = None
        self.is_running = False
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.frame_queue = Queue(maxsize=MAX_QUEUE_SIZE)

        # Prediction/shared state
        self.current_action = "..."
        self.current_confidence = 0.0
        self.cooldown_status = "READY"
        self.frame_count = 0
        self.sentence = ""
        self.draft_text = ""
        self.last_sign_time = time.time()
        self.tts_threshold = 7 
        self.silence_timeout = 2.0

        # --- Bottom sentence text settings (live) ---
        self.text_enabled = True
        self.text_color = "#FFFFFF"
        self.text_size = 40
        self.text_y = self.height - 20 
        self.text_effect = "none"

        # Canvas text items:
        self.prediction_text_id = self.canvas.create_text(
            10, 10,
            anchor='nw', fill='lime', font=('Helvetica', 16, 'bold'),
            text="Ready"
        )

        self.sentence_text_id = self.canvas.create_text(
            self.width // 2, self.height - 20,
            anchor='s', fill=self.text_color, font=('Helvetica', self.text_size, 'bold'),
            text="Start Signing..."
        )

        self._shadow_item = self.canvas.create_text(
            self.width // 2 + 3, (self.height - 20) + 3,
            anchor='s', fill='black', font=('Helvetica', self.text_size, 'bold'),
            text="" 
        )

        self._outline_items = [
            self.canvas.create_text(self.width // 2 - 2, (self.height - 20), anchor='s',
                                     fill='black', font=('Helvetica', self.text_size, 'bold'), text=""),
            self.canvas.create_text(self.width // 2 + 2, (self.height - 20), anchor='s',
                                     fill='black', font=('Helvetica', self.text_size, 'bold'), text=""),
            self.canvas.create_text(self.width // 2, (self.height - 20) - 2, anchor='s',
                                     fill='black', font=('Helvetica', self.text_size, 'bold'), text=""),
            self.canvas.create_text(self.width // 2, (self.height - 20) + 2, anchor='s',
                                     fill='black', font=('Helvetica', self.text_size, 'bold'), text="")
        ]

        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.log("MediaPipe initialized.")

        self.prediction_history = []  # This stores the last 5 predictions
        self.STABILITY_FRAMES = 5  
        
    def set_engine(self, engine_type):
        """Swaps the active engine and resets buffers."""
        from app.engine import AlphabetEngine, WordEngine
        
        if engine_type == "alphabet":
            self.engine = AlphabetEngine(self.model_wrapper)
            self.log("Engine Switched: ALPHABET (Letters)")
        elif engine_type == "word":
            self.engine = WordEngine(self.model_wrapper)
            self.log("Engine Switched: WORDS (Full Gestures)")
        
        # Reset sentence and internal frame counters to prevent glitches
        self.frame_count = 0
        self.sentence = ""
        self.current_action = "..."
        self.current_confidence = 0.0

    # ... (Keep apply_custom_text_settings, start_feed, stop_feed as they are) ...
    def apply_custom_text_settings(self, enabled, color, size, y, effect):
        """
        Called by GUI to push live settings.
        - enabled: bool
        - color: hex color string (e.g. "#FFFFFF")
        - size: int or string convertible to int
        - y: pixel Y position for the baseline (anchor='s')
        - effect: "none" | "shadow" | "outline"
        """
        try:
            self.text_enabled = bool(enabled)
            # Keep color as-is (expecting hex like "#RRGGBB")
            if color is None:
                color = "#FFFFFF"
            self.text_color = str(color)
            self.text_size = int(size)
            # clamp y to canvas
            y_pix = int(y)
            if y_pix < 0:
                y_pix = 0
            if y_pix > self.height:
                y_pix = self.height
            self.text_y = y_pix
            if effect in ("none", "shadow", "outline"):
                self.text_effect = effect
            else:
                self.text_effect = "none"

            # Update fonts for helper items so outline/shadow sizes stay consistent
            font_spec = ('Helvetica', self.text_size, 'bold')
            self.canvas.itemconfig(self.sentence_text_id, font=font_spec)
            self.canvas.itemconfig(self._shadow_item, font=font_spec)
            for oid in self._outline_items:
                self.canvas.itemconfig(oid, font=font_spec)

        except Exception as e:
            self.log(f"apply_custom_text_settings error: {e}")

    # -------------------------
    # Camera control
    # -------------------------
    def start_feed(self):
            if self.is_running:
                return
            try:
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    raise Exception(f"Could not open camera id {self.camera_id}")
                # set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                self.is_running = True
                self.stop_event.clear()
                self.frame_count = 0
                self.sentence = ""

                # clear queue
                with self.frame_queue.mutex:
                    self.frame_queue.queue.clear()

                # start worker thread
                self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
                self.processing_thread.start()
                threading.Thread(target=self._tts_monitor, daemon=True).start()

                self.log(f"Camera started (ID {self.camera_id})")
                # start GUI update loop (non-blocking)
                self._gui_update_loop()

            except Exception as e:
                self.log(f"Camera start error: {e}")
                self.stop_feed()

    def stop_feed(self):
            if not self.is_running:
                return
            self.is_running = False
            self.stop_event.set()
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            if self.cap:
                self.cap.release()
                self.cap = None
            self.log("Camera stopped.")

    def _process_frame(self, frame_rgb, results):
            if results is None:
                self.prediction_history = [] 
                return

            # 1. GET PREDICTION
            if isinstance(self.engine, AlphabetEngine):
                keypoints, hand_detected = self.extract_keypoints(results)
                if hand_detected:
                    action, confidence = self.engine.predict(keypoints)
                    is_confident = (confidence >= PREDICTION_THRESHOLD * 100)
                else:
                    action, confidence = "NO HAND", 0.0 # Changed from "..." for better UI feedback
                    is_confident = False
            else:
                action, confidence = self.engine.predict(results)
                is_confident = (confidence >= 75) 

            # 2. UPDATE STABILITY HISTORY
            if is_confident and action not in ['NOTHING', 'Error', 'Thinking...', 'NO HAND']:
                self.prediction_history.append(action)
            else:
                self.prediction_history.append("...") 

            if len(self.prediction_history) > self.STABILITY_FRAMES:
                self.prediction_history.pop(0)

            # 3. STABILITY CHECK
            if (len(self.prediction_history) == self.STABILITY_FRAMES and 
                len(set(self.prediction_history)) == 1 and 
                self.prediction_history[0] != "..."):
                
                stable_action = self.prediction_history[0]
                
                # 4. COMMIT ONLY IF READY
                if self.frame_count >= COOLDOWN_FRAMES:
                    self._commit_to_sentence(stable_action)
                    self.frame_count = 0  # Reset Cooldown
                    self.prediction_history = [] # Reset History so it doesn't double-trigger
                    self.log(f"STABLE ACCEPTED: {stable_action}")

            # IMPORTANT: Always increment frame_count so the "READY" state can be reached
            self.current_action = action
            self.current_confidence = confidence
            self.frame_count += 1
            
            # UI Status Text
            if self.frame_count < COOLDOWN_FRAMES:
                self.cooldown_status = f"WAIT ({COOLDOWN_FRAMES - self.frame_count})"
            else:
                self.cooldown_status = "READY"

    def _process_loop(self):
            while not self.stop_event.is_set():
                try:
                    # 1. Pull the 'package' from the queue
                    data = self.frame_queue.get(timeout=0.1)
                    
                    # 2. Unpack it into the frame and the mediapipe results
                    frame_rgb, results = data
                    
                    # 3. Send BOTH to the process function
                    self._process_frame(frame_rgb, results)
                    
                except Empty:
                    continue
                except Exception as e:
                    self.log(f"Processing thread error: {e}")
                    time.sleep(0.1)

    # -------------------------
    # GUI update loop (main thread)
    # -------------------------
    def _gui_update_loop(self):
            if not self.is_running:
                self.canvas.itemconfig(self.sentence_text_id, text="Start Signing...")
                self.canvas.itemconfig(self.prediction_text_id, text="Camera Stopped")
                return
            
            results = None
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # --- SKELETON DRAWING ---
                results = self.holistic.process(frame_rgb)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_rgb, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=2)
                    )
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_rgb, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame_rgb, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # Render frame to canvas
                img = Image.fromarray(frame_rgb)
                img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
                
                # --- VIRTUAL CAMERA BLOCK ---
                if self.vcam_manager and self.vcam_manager.is_active:
                    v_img = img.copy() 
                    draw = ImageDraw.Draw(v_img)
                    if self.text_enabled:
                        txt = self.sentence if self.sentence else "Start Signing..."
                        pos = (self.width // 2, self.text_y)
                        if not hasattr(self, '_cached_font') or self._cached_font_size != self.text_size:
                            try:
                                self._cached_font = ImageFont.truetype("arial.ttf", self.text_size)
                            except:
                                self._cached_font = ImageFont.load_default()
                            self._cached_font_size = self.text_size
                        
                        if self.text_effect == "shadow":
                            draw.text((pos[0] + 3, pos[1] + 3), txt, fill="black", font=self._cached_font, anchor="ms")
                        elif self.text_effect == "outline":
                            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                                draw.text((pos[0] + dx, pos[1] + dy), txt, fill="black", font=self._cached_font, anchor="ms")
                        draw.text(pos, txt, fill=self.text_color, font=self._cached_font, anchor="ms")
                    
                    v_frame = np.array(v_img)
                    v_frame = np.fliplr(v_frame) 
                    self.vcam_manager.send_frame(v_frame)

                # Enqueue for processing thread
                try:
                    # OPTIMIZATION: Pass the results we already calculated to save CPU!
                    self.frame_queue.put_nowait((frame_rgb.copy(), results))
                except Exception:
                    pass

                # Update prediction text
                prediction_display = f"{self.current_action} ({self.current_confidence:.1f}%) | {self.cooldown_status}"
                self.canvas.itemconfig(self.prediction_text_id, text=prediction_display)

                # Update sentence text
                text = self.sentence if self.sentence else "Start Signing..."
                x, y = self.width // 2, self.text_y
                font_spec = ('Helvetica', self.text_size, 'bold')

                if self.text_enabled:
                    self.canvas.itemconfig(self.sentence_text_id, text=text, fill=self.text_color, font=font_spec)
                    self.canvas.coords(self.sentence_text_id, x, y)
                    
                    if self.text_effect == "shadow":
                        self.canvas.itemconfig(self._shadow_item, text=text, fill='black', font=font_spec)
                        self.canvas.coords(self._shadow_item, x + 3, y + 3)
                        self.canvas.tag_lower(self._shadow_item, self.sentence_text_id)
                    elif self.text_effect == "outline":
                        for oid, (dx, dy) in zip(self._outline_items, [(-2, 0), (2, 0), (0, -2), (0, 2)]):
                            self.canvas.itemconfig(oid, text=text, fill='black', font=font_spec)
                            self.canvas.coords(oid, x + dx, y + dy)
                            self.canvas.tag_lower(oid, self.sentence_text_id)
                else:
                    self.canvas.itemconfig(self.sentence_text_id, text="")

                self.canvas.tag_raise(self.sentence_text_id)
                self.canvas.tag_raise(self.prediction_text_id)

            # THIS LINE MUST BE INDENTED 8 SPACES (ALIGNED WITH 'if ret:')
            self.canvas.after(16, self._gui_update_loop)

    # -------------------------
    # Helpers
    # -------------------------
    def extract_keypoints(self, results):
        hand_keypoints = np.zeros(LANDMARK_DIM, dtype=np.float32)
        hand_landmarks_detected = False

        if results is None:
            return hand_keypoints, hand_landmarks_detected

        if getattr(results, 'right_hand_landmarks', None):
            hand_landmarks_detected = True
            kp = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32)
            hand_keypoints = kp.flatten()
        elif getattr(results, 'left_hand_landmarks', None):
            hand_landmarks_detected = True
            kp = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32)
            # mirror left to match right
            kp[:, 0] = -kp[:, 0]
            hand_keypoints = kp.flatten()

        return hand_keypoints, hand_landmarks_detected
    
    def _tts_monitor(self):
        """Background loop to check for silence."""
        while True:
            time.sleep(0.5)
            if not self.draft_text:
                continue
            
            if (time.time() - self.last_sign_time) >= self.silence_timeout:
                self._trigger_draft_speech(" (Silence Timeout)")

    def _trigger_draft_speech(self, reason=""):
        if self.draft_text and self.tts_enabled:
            text_to_say = self.draft_text
            self.log(f"Speaking Draft{reason}: {text_to_say}")
            # Start speech in a background thread
            threading.Thread(target=self._speak, args=(text_to_say,), daemon=True).start()
            self.draft_text = "" # Clear draft buffer

    def _speak(self, text):
        """Internal helper to handle TTS output and route to Virtual Cable."""
        try:
            import pyttsx3
            import os
            from pygame import mixer
            import time

            # 1. Initialize TTS Engine
            engine = pyttsx3.init()
            
            # Sync voice type (Male/Female) from GUI variables
            voices = engine.getProperty('voices')
            if hasattr(self, 'voice_type') and self.voice_type == "M":
                engine.setProperty('voice', voices[0].id)
            else:
                engine.setProperty('voice', voices[1].id)

            # 2. Render speech to a temporary file
            # pyttsx3.say() goes to default hardware; save_to_file allows manual routing
            temp_wav = "tts_output.wav"
            engine.save_to_file(text, temp_wav)
            engine.runAndWait()
            engine.stop()
            del engine

            # 3. Initialize Pygame Mixer on the Virtual Cable
            virtual_device = 'CABLE Input (VB-Audio Virtual Cable)'
            
            try:
                # Initialize mixer specifically for the virtual cable
                mixer.init(devicename=virtual_device)
                speech_sound = mixer.Sound(temp_wav)
                speech_sound.play()
                
                # Block thread until speaking is finished to prevent file cleanup errors
                while mixer.get_busy():
                    time.sleep(0.1)
                    
                mixer.quit()
            except Exception as mixer_e:
                self.log(f"Mixer Error: {mixer_e}. Falling back to default audio.")
                mixer.init() # Fallback to default
                mixer.Sound(temp_wav).play()
            
            # 4. Cleanup
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        except Exception as e:
            self.log(f"TTS Routing Error: {e}")

    def _commit_to_sentence(self, action):
            """Handles adding characters/words to the string and TTS buffer."""
            char_to_add = action
            
            if action == 'SPACE':
                char_to_add = ' '
            elif action == 'DELETE':
                if self.sentence:
                    self.sentence = self.sentence[:-1]
                if self.draft_text: # Keep TTS draft in sync with the sentence
                    self.draft_text = self.draft_text[:-1]
                char_to_add = None # Don't add "DELETE" as a word
            
            if char_to_add:
                # Word Mode Logic
                if isinstance(self.engine, WordEngine) and self.sentence and not self.sentence.endswith(" "):
                    self.sentence += " " + char_to_add
                    self.draft_text += " " + char_to_add
                else:
                    # Alphabet Mode Logic
                    self.sentence += char_to_add
                    self.draft_text += char_to_add
                
                self.last_sign_time = time.time()