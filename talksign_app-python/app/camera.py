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

mp_holistic = mp_solutions.holistic

SEQUENCE_LENGTH = 30
LANDMARK_DIM = 63
PREDICTION_THRESHOLD = 0.60
COOLDOWN_FRAMES = 10
MAX_QUEUE_SIZE = 2

def normalize_keypoints(kp):
    pts = kp.reshape(21, 3)
    base = pts[0].copy()
    pts -= base
    maxv = np.max(np.abs(pts))
    if maxv > 0:
        pts /= maxv
    return pts.flatten()


class CameraProcessor:
    """
    Camera feed + MediaPipe processing + canvas overlay rendering.
    The bottom sentence text supports live-updated settings via
    apply_custom_text_settings(enabled, color, size, y, effect).
    """

    def __init__(self, canvas, inference_model, log_callback,
                 width=480, height=320, camera_id=0):
        self.canvas = canvas
        self.model = inference_model
        self.log = log_callback
        self.width = width
        self.height = height
        self.camera_id = camera_id
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
        # Defaults (will be overridden by GUI on startup)
        self.text_enabled = True
        self.text_color = "#FFFFFF"
        self.text_size = 40
        self.text_y = self.height - 20  # pixel Y (GUI maps slider -> pixel Y)
        self.text_effect = "none"  # "none", "shadow", "outline"

        # Canvas text items:
        # 1) top-left prediction (unchanged)
        self.prediction_text_id = self.canvas.create_text(
            10, 10,
            anchor='nw', fill='lime', font=('Helvetica', 16, 'bold'),
            text="Ready"
        )

        # 2) bottom sentence main text item (single object)
        # We'll create the main text and also a set of pre-created shadow/outline items (updated, not recreated).
        self.sentence_text_id = self.canvas.create_text(
            self.width // 2, self.height - 20,
            anchor='s', fill=self.text_color, font=('Helvetica', self.text_size, 'bold'),
            text="Start Signing..."
        )

        # Shadow / Outline helper items (pre-created)
        # One shadow item (single offset) for 'shadow' effect:
        self._shadow_item = self.canvas.create_text(
            self.width // 2 + 3, (self.height - 20) + 3,
            anchor='s', fill='black', font=('Helvetica', self.text_size, 'bold'),
            text=""  # initially empty; we'll update when needed
        )

        # Four outline items (N, S, E, W) for outline effect:
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

        # MediaPipe
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.log("MediaPipe initialized.")

    # -------------------------
    # Public API: GUI calls this
    # -------------------------
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

    # -------------------------
    # Processing thread
    # -------------------------
    def _process_frame(self, frame_rgb):
        # process with mediapipe
        try:
            results = self.holistic.process(frame_rgb)
        except Exception as e:
            # if processing fails, do not crash worker
            self.log(f"MediaPipe process error: {e}")
            return

        keypoints, hand_detected = self.extract_keypoints(results)
        action = "NO HAND"
        confidence = 0.0

        if hand_detected:
            kp_norm = normalize_keypoints(np.array(keypoints, dtype=np.float32))
            input_sequence = kp_norm.reshape(1, 1, LANDMARK_DIM)
            action, confidence = self.model.predict_action(input_sequence)

        # cooldown and sentence building logic
        self.frame_count += 1
        cooldown_status = f"CD: {COOLDOWN_FRAMES - self.frame_count}" if self.frame_count < COOLDOWN_FRAMES else "READY"

        if (self.frame_count >= COOLDOWN_FRAMES and
                confidence >= (PREDICTION_THRESHOLD * 100) and
                action != 'NOTHING'):
            char_to_add = action
            if action == 'SPACE':
                char_to_add = ' '
                self.sentence += char_to_add
            elif action == 'DELETE':
                if self.sentence:
                    self.sentence = self.sentence[:-1]
                char_to_add = None
            else:
                self.sentence += char_to_add

            self.frame_count = 0
            self.log(f"Accepted action '{action}'. Sentence: \"{self.sentence}\"")

            if action not in ["DELETE", "NOTHING"]:
                char = action if action != "SPACE" else " "
                self.draft_text += char
                self.last_sign_time = time.time() 
                
                # Check if threshold reached
                if len(self.draft_text) >= self.tts_threshold:
                    self._trigger_draft_speech(" (Threshold Reached)")

        self.current_action = action
        self.current_confidence = confidence
        self.cooldown_status = cooldown_status

    def _process_loop(self):
        while not self.stop_event.is_set():
            try:
                frame_rgb = self.frame_queue.get(timeout=0.1)
                self._process_frame(frame_rgb)
            except Empty:
                continue
            except Exception as e:
                self.log(f"Processing thread error: {e}")
                time.sleep(0.2)
        self.log("Processing thread finished.")

    # -------------------------
    # GUI update loop (main thread)
    # -------------------------
    def _gui_update_loop(self):
        if not self.is_running:
            # ensure UI indicates stopped state
            self.canvas.itemconfig(self.sentence_text_id, text="Start Signing...")
            self.canvas.itemconfig(self.prediction_text_id, text="Camera Stopped")
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # render frame to canvas (resized)
            img = Image.fromarray(frame_rgb)
            img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image=img)
            # keep a reference to avoid GC
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
            
            # --- UPDATED VIRTUAL CAMERA BLOCK (FIXED) ---
            if self.vcam_manager and self.vcam_manager.is_active:
                v_img = img.copy() 
                draw = ImageDraw.Draw(v_img)
                
                if self.text_enabled:
                    # 1. Define variables BEFORE use
                    txt = self.sentence if self.sentence else "Start Signing..."
                    pos = (self.width // 2, self.text_y)
                    
                    # 2. Font Caching Logic
                    if not hasattr(self, '_cached_font') or self._cached_font_size != self.text_size:
                        try:
                            # Ensure you use a valid font path for your OS
                            self._cached_font = ImageFont.truetype("arial.ttf", self.text_size)
                        except:
                            self._cached_font = ImageFont.load_default()
                        self._cached_font_size = self.text_size
                    
                    # 3. Apply Styles (Shadow/Outline)
                    if self.text_effect == "shadow":
                        draw.text((pos[0] + 3, pos[1] + 3), txt, fill="black", font=self._cached_font, anchor="ms")
                    elif self.text_effect == "outline":
                        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                            draw.text((pos[0] + dx, pos[1] + dy), txt, fill="black", font=self._cached_font, anchor="ms")
                    
                    # 4. Draw main text
                    draw.text(pos, txt, fill=self.text_color, font=self._cached_font, anchor="ms")
                
                # 5. CONVERT & FIX MIRRORING
                v_frame = np.array(v_img)
                v_frame = np.fliplr(v_frame) 
                
                self.vcam_manager.send_frame(v_frame)
            # ---------------------------------------------

            # enqueue for processing thread
            try:
                self.frame_queue.put_nowait(frame_rgb.copy())
            except Exception:
                # queue full, drop frame
                pass

            # update top-left prediction text
            prediction_display = f"{self.current_action} ({self.current_confidence:.1f}%) | {self.cooldown_status}"
            self.canvas.itemconfig(self.prediction_text_id, text=prediction_display)
            self.canvas.tag_raise(self.prediction_text_id)

            # update bottom sentence text and effects based on current live settings
            # We update pre-created shadow/outline items rather than creating new items each frame.
            text = self.sentence if self.sentence else "Start Signing..."
            x = self.width // 2
            y = self.text_y  # GUI already mapped slider -> pixel Y

            font_spec = ('Helvetica', self.text_size, 'bold')

            # Update main sentence item
            if self.text_enabled:
                self.canvas.itemconfig(self.sentence_text_id, text=text, fill=self.text_color, font=font_spec)
                self.canvas.coords(self.sentence_text_id, x, y)
            else:
                # Hide text by setting it empty
                self.canvas.itemconfig(self.sentence_text_id, text="")

            # Reset helper items to empty/hide them unless needed
            for oid in self._outline_items:
                self.canvas.itemconfig(oid, text="")
            self.canvas.itemconfig(self._shadow_item, text="")

            # Apply effect: update pre-created items positions and text (so they move with main text)
            if self.text_enabled and self.text_effect == "shadow":
                # draw single shadow slightly offset behind main text
                sx, sy = x + 3, y + 3
                self.canvas.coords(self._shadow_item, sx, sy)
                self.canvas.itemconfig(self._shadow_item, text=text, fill='black', font=font_spec)
                # main text already set above (will draw over shadow)
                # ensure ordering: shadow below main
                self.canvas.tag_lower(self._shadow_item, self.sentence_text_id)

            elif self.text_enabled and self.text_effect == "outline":
                # place outline items around the main text
                offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]
                for oid, (dx, dy) in zip(self._outline_items, offsets):
                    self.canvas.coords(oid, x + dx, y + dy)
                    self.canvas.itemconfig(oid, text=text, fill='black', font=font_spec)
                    # ensure outline is below main text
                    self.canvas.tag_lower(oid, self.sentence_text_id)

            # ensure main text is on top of frame and overlay helpers
            self.canvas.tag_raise(self.sentence_text_id)

        # schedule next update (~60 FPS)
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