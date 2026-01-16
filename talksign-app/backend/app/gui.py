import tkinter as tk
from tkinter import scrolledtext, colorchooser
from app.camera import CameraProcessor
from app.virtual_cam import VirtualCamManager
import time
import cv2
import os

# --- DEFAULT SETTINGS ---
DEFAULT_TEXT_COLOR = "#FFFFFF"
DEFAULT_TEXT_SIZE = 40
DEFAULT_TTS_ENABLED = True
DEFAULT_VOICE_TYPE = "F"


class AppGUI:
    """Main GUI class."""

    def __init__(self, master, inference_model, log_callback):
        self.master = master
        self.log = log_callback

        master.title("TalkSign – Customization")
        master.configure(bg="#f0f0f0")
        master.geometry("1100x550")
        # --- GUI VARIABLES ---
        self.text_enabled_var = tk.BooleanVar(value=True)
        self.text_color_var = tk.StringVar(value=DEFAULT_TEXT_COLOR)
        self.text_size_var = tk.StringVar(value=str(DEFAULT_TEXT_SIZE))
        self.text_vertical_slider = tk.IntVar(value=0)  # 0 bottom → 100 top
        self.text_effect_var = tk.StringVar(value="none")

        self.tts_enabled_var = tk.BooleanVar(value=DEFAULT_TTS_ENABLED)
        self.voice_type_var = tk.StringVar(value=DEFAULT_VOICE_TYPE)

        # --- MAIN FRAME ---
        main_frame = tk.Frame(master, bg="#f0f0f0", padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left and right
        self._create_left_video_column(main_frame)
        self._create_right_customization_column(main_frame)
        self.vcam_manager = VirtualCamManager(self.video_width, self.video_height)

        # Camera initialization
        # Check if cameras exist, fallback to 0 if list is empty or weird
        if self.camera_options:
            try:
                initial_camera_id = int(self.camera_id_var.get().split(":")[0])
            except:
                initial_camera_id = 0
        else:
            initial_camera_id = 0

        self.camera_processor = CameraProcessor(
            self.video_canvas,
            inference_model,
            self.update_log,
            width=self.video_width,
            height=self.video_height,
            camera_id=initial_camera_id
        )

        self.camera_processor.vcam_manager = self.vcam_manager

        # Apply defaults BEFORE camera starts
        self._apply_live_updates()

        master.after(100, self._start_initial_camera)
        self.update_log("GUI initialized. Starting camera...")

        # --- KEYBOARD BINDINGS ---
        self.master.bind("<Escape>", self._keyboard_clear_sentence)
        self.master.bind("<BackSpace>", self._keyboard_backspace)
        self.master.bind("<space>", self._keyboard_toggle_mode)  # NEW
        self.master.bind("c", self._keyboard_clear_sentence)     # NEW (alternative to ESC)
        self.master.bind("C", self._keyboard_clear_sentence)     # NEW (alternative to ESC)

    def _keyboard_toggle_mode(self, event=None):
        """Toggle between modes via spacebar with debounce."""
        if hasattr(self, 'camera_processor'):
            # Debounce: Check if 0.5s has passed since last toggle
            current_time = time.time()
            if hasattr(self, '_last_mode_toggle_time'):
                if current_time - self._last_mode_toggle_time < 0.5:
                    return  # Ignore rapid presses
            
            self._last_mode_toggle_time = current_time
            
            # Toggle the mode
            self.toggle_prediction_mode()
            self.update_log("Mode toggled via SPACEBAR")

    # ----------------------------------------------------------
    # LEFT COLUMN
    # ----------------------------------------------------------
    def _create_left_video_column(self, parent):
        left = tk.Frame(parent, bg="#e0e0e0", padx=10, pady=10)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.video_width = 640
        self.video_height = 360

        self.video_canvas = tk.Canvas(
            left,
            width=self.video_width,
            height=self.video_height,
            bg="black",
            highlightthickness=0
        )
        self.video_canvas.pack(pady=5)

        controls = tk.Frame(left, bg="#e0e0e0")
        controls.pack(fill=tk.X)
        self._create_camera_controls(controls)
        
        # --- NEW: SENTENCE CONTROL BUTTONS ---
        self._create_sentence_controls(left)


    def _create_camera_controls(self, parent):
        tk.Label(parent, text="📷 Select Camera:", bg="#e0e0e0").pack(side=tk.LEFT)

        self.camera_options = self.find_cameras()
        if not self.camera_options:
            self.camera_options = ["0: Default"]
            
        self.camera_id_var = tk.StringVar(value=self.camera_options[0])
        self.camera_id_var.trace_add("write", self._on_camera_change)

        menu = tk.OptionMenu(parent, self.camera_id_var, *self.camera_options)
        menu.config(width=20)
        menu.pack(side=tk.LEFT, padx=10)
        
        # --- MODE TOGGLE BUTTON ---
        self.mode_var = tk.StringVar(value="SPELLING")
        self.mode_btn = tk.Button(
            parent,
            textvariable=self.mode_var,
            command=self.toggle_prediction_mode,
            width=12,
            bg="#9b59b6",  # Purple for Alphabet
            fg="white",
            font=('Helvetica', 10, 'bold'),
            cursor="hand2"
        )
        self.mode_btn.pack(side=tk.LEFT, padx=5)

        # --- START VIRTUAL CAMERA BUTTON ---
        self.start_virtual_btn = tk.Button(
            parent,
            text="START",
            bg="#4CAF50",
            fg="white",
            font=("Helvetica", 10, "bold"),
            padx=15,
            command=self._handle_virtual_start
        )
        self.start_virtual_btn.pack(side=tk.RIGHT, padx=5)

    def _create_sentence_controls(self, parent):
        """New control panel for sentence management."""
        control_frame = tk.LabelFrame(
            parent,
            text="📝 Sentence Controls",
            bg="#e0e0e0",
            padx=10,
            pady=10,
            font=('Helvetica', 10, 'bold')
        )
        control_frame.pack(fill=tk.X, pady=10)

        # Button row
        btn_frame = tk.Frame(control_frame, bg="#e0e0e0")
        btn_frame.pack(fill=tk.X)

        # Clear All Button
        clear_btn = tk.Button(
            btn_frame,
            text="🗑️ Clear All (ESC)",
            command=self._keyboard_clear_sentence,
            bg="#f44336",
            fg="white",
            font=('Helvetica', 9, 'bold'),
            cursor="hand2",
            padx=10,
            pady=5
        )
        clear_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Delete Last Button
        delete_btn = tk.Button(
            btn_frame,
            text="⌫ Delete Last (Backspace)",
            command=self._keyboard_backspace,
            bg="#ff9800",
            fg="white",
            font=('Helvetica', 9, 'bold'),
            cursor="hand2",
            padx=10,
            pady=5
        )
        delete_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Info label
        info_label = tk.Label(
            control_frame,
            text="Tip: Use keyboard shortcuts for quick access",
            bg="#e0e0e0",
            fg="#666",
            font=('Helvetica', 8, 'italic')
        )
        info_label.pack(pady=(5, 0))

    def toggle_prediction_mode(self):
        """Swaps between Alphabet and Word recognition modes."""
        
        # JUST USE FILENAMES - NO PATHS
        # The inference.py will automatically look inside the 'model' folder
        ALPHA_FILE = 'asl_alphabet_model.keras'
        WORD_FILE = 'robust_word_model.keras'  # FIXED: Changed from 'word_model.keras'

        if self.mode_var.get() == "SPELLING":
            self.update_log("Switching to SIGNING mode...")
            self.mode_var.set("SIGNING")
            self.mode_btn.config(bg="#2ecc71")  # Green for Words
            
            # Load Word Model
            self.camera_processor.model_wrapper.load_model(WORD_FILE)
            self.camera_processor.set_engine("word")
            
        else:
            self.update_log("Switching to SPELLING mode...")
            self.mode_var.set("SPELLING")
            self.mode_btn.config(bg="#9b59b6")  # Purple for Alphabet
            
            # Load Alphabet Model
            self.camera_processor.model_wrapper.load_model(ALPHA_FILE)
            self.camera_processor.set_engine("alphabet")


    def _handle_virtual_start(self):
        """Toggle the virtual camera through the manager."""
        new_text = self.vcam_manager.toggle(self.update_log)
        self.start_virtual_btn.config(text=new_text)
        self.start_virtual_btn.config(bg="#f44336" if new_text == "STOP" else "#4CAF50")


    # ----------------------------------------------------------
    # RIGHT COLUMN
    # ----------------------------------------------------------
    def _create_right_customization_column(self, parent):
        right = tk.Frame(parent, bg="#f0f0f0")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._create_text_settings(right).pack(fill=tk.X, pady=10)
        self._create_tts_settings(right).pack(fill=tk.X, pady=10)

        tk.Label(right, text="Application Log:", bg="#f0f0f0").pack(anchor="w")

        self.log_area = scrolledtext.ScrolledText(
            right,
            height=8,
            state=tk.DISABLED,
            bg="white",
            font=("Consolas", 10)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=5)


    # ----------------------------------------------------------
    # TEXT SETTINGS
    # ----------------------------------------------------------
    def _create_text_settings(self, parent):
        frame = tk.LabelFrame(
            parent,
            text="📝 On-Screen Text (Bottom Sentence)",
            bg="#e0e0e0",
            padx=10,
            pady=10
        )

        tk.Checkbutton(
            frame,
            text="Enable Text",
            variable=self.text_enabled_var,
            bg="#e0e0e0",
            command=self._apply_live_updates
        ).pack(anchor="w")

        # Position slider
        pos_frame = tk.Frame(frame, bg="#e0e0e0")
        pos_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            pos_frame,
            text="Position (Bottom ↑ Top):",
            bg="#e0e0e0",
            width=20
        ).pack(side=tk.LEFT)

        tk.Scale(
            pos_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.text_vertical_slider,
            length=200,
            command=self._live_position_update,
            bg="#e0e0e0"
        ).pack(side=tk.LEFT, padx=5)

        # Text size
        size_frame = tk.Frame(frame, bg="#e0e0e0")
        size_frame.pack(fill=tk.X, pady=3)

        tk.Label(size_frame, text="Size:", bg="#e0e0e0", width=10).pack(side=tk.LEFT)

        tk.Spinbox(
            size_frame,
            from_=10,
            to=120,
            textvariable=self.text_size_var,
            command=self._apply_live_updates
        ).pack(side=tk.LEFT)

        # Color
        color_frame = tk.Frame(frame, bg="#e0e0e0")
        color_frame.pack(fill=tk.X, pady=5)

        tk.Label(color_frame, text="Color:", width=10, bg="#e0e0e0").pack(side=tk.LEFT)

        tk.Radiobutton(
            color_frame,
            text="White",
            value="#FFFFFF",
            variable=self.text_color_var,
            bg="#e0e0e0",
            command=self._apply_live_updates
        ).pack(side=tk.LEFT)

        tk.Radiobutton(
            color_frame,
            text="RGB",
            value="RGB",
            variable=self.text_color_var,
            bg="#e0e0e0",
            command=self._choose_color
        ).pack(side=tk.LEFT, padx=10)

        # Effects
        effect_frame = tk.Frame(frame, bg="#e0e0e0")
        effect_frame.pack(fill=tk.X, pady=5)

        tk.Label(effect_frame, text="Style:", width=10, bg="#e0e0e0").pack(side=tk.LEFT)

        for label, val in [("None", "none"), ("Shadow", "shadow"), ("Outline", "outline")]:
            tk.Radiobutton(
                effect_frame,
                text=label,
                value=val,
                variable=self.text_effect_var,
                bg="#e0e0e0",
                command=self._apply_live_updates
            ).pack(side=tk.LEFT)

        return frame


    # ----------------------------------------------------------
    # LIVE UPDATE LOGIC
    # ----------------------------------------------------------
    def _live_position_update(self, _):
        self._apply_live_updates()

    def _choose_color(self):
        self.text_color_var.set("RGB")
        chosen = colorchooser.askcolor()[1]
        if chosen:
            self.text_color_var.set(chosen)
        self._apply_live_updates()

    def _apply_live_updates(self):
        slider = self.text_vertical_slider.get()

        # Clamp effective max at 80
        effective_slider = min(slider, 80)

        bottom_y = self.video_height - 10
        top_y = 60  # tuned visual max (matches your reference)

        y = int(
            bottom_y - (effective_slider / 80) * (bottom_y - top_y)
        )

        if hasattr(self, "camera_processor"):
            self.camera_processor.tts_enabled = self.tts_enabled_var.get()
            self.camera_processor.voice_type = self.voice_type_var.get()
            
            self.camera_processor.apply_custom_text_settings(
                enabled=self.text_enabled_var.get(),
                color=self.text_color_var.get(),
                size=self.text_size_var.get(),
                y=y,
                effect=self.text_effect_var.get()
            )


    # ----------------------------------------------------------
    # TTS SETTINGS
    # ----------------------------------------------------------
    def _create_tts_settings(self, parent):
        frame = tk.LabelFrame(
            parent,
            text="🔊 Text-to-Speech",
            bg="#e0e0e0",
            padx=10,
            pady=10
        )

        tk.Checkbutton(
            frame,
            text="Enable TTS",
            variable=self.tts_enabled_var,
            bg="#e0e0e0",
            command=self._apply_live_updates
        ).pack(anchor="w")

        type_frame = tk.Frame(frame, bg="#e0e0e0")
        type_frame.pack(fill=tk.X, pady=3)

        tk.Label(type_frame, text="Voice:", width=10, bg="#e0e0e0").pack(side=tk.LEFT)

        for label, val in [("Male", "M"), ("Female", "F")]:
            tk.Radiobutton(
                type_frame,
                text=label,
                value=val,
                variable=self.voice_type_var,
                bg="#e0e0e0",
                command=self._apply_live_updates
            ).pack(side=tk.LEFT)

        return frame


    # ----------------------------------------------------------
    # CAMERA CONTROL
    # ----------------------------------------------------------
    def _start_initial_camera(self):
        self.camera_processor.start_feed()

    def _on_camera_change(self, *_):
        try:
            cam_id = int(self.camera_id_var.get().split(":")[0])
            self.camera_processor.stop_feed()
            self.master.after(120, lambda: self._restart_camera(cam_id))
        except:
            pass

    def _restart_camera(self, cam_id):
        self.camera_processor.camera_id = cam_id
        self.camera_processor.start_feed()

    def find_cameras(self):
        cams = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                name = "Webcam" if i == 0 else f"Camera {i}"
                cams.append(f"{i}: {name}")
                cap.release()
        return cams if cams else []


    # ----------------------------------------------------------
    # LOGGING
    # ----------------------------------------------------------
    def update_log(self, msg):
        timestamp = time.strftime("[%H:%M:%S]")
        if hasattr(self, "log_area"):
            self.log_area.config(state=tk.NORMAL)
            self.log_area.insert(tk.END, f"{timestamp} {msg}\n")
            self.log_area.see(tk.END)
            self.log_area.config(state=tk.DISABLED)
        else:
            print(msg)
    
    # ----------------------------------------------------------
    # KEYBOARD SHORTCUTS & SENTENCE MANAGEMENT
    # ----------------------------------------------------------
    def _keyboard_clear_sentence(self, event=None):
        """Clears the entire sentence and logs the action."""
        if hasattr(self, "camera_processor"):
            self.camera_processor.raw_tokens = []  # Clear token list
            self.camera_processor.draft_text = ""  # Clear draft for TTS
            self.update_log("✓ Sentence CLEARED (All tokens removed)")

    def _keyboard_backspace(self, event=None):
        """Deletes the last token (word or letter) from the sentence."""
        if hasattr(self, "camera_processor"):
            if self.camera_processor.raw_tokens:
                deleted = self.camera_processor.raw_tokens.pop()
                
                # Also update draft_text for TTS
                if self.camera_processor.draft_text:
                    words = self.camera_processor.draft_text.split()
                    if words:
                        words.pop()
                        self.camera_processor.draft_text = " ".join(words)
                
                self.update_log(f"✓ Deleted last token: '{deleted}'")
            else:
                self.update_log("⚠ Nothing to delete (sentence is empty)")