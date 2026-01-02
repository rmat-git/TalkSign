import tkinter as tk
from tkinter import scrolledtext, colorchooser
from app.camera import CameraProcessor
from app.virtual_cam import VirtualCamManager
import time
import cv2


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
        initial_camera_id = int(self.camera_id_var.get().split(":")[0])
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

        # --- INSERT THESE KEYBOARD BINDINGS ---
        self.master.bind("<Escape>", self._keyboard_clear_sentence)
        self.master.bind("<BackSpace>", self._keyboard_backspace)
        # --------------------------------------

        master.after(100, self._start_initial_camera)


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


    def _create_camera_controls(self, parent):
        tk.Label(parent, text="📷 Select Camera:", bg="#e0e0e0").pack(side=tk.LEFT)

        self.camera_options = self.find_cameras()
        self.camera_id_var = tk.StringVar(value=self.camera_options[0])
        self.camera_id_var.trace_add("write", self._on_camera_change)

        menu = tk.OptionMenu(parent, self.camera_id_var, *self.camera_options)
        menu.config(width=20)
        menu.pack(side=tk.LEFT, padx=10)

        # --- NEW START BUTTON (FAR RIGHT) ---
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
            text="Position (Bottom ↔ Top):",
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
        return cams if cams else ["0: Default Camera"]


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
    # KEYBOARD SHORTCUTS
    # ----------------------------------------------------------
    def _keyboard_clear_sentence(self, event=None):
        """Clears the entire sentence and logs the action."""
        if hasattr(self, "camera_processor"):
            self.camera_processor.sentence = ""
            self.update_log("Sentence CLEARED via Keyboard (ESC)")

    def _keyboard_backspace(self, event=None):
        """Deletes the last character from the sentence."""
        if hasattr(self, "camera_processor"):
            if self.camera_processor.sentence:
                self.camera_processor.sentence = self.camera_processor.sentence[:-1]
                self.update_log("Last character DELETED via Keyboard (Backspace)")