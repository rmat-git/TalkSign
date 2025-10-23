# --- cd "C:\Users\mat-u\Documents\TalkSign_Thesis\app" ---
# --- py v2.py ---
# hello world, sample text is placed here. ts pmo, sybau

#check test

import sys
import cv2
import pyvirtualcam
import pyttsx3
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLineEdit, QCheckBox, QSlider
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter


class TalkSignApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TalkSign")

        self.webcam_width = 640
        self.webcam_height = 360
        self.setFixedSize(1100, self.webcam_height + 250)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Defaults for customization
        self.overlay_text = ""
        self.font_color = (0, 255, 255)  # Yellow default
        self.bg_enabled = False
        self.bg_alpha = 0.5
        self.font_scale = 1.35

        # Left Layout (Webcam + Title + Selector + Virtual Cam Button)
        left_layout = QVBoxLayout()
        webcam_title = QLabel("TalkSign")
        webcam_title.setFont(QFont("Arial", 24, QFont.Bold))
        left_layout.addWidget(webcam_title)

        self.image_label = QLabel()
        self.image_label.setFixedSize(self.webcam_width, self.webcam_height)
        self.image_label.setStyleSheet("background-color: black; border: 1px solid #ccc;")
        self.image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.image_label)

        # Overlay Text Input
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter overlay text (e.g. ASL translation)")
        self.text_input.textChanged.connect(self.update_overlay_text)
        left_layout.addWidget(self.text_input)

        select_layout = QHBoxLayout()
        select_label = QLabel("Select Webcam")
        select_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.cam_select = QComboBox()
        select_layout.addWidget(select_label)
        select_layout.addWidget(self.cam_select)
        select_layout.addStretch()
        left_layout.addLayout(select_layout)

        # Virtual camera toggle button
        self.vcam_button = QPushButton("Start Virtual Camera")
        self.vcam_button.setCheckable(True)
        self.vcam_button.clicked.connect(self.toggle_virtual_camera)
        left_layout.addWidget(self.vcam_button)

        left_layout.addStretch()

        # Right Layout (Customization controls)
        right_layout = QVBoxLayout()
        customization_label = QLabel("Customization")
        customization_label.setFont(QFont("Arial", 14, QFont.Bold))
        customization_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(customization_label)

        # Background toggle
        self.bg_toggle = QCheckBox("Background")
        self.bg_toggle.stateChanged.connect(self.toggle_background)
        right_layout.addWidget(self.bg_toggle)

        # Transparency slider
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setMinimum(0)
        self.transparency_slider.setMaximum(100)
        self.transparency_slider.setValue(int(self.bg_alpha * 100))
        self.transparency_slider.valueChanged.connect(self.change_transparency)
        right_layout.addWidget(QLabel("Transparency"))
        right_layout.addWidget(self.transparency_slider)

        # Font size slider
        self.font_slider = QSlider(Qt.Horizontal)
        self.font_slider.setMinimum(1)
        self.font_slider.setMaximum(5)
        self.font_slider.setValue(int(self.font_scale))
        self.font_slider.valueChanged.connect(self.change_font_size)
        right_layout.addWidget(QLabel("Size"))
        right_layout.addWidget(self.font_slider)

        # Font color selection
        self.color_box = QComboBox()
        self.color_box.addItems(["Red", "Green", "Yellow"])
        self.color_box.currentTextChanged.connect(self.change_color)
        right_layout.addWidget(QLabel("Color"))
        right_layout.addWidget(self.color_box)

        # --- TTS Setup ---
        self.tts_engine = pyttsx3.init()
        self.voice_map = {}
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            self.voice_map[voice.name] = voice.id

        # Audio Output Selector
        self.voice_box = QComboBox()
        self.voice_box.addItems([voice.name for voice in voices])
        self.voice_box.currentTextChanged.connect(self.change_tts_voice)
        right_layout.addWidget(QLabel("TTS Voice"))
        right_layout.addWidget(self.voice_box)

        # TTS Test Button
        self.tts_test_button = QPushButton("Test TTS")
        self.tts_test_button.clicked.connect(self.test_tts)
        right_layout.addWidget(self.tts_test_button)

        right_layout.addStretch()

        # Combine Layouts
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout, 1)
        central_widget.setLayout(main_layout)

        # Virtual camera setup
        self.vcam = None
        self.vcam_active = False

        # Camera setup
        self.available_cams = self.scan_cameras()
        for idx, name in self.available_cams:
            self.cam_select.addItem(name, userData=idx)

        initial_cam_index = self.available_cams[0][0] if self.available_cams else 0
        self.cap = cv2.VideoCapture(initial_cam_index)
        self.set_camera_resolution(1280, 720)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.cam_select.currentIndexChanged.connect(self.change_camera)

    # ---------------- Camera Methods ----------------
    def scan_cameras(self):
        cams = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap and cap.isOpened():
                cams.append((i, f"Camera {i}"))
                cap.release()
        if not cams:
            cams.append((0, "Default Webcam (0)"))
        return cams

    def set_camera_resolution(self, width, height):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.actual_camera_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_camera_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera resolution: {self.actual_camera_width}x{self.actual_camera_height}")

    def change_camera(self, index):
        cam_index = self.cam_select.itemData(index)
        if cam_index is None:
            cam_index = 0
        self.cap.release()
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            print(f"[WARNING] Cannot open camera index {cam_index}, falling back to 0")
            self.cap = cv2.VideoCapture(0)
            self.cam_select.blockSignals(True)
            self.cam_select.setCurrentIndex(0)
            self.cam_select.blockSignals(False)
        self.set_camera_resolution(1280, 720)

    def toggle_virtual_camera(self):
        if self.vcam_active:
            self.vcam_active = False
            self.vcam_button.setText("Start Virtual Camera")
            if self.vcam:
                self.vcam.close()
                self.vcam = None
            print("[INFO] Virtual camera stopped.")
        else:
            try:
                self.vcam = pyvirtualcam.Camera(
                    width=self.actual_camera_width,
                    height=self.actual_camera_height,
                    fps=30,
                    fmt=pyvirtualcam.PixelFormat.BGR,
                    device="OBS Virtual Camera"
                )
                self.vcam_active = True
                self.vcam_button.setText("Stop Virtual Camera")
                print("[INFO] Virtual camera started.")
            except Exception as e:
                print(f"[ERROR] Failed to start virtual camera: {e}")
                self.vcam_active = False
                self.vcam_button.setChecked(False)

    # ---------------- Overlay Methods ----------------
    def update_overlay_text(self, text):
        self.overlay_text = text

    def change_color(self, color):
        if color == "Red":
            self.font_color = (0, 0, 255)
        elif color == "Green":
            self.font_color = (0, 255, 0)
        elif color == "Yellow":
            self.font_color = (0, 255, 255)

    def toggle_background(self, state):
        self.bg_enabled = state == Qt.Checked

    def change_transparency(self, value):
        self.bg_alpha = value / 100.0

    def change_font_size(self, value):
        self.font_scale = float(value)

    def apply_overlay(self, frame):
        """Apply text + background overlay to a frame"""
        if self.overlay_text:
            font = cv2.FONT_HERSHEY_DUPLEX
            font_thickness = 3
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, font, self.font_scale, font_thickness)
            text_x = (frame.shape[1] - text_width) // 2
            text_y = frame.shape[0] - 40

            if self.bg_enabled:
                overlay = frame.copy()
                cv2.rectangle(overlay,
                              (text_x - 10, text_y - text_height - 10),
                              (text_x + text_width + 10, text_y + 10),
                              (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, self.bg_alpha, frame, 1 - self.bg_alpha, 0)

            cv2.putText(frame, self.overlay_text, (text_x, text_y), font,
                        self.font_scale, self.font_color, font_thickness, cv2.LINE_AA)
        return frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Apply overlay to both preview and vcam
            preview_frame = self.apply_overlay(frame.copy())
            vcam_frame = self.apply_overlay(frame.copy())

            # Send to virtual camera (mirrored)
            if self.vcam_active and self.vcam:
                try:
                    self.vcam.send(cv2.flip(vcam_frame, 1))
                    self.vcam.sleep_until_next_frame()
                except Exception as e:
                    print(f"[ERROR] Virtual camera frame send failed: {e}")
                    self.vcam_active = False
                    self.vcam_button.setChecked(False)

            # Show preview in Qt
            rgb_image = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            label_width = self.image_label.width()
            label_height = self.image_label.height()
            scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            background = QPixmap(label_width, label_height)
            background.fill(Qt.black)
            painter = QPainter(background)
            x_offset = (label_width - scaled_pixmap.width()) // 2
            y_offset = (label_height - scaled_pixmap.height()) // 2
            painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
            painter.end()
            self.image_label.setPixmap(background)

    # ---------------- TTS Methods ----------------
    def change_tts_voice(self, voice_name):
        """Set selected TTS voice"""
        if voice_name in self.voice_map:
            self.tts_engine.setProperty('voice', self.voice_map[voice_name])

    def test_tts(self):
        """Play TTS for current overlay text or sample text"""
        text = self.overlay_text if self.overlay_text else "Hello, this is a TTS test."
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[ERROR] TTS failed: {e}")

    # ---------------- Cleanup ----------------
    def closeEvent(self, event):
        self.cap.release()
        if self.vcam:
            self.vcam.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TalkSignApp()
    window.show()
    sys.exit(app.exec_())
