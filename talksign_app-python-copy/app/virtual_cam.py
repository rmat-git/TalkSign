import pyvirtualcam
import numpy as np

class VirtualCamManager:
    """Handles streaming the processed application feed to a virtual camera device."""
    def __init__(self, width=640, height=360, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = None
        self.is_active = False

    def toggle(self, log_callback):
        """Starts or stops the virtual camera device."""
        if not self.is_active:
            try:
                # Initialize the virtual camera device
                self.device = pyvirtualcam.Camera(
                    width=self.width, 
                    height=self.height, 
                    fps=self.fps,
                    fmt=pyvirtualcam.PixelFormat.RGB
                )
                self.is_active = True
                log_callback(f"Virtual Camera Started: {self.device.device}")
                return "STOP"  # Return new button text
            except Exception as e:
                log_callback(f"Virtual Cam Error: {e}")
                return "START"
        else:
            self.stop()
            log_callback("Virtual Camera Stopped.")
            return "START"

    def stop(self):
        """Gracefully shuts down the virtual camera."""
        self.is_active = False
        if self.device:
            self.device.close()
            self.device = None

    def send_frame(self, frame_rgb):
        """Sends an RGB frame to the virtual camera output."""
        if self.is_active and self.device:
            # Frame must be RGB and match the initialized width/height
            self.device.send(frame_rgb)
            self.device.sleep_until_next_frame()