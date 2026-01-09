import tkinter as tk
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.inference import InferenceModel
from app.gui import AppGUI

def null_logger(message):
    pass

def main():
    """Main function to initialize the application."""
    
    root = tk.Tk()
    
    model_instance = InferenceModel(null_logger)
    
    gui_instance = AppGUI(root, model_instance, null_logger) 
    
    real_log_callback = gui_instance.update_log
    
    model_instance.log = real_log_callback
    gui_instance.camera_processor.log = real_log_callback 
    gui_instance.log = real_log_callback 
    
    real_log_callback("Switching to real application log.")
    model_instance.load_model()
    
    root.mainloop()

if __name__ == '__main__':
    main()