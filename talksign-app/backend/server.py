import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

# Load environment variables from .env file at the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    print(f"System: Loading environment variables from {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    print("System: .env file not found. Relying on system environment variables.")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.inference import InferenceModel
    from camera_service import CameraService
except ImportError as e:
    print(f"\nCRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("System: Backend Starting...")
    try:
        camera_service.start_camera(0)
    except Exception as e:
        print(f"Error opening default camera: {e}")
    
    yield
    
    # Code to run on shutdown
    print("System: Shutting down...")
    camera_service.stop_camera()
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("System: Initializing AI Models...")
inference_model = InferenceModel()
inference_model.load_model('asl_alphabet_model.keras') 

camera_service = CameraService(inference_model)

class CameraSwitchRequest(BaseModel):
    camera_id: Optional[int] = None

class SettingsRequest(BaseModel):
    ttsEnabled: Optional[bool] = None
    voiceType: Optional[str] = None
    textEnabled: Optional[bool] = None
    textSize: Optional[int] = None
    textColor: Optional[str] = None
    textPosition: Optional[int] = None

# --- ROUTES ---
@app.get("/cameras")
def get_cameras():
    devices = camera_service.get_available_cameras()
    return JSONResponse(content=devices)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        camera_service.generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/status")
def get_status():
    # Thread-safe read optional but good practice
    return JSONResponse(content=camera_service.status)

@app.post("/control/set_mode/{mode}")
def set_mode(mode: str):
    camera_service.set_mode(mode)
    return {"status": "success", "mode": mode}

@app.post("/control/clear_sentence")
def clear_sentence():
    # CALLING THE NEW METHOD
    camera_service.clear_input()
    return {"status": "cleared"}

@app.post("/control/switch_camera")
def switch_camera(request: CameraSwitchRequest):
    new_id = camera_service.trigger_camera_switch(request.camera_id)
    return {"status": "switched", "new_camera_id": new_id}

@app.post("/control/toggle_vcam")
def toggle_vcam():
    is_enabled = camera_service.toggle_vcam()
    return {"status": "success", "vcam_enabled": is_enabled}

@app.post("/control/backspace")
def backspace():
    # CALLING THE NEW METHOD
    camera_service.process_backspace()
    return {"status": "success", "new_sentence": camera_service.status["sentence"]}

@app.post("/control/trigger_gemini")
def trigger_gemini():
    success = camera_service.trigger_gemini()
    return {"status": "triggered" if success else "empty"}

@app.post("/control/update_settings")
def update_settings(settings: SettingsRequest):
    if settings.ttsEnabled is not None:
        camera_service.tts_enabled = settings.ttsEnabled
    if settings.voiceType is not None:
        camera_service.voice_type = settings.voiceType
    
    if settings.textEnabled is not None:
        camera_service.text_enabled = settings.textEnabled
    if settings.textSize is not None:
        camera_service.text_size = settings.textSize
    if settings.textColor is not None:
        camera_service.text_color = settings.textColor
    if settings.textPosition is not None:
        camera_service.text_position = settings.textPosition
        
    return {"status": "updated"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)