import os
import io
import time
import sys
import logging
import threading
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("ai-detector-api")

import json

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'ai_image_detector_final.keras')

# Load environment variables
load_dotenv()

# Persistent history storage
scan_history = []
history_lock = threading.Lock()

def load_history():
    global scan_history
    # Clear history to avoid format conflicts
    scan_history = []
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            logger.info("Cleared old history file.")
        except:
            pass

def save_history():
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(scan_history[-100:], f)
    except Exception as e:
        logger.error(f"Error saving history file: {e}")

load_history()

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="AI Image Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

# Mount static frontend files if they exist
STATIC_DIR = os.path.join(BASE_DIR, 'static')
if os.path.exists(STATIC_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(STATIC_DIR, 'assets')), name="assets")

import tensorflow as tf

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    return None

MODEL = load_model()

def preprocess_image(image: Image.Image):
    image = image.resize((128, 128)) 
    img_array = np.array(image)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_is_ai_generated(image: Image.Image):
    if MODEL:
        try:
            processed_img = preprocess_image(image)
            prediction = MODEL.predict(processed_img, verbose=0)
            score = float(prediction[0][0]) 
            logger.info(f"ML Prediction Raw Score: {score}")
            is_ai = score > 0.5
            confidence = score if is_ai else (1.0 - score)
            return is_ai, confidence, score
        except Exception as e:
            import traceback
            logger.error(f"Prediction error: {e}")
            logger.error(traceback.format_exc())
            return False, 0.0, 0.0
    else:
        width, height = image.size
        is_ai = (width == 1024 and height == 1024)
        return is_ai, 0.5, 0.5 if is_ai else 0.0

@app.get("/")
async def serve_frontend():
    """Serve the frontend React app"""
    static_dir = os.path.join(BASE_DIR, 'static')
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "online", "message": "AI Image Detector API is running (FastAPI). Use /predict to analyze images.", "docs_url": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/history")
def get_history():
    with history_lock:
        return scan_history[-10:]

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    client_id: str = Form("anonymous")
):
    filename = file.filename
    try:
        contents = await file.read()
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        is_ai, confidence, raw_score = predict_is_ai_generated(image)
        
        base_url = str(request.base_url).rstrip('/')
        image_url = f"{base_url}/uploads/{unique_filename}"
        
        result = {
            "filename": filename,
            "is_ai_generated": bool(is_ai),
            "confidence": float(confidence),
            "prediction_label": "AI-Generated" if is_ai else "Authentic",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_url": image_url
        }
        with history_lock:
            scan_history.append(result)
            save_history()
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
