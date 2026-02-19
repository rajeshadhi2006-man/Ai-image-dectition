import os
import io
import time
import sys
import logging
import threading
import numpy as np
from PIL import Image, ExifTags
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
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                scan_history = json.load(f)
            logger.info(f"Loaded {len(scan_history)} history entries.")
        except Exception as e:
            logger.error(f"Error loading history file: {e}")
            scan_history = []
    else:
        scan_history = []

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

@app.get("/")
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Detector API is running. Frontend not found."}

if os.path.exists(STATIC_DIR):
    # Mount everything else in static to /
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

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

def perform_ela_analysis(image: Image.Image, quality: int = 90):
    """
    Error Level Analysis (ELA) detects if an image has been resaved at different compression levels.
    Synthetic or edited regions often show higher error levels.
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resave at target quality
        tmp_buffer = io.BytesIO()
        image.save(tmp_buffer, format='JPEG', quality=quality)
        tmp_buffer.seek(0)
        resaved_image = Image.open(tmp_buffer)
        
        # Calculate pixel difference
        # Use ImageChops or manual numpy for more control
        from PIL import ImageChops
        diff = ImageChops.difference(image, resaved_image)
        
        # Get mean pixel difference (normalized)
        # Note: Higher mean error indicates potential manipulation/synthetic nature
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        
        # Boost contrast for visualization (optional)
        # scale = 255.0 / max_diff
        # diff = ImageEnhance.Brightness(diff).enhance(scale)
        
        # Calculate error metric
        img_array = np.array(diff)
        mean_error = np.mean(img_array)
        
        # Heuristic threshold: higher mean error is more likely to be synthetic/compressed/AI
        # Real high-res photos usually have low error consistency
        # AI images often have uniform pixel distributions that react differently to re-compression
        return float(mean_error), float(max_diff)
    except Exception as e:
        logger.error(f"ELA error: {e}")
        return 0.0, 0.0

def extract_noise_pattern(image: Image.Image):
    """
    Advanced Noise Score Algorithm (User Requested):
    1. noise = image - denoise(image)
    2. V (Variance), H (High-Freq Energy), C (PRNU Correlation), G (Gaussian), F (CFA)
    Final Noise_Score = (0.25*V) + (0.20*H) + (0.25*C) + (0.15*G) + (0.15*F)
    > 0.65 = REAL
    """
    try:
        from scipy import ndimage, stats
        
        # Step 1: Isolate Noise Residual
        img_gray = image.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)
        
        # Denoising to find 'residual'
        denoised = ndimage.median_filter(img_array, size=3)
        noise = img_array - denoised
        
        # Step 2: Metric calculation
        V_val = noise.var()
        
        # High Frequency Energy (normalized sum of squares)
        H_val = np.sum(np.square(noise)) / (noise.size + 1e-6)
        
        # PRNU Correlation (simulated by checking low spatial correlation)
        # Real sensor noise is spatially independent (white), AI has patterns (correlated)
        # We check correlation between neighboring pixels of the residual
        sample_noise = noise.flatten()[:10000]
        c_val = 1.0 - abs(np.corrcoef(sample_noise[:-1], sample_noise[1:])[0, 1])
        
        # Gaussian Score (0 to 1)
        # Real camera noise is typically Gaussian
        _, p_val = stats.normaltest(sample_noise)
        G_val = min(p_val * 5, 1.0) # Map p-value to 0-1 range
        
        # CFA Pattern Score (0 to 1)
        # Check for Bayer grid interpolation residuals (common in real sensors)
        # Simple check for 2x2 grid variance consistency
        if noise.shape[0] > 4 and noise.shape[1] > 4:
            cfa_res = np.abs(noise[0::2, 0::2] - noise[0::2, 1::2]).mean()
            F_val = 0.8 if cfa_res > 0.05 else 0.4
        else:
            F_val = 0.5
            
        # Normalization
        norm_V = min(V_val / 40.0, 1.0)
        norm_H = min(H_val / 80.0, 1.0)
        
        # Final Noise_Score
        noise_score = (0.25 * norm_V) + (0.20 * norm_H) + (0.25 * c_val) + (0.15 * G_val) + (0.15 * F_val)
        
        is_natural_sensor = noise_score > 0.65
        
        return {
            "variance": float(V_val),
            "high_freq_energy": float(H_val),
            "noise_score": float(noise_score),
            "is_natural_sensor": bool(is_natural_sensor),
            "gaussian_fit": float(G_val),
            "cfa_match": float(F_val)
        }
    except Exception as e:
        logger.error(f"Noise Score algorithm error: {e}")
        return None

def analyze_frequency_domain(image: Image.Image):
    """
    Detect AI upscaling 'checkerboard' artifacts in the frequency domain.
    GANs/Diffusion models often leave periodic spikes in high-frequency bands.
    """
    try:
        img_gray = image.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)
        
        # 1. 2D Fast Fourier Transform
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # 2. Detect high-frequency spikes (periodic artifacts)
        # Check corners of the frequency plot for artificial peaks
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Define high-frequency regions (excluding the center)
        # Periodic 'checkerboards' show up as bright dots in high-freq bands
        mask = np.ones((h, w), np.uint8)
        r = min(h, w) // 10
        mask[center_h-r:center_h+r, center_w-r:center_w+r] = 0
        
        high_freq_signal = magnitude_spectrum * mask
        peak_value = np.max(high_freq_signal)
        mean_value = np.mean(high_freq_signal)
        
        # If the peak is significantly higher than the mean, it's a repeating artificial pattern
        ratio = peak_value / (mean_value + 1e-6)
        has_checkerboard = ratio > 3.5 # Heuristic for spike detection
        
        return {
            "peak_ratio": float(ratio),
            "has_checkerboard": bool(has_checkerboard)
        }
    except Exception as e:
        logger.error(f"FFT Analysis error: {e}")
        return None

def analyze_structural_consistency(image: Image.Image):
    """
    Search for 'Structural/Anatomical Inconsistencies'.
    Analyzes local entropy and edge continuity. AI often creates 'blur-pools' 
    or disjointed edges in complex regions (hands, eyes).
    """
    try:
        from scipy.stats import entropy
        img_gray = image.convert('L').resize((128, 128))
        img_array = np.array(img_gray)
        
        # Calculate local entropy variance
        # AI images often have extremely low entropy pockets mixed with high-noise edges
        flat_img = img_array.flatten()
        counts = np.bincount(flat_img, minlength=256)
        global_entropy = entropy(counts + 1e-6)
        
        # Check for 'dead zones' (anatomical discontinuities)
        # We calculate variance of local 8x8 patches
        patch_size = 16
        variances = []
        for i in range(0, 128, patch_size):
            for j in range(0, 128, patch_size):
                patch = img_array[i:i+patch_size, j:j+patch_size]
                variances.append(np.var(patch))
        
        var_coeff = np.std(variances) / (np.mean(variances) + 1e-6)
        
        # High coefficient of variance in local textures is suspicious (inconsistency)
        is_inconsistent = var_coeff > 1.2 or global_entropy < 3.5
        
        return {
            "entropy": float(global_entropy),
            "structural_variance_coeff": float(var_coeff),
            "is_inconsistent": bool(is_inconsistent)
        }
    except Exception as e:
        logger.error(f"Structural analysis error: {e}")
        return None

def extract_metadata(image: Image.Image):
    metadata = {}
    try:
        # 1. Try standard getexif()
        exif = image.getexif()
        if exif:
            for tag, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag, tag)
                if isinstance(value, (bytes, bytearray)) or len(str(value)) > 500:
                    continue
                metadata[str(tag_name)] = str(value)
        
        # 2. Check image.info for extra metadata (common in PNG/WebP)
        for key, value in image.info.items():
            if key in ['exif', 'icc_profile', 'photoshop', 'xmp']:
                continue # Skip binary blobs
            if len(str(value)) < 500:
                metadata[str(key)] = str(value)
                
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}")
    
    # Analyze metadata presence
    has_metadata = len(metadata) > 0
    
    # Generic fields list (not used for authenticity check anymore but kept for reference)
    generic_fields = ['ExifOffset', 'Orientation', 'ColorSpace']
    
    # Required Fields that prove authenticity per user request
    important_fields = [
        'ResolutionUnit',
        'YResolution',
        'XResolution',
        'YCbCrPositioning',
        'Make',
        'Model',
        'Software',
        'DateTime',
        'DateTimeOriginal',
        'GPSInfo'
    ]
    
    has_camera_info = any(field in metadata for field in important_fields)
    
    # DIGITAL FOOTPRINT ANALYSIS
    software_ref = metadata.get('Software', '').lower()
    is_edited = any(s in software_ref for s in ['adobe', 'photoshop', 'gimp', 'canva', 'framer'])
    
    # Determine metadata/footprint verdict
    if has_camera_info:
        if is_edited:
            metadata_verdict = "Edited Footprint"
            metadata_confidence = 0.75
            metadata_is_ai = False
        else:
            metadata_verdict = "Physical Footprint"
            metadata_confidence = 0.98
            metadata_is_ai = False
    else:
        # User: Missing EXIF is only AI if zero footprint exists
        if has_metadata:
            metadata_verdict = "Anomalous Footprint"
            metadata_confidence = 0.60
            metadata_is_ai = True
        else:
            metadata_verdict = "Zero Footprint"
            metadata_confidence = 0.85 # Strong AI indicator
            metadata_is_ai = True
    
    return {
        "data": metadata,
        "verdict": metadata_verdict,
        "confidence": metadata_confidence,
        "is_ai_generated": metadata_is_ai,
        "has_metadata": has_metadata,
        "has_camera_info": has_camera_info,
        "is_edited": is_edited
    }

HF_API_URL = "https://rajesh9698-ai-image-detector-api.hf.space/predict"
IS_ON_HF = os.environ.get('SPACE_ID') is not None or os.environ.get('HF_HUB_OFFLINE') is not None

def predict_with_hf_api(image_bytes, filename="image.jpg"):
    """
    Call the User's Custom Hosted API on Hugging Face Spaces.
    URL: https://rajesh9698-ai-image-detector-api.hf.space/predict
    """
    try:
        import requests
        # The user's API expects a multipart/form-data upload with key 'file'
        files = {'file': (filename, image_bytes, 'image/jpeg')}
        
        # We don't need the HF_API_KEY for this public space unless it's protected.
        # If it is protected, we'd add the bearer token. 
        # For now, we try public access or standard Space access.
        response = requests.post(HF_API_URL, files=files, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"Remote Node Error ({response.status_code}): {response.text}")
            return None
            
        # Parse the custom API response (matches our local schema)
        try:
            result = response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON from remote node: {e}")
            logger.error(f"Response snippet: {response.text[:200]}")
            return None
        
        # Check if we have the specific ml_analysis block we need
        if 'ml_analysis' in result:
            ml_data = result['ml_analysis']
            return {
                "is_ai": ml_data.get('is_ai_generated', False),
                "confidence": ml_data.get('confidence', 0.0),
                "score": ml_data.get('raw_score', 0.0)
            }
        
        # Fallback: Use the top-level result if ml_analysis is missing
        return {
            "is_ai": result.get('is_ai_generated', False),
            "confidence": result.get('confidence', 0.0),
            "score": result.get('confidence', 0.0) if result.get('is_ai_generated') else (1.0 - result.get('confidence', 0.0))
        }

    except Exception as e:
        logger.error(f"Remote Node Exception: {e}")
        return None

def predict_is_ai_generated(image: Image.Image, image_bytes: bytes = None):
    # 1. Try Remote Node first (User's HF Space)
    # ONLY if we are NOT already running on HF Spaces to avoid infinite recursion
    if image_bytes and not IS_ON_HF:
        remote_result = predict_with_hf_api(image_bytes)
        if remote_result:
            logger.info(f"Remote ML Prediction: {remote_result}")
            return remote_result['is_ai'], remote_result['confidence'], remote_result['score']
            
    # 2. Fallback to Local Model
    if MODEL:
        try:
            processed_img = preprocess_image(image)
            prediction = MODEL.predict(processed_img, verbose=0)
            score = float(prediction[0][0]) 
            logger.info(f"Local ML Prediction Raw Score: {score}")
            is_ai = score > 0.5
            confidence = score if is_ai else (1.0 - score)
            return is_ai, confidence, score
        except Exception as e:
            import traceback
            logger.error(f"Prediction error: {e}")
            logger.error(traceback.format_exc())
            return False, 0.0, 0.0
    else:
        # Fallback if model not loaded
        return False, 0.5, 0.0

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
            
        # Open image for analysis
        image = Image.open(io.BytesIO(contents))
        
        # 1. Forensic Analysis (Deep Analysis)
        ela_mean, ela_max = perform_ela_analysis(image)
        noise_profile = extract_noise_pattern(image)
        fft_analysis = analyze_frequency_domain(image)
        struct_analysis = analyze_structural_consistency(image)
        
        # --- NEW THREE-BRANCH FORENSIC SCORING (User Requested) ---
        
        # BRANCH 1: Domain Integrity (ELA + FFT checkerboard detection)
        domain_score = 100.0
        if ela_mean > 1.2: domain_score -= 15
        if ela_mean > 2.0: domain_score -= 20
        if fft_analysis and fft_analysis.get('has_checkerboard'): domain_score -= 50
        domain_score = max(0.0, domain_score)
        
        # BRANCH 2: Sensor Authenticity (Noise Pattern Correlation)
        sensor_score = 0.0
        if noise_profile:
            sensor_score = noise_profile.get('noise_score', 0.0) * 100.0
        sensor_score = min(max(sensor_score, 0.0), 100.0)
            
        # BRANCH 3: Structural Consistency (Local entropy & Anatomical cues)
        structure_score = 100.0
        if struct_analysis:
            if struct_analysis.get('is_inconsistent'): structure_score -= 60
            ent = struct_analysis.get('entropy', 5.0)
            if ent < 3.8: structure_score -= 20
            if ent < 3.0: structure_score -= 20
        structure_score = max(0.0, structure_score)
        
        # FINAL FORENSIC INTEGRITY SCORE (Weighted Ensemble)
        # Weights: Domain (40%), Sensor (30%), Structure (30%)
        final_forensic_integrity = (domain_score * 0.4) + (sensor_score * 0.3) + (structure_score * 0.3)
        
        # Convert to probability for internal logic
        forensic_ai_probability = 1.0 - (final_forensic_integrity / 100.0)
        
        # 2. Extract metadata
        raw_metadata = extract_metadata(Image.open(io.BytesIO(contents)))
        
        # 3. Get ML prediction
        ml_is_ai, ml_confidence, raw_score = predict_is_ai_generated(image.convert('RGB'), image_bytes=contents)
        
        # 4. Hybrid Decision Logic (High Accuracy Ensemble)
        # Final combined AI score based on ML + Forensic signals
        adjusted_ai_score = (raw_score * 0.4) + (forensic_ai_probability * 0.6)
        
        # Threshold checks for manual boosts
        checkerboard_alert = fft_analysis and fft_analysis.get('has_checkerboard', False)
        if checkerboard_alert or (sensor_score < 40):
            adjusted_ai_score = min(adjusted_ai_score + 0.3, 1.0)
        
        if sensor_score > 80 and not checkerboard_alert:
            adjusted_ai_score = max(adjusted_ai_score - 0.2, 0.0)

        metadata_is_ai = raw_metadata.get('is_ai_generated', True)
        metadata_confidence = raw_metadata.get('confidence', 0.0)
        
        # USER REQUEST: Footprint_Score Logic
        # (metadata_score + source_score + file_structure_score + sensor_score + ai_artifact_score)
        
        # 1. Metadata Score (0-20)
        # 20 = Rich Camera Info, 10 = Basic Metadata, 0 = None/Stripped
        md_score = 0
        if raw_metadata.get('has_camera_info'): md_score = 20
        elif raw_metadata.get('has_metadata'): md_score = 10
        
        # 2. Source Score (0-20)
        # 20 = Clean/Firmware, 0 = Edited Software Signature
        src_score = 0
        if not raw_metadata.get('is_edited', False): src_score = 20
        
        # 3. File Structure Score (0-20)
        # 20 = Standard JPEG/TIFF, 10 = PNG/WebP (Common for AI exports), 5 = Other
        fs_score = 10
        if filename.lower().endswith(('.jpg', '.jpeg', '.tiff', '.dng')): fs_score = 20
        
        # 4. Sensor Score (0-20)
        # Derived from Noise Analysis (is_natural_sensor)
        # If noise_score > 0.65 it was "Real", so map that roughly
        ns_score = 0
        if noise_profile:
             # Map noise_score (0.0-1.0) to 0-20
             raw_ns = noise_profile.get('noise_score', 0.0)
             ns_score = min(raw_ns * 20, 20)
        
        # 5. AI Artifact Score (0-20) 
        # 20 = Clean, 0 = Artifacts Found
        fft_hit = fft_analysis.get('has_checkerboard', False) if fft_analysis else False
        struct_hit = struct_analysis.get('is_inconsistent', False) if struct_analysis else False
        art_score = 20
        if fft_hit: art_score -= 10
        if struct_hit: art_score -= 10
        art_score = max(art_score, 0)
        
        footprint_score = md_score + src_score + fs_score + ns_score + art_score
        # Threshold > 50 => REAL
        is_footprint_real = footprint_score > 50
        
        # Update Metadata Verdict based on deep footprint score
        raw_metadata['footprint_score'] = float(footprint_score)
        if is_footprint_real:
            raw_metadata['verdict'] = "Physical Footprint"
            raw_metadata['confidence'] = min(footprint_score / 100.0 + 0.2, 0.99)
            raw_metadata['is_ai_generated'] = False
        else:
            raw_metadata['verdict'] = "Digital/Artificial Footprint"
            raw_metadata['confidence'] = min((100 - footprint_score) / 100.0 + 0.2, 0.98) 
            raw_metadata['is_ai_generated'] = True
        
        # 4. Backend Simplification (User Requested)
        # Bypassing the Hybrid Decision Logic / Ensemble Engine.
        # The top-level verdict is now just the ML prediction.
        final_is_ai = ml_is_ai
        final_confidence = ml_confidence

        base_url = str(request.base_url).rstrip('/')
        image_url = f"{base_url}/uploads/{unique_filename}"
        
        # GLOBAL AUTHENTICITY SCORE (100 = Authentic, 0 = AI-Generated)
        # This combines all signals: ML, Forensics, and Metadata
        # Invert the AI score to get Authenticity
        global_score = (1.0 - adjusted_ai_score) * 100.0
        
        result = {
            "filename": filename,
            "is_ai_generated": bool(final_is_ai),
            "confidence": float(final_confidence),
            "global_score": float(global_score),
            "prediction_label": "AI GENERATED" if final_is_ai else "REAL IMAGE",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_url": image_url,
            "metadata": {
                "has_metadata": raw_metadata.get('has_metadata', False),
                "verdict": raw_metadata.get('verdict', "No Footprint Available"),
                "confidence": float(raw_metadata.get('confidence', 0.0)),
                "is_ai_generated": bool(raw_metadata.get('is_ai_generated', False)),
                "has_camera_info": bool(raw_metadata.get('has_camera_info', False)),
                "is_edited": bool(raw_metadata.get('is_edited', False)),
                "footprint_score": float(raw_metadata.get('footprint_score', 0.0)),
                "data": raw_metadata.get('data', {})
            },
            "forensics": {
                "branch_scores": {
                    "domain": float(domain_score),
                    "sensor": float(sensor_score),
                    "structure": float(structure_score)
                },
                "final_integrity": float(final_forensic_integrity),
                "ela_score": float(ela_mean),
                "noise_profile": {
                    "noise_score": float(noise_profile.get('noise_score', 0.0)) if noise_profile else 0.0,
                    "is_natural_sensor": bool(noise_profile.get('is_natural_sensor', False)) if noise_profile else False,
                    "cfa_match": float(noise_profile.get('cfa_match', 0.0)) if noise_profile else 0.0,
                    "variance": float(noise_profile.get('variance', 0.0)) if noise_profile else 0.0,
                    "high_freq_energy": float(noise_profile.get('high_freq_energy', 0.0)) if noise_profile else 0.0,
                    "gaussian_fit": float(noise_profile.get('gaussian_fit', 0.0)) if noise_profile else 0.0
                },
                "fft_analysis": {
                    "peak_ratio": float(fft_analysis.get('peak_ratio', 0.0)) if fft_analysis else 0.0,
                    "has_checkerboard": bool(fft_analysis.get('has_checkerboard', False)) if fft_analysis else False
                },
                "structural_analysis": {
                    "entropy": float(struct_analysis.get('entropy', 0.0)) if struct_analysis else 0.0,
                    "is_inconsistent": bool(struct_analysis.get('is_inconsistent', False)) if struct_analysis else False,
                    "structural_variance_coeff": float(struct_analysis.get('structural_variance_coeff', 0.0)) if struct_analysis else 0.0
                },
                "forensic_probability": float(forensic_ai_probability),
                "sensor_match": sensor_score > 50
            },
            "ml_analysis": {
                "is_ai_generated": bool(ml_is_ai),
                "confidence": float(ml_confidence or 0.0),
                "raw_score": float(raw_score or 0.0)
            }
        }
        
        with history_lock:
            scan_history.append(result)
            save_history()
            
        return result
    except Exception as e:
        import traceback
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Deep analysis failed: {str(e)}")
