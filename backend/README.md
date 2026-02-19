python -m uvicorn app:app --reload --port 8000# AI Image Detector API

This is a FastAPI-based API for detecting AI-generated images using a TensorFlow model.

## Project Structure

- `app.py`: Main application file containing the FastAPI endpoints and model logic.
- `model/`: Directory containing the trained ML model (`ai_image_detector_final.keras`).
- `uploads/`: Directory for temporary image processing.
- `requirements.txt`: Python dependencies (FastAPI, TensorFlow-CPU, etc.).
- `render.yaml`: Configuration for deployment on Render.
- `runtime.txt`: Specifies the Python runtime version.

## Setup

1.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    python app.py
    ```

The API will be available at `http://localhost:8000`. You can also access the interactive documentation at `http://localhost:8000/docs`.

## API Endpoints

### 1. Root
- **URL**: `/`
- **Method**: `GET`
- **Response**: Status and basic info.

### 2. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy",
    "service": "AI Image Detector API"
  }
  ```

### 3. Predict Image
- **URL**: `/predict`
- **Method**: `POST`
- **Body**: `multipart/form-data` with a key `file` containing the image.
- **Response**:
  ```json
  {
    "filename": "image.jpg",
    "is_ai_generated": true,
    "confidence": 0.92,
    "prediction_label": "AI-Generated",
    "timestamp": 1707923456.789
  }
  ```

## Deployment

The project is configured for deployment on **Render** (via `render.yaml`). It uses `tensorflow-cpu` to keep the deployment size manageable and compatible with CPU environments.

## Model Logic

The API uses a Keras model located at `model/ai_image_detector_final.keras`. It expects input images to be resized to 224x224 and normalized.

