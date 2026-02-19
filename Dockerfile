# Multi-stage build for AI Image Detector on Hugging Face

FROM node:18-slim AS frontend-builder

# Build frontend
WORKDIR /app/frontend
COPY ai-image-detector-ui/package*.json ./
RUN npm install
COPY ai-image-detector-ui/ ./
RUN npm run build

# Python backend stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY backend/ ./

# Copy built frontend to static directory
COPY --from=frontend-builder /app/frontend/dist ./static

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create uploads directory and history file if needed
RUN mkdir -p uploads && touch history.json && chmod 777 history.json uploads

# Expose port (Hugging Face uses 7860 by default)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
