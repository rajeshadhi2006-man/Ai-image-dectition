---
<<<<<<< HEAD
title: AI Image Detector
emoji: 🔍
colorFrom: blue
=======
title: Detetor2.0
emoji: 📈
colorFrom: yellow
>>>>>>> fbf60e198b2dd2a2f660077e1d8173bf5a360662
colorTo: purple
sdk: docker
pinned: false
license: mit
---

<<<<<<< HEAD
# AI Image Detector 🔍

An advanced AI-powered image authenticity verification system that uses machine learning to detect AI-generated images vs. authentic photos.

## Features

✨ **Dual Analysis Dashboard**
- Machine Learning model for image classification
- OpenAI Vision API integration for detailed forensic analysis
- Real-time confidence scoring
- Comprehensive scan history

🎨 **Premium UI/UX**
- Modern, responsive design
- Dark/Light theme support
- Smooth animations and transitions
- Interactive data visualizations

🔬 **Advanced Detection**
- Deep learning model trained on AI vs. real images
- Multi-factor analysis
- Detailed forensic reports
- Processing time metrics

## Technology Stack

### Backend
- FastAPI
- TensorFlow/Keras
- OpenAI Vision API
- Python 3.9+

### Frontend
- React + TypeScript
- Vite
- Modern CSS with glassmorphism effects

## Local Development

### Backend Setup
```bash
cd ai-image-detector-api
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app:app --reload
```

### Frontend Setup
```bash
cd ai-image-detector-ui
npm install
npm run dev
```

## Environment Variables

Create a `.env` file in the `ai-image-detector-api` directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Upload an image (PNG, JPG, JPEG, GIF)
2. View ML model prediction with confidence score
3. Review detailed forensic analysis from Vision API
4. Check scan history for past analyses

## Model Information

The ML model is trained to classify images as:
- **AI-Generated**: Created by AI systems (DALL-E, Midjourney, etc.)
- **Authentic**: Real photographs taken by cameras

## License

MIT License - See LICENSE file for details

## Author

Built with ❤️ for image authenticity verification
=======
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> fbf60e198b2dd2a2f660077e1d8173bf5a360662
