# Hugging Face Deployment Guide

## Prerequisites

1. **Create a Hugging Face account**: https://huggingface.co/join
2. **Install Hugging Face CLI**:
   ```bash
   pip install huggingface_hub
   ```
3. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```
   - You'll be prompted to enter your token
   - Get your token from: https://huggingface.co/settings/tokens

## Steps to Deploy

### 1. Initialize Git Repository (if not already done)
```bash
cd "c:\Users\Rajesh s\Downloads\ai image detector"
git init
git add .
git commit -m "Initial commit: AI Image Detector"
```

### 2. Create a Hugging Face Space

Visit: https://huggingface.co/new-space

**Fill in the details:**
- **Space name**: `ai-image-detector` (or your preferred name)
- **License**: MIT
- **SDK**: Docker
- **Visibility**: Public (or Private)

Click "Create Space"

### 3. Add Hugging Face as Remote and Push

After creating the Space, Hugging Face will give you a repository URL like:
`https://huggingface.co/spaces/YOUR_USERNAME/ai-image-detector`

```bash
# Add Hugging Face remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/ai-image-detector

# Push to Hugging Face
git push hf main
```

If your default branch is `master` instead of `main`:
```bash
git branch -M main
git push hf main
```

### 4. Set Environment Variables (Important!)

After pushing, go to your Space settings:
1. Navigate to: `https://huggingface.co/spaces/YOUR_USERNAME/ai-image-detector/settings`
2. Click on "Repository secrets"
3. Add your secrets:
   - Name: `OPENAI_API_KEY`
   - Value: Your OpenAI API key

### 5. Monitor Deployment

- Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/ai-image-detector`
- Watch the "Building" logs
- Once complete, your app will be live!

## Troubleshooting

### Build Fails
- Check the build logs in your Space
- Ensure all dependencies are in `requirements.txt`
- Verify the Dockerfile syntax

### Model Not Found
- Make sure the model file is committed to git
- Check file size limits (Hugging Face has limits)
- If model is large (>10MB), use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.keras"
  git add .gitattributes
  git commit -m "Track model with LFS"
  ```

### Port Issues
- Hugging Face Spaces use port 7860 by default
- The Dockerfile is already configured for this

## Alternative: Using Hugging Face CLI

```bash
# Create and upload space in one command
huggingface-cli repo create ai-image-detector --type space --space_sdk docker

# Add remote and push
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/ai-image-detector
git push hf main
```

## Updating Your Space

To update after making changes:
```bash
git add .
git commit -m "Description of changes"
git push hf main
```

The Space will automatically rebuild and redeploy!
