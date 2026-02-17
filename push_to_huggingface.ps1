# Quick Start: Push to Hugging Face
# Run this script step by step

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI Image Detector - Hugging Face Deploy" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check if logged in
Write-Host "[Step 1] Checking Hugging Face login status..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Run this command to login:" -ForegroundColor Green
Write-Host "  huggingface-cli login" -ForegroundColor White
Write-Host ""
Write-Host "You'll need a Hugging Face token from:" -ForegroundColor Cyan
Write-Host "  https://huggingface.co/settings/tokens" -ForegroundColor White
Write-Host ""

# Step 2: Instructions
Write-Host "[Step 2] Create a Hugging Face Space" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Go to: https://huggingface.co/new-space" -ForegroundColor Cyan
Write-Host "2. Fill in:" -ForegroundColor Cyan
Write-Host "   - Space name: ai-image-detector (or your choice)" -ForegroundColor White
Write-Host "   - License: MIT" -ForegroundColor White
Write-Host "   - SDK: Docker" -ForegroundColor White
Write-Host "   - Visibility: Public" -ForegroundColor White
Write-Host "3. Click 'Create Space'" -ForegroundColor Cyan
Write-Host ""

# Step 3: Add remote and push
Write-Host "[Step 3] After creating the Space, run these commands:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Replace YOUR_USERNAME with your Hugging Face username:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/ai-image-detector" -ForegroundColor White
Write-Host "  git push hf master:main" -ForegroundColor White
Write-Host ""

# Step 4: Environment variables
Write-Host "[Step 4] Set up environment variables (IMPORTANT!)" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Go to your Space settings:" -ForegroundColor Cyan
Write-Host "   https://huggingface.co/spaces/YOUR_USERNAME/ai-image-detector/settings" -ForegroundColor White
Write-Host ""
Write-Host "2. Add Repository secrets:" -ForegroundColor Cyan
Write-Host "   - Name: OPENAI_API_KEY" -ForegroundColor White
Write-Host "   - Value: Your OpenAI API key" -ForegroundColor White
Write-Host ""

# Step 5: Monitor
Write-Host "[Step 5] Monitor your deployment" -ForegroundColor Yellow
Write-Host ""
Write-Host "Your Space will be available at:" -ForegroundColor Cyan
Write-Host "  https://huggingface.co/spaces/YOUR_USERNAME/ai-image-detector" -ForegroundColor White
Write-Host ""
Write-Host "Watch the build logs to ensure everything deploys correctly!" -ForegroundColor Green
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Ready to deploy! Good luck! 🚀" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
