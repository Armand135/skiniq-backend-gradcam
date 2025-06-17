SkinIQ Backend with Grad-CAM (Render Deployment)

1️⃣ Replace MODEL_URL inside main.py with your Hugging Face model URL.

2️⃣ Push to GitHub.

3️⃣ Render Web Service:
- Build command: pip install -r requirements.txt
- Start command: uvicorn main:app --host 0.0.0.0 --port 8000
- Python 3.9+

Returns prediction + Grad-CAM heatmap.