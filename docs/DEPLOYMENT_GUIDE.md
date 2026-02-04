# Deployment Guide: CSP Timing Web App

Deploy your CSP predictor so you can access it from work! The backend runs on your PC (using your GPU), while the frontend is hosted on GitHub Pages (free).

## Architecture

```
Your PC (Backend)          Internet          GitHub Pages (Frontend)
┌─────────────────┐       ┌────────┐        ┌──────────────────┐
│  API Server     │◄─────►│ ngrok  │◄──────►│  Web Interface   │
│  (FastAPI)      │       │        │        │  (HTML/CSS/JS)   │
│  Uses GPU       │       └────────┘        └──────────────────┘
└─────────────────┘           ▲                      ▲
                              │                      │
                         Your Work ─────────────────┘
```

## Step 1: Install Dependencies

```bash
cd /home/cason/model
pip install fastapi uvicorn pydantic
```

## Step 2: Start the Backend Server

On your PC, run:

```bash
python api_server.py
```

This starts the API server on `http://localhost:8000`. You should see:

```
CSP TIMING API SERVER
======================================================================
Starting server on http://0.0.0.0:8000
GPU Available: True
GPU: NVIDIA GeForce RTX 5070 Ti

API Documentation: http://localhost:8000/docs
======================================================================
```

**Test it locally**: Open http://localhost:8000 in your browser. You should see a status page.

## Step 3: Expose Your Server with ngrok

### Install ngrok

1. Go to https://ngrok.com/ and sign up (free)
2. Download ngrok for your OS
3. Follow their setup instructions to authenticate

### Start ngrok

In a new terminal:

```bash
ngrok http 8000
```

You'll see something like:

```
Session Status                online
Forwarding                    https://abc123.ngrok.io -> http://localhost:8000
```

**Copy that `https://abc123.ngrok.io` URL** - this is your public API endpoint!

### Keep ngrok Running

Leave this terminal open. Your server is now accessible from anywhere.

### Alternative: ngrok config for stable URLs

Free ngrok gives you random URLs. For a stable URL, you can:
- Upgrade to ngrok paid ($8/month) for static domains
- Use alternatives like localtunnel, serveo, or localhost.run (free)

## Step 4: Deploy Frontend to GitHub Pages

### Create GitHub Repository

1. Go to GitHub.com and create a new repository
2. Name it: `csp-timing-predictor`
3. Make it **Public**
4. Don't initialize with README

### Push Your Web App

```bash
cd /home/cason/model
git init
git add web/
git commit -m "Initial commit: CSP Timing web interface"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/csp-timing-predictor.git
git push -u origin main
```

### Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings**
3. Scroll to **Pages** (left sidebar)
4. Under "Source", select: **Deploy from a branch**
5. Select branch: **main**
6. Select folder: **/ (root)** or **/docs** (if you move web/ to docs/)
7. Click **Save**

Wait 1-2 minutes. Your site will be live at:
```
https://YOUR_USERNAME.github.io/csp-timing-predictor/web/
```

**Note**: If the web folder structure doesn't work, move `web/index.html` to the root:
```bash
mv web/index.html .
git add index.html
git commit -m "Move index to root"
git push
```

Then access at:
```
https://YOUR_USERNAME.github.io/csp-timing-predictor/
```

## Step 5: Configure and Use

### First Time Setup

1. Open your GitHub Pages URL at work
2. Enter your ngrok URL (e.g., `https://abc123.ngrok.io`) in the API URL field
3. Select a model (Hybrid recommended)
4. Try a prediction!

### Daily Workflow

**Before work:**
1. Start API server: `python api_server.py`
2. Start ngrok: `ngrok http 8000`
3. Copy the new ngrok URL (it changes each time on free plan)

**At work:**
1. Open your GitHub Pages URL
2. Update the API URL with today's ngrok URL
3. Use the app!

**After work:**
- Stop both servers (Ctrl+C in each terminal)

## Step 6: Keep Your PC Server Running 24/7 (Optional)

### Option A: Leave PC On

Simple! Just leave terminals running.

### Option B: Run as System Service (Linux/WSL)

Create a systemd service:

```bash
sudo nano /etc/systemd/system/csp-api.service
```

Add:
```ini
[Unit]
Description=CSP Timing API Server
After=network.target

[Service]
Type=simple
User=cason
WorkingDirectory=/home/cason/model
ExecStart=/usr/bin/python3 /home/cason/model/api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable csp-api
sudo systemctl start csp-api
sudo systemctl status csp-api
```

### Option C: Run with Docker

Create `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:latest

WORKDIR /app
COPY requirements_gpu.txt .
RUN pip install -r requirements_gpu.txt
RUN pip install fastapi uvicorn

COPY . .

CMD ["python", "api_server.py"]
```

Build and run:
```bash
docker build -t csp-api .
docker run -d -p 8000:8000 --gpus all csp-api
```

## Security Considerations

### Important Security Notes

Your API is exposed to the internet via ngrok. Here's how to secure it:

1. **Add API Key Authentication**

Edit `api_server.py`:

```python
from fastapi import Header, HTTPException

API_KEY = "your-secret-key-here"  # Change this!

async def verify_token(x_api_key: str = Header()):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

# Add dependency to endpoints:
@app.post("/predict", dependencies=[Depends(verify_token)])
def predict(request: PredictionRequest):
    ...
```

Update frontend to send API key:
```javascript
headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-secret-key-here'
}
```

2. **Rate Limiting**

```bash
pip install slowapi
```

Add to `api_server.py`:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
def predict(request: Request, ...):
    ...
```

3. **HTTPS Only**

ngrok provides HTTPS by default. In production, don't use HTTP.

## Troubleshooting

### "Connection refused" error
- Check if API server is running: `curl http://localhost:8000`
- Check if ngrok is running: Look for "Forwarding" line
- Verify ngrok URL is correct in web interface

### "CORS error" in browser
- API server already has CORS enabled for all origins
- If you restrict origins, update `allow_origins` in `api_server.py`

### "Model not found" error
- Ensure models are trained: Run `python deep_learning_model.py`
- Check model files exist in the same directory as `api_server.py`

### Slow predictions
- First prediction is slow (model loading)
- Subsequent predictions are fast (cached)
- Consider using TabNet for faster inference

### ngrok URL changes daily
- Free ngrok gives random URLs
- Solutions:
  1. Upgrade to ngrok paid ($8/mo) for static URL
  2. Use a URL shortener (e.g., bit.ly) and update the redirect
  3. Store the URL in a GitHub Gist and fetch it from frontend

### GPU not being used
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure PyTorch is installed with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Alternative: Free Cloud Hosting

If you want to avoid running servers at home:

### Option 1: Hugging Face Spaces (Free GPU)
- Deploy as a Gradio app
- Free GPU inference
- Always online

### Option 2: Google Colab (Free)
- Run notebook with API server
- Use ngrok in Colab
- Limited runtime (12 hours)

### Option 3: AWS/Azure Free Tier
- Free for 12 months
- Small instances only (no GPU)
- Use CPU inference

## Cost Comparison

| Option | Cost | GPU | Always On | Setup Difficulty |
|--------|------|-----|-----------|------------------|
| Your PC + ngrok (free) | $0 | Yes | Manual | Easy |
| Your PC + ngrok (paid) | $8/mo | Yes | Manual | Easy |
| Hugging Face Spaces | $0 | Yes | Yes | Medium |
| AWS EC2 (GPU) | ~$200/mo | Yes | Yes | Hard |
| Your PC 24/7 | ~$30/mo electricity | Yes | Yes | Easy |

**Recommended for you**: Your PC + ngrok free tier + GitHub Pages = **$0/month**

Just update the ngrok URL each day at work (takes 10 seconds).

## Next Steps

1. **Automate ngrok URL update**
   - Store URL in a GitHub Gist
   - Frontend fetches from gist
   - Update gist automatically on server start

2. **Add authentication**
   - JWT tokens
   - OAuth with Google

3. **Add features**
   - Historical predictions tracking
   - Email/SMS alerts when opportunities arise
   - Backtesting results visualization

4. **Mobile app**
   - Wrap in React Native
   - iOS/Android app
   - Push notifications

## Questions?

Test everything works:
1. API server running: http://localhost:8000
2. API docs: http://localhost:8000/docs (test endpoints here!)
3. ngrok running: Check for public URL
4. Test API via ngrok: `curl https://your-ngrok-url.ngrok.io/health`
5. Frontend on GitHub Pages: Check browser console for errors

Your GPU-powered CSP predictor is now accessible from anywhere! 🚀
