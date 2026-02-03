# GitHub Deployment Instructions

Your repository is ready to push to GitHub! Follow these steps:

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `csp-timing-predictor`
3. Description: `AI-powered CSP timing predictions with machine learning`
4. Visibility: **Public** (required for free GitHub Pages)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click **Create repository**

## Step 2: Push to GitHub

### Option A: Using SSH (Recommended)

SSH is more convenient - no need to enter credentials every time.

**First-time SSH setup:**

1. Check if you already have an SSH key:
   ```bash
   ls ~/.ssh/id_*.pub
   ```

2. If not, generate one:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Press Enter to accept default location
   # Enter passphrase (optional but recommended)
   ```

3. Copy your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

4. Add to GitHub:
   - Go to https://github.com/settings/keys
   - Click "New SSH key"
   - Title: "CSP Timing PC" (or any name)
   - Paste your public key
   - Click "Add SSH key"

5. Test connection:
   ```bash
   ssh -T git@github.com
   # Should see: "Hi USERNAME! You've successfully authenticated..."
   ```

**Push with SSH:**

```bash
# Add your GitHub repo as remote (replace YOUR_USERNAME)
git remote add origin git@github.com:YOUR_USERNAME/csp-timing-predictor.git

# Push your code
git push -u origin main
```

### Option B: Using HTTPS

HTTPS requires a Personal Access Token each time (unless cached).

**Creating a Personal Access Token:**

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name: "CSP Timing Predictor"
4. Scopes: Select `repo` (full control)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

**Push with HTTPS:**

```bash
# Add your GitHub repo as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/csp-timing-predictor.git

# Push your code
git push -u origin main
# Username: your GitHub username
# Password: paste your Personal Access Token
```

## Step 3: Enable GitHub Pages

1. Go to your repo on GitHub
2. Click **Settings** (top right)
3. Click **Pages** (left sidebar)
4. Under "Source":
   - Branch: `main`
   - Folder: `/ (root)`
5. Click **Save**

Wait 1-2 minutes, then your site will be live at:
```
https://YOUR_USERNAME.github.io/csp-timing-predictor/
```

## Step 4: Test Your Web Interface

1. Open your GitHub Pages URL
2. You should see the CSP Timing Predictor interface
3. To use it:
   - Start your API server: `./start_web_system.sh`
   - In new terminal: `ngrok http 8000`
   - Enter the ngrok URL in your web interface
   - Get predictions!

## What's Been Pushed

✅ **30 files** including:
- All Python scripts (training, prediction, API)
- Web interface (index.html)
- Documentation (README, guides)
- Setup scripts (.sh files)

❌ **NOT pushed** (thanks to .gitignore):
- Model files (*.pkl) - too large
- Training data (*.csv)
- Logs (*.log)
- Training outputs (*.png) - except feature importance

## Updating Your Repository

After making changes locally:

```bash
# See what changed
git status

# Add changes
git add .

# Commit with a message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Common Issues

### Authentication Failed
- Use Personal Access Token, not password
- Token needs `repo` scope

### GitHub Pages Not Working
- Make sure repo is Public
- Wait 2-3 minutes after enabling Pages
- Check Settings → Pages for deployment status

### Web Interface Shows 404
- Make sure you're using the correct URL: `https://YOUR_USERNAME.github.io/csp-timing-predictor/`
- Check that index.html is in the root directory (it is)

## Repository Structure

```
csp-timing-predictor/
├── index.html              # Web interface (GitHub Pages serves this)
├── README.md               # GitHub displays this on repo homepage
├── api_server.py           # Backend API (runs locally)
├── data_collector.py       # Feature engineering
├── model_trainer.py        # Model training
├── predictor.py            # CLI predictions
├── daily_alerts.py         # Automated alerts
├── train_multi_ticker.py   # Multi-ticker training
└── [other files...]
```

## Next Steps After Deployment

1. **Update README.md** with your actual GitHub Pages URL:
   ```bash
   # Edit README.md and replace:
   # **Live Demo**: [Your GitHub Pages URL]
   # with:
   # **Live Demo**: https://YOUR_USERNAME.github.io/csp-timing-predictor/

   git add README.md
   git commit -m "Update live demo URL"
   git push
   ```

2. **Test from work:**
   - Bookmark your GitHub Pages URL
   - Start API server at home
   - Start ngrok
   - Access from work and enter ngrok URL

3. **Share with others:**
   - Your repo is public on GitHub
   - Anyone can clone and train their own model
   - Web interface requires your API server to work

## Security Notes

- ✅ Model files NOT pushed (too large, contains your training)
- ✅ No API keys or credentials in code
- ✅ ngrok URL is entered by user (not hardcoded)
- ⚠️ Keep your ngrok URL private (anyone with it can use your API)

## Making the Repo Private (Optional)

If you want to keep your code private:
1. Upgrade to GitHub Pro ($4/mo) for private Pages
2. Or keep repo public but use different API authentication

Current setup works great as-is with public repo + private ngrok URL.

## Success Checklist

- [ ] Created GitHub repository
- [ ] Pushed code to GitHub
- [ ] Enabled GitHub Pages
- [ ] Verified web interface loads
- [ ] Started API server locally
- [ ] Started ngrok
- [ ] Connected web interface to API
- [ ] Got predictions working!

🎉 Once all checked, you're live on GitHub!
