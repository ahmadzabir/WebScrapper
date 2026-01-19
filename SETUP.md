# 🛠️ Setup Guide

Quick setup instructions for getting your web scraper running locally or deploying online.

## 📦 Local Setup

### Step 1: Install Python
- Download Python 3.10 or higher from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"
- Verify installation: `python --version` (should show 3.10+)

### Step 2: Create Virtual Environment (Recommended)
```bash
# Navigate to project folder
cd WebScrapper

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run python-scraper.py
```

The app will open automatically in your browser at `http://localhost:8501`

## 🌐 Online Deployment

### Option A: Streamlit Cloud (Easiest - Recommended)

1. **Create GitHub Repository**
   - Go to [github.com](https://github.com)
   - Create a new repository
   - Upload all files (or use Git)

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `python-scraper.py`
   - Click "Deploy"

3. **Done!** Your app is live at `https://YOUR_APP_NAME.streamlit.app`

### Option B: Render.com

1. Push code to GitHub
2. Go to [render.com](https://render.com) and sign up
3. Click "New +" → "Web Service"
4. Connect your GitHub repo
5. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run python-scraper.py --server.port=$PORT --server.address=0.0.0.0`
6. Click "Create Web Service"

### Option C: Railway.app

1. Push code to GitHub
2. Go to [railway.app](https://railway.app) and sign up
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects and deploys!

## 🏠 Hostinger Shared Hosting

**Important Note:** Most shared hosting plans don't support long-running Python applications like Streamlit. However:

### If you have VPS/SSH access:

1. **Upload files via FTP/SFTP** to your hosting directory

2. **SSH into server:**
   ```bash
   ssh username@your-hostinger-ip
   ```

3. **Set up Python environment:**
   ```bash
   cd /path/to/your/app
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Run with screen (keeps it running):**
   ```bash
   screen -S scraper
   streamlit run python-scraper.py --server.port=8501 --server.address=0.0.0.0
   # Press Ctrl+A, then D to detach
   ```

5. **Access your app:**
   - If port 8501 is open: `http://your-domain.com:8501`
   - Or set up reverse proxy with nginx/apache

### Limitations:
- ⚠️ Regular shared hosting usually won't work
- ✅ VPS or dedicated server required
- ✅ Need SSH access
- ✅ Need ability to install Python packages

**Recommendation:** Use Streamlit Cloud (free) or Render (free tier) instead - they're much easier!

## ✅ Verification Checklist

After setup, verify:
- [ ] Python 3.10+ installed
- [ ] All dependencies installed (`pip list` shows aiohttp, pandas, streamlit)
- [ ] App starts without errors
- [ ] Can upload CSV file
- [ ] Scraping works with test URLs
- [ ] Output files are generated

## 🐛 Common Issues

### "streamlit: command not found"
- Make sure virtual environment is activated
- Reinstall: `pip install streamlit`

### "Module not found" errors
- Reinstall dependencies: `pip install -r requirements.txt`
- Check virtual environment is activated

### Port already in use
- Change port: `streamlit run python-scraper.py --server.port=8502`
- Or kill existing process using port 8501

### Permission errors (Linux/Mac)
- Use `python3` instead of `python`
- May need `sudo` for system-wide installs (not recommended)

## 📚 Next Steps

- Read [README.md](README.md) for usage instructions
- Read [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment guide
- Test with a small CSV file first
- Adjust settings in the UI for your needs

---

**Need help?** Check the troubleshooting section in [README.md](README.md) or [DEPLOYMENT.md](DEPLOYMENT.md)

