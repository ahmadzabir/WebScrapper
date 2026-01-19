# 🚀 Deployment Guide - Web Scraper Application

This guide will help you deploy your Streamlit web scraper application to various hosting platforms. The application is ready for deployment and includes all necessary configuration files.

## 📋 Prerequisites

- A GitHub account (for most free hosting options)
- Your code pushed to a GitHub repository
- Basic understanding of Git (optional, but helpful)

---

## ❌ Vercel - Not Supported

**Important:** Vercel does **NOT** support Streamlit apps. Vercel is serverless and designed for short-lived functions, while Streamlit requires a persistent server. See `VERCEL_DEPLOYMENT.md` for details.

---

## 🌟 Option 1: Streamlit Cloud (RECOMMENDED - Easiest & Free)

**Best for:** Quick deployment, zero configuration, perfect for Streamlit apps

### Steps:

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set **Main file path** to: `python-scraper.py`
   - Click "Deploy"

3. **That's it!** Your app will be live at `https://YOUR_APP_NAME.streamlit.app`

### Advantages:
- ✅ Completely free
- ✅ Automatic deployments on git push
- ✅ No configuration needed
- ✅ Built specifically for Streamlit
- ✅ HTTPS included

### Limitations:
- Free tier has some usage limits (but generous for personal use)
- Apps sleep after inactivity (wake up quickly when accessed)

---

## 🌐 Option 2: Render (Free Tier Available)

**Best for:** More control, similar to Heroku

### Steps:

1. **Push your code to GitHub** (same as above)

2. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Deploy Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Use these settings:
     - **Name:** web-scraper (or your choice)
     - **Environment:** Python 3
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `streamlit run python-scraper.py --server.port=$PORT --server.address=0.0.0.0`
     - **Plan:** Free

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (5-10 minutes)

### Advantages:
- ✅ Free tier available
- ✅ More control over environment
- ✅ Can add databases, Redis, etc.
- ✅ Custom domains supported

### Limitations:
- Free tier apps sleep after 15 minutes of inactivity
- Slower cold starts than Streamlit Cloud

---

## 🚂 Option 3: Railway (Free Trial)

**Best for:** Modern platform, easy deployment

### Steps:

1. **Push your code to GitHub**

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect Python and use `railway.json`

3. **Configure (if needed)**
   - Railway should auto-detect the start command
   - If not, add: `streamlit run python-scraper.py --server.port=$PORT --server.address=0.0.0.0`

### Advantages:
- ✅ Very easy setup
- ✅ Good free trial ($5 credit)
- ✅ Fast deployments
- ✅ Great developer experience

### Limitations:
- Free trial credits expire
- Need to add payment method for continued use

---

## 🏠 Option 4: Hostinger Shared Hosting (If Available)

**Note:** Shared hosting typically doesn't support long-running Python applications like Streamlit. However, if you have VPS hosting or can upgrade, here's how:

### Requirements:
- Python 3.10+ installed
- Ability to install pip packages
- SSH access
- Ability to run processes in background

### Steps (if you have VPS/SSH access):

1. **Upload files via FTP/SFTP**
   - Upload all project files to your hosting directory

2. **SSH into your server**
   ```bash
   ssh your_username@your_hostinger_ip
   ```

3. **Set up Python environment**
   ```bash
   cd /path/to/your/app
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run Streamlit**
   ```bash
   streamlit run python-scraper.py --server.port=8501 --server.address=0.0.0.0
   ```

5. **Keep it running (use screen or tmux)**
   ```bash
   screen -S scraper
   streamlit run python-scraper.py --server.port=8501 --server.address=0.0.0.0
   # Press Ctrl+A then D to detach
   ```

### Alternative: Use a Process Manager
If you have systemd access:
```bash
# Create service file
sudo nano /etc/systemd/system/web-scraper.service
```

Add:
```ini
[Unit]
Description=Web Scraper Streamlit App
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/your/app
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/streamlit run python-scraper.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable web-scraper
sudo systemctl start web-scraper
```

### Limitations:
- ⚠️ Most shared hosting doesn't support this
- ⚠️ Need VPS or dedicated server
- ⚠️ More technical setup required

---

## 🔧 Configuration Files Included

This project includes deployment-ready configuration files:

- **`Procfile`** - For Heroku/Render deployment
- **`runtime.txt`** - Python version specification
- **`render.yaml`** - Render.com configuration
- **`railway.json`** - Railway.app configuration
- **`.streamlit/config.toml`** - Streamlit settings
- **`.gitignore`** - Git ignore patterns

---

## 📝 Post-Deployment Checklist

After deploying, verify:

- [ ] App loads without errors
- [ ] File upload works
- [ ] Scraping functionality works
- [ ] Output files are generated
- [ ] Progress bar updates correctly
- [ ] App handles errors gracefully

---

## 🐛 Troubleshooting

### App won't start
- Check that `python-scraper.py` is in the root directory
- Verify `requirements.txt` has all dependencies
- Check platform logs for error messages

### Port issues
- Ensure start command uses `$PORT` environment variable
- Some platforms require `0.0.0.0` as address

### Memory issues
- Reduce concurrency in the UI
- Process smaller batches of URLs
- Increase timeout values

### File upload limits
- Some platforms have file size limits
- Consider processing in smaller chunks
- Use external storage (S3, etc.) for large outputs

---

## 💡 Recommendations

**For beginners:** Use **Streamlit Cloud** - it's the easiest and requires zero configuration.

**For more control:** Use **Render** - good balance of features and ease of use.

**For production:** Consider paid tiers or VPS hosting for better performance and reliability.

---

## 📞 Need Help?

If you encounter issues:
1. Check the platform's documentation
2. Review error logs in the platform dashboard
3. Ensure all files are committed to your repository
4. Verify Python version compatibility (3.10+)

---

**Happy Deploying! 🎉**

