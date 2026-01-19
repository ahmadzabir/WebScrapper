# 🚀 Deploy to Streamlit Cloud - Simple Guide

Follow these easy steps to deploy your web scraper online for free!

## 📋 Prerequisites
- GitHub account (free at github.com)
- Your code ready in this folder

---

## Step 1: Test Locally First ✅

**Before deploying, make sure it works on your computer:**

### 1.1 Install Python (if not already installed)
- Download from [python.org](https://www.python.org/downloads/)
- During installation, **check "Add Python to PATH"**
- Verify: Open terminal and type `python --version` (should show 3.10+)

### 1.2 Install Dependencies
Open terminal/PowerShell in this folder and run:
```bash
pip install -r requirements.txt
```

### 1.3 Run the App
```bash
streamlit run python-scraper.py
```

### 1.4 Test It
- Browser should open automatically at `http://localhost:8501`
- Create a test CSV file `test.csv` with this content:
  ```csv
  URL
  https://example.com
  https://www.python.org
  ```
- Upload the CSV in the app
- Click "Start Scraping"
- Wait for it to finish
- ✅ If it works, you're ready to deploy!

**If you see errors:** Make sure Python 3.10+ is installed and all packages installed correctly.

---

## Step 2: Push to GitHub 📤

### 2.1 Create GitHub Repository
1. Go to [github.com](https://github.com) and sign in
2. Click the **"+"** button → **"New repository"**
3. Name it: `web-scraper` (or any name you like)
4. Make it **Public** (required for free Streamlit Cloud)
5. **Don't** check "Initialize with README"
6. Click **"Create repository"**

### 2.2 Upload Your Code
**Option A: Using GitHub Desktop (Easiest)**
1. Download [GitHub Desktop](https://desktop.github.com/)
2. Install and sign in
3. Click "File" → "Add Local Repository"
4. Select this folder: `C:\Users\MG\OneDrive\Documents\Code Play\WebScrapper`
5. Click "Publish repository"
6. Choose your repository name
7. Click "Publish"

**Option B: Using Git Command Line**
Open terminal in this folder and run:
```bash
git init
git add .
git commit -m "Initial commit - Web Scraper"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```
(Replace YOUR_USERNAME and YOUR_REPO_NAME with your actual GitHub username and repository name)

---

## Step 3: Deploy to Streamlit Cloud 🎉

### 3.1 Go to Streamlit Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in"** → Sign in with **GitHub**

### 3.2 Create New App
1. Click **"New app"** button
2. Fill in:
   - **Repository:** Select your repository (web-scraper)
   - **Branch:** `main` (or `master`)
   - **Main file path:** `python-scraper.py` ⚠️ **Important!**
   - **App URL:** Choose a name (e.g., `my-web-scraper`)
3. Click **"Deploy"**

### 3.3 Wait for Deployment
- First deployment takes 2-5 minutes
- You'll see build logs
- When it says "Your app is live!", you're done! 🎉

### 3.4 Access Your App
Your app will be live at:
```
https://YOUR_APP_NAME.streamlit.app
```

---

## ✅ Success Checklist

After deployment, verify:
- [ ] App loads without errors
- [ ] Can upload CSV file
- [ ] Scraping works
- [ ] Progress bar shows
- [ ] Results download works

---

## 🔄 Update Your App

Whenever you make changes:
1. Save your files
2. Push to GitHub (same as Step 2)
3. Streamlit Cloud **automatically redeploys** your app!

---

## 🐛 Troubleshooting

### "App not found" error
- Check that `python-scraper.py` is in the root folder
- Verify the main file path is exactly `python-scraper.py`

### "Module not found" error
- Check `requirements.txt` has all packages
- Make sure file is in root folder

### App won't start
- Check the build logs in Streamlit Cloud dashboard
- Look for error messages
- Verify Python version (should be 3.10+)

### Can't push to GitHub
- Make sure you're signed in
- Check repository name is correct
- Try using GitHub Desktop instead

---

## 📞 Need Help?

1. Check Streamlit Cloud docs: [docs.streamlit.io](https://docs.streamlit.io)
2. Review error messages in build logs
3. Make sure all files are in the root folder

---

**That's it! Your web scraper is now live on the internet! 🎉**

