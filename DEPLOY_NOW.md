# 🚀 Deploy to Streamlit Cloud - Quick Guide

Get your web scraper live in 5 minutes!

## Step 1: Push Code to GitHub (2 minutes)

### Option A: Using GitHub Desktop (Easiest)

1. **Download GitHub Desktop**
   - Go to [desktop.github.com](https://desktop.github.com/)
   - Download and install

2. **Sign in to GitHub**
   - Create account at [github.com](https://github.com) if needed
   - Sign in to GitHub Desktop

3. **Add Your Project**
   - In GitHub Desktop: **File → Add Local Repository**
   - Click **Choose** and select: `C:\Users\MG\OneDrive\Documents\Code Play\WebScrapper`
   - Click **Add repository**

4. **Publish to GitHub**
   - Click **Publish repository** button
   - Name it: `web-scraper` (or any name)
   - ✅ **IMPORTANT:** Make it **Public** (required for free Streamlit Cloud)
   - Click **Publish repository**

### Option B: Using Git Command Line

Open PowerShell in your project folder and run:

```powershell
cd "C:\Users\MG\OneDrive\Documents\Code Play\WebScrapper"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Web scraper app - ready to deploy"

# Create repository on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/web-scraper.git
git branch -M main
git push -u origin main
```

(Replace `YOUR_USERNAME` with your GitHub username)

---

## Step 2: Deploy to Streamlit Cloud (3 minutes)

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click **Sign in** → Sign in with **GitHub**

2. **Create New App**
   - Click **New app** button (top right)

3. **Configure Your App**
   - **Repository:** Select your repository (`web-scraper`)
   - **Branch:** `main` (or `master` if that's your branch)
   - **Main file path:** `python-scraper.py` ⚠️ **CRITICAL - Must be exact!**
   - **App URL:** Choose a name (e.g., `my-web-scraper`)
     - This will be: `https://my-web-scraper.streamlit.app`

4. **Deploy!**
   - Click **Deploy** button
   - Wait 2-5 minutes for first deployment
   - Watch the build logs

5. **Done!** 🎉
   - When it says "Your app is live!", you're done!
   - Your app URL: `https://YOUR_APP_NAME.streamlit.app`

---

## ✅ Checklist

Before deploying, make sure:
- [ ] All files are in the project folder
- [ ] `python-scraper.py` is in the root folder
- [ ] `requirements.txt` exists and has all packages
- [ ] Repository is **Public** (for free tier)
- [ ] Code is pushed to GitHub

---

## 🔄 Updating Your App

Whenever you make changes:

1. **Save your files**
2. **Commit and push to GitHub:**
   ```powershell
   git add .
   git commit -m "Updated app"
   git push
   ```
3. **Streamlit Cloud automatically redeploys!** (takes 1-2 minutes)

---

## 🐛 Troubleshooting

### "App not found" error
- ✅ Check that `python-scraper.py` is in the root folder
- ✅ Verify main file path is exactly `python-scraper.py`

### "Module not found" error
- ✅ Check `requirements.txt` has all packages
- ✅ Make sure file is in root folder

### Build fails
- ✅ Check build logs in Streamlit Cloud dashboard
- ✅ Look for error messages
- ✅ Verify Python version (should be 3.10+)

### Can't push to GitHub
- ✅ Make sure you're signed in
- ✅ Check repository name is correct
- ✅ Try using GitHub Desktop instead

---

## 📝 Important Notes

- **Free tier:** Completely free, no credit card needed
- **Public repos only:** Free tier requires public GitHub repository
- **Auto-deploy:** Every push to GitHub automatically redeploys
- **HTTPS included:** Your app gets a secure HTTPS URL
- **Custom domain:** Can add custom domain (paid feature)

---

## 🎯 Your App Will Be Live At:

```
https://YOUR_APP_NAME.streamlit.app
```

Share this URL with anyone - they can use your web scraper!

---

**Need more help?** See `STREAMLIT_DEPLOY.md` for detailed instructions.

**That's it! Your app will be live in minutes! 🚀**

