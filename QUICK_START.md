# ⚡ Quick Start Guide

Get your web scraper running in 5 minutes!

## 🚀 Deploy Online (Recommended - 2 minutes)

### Streamlit Cloud (Easiest)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Web scraper app"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Deploy**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repo
   - Main file: `python-scraper.py`
   - Click "Deploy"

**Done!** Your app is live! 🎉

---

## 💻 Run Locally (5 minutes)

1. **Install Python 3.10+** from [python.org](https://www.python.org/downloads/)

2. **Open terminal in project folder**

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run python-scraper.py
   ```

5. **Use the app**
   - Browser opens automatically
   - Upload a CSV with URLs in first column
   - Click "Start Scraping"
   - Download results when done!

---

## 📝 Create Test CSV

Create a file `test_urls.csv`:
```csv
URL
https://example.com
https://www.python.org
```

Upload this in the app to test!

---

## 🆘 Need Help?

- **Detailed setup:** See [SETUP.md](SETUP.md)
- **Deployment options:** See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Full documentation:** See [README.md](README.md)

---

**That's it! Happy scraping! 🎉**

