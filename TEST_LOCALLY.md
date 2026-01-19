# 🧪 Test Locally - Step by Step

Test your web scraper on your computer before deploying online.

---

## Step 1: Check Python Installation ✅

1. Open **PowerShell** or **Command Prompt**
2. Type: `python --version`
3. Should show: `Python 3.10.x` or higher

**If not installed:**
- Download from [python.org](https://www.python.org/downloads/)
- During installation, **check "Add Python to PATH"**
- Restart terminal and check again

---

## Step 2: Navigate to Project Folder 📁

Open terminal and go to your project:
```bash
cd "C:\Users\MG\OneDrive\Documents\Code Play\WebScrapper"
```

---

## Step 3: Install Dependencies 📦

Run this command:
```bash
pip install -r requirements.txt
```

**Wait for it to finish** - this installs:
- streamlit (web UI)
- aiohttp (web scraping)
- pandas (data handling)

**If you see errors:**
- Try: `python -m pip install -r requirements.txt`
- Or: `pip3 install -r requirements.txt`

---

## Step 4: Create Test CSV File 📝

Create a file named `test.csv` in this folder with this content:

```csv
URL
https://example.com
https://www.python.org
https://github.com
```

**How to create:**
- Open Notepad
- Paste the content above
- Save as `test.csv` (make sure it's `.csv`, not `.txt`)
- Save it in the WebScrapper folder

---

## Step 5: Run the App 🚀

In the terminal, run:
```bash
streamlit run python-scraper.py
```

**What happens:**
- Terminal shows: "You can now view your Streamlit app in your browser"
- Browser opens automatically at `http://localhost:8501`
- If browser doesn't open, copy the URL and paste in browser

---

## Step 6: Test the App 🧪

1. **Upload CSV:**
   - Click "Browse files" or drag `test.csv` into the upload area

2. **Adjust Settings (optional):**
   - Concurrency: 10 (lower for testing)
   - Timeout: 10 seconds
   - Other settings can stay default

3. **Start Scraping:**
   - Click **"🚀 Start Scraping"** button
   - Watch the progress bar
   - Wait for completion (1-2 minutes for 3 URLs)

4. **Check Results:**
   - Should see "✅ Scraping finished" message
   - Check `outputs` folder for CSV files
   - Download the ZIP file if shown

---

## ✅ Success Indicators

- ✅ App opens in browser
- ✅ Can upload CSV file
- ✅ Progress bar shows and updates
- ✅ Scraping completes without errors
- ✅ CSV files appear in `outputs` folder
- ✅ No error messages in terminal

---

## 🐛 Common Issues & Fixes

### "streamlit: command not found"
**Fix:**
```bash
python -m streamlit run python-scraper.py
```

### "Module not found: streamlit"
**Fix:**
```bash
pip install streamlit
pip install -r requirements.txt
```

### "Port 8501 already in use"
**Fix:**
- Close other Streamlit apps
- Or use different port: `streamlit run python-scraper.py --server.port=8502`

### App opens but shows errors
**Fix:**
- Check terminal for error messages
- Make sure all files are in the same folder
- Verify `python-scraper.py` exists

### Scraping fails for all URLs
**Fix:**
- Check internet connection
- Try with just `https://example.com`
- Increase timeout to 20 seconds
- Reduce concurrency to 5

---

## 🎯 Next Steps

Once local testing works:
1. ✅ Everything is ready!
2. Follow **STREAMLIT_DEPLOY.md** to deploy online
3. Your app will work the same way online!

---

**Need help?** Check the error message in terminal and try the fixes above.

