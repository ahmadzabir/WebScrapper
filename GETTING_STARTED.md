# How to run this app

Get the web scraper running on your machine in a few minutes.

---

## What you need

- **Python 3.10 or newer**  
  Download: [python.org/downloads](https://www.python.org/downloads/)  
  ⚠️ During install, **check "Add Python to PATH"**.

---

## Step 1: Get the app

- **From GitHub:** Click **Code** → **Download ZIP**, then unzip the folder.
- Or clone: `git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git`

---

## Step 2: Open the app folder

Open a terminal (or File Explorer) and go into the folder that contains `python-scraper.py` and `requirements.txt`.

---

## Step 3: Start the app

**Windows**

- Double-click **`run.bat`**  
  or in a terminal: `run.bat`

**Mac / Linux**

- In a terminal:  
  `chmod +x run.sh`  
  then  
  `./run.sh`

**If you don't have run.bat / run.sh**

- In a terminal in the app folder:  
  `pip install -r requirements.txt`  
  then  
  `streamlit run python-scraper.py`

The first run may take a minute to install dependencies. After that it starts quickly.

---

## Step 4: Open the app in your browser

When the app starts, the terminal shows something like:

```text
Local URL: http://localhost:8501
```

Open that link in your browser (or go to **http://localhost:8501**).

---

## Step 5: Use the app

1. Upload a **CSV** with URLs in the first column (header name doesn’t matter).
2. Click **Start** and wait. Progress and ETA are shown on the page.
3. When it’s done, use the **download** options on the same page (ZIP, Excel, or combined CSV).

No account or login is required to run it locally.

---

## Local vs running in the cloud

When you run the app **on your own machine** (as above):

- **Auto-tuning:** The app detects your RAM and CPU and picks a “tier” (low / medium / high). On a **fast machine** it runs at **full force** (more workers, higher limits). On **modest hardware** (e.g. i5 5th gen, 8 GB RAM) it **optimizes automatically** so it doesn’t crash and doesn’t run out of memory, while still finishing in a reasonable time.
- **300k leads:** On capable hardware, a 300k list runs at full speed. On a typical **i5 + 8 GB RAM** machine, the same 300k list is designed to complete in **under 6 hours** without crashing.
- **Progress is saved** every few seconds; you can re-upload the same CSV and **resume** if the app or browser is closed.
- The app is built so it **doesn’t crash** and **doesn’t lose data** (checkpoints + safe writer). If your machine has very little RAM (e.g. 4 GB), consider splitting the CSV or using a machine with more memory.

Running in the cloud (e.g. Streamlit Cloud) uses gentler, fixed settings so it stays within free-tier limits; it’s stable but slower than running locally.

---

## Need help?

- **Project layout (Git, Streamlit, local):** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)  
- **Setup details:** [SETUP.md](SETUP.md)  
- **Deploying online:** [DEPLOYMENT.md](DEPLOYMENT.md)  
- **Full docs:** [README.md](README.md)
