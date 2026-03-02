# 🌐 Async Web Scraper

A high-performance asynchronous web scraper built with Python and **Streamlit**. Upload a CSV of URLs, run in the browser (or on Streamlit Cloud), and get cleaned text plus optional AI summaries. No installers—just Python and two commands to run locally.

## ✨ Features

- ⚡ **Async scraping** with configurable concurrency (handle 50k+ URLs efficiently)
- 🤖 **AI-Powered Company Summaries** (OpenAI & Gemini support) - Generate intelligent company summaries in parallel
- 🧹 **HTML cleanup** (removes tags, scripts, headers, footers)
- 🔗 **Keyword-based link following** (about, service, product pages)
- 📊 **Batch processing** - Split output into multiple CSV files (e.g. 2000 rows each)
- 🎨 **Beautiful Streamlit UI** with:
  - File upload for input CSV
  - Real-time progress bar with fun status messages + ETA
  - Customizable output folder names
  - Automatic ZIP file generation
  - AI summary generation (optional)
- 🛡️ **Error handling** with retry logic
- ⚙️ **Highly configurable** - Adjust all parameters via UI
- 📈 **Excel & Google Sheets compatible** - Perfect formatting, no errors

## 📥 Input Format

Upload a CSV file with URLs in the first column. The header name doesn't matter - the first column will always be used.

**Example CSV:**
```csv
URL
https://example.com
https://openai.com
https://github.com
```

## 📤 Output

Results are saved in parts:
- `output_part_1.csv`
- `output_part_2.csv`
- `output_part_3.csv`
- ...

Each file contains two columns:
- **Website** - The URL that was scraped
- **ScrapedText** - The cleaned text content from the website

All CSV files are automatically zipped into a single archive for easy download.

## 🚀 How to use the app

### Option 1: Use online (easiest)

Deploy to **Streamlit Cloud** (free) and use it in the browser—no local setup.

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and deploy this repo.
3. Open the app URL and upload your CSV.

See **[STREAMLIT_DEPLOY.md](STREAMLIT_DEPLOY.md)** or **[DEPLOYMENT.md](DEPLOYMENT.md)** for step-by-step deploy instructions.

### Option 2: Run locally (simple)

**You need:** Python 3.10+ from [python.org](https://www.python.org/downloads/).

1. **Open a terminal** in the project folder (where `python-scraper.py` and `requirements.txt` are).
2. **Install dependencies** (one time): `pip install -r requirements.txt`
3. **Start the app:** `streamlit run python-scraper.py`
4. **Open the URL** shown in the terminal (usually `http://localhost:8501`). Upload your CSV and run.

**Optional — virtual environment:**

```bash
python -m venv venv
# Windows:  .\venv\Scripts\activate   |   macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
streamlit run python-scraper.py
```

No installer, no packaging—just Python and these two commands.

## 🌐 Deploy online

- **Streamlit Cloud** — recommended; see [STREAMLIT_DEPLOY.md](STREAMLIT_DEPLOY.md).
- **Render / Railway** — see [DEPLOYMENT.md](DEPLOYMENT.md).
- **Vercel** does not support Streamlit; see [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md).

## ⚙️ Configuration

Inside the UI you can adjust:

- **Keywords** - Comma-separated keywords to find in links (default: about,service,product)
- **Concurrency** - Number of parallel workers (recommended: 20-50)
- **Retry attempts** - How many times to retry failed requests (default: 3)
- **Depth** - How many link levels to follow (default: 3)
- **Timeout** - Request timeout in seconds (default: 10)
- **Max chars per site** - Limit text length per website (default: 50,000)
- **Rows per CSV file** - Chunk size for output files (default: 2,000)
- **User-Agent** - Custom user agent string
- **Output folder/zip name** - Custom name for output folder and ZIP file

## 📋 Requirements

- Python 3.10 or higher
- See `requirements.txt` for package dependencies:
  - `aiohttp>=3.9.0` - Async HTTP client
  - `pandas>=2.0.0` - Data manipulation
  - `streamlit>=1.28.0` - Web UI framework
  - `openpyxl>=3.1.0` - Excel file support (if needed)

## 🎯 Use Cases

- **SEO Analysis** - Scrape competitor websites for content analysis
- **Data Collection** - Gather text content from multiple websites
- **Content Research** - Extract information from various sources
- **Link Analysis** - Follow and analyze internal website structures
- **Market Research** - Collect data from multiple business websites

## ⚠️ Important Notes

- **Respect robots.txt** - Always check and respect website robots.txt files
- **Rate limiting** - The app includes built-in rate limiting via concurrency control
- **Legal compliance** - Ensure you have permission to scrape target websites
- **Large datasets** - For very large URL lists (10k+), consider processing in batches
- **Windows users** - Avoid very high concurrency (stick to 30-50)

## 🐛 Troubleshooting

### App won't start
- Ensure Python 3.10+ is installed
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify you're in the correct directory

### Scraping fails
- Check your internet connection
- Verify URLs are valid and accessible
- Try reducing concurrency
- Increase timeout values
- Some websites may block automated requests

### Memory issues
- Reduce concurrency
- Process smaller batches
- Lower max_chars per site
- Reduce rows_per_file

## 📁 Project structure

```
WebScrapper/
├── python-scraper.py      # Main Streamlit app
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── .streamlit/config.toml
├── DEPLOYMENT.md, STREAMLIT_DEPLOY.md
└── outputs/              # Generated files (gitignored)
```

## 🔒 Security Notes

- The app runs locally or on your chosen hosting platform
- No data is sent to third-party services
- All processing happens in your environment
- File uploads are handled securely by Streamlit

## 📝 License

This project is provided as-is for personal and commercial use.

## 🤝 Contributing

Feel free to fork, modify, and use this project for your needs!

## 📞 Support

For deployment help, see [DEPLOYMENT.md](DEPLOYMENT.md).

For issues or questions:
1. Check the troubleshooting section
2. Review error messages in the terminal/console
3. Verify your Python version and dependencies

---

**Happy Scraping! 🎉**
