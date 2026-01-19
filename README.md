# 🌐 Async Web Scraper

A high-performance asynchronous web scraper built with Python, aiohttp, asyncio, pandas, and Streamlit. It fetches website content, cleans HTML, follows internal links, and saves results in chunked CSV files for easy download.

## ✨ Features

- ⚡ **Async scraping** with configurable concurrency (handle 50k+ URLs efficiently)
- 🧹 **HTML cleanup** (removes tags, scripts, headers, footers)
- 🔗 **Keyword-based link following** (about, service, product pages)
- 📊 **Batch processing** - Split output into multiple CSV files (e.g. 2000 rows each)
- 🎨 **Beautiful Streamlit UI** with:
  - File upload for input CSV
  - Real-time progress bar with fun status messages + ETA
  - Customizable output folder names
  - Automatic ZIP file generation
- 🛡️ **Error handling** with retry logic
- ⚙️ **Highly configurable** - Adjust all parameters via UI

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

## 🚀 Quick Start (Local)

1. **Clone or download this repository**
   ```bash
   cd WebScrapper
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run python-scraper.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in the terminal

## 🌐 Deploy Online (Free Options)

This app is ready to deploy! See **[DEPLOYMENT.md](DEPLOYMENT.md)** for detailed instructions.

### Quick Deploy Options:

1. **Streamlit Cloud** ⭐ (Recommended - Easiest)
   - Push to GitHub
   - Deploy at [share.streamlit.io](https://share.streamlit.io)
   - Zero configuration needed!
   - 📖 See: `STREAMLIT_DEPLOY.md` for step-by-step guide

2. **Render** (Free tier available)
   - Push to GitHub
   - Deploy at [render.com](https://render.com)
   - Uses included `render.yaml`

3. **Railway** (Free trial)
   - Push to GitHub
   - Deploy at [railway.app](https://railway.app)
   - Uses included `railway.json`

### ⚠️ Not Supported:
- **Vercel** - Does not support Streamlit apps (see `VERCEL_DEPLOYMENT.md` for details)

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

## 📁 Project Structure

```
WebScrapper/
├── python-scraper.py      # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── DEPLOYMENT.md          # Deployment guide
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── Procfile               # For Heroku/Render
├── render.yaml            # Render.com config
├── railway.json           # Railway.app config
├── runtime.txt            # Python version
├── .gitignore             # Git ignore rules
└── outputs/               # Generated output files (gitignored)
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
