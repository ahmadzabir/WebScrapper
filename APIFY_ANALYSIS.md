# How Apify Scrapers Work vs. Your Current Approach

## TL;DR

**Apify's scrapers succeed (99.3%) because they're simple: one method, clear timeouts, bounded retries, fail fast.** Your script has a long fallback chain that burns time on each failure.

---

## Apify's Architecture (What Makes It Work)

### 1. **Single fetch method per run**
- **Web Scraper**: Browser (Chromium) only. No HTTP → archive.org → Playwright → Google → Common Crawl chain.
- **Cheerio Scraper**: HTTP only. Fast, no browser.
- They pick the right tool for the job and stick with it.

### 2. **Bounded, predictable timeouts**
From Apify Web Scraper input schema:
- `pageLoadTimeoutSecs`: **60** (page load)
- `pageFunctionTimeoutSecs`: **60** (extraction)
- `maxRequestRetries`: **3**

Each URL gets at most ~60s × (1 + 3 retries) = ~4 minutes in the worst case, but typically fails or succeeds in 60–90s.

### 3. **No cascading fallbacks**
When a page fails:
1. Retry (up to 3 times)
2. Mark failed, move on

No "timeout → try archive.org → timeout → try Playwright → timeout → try Common Crawl". Each fallback adds latency and complexity.

### 4. **Request queue + resume**
- Failed requests stay in the queue
- Runs can resume from checkpoint
- No need to re-scrape successful URLs

### 5. **Proxy by default**
- Apify Proxy rotates IPs
- Reduces blocks and rate limits

### 6. **Controlled concurrency**
- `maxConcurrency: 50` (configurable)
- Avoids overloading target sites and the scraper

---

## Your Current Approach (Why It Struggles)

### Fetch chain (simplified)
```
1. aiohttp direct fetch (45s budget in fast mode, else unbounded)
   → timeout or fail
2. archive.org (4 URLs × 20s = 80s)
   → fail
3. Playwright (25s goto + 4s sleep)
   → fail
4. Google cache (20s)
   → fail
5. Common Crawl (6 indexes × 15s)
   → fail
```

When the **whole** `scrape_site` hits 180s:
- You call cache fallback **again** (another 60s+)
- Total: 180s + 60s = **240s per failed URL**

With 100+ failed URLs: **6+ hours** just on cache fallbacks.

### Problems
1. **Too many fallbacks** – Each adds latency; most fail for the same reason (site down, slow, or blocking).
2. **Unbounded fetch** – When not in fast mode, direct fetch has no timeout.
3. **Double cache attempt** – Cache is tried inside `scrape_site` and again after timeout.
4. **No proxy** – Easier to get blocked.
5. **Heavy Playwright use** – Semaphore of 2, but still resource-heavy on Streamlit Cloud.

---

## Recommendations

### Option A: Simplify your fetch logic (Apify-style)

**Single path, fail fast:**

1. **Primary: aiohttp** with a strict timeout (e.g. 30s).
2. **Fallback: Playwright** with a strict timeout (e.g. 45s), **or** archive.org (2 snapshots, 15s each).
3. **No** Google cache or Common Crawl in the hot path (or make them opt‑in).
4. **Total budget per URL**: 60–90s. If both fail, mark failed and continue.

```python
# Pseudocode
async def fetch_url(url):
    # 1. Try HTTP (30s)
    result = await try_http(url, timeout=30)
    if result: return result

    # 2. Try ONE fallback: Playwright (45s) OR archive.org (30s)
    result = await try_playwright(url, timeout=45)  # or try_archive(url, timeout=30)
    if result: return result

    return None  # Fail, move on. Don't chain 4 more sources.
```

### Option B: Use Apify API for scraping

Call Apify's Web Scraper or Website Content Crawler from your Python app:

```python
from apify_client import ApifyClient

client = ApifyClient("YOUR_API_TOKEN")
run = client.actor("apify/web-scraper").call(
    run_input={
        "startUrls": [{"url": url} for url in your_urls],
        "pageFunction": """
        async function pageFunction(context) {
            return { html: document.documentElement.outerHTML };
        }
        """,
        "maxRequestRetries": 3,
        "pageLoadTimeoutSecs": 60,
    }
)
# Poll for completion, fetch dataset
```

- **Pros**: 99.3% success, no maintenance, proxies, retries, timeouts handled.
- **Cons**: Pay per use (~$0.25 per 1000 pages on free tier).

### Option C: Hybrid

- Use your script for small runs (< 100 URLs).
- Use Apify API for large runs (500+ URLs) where reliability matters.

---

## Concrete Changes to Your Script (Option A)

If you keep the current script, these changes align it with Apify’s approach:

| Change | Current | Apify-style |
|--------|---------|-------------|
| Direct fetch timeout | 45s (fast) / unbounded | Always 30s |
| Fallback chain | 4 sources (archive, Playwright, Google, CC) | 1 fallback: Playwright **or** archive |
| Post-timeout cache | 60s, quick_mode | Remove; fail after primary + 1 fallback |
| Total per-URL budget | 180s + 60s | 90s |
| Retries | Many, with backoff | 2–3, simple |

### Code changes (high level)

1. **Always use `fetch_budget`** (e.g. 30s) for direct fetch, not only in fast mode.
2. **Replace the 4-step cache chain** with a single fallback: Playwright **or** archive.org.
3. **Remove post-timeout cache** – when the main flow times out, mark failed and continue.
4. **Cap total time per URL** at 90s.

---

## Summary

| Aspect | Apify | Your script |
|--------|-------|-------------|
| Fetch strategy | One method (browser or HTTP) | Long chain of fallbacks |
| Timeout per URL | ~60s | 180s + 60s |
| Retries | 3, bounded | Many, complex |
| On failure | Fail, move on | Try 4 more sources |
| Proxy | Yes | No |
| Success rate | 99.3% | Lower (many timeouts) |

**Core idea**: Apify succeeds by being simple and failing fast. Your script spends a lot of time on fallbacks that rarely help.
