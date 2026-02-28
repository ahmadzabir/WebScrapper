# Web Scraper Improvement Plan

## Overview
A systematic approach to improving the web scraper's reliability, success rate, and cost-effectiveness.

---

## Phase 0: Measurement First
**Goal:** Turn "it failed" into 6-10 failure buckets with counts so we fix the right things.

### Implementation
- [ ] Create structured scrape result object per URL:
  - `result_status` (success, failed, partial)
  - `http_status` (200, 403, 429, etc.)
  - `exception_type` (timeout, decode_error, etc.)
  - `stage_failed` (connect, read, parse, etc.)
  - `html_bytes` (size of raw response)
  - `cleaned_chars` (final text length)
  - `detected_js_shell` (boolean)
  - `detected_block_page` (boolean)
  - `retries_used`
  - `total_seconds`

- [ ] Add "Top Failure Reasons" panel in UI
- [ ] Add downloadable failures CSV export
- [ ] Log sample HTML snippets (first 2,000 chars) per failure bucket for pattern analysis

### Success Criteria
Can answer in one screenshot: "Top 3 failure buckets are X (45%), Y (23%), Z (12%)."

---

## Phase 1: Fix Compression & Decoding
**Goal:** Eliminate mysterious failures from Brotli edge cases and content-type confusion.

### Implementation
- [ ] Stop advertising Brotli unless runtime truly supports it
- [ ] In "cloud mode", force `Accept-Encoding: gzip, deflate` only
- [ ] Capture response `Content-Type` header
- [ ] Only treat response as HTML if `Content-Type` contains `text/html` or `application/xhtml`
- [ ] Reject non-HTML content (PDF, images, JSON) early with clear error message

### Success Criteria
"Decode error" bucket drops close to zero.

---

## Phase 2: Stable Browser Identity
**Goal:** Become a believable browser session instead of rotating identities aggressively.

### Implementation
- [ ] Create per-domain identity object:
  - User agent
  - Accept language
  - Accept headers
  - Accept encoding
  - Viewport
  - Referer behavior
- [ ] Keep identity stable across retries for same domain
- [ ] Maintain cookie jar stability per session
- [ ] Reduce header variations - use 3-4 consistent profiles instead of 15+ random ones
- [ ] Add domain fingerprinting to cache identity choices

### Success Criteria
403 and "soft block with empty template" errors reduce. Fewer "works on retry 4 but fails on retry 1" patterns.

---

## Phase 3: Real Extraction Pipeline
**Goal:** Stop falsely rejecting pages as "under 200 chars" when meaningful content exists.

### Implementation

#### Layer 1: DOM Text Extraction
- [ ] Parse HTML with BeautifulSoup or similar
- [ ] Remove script, style, nav, footer elements
- [ ] Extract visible text preserving headings and lists
- [ ] Preserve structure (headings, paragraphs, lists)

#### Layer 2: Main Content Extraction
- [ ] Implement readability-style extraction
- [ ] Score elements by text density, link density, class names
- [ ] Extract main content container
- [ ] Fallback to Layer 1 if extraction yields <100 chars

#### Layer 3: Head-Based Signals
- [ ] Extract `<title>` tag
- [ ] Extract `<meta name="description">`
- [ ] Extract OpenGraph tags (`og:title`, `og:description`)
- [ ] Extract JSON-LD scripts
- [ ] Combine head signals when body is thin

### Success Criteria
"Insufficient content (<200 chars)" bucket drops materially. Small landing pages stop being falsely classified as failures.

---

## Phase 4: JavaScript Shell Handling
**Goal:** Extract value from JS-heavy sites without running a browser.

### Implementation
- [ ] Detect JavaScript shell patterns:
  - Script-to-text ratio > 80%
  - Tiny visible text (<100 chars)
  - Presence of framework markers (React, Vue, Next.js)
- [ ] Extract JSON-LD data (often rich and structured)
- [ ] Extract Next.js/Nuxt.js payload data from `<script id="__NEXT_DATA__">`
- [ ] Parse embedded JSON for page content fragments
- [ ] Mark as "partial success" when head + embedded data yields useful content
- [ ] Distinguish between "full success", "partial success", and "failure"

### Success Criteria
"JS heavy empty page" bucket becomes "partial success" for meaningful share of sites.

---

## Phase 5: Smarter Retries
**Goal:** Make retries strategic, not just longer.

### Implementation
- [ ] Separate connect timeout (5-10s) from read timeout (30-60s)
- [ ] Add per-domain circuit breaker:
  - Track 429, 403 responses
  - Cool down domain for 60s after repeated blocks
  - Reduce concurrency for that domain
- [ ] Reduce per-host concurrency (current global setting is too high for single hosts)
- [ ] Add maximum total time budget per URL (120s)
- [ ] Return best partial result when budget exceeded instead of empty failure

### Success Criteria
Timeouts reduce, average seconds per URL drops, success rate rises.

---

## Phase 6: Intelligent Link Discovery
**Goal:** Get better content with fewer page fetches.

### Implementation
- [ ] Define priority page intents: `about`, `services`, `product`, `pricing`, `case-studies`, `contact`
- [ ] Score links by:
  - Keyword match in URL path
  - Path depth (shorter = better)
  - Same domain preference
  - Not asset files (ignore .pdf, .jpg, etc.)
- [ ] Visit only top 3-5 scored links per site
- [ ] Remove "depth * 2" approach - replace with intent-based scoring

### Success Criteria
Better content per site with fewer total page fetches.

---

## Phase 7: Cloud-Ready Deployment
**Goal:** Design around Streamlit Community Cloud limitations.

### Cloud Mode (Default)
- [ ] Hard cap batch size: 50-300 URLs per run
- [ ] Disable Playwright fallback by default (browser binary issues)
- [ ] Do not rely on disk persistence:
  - Stream results as they complete
  - Offer immediate download
  - Store failures in session memory
  - Download available before run ends
- [ ] Add memory usage monitoring
- [ ] Auto-stop if memory exceeds threshold

### Local Mode (Power Users)
- [ ] Provide downloadable runner script
- [ ] Streamlit app becomes "frontend and configurator"
- [ ] Export config to run locally with full power
- [ ] Keep cloud costs near zero

### Success Criteria
Public Streamlit app stays responsive under light multi-user usage.

---

## Phase 8: Ethics & Stability Guardrails
**Goal:** Improve deliverability through simple compliance.

### Implementation
- [ ] Optional robots.txt check for domains that repeatedly block
- [ ] Slow down (exponential backoff) on blocked domains
- [ ] Respect `Crawl-delay` when present
- [ ] Track domain health score:
  - Successful requests increase score
  - 429/403 decrease score
  - Adjust rate based on score
- [ ] Add user-agent rotation only when identity is blocked

### Success Criteria
Fewer domains escalate into permanent blocks.

---

## Priority Order

| Phase | Impact | Effort | Priority |
|-------|--------|--------|----------|
| 0 - Measurement | High | Low | **P0** |
| 1 - Compression/Decoding | High | Low | **P0** |
| 3 - Extraction Pipeline | High | Medium | **P1** |
| 5 - Smarter Retries | High | Medium | **P1** |
| 2 - Stable Identity | Medium | Medium | P2 |
| 4 - JS Shell Handling | Medium | High | P2 |
| 6 - Link Discovery | Medium | Low | P2 |
| 7 - Cloud Deployment | High | Medium | **P1** |
| 8 - Ethics Guardrails | Low | Low | P3 |

---

## Implementation Notes

### Quick Wins (Do First)
1. **Phase 0** - Add measurement to know what to fix
2. **Phase 1** - Disable Brotli, check Content-Type
3. **Phase 7** - Cap batch size, disable Playwright in cloud

### Dependencies
- Phase 0 enables all other phases (measurement first)
- Phase 3 helps Phase 4 (better extraction enables JS shell detection)
- Phase 2 helps Phase 5 (stable identity enables smarter retries)

### Testing Strategy
1. Run measurement phase on 100-500 URL sample
2. Identify top 3 failure buckets
3. Implement fix for #1 bucket
4. Re-measure, verify bucket reduced
5. Repeat

---

## Success Metrics

Track these over time:
- **Success rate** (target: >85%)
- **Avg time per URL** (target: <30s)
- **False failure rate** (under 200 chars but actually valid)
- **Cost per 1000 URLs** (keep minimal)
- **User-facing error rate** (target: <5%)
