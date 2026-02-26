# Signal Packs and Provider Architecture — Full Plan

*Saved for later implementation. See APP_FEATURES_AND_DESIGN_FOR_UI_DESIGNER.md for current app features and design.*

## Current state (brief)

- Single file: python-scraper.py (~5.8k lines).
- Worker outputs a **fixed tuple** `(url, scraped_text, ai_summary, email_copy)`; writer expects 3–4 columns (`Website`, `ScrapedText`, `CompanySummary`, `EmailCopy`).
- Fetch flow: `fetch()` then on failure `_fetch_from_cache_fallbacks()` (Google cache → archive.org → Playwright → Common Crawl). No explicit `source_type` or confidence.
- Discovery: `extract_links(html, base_url)` from homepage + keyword-matched links; no sitemap, no heuristic `/about`, `/contact`, `/careers`.
- Text: `cleanup_html()` on fetched HTML; optional multi-page (homepage + links by keywords/depth).

---

## Step 1: Core data types and fetch pipeline (refactor)

- Add **FetchArtifacts**, **SiteProfile**, **SignalResult**, **BaseSignal** types.
- Fetch pipeline as explicit state machine with `source_type` (live | google_cache | archive | playwright | common_crawl).
- Refactor worker/scrape_site to build FetchArtifacts and pass to signal runners.

## Step 2: Domain and site health signals (DomainProbe + TLS)

- DomainProbe: HEAD/GET, redirect chain, status, headers, timings.
- TLSInspector: ssl certificate fields (issuer, expiry).
- Output columns: final_url, redirect_count, http_status, tls_issuer, tls_expiry, response_time_ms.

## Step 3: Metadata extraction pack

- Parse head with BeautifulSoup: title, meta description, Open Graph, Twitter cards, canonical, language, robots.
- Columns: meta_title, meta_description, og_title, og_description, canonical_url, language, robots_directive.

## Step 4: Structured data and entity pack (JSON-LD)

- Extract script type="application/ld+json"; flatten Organization, Product, Person.
- Columns: org_name, org_logo, same_as_links, etc.

## Step 5: Contact discovery pack

- Regex for email/phone; prioritize /contact, /about, /team; dedupe and confidence.
- Columns: emails, phones, contact_page_url, support_url.

## Step 6: Social presence discovery pack

- Normalize outbound links to LinkedIn, Instagram, X, YouTube, GitHub, Crunchbase.
- Columns: linkedin_url, twitter_url, etc.

## Step 7: Tech stack detection pack

- Wappalyzer-style patterns on HTML, headers, script URLs.
- Columns: tech_cms, tech_ecommerce, tech_analytics, tech_chat, tech_cdn.

## Step 8: Hiring signals pack (careers + ATS)

- Prioritize careers/jobs URLs; detect ATS; parse job list or HTML job cards.
- Columns: careers_page_url, open_roles_count, ats_vendor, departments, locations, remote_hint.

## Step 9: Content velocity and freshness pack

- Discover blog/press; parse dates; content_freshness_days; optional PageDiff (hash) for “recently updated”.

## Step 10: Archive fallback enrichment

- Label source_type and confidence when using archive; same extraction pipeline on archived HTML.

## Step 11: News mention pack (GDELT)

- GDELT API by company/domain; mention counts, top story URLs, headlines.

## Step 12: Domain registration (RDAP)

- RDAP query; registration/expiry dates, registrar.

## Step 13: Extraction quality and confidence scoring

- QualityRater: confidence_score, failure_reason (blocked, thin_content, cookie_wall).

## Architecture: Discovery, output, UI, caching, concurrency

- **Discovery:** sitemap.xml + nav links + heuristic paths (/about, /contact, /careers, etc.); capped queue per domain.
- **Output:** Dynamic columns (core + one per signal + signal_evidence JSON); writer accepts dict rows.
- **UI:** Checkbox group for signal packs; advanced toggles; persist in run config.
- **Caching:** Fetch cache (url/source_type); signal cache (domain + settings hash); TTL 7–30 days.
- **Concurrency:** Per-domain limit; global limit; backoff; bot-wall detection.

## Suggested implementation order

1. Foundation (types, state machine, writer dict rows).
2. Domain + metadata + quality.
3. Discovery pipeline.
4. Remaining packs.
5. UI (pack toggles).
6. Caching and concurrency.

---

*Full detail is in the original plan file. This copy is for quick reference in the project.*
