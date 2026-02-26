# Website Scraper App — Features and Design Brief for UI Designer

This document describes all features, user flows, and current UI design of the app so a designer can propose a better visual and interaction design. The app is built with **Streamlit** (Python), single-page with vertical scroll; layout is wide (max-width 1200px), sidebar collapsed by default.

---

## 1. Product purpose

- **What it does:** Users upload a CSV of website URLs (and optional lead columns like company name). The app scrapes each URL, extracts clean text, and optionally runs AI to generate company summaries and/or personalized email copy. Output is downloadable as Excel and CSV (part files + combined file).
- **Who it’s for:** Sales, marketing, or ops users who need bulk website content and AI-generated summaries/emails from a list of URLs.
- **Key value:** One place to configure CSV → scrape → AI → download; resume after crash; no vendor lock-in (user brings API keys).

---

## 2. Feature list (complete)

### 2.1 Resume and partial results (top of page)

- **Feature:** If previous runs exist in `outputs/`, the app shows an expander **“Resume or Download Partial Results”**.
- **Content:** Up to 10 run folders; for each: run name, completed/total row count, and a **Download** button (ZIP of CSV + Excel + crash logs). For incomplete runs, a caption explains: “Re-upload your CSV and click Start to resume.”
- **Interaction:** Expand/collapse; per-run download button. No “Resume” button — resume happens automatically when user clicks Start Scraping with same CSV.

### 2.2 Step 1: Upload CSV

- **File uploader:** “Choose CSV file”, type CSV only, single file.
- **After upload:** CSV configuration section appears:
  - **Radio:** “First row has column names?” — Yes / No.
  - **Success/Info:** “Found N columns” with column list preview.
  - **Select box:** “Which column has URLs?” — options from column names (or “Column 1”, “Column 2” if no headers). Help text: “Output Column A (Website) will be this column. Pick the one with website addresses (e.g. https://...), not lead names.”
  - **Warning (conditional):** If the selected column doesn’t look like URLs (e.g. mostly names), a warning explains that Column A will be this column and to pick the URL column if they want websites.
  - **Expander “Variables from your sheet”:** Lists placeholder names derived from non-URL columns (e.g. `{company_name}`, `{website_name}`) for use in Steps 3 and 4.

### 2.3 Step 2: Settings

- **Layout:** Two columns (Basic Settings | Advanced Settings).
- **Basic:**
  - **Keywords (optional):** Text input, comma-separated (e.g. “about, service, product”). Caption shows “Looking for: …” or “Leave empty to scrape homepage only.”
  - **Parallel workers:** Slider 1–N (N = CPU-based max, e.g. 96). Default ~20. Help: “Sites scraped at once.”
  - **Force maximum workers:** Checkbox to override to max (for testing).
  - **Pages to scrape per site:** Slider 0–5 (0 = homepage only; 3 = homepage + 3 more). Default 3.
  - **Retries if failed:** Number input 0–5, default 2.
- **Advanced:**
  - **Wait time per site (seconds):** Number input 5–120, default 30. Help mentions fallback to Google cache / archive.org.
  - **Expander “About Google cache & context”:** Explains that only the provided URLs are scraped; Google cache and Archive.org are used when a site is unreachable; no separate “Google search” feature.
  - **Text limit per site:** Number input 1000–50000, step 1000, default 10000 (characters).
  - **Split files every X rows:** Number input 500–50000, default 2000 (or 1000 on low-resource). Help: “Lower = less memory per write.”
  - **Expander “User-Agent (Advanced)”:** Single text input for browser User-Agent string.

### 2.4 Step 3: Company Summary Generator (optional)

- **Checkbox:** “Enable company summaries” — turns on AI company summaries.
- **When enabled:**
  - **Provider:** Select box — OpenAI, Gemini, OpenRouter.
  - **API key:** Password input; captions link to each provider’s key page. “API key saved” success when entered.
  - **Model:** Select box (populated from API when key is present; else defaults). Caption “N models available.”
  - **Prompt:** Radio — “Use default prompt” | “Customize prompt”.
  - **If Customize:**
    - Caption with “Available variables” list.
    - Large text area “Prompt template (edit below)”.
    - Optional “Complete your variable” selectbox + “Insert variable” button when prompt ends with `{` or `{{`.
    - **Final prompt (preview with sample data):** Read-only text area with sample-filled prompt.
    - **Preview row** selectbox (Row 1, Row 2, …) and **“Preview with sample lead”** button — opens a **dialog** “Sample prompt — filled with one lead’s data” with full filled prompt and Close button.
  - **Session:** `ai_prompt` and provider/key/model stored in session state.

### 2.5 Step 4: Email Copy Writer (optional)

- **Checkbox:** “Enable email copy generation” — turns on AI email copy per lead.
- **When enabled:**
  - **Checkbox:** “Use same API key and model as Step 3” (if Step 3 enabled, shows caption “Using Step 3: …”).
  - **Else:** Provider select, API key input, Model select/input (same pattern as Step 3).
  - **Prompt template:** Text area; “Available variables” caption; optional variable completion when prompt ends with `{`/`{{`; “Insert variable” button.
  - **Final prompt (preview with sample data):** Read-only preview.
  - **Preview row** + **“Preview with sample lead”** button — same dialog pattern as Step 3 for email prompt.

### 2.6 Token usage estimator

- **Expander “Token usage estimator”** (collapsed by default).
- **Inputs:** “Number of URLs” (default from CSV if uploaded); rest derived from session (max_chars, depth, retries, prompts, enabled steps, models).
- **Outputs:** Totals (input tokens, output tokens, total); per-lead tokens; estimated cost per step (Company summary, Email copy) and total. Caption with assumptions (success rate, etc.). If neither Step 3 nor 4 enabled, shows info message.

### 2.7 Test run

- **Number input:** “Test with X rows” (1–20, default 5).
- **Caption:** “Run a quick test to see scraped content and AI summary in the browser (no files created).”
- **Button:** “Test run” — runs scraping + AI on first X URLs in a background thread.
- **When running:** Info message “Test is running in the background (1–3 min). Click below to check if results are ready.” + **“Check for results”** button.
- **When done:** Section “Test results” with expandable “Logs” and one expander per result row: URL, Scraped content (text area), Company summary (formatted), Email copy (if enabled). Errors shown at top if any.

### 2.8 Start Scraping (main action)

- **Section:** “Ready to Start” with short copy and one full-width button **“Start Scraping”**.
- **When clicked (with CSV configured):**
  - Validates CSV config and URL column; builds URL list and lead_data_map; shows large-file tips if URLs > 5000 or > 10000.
  - **Auto-resume:** If a run with the same URL set exists and is incomplete, app uses that run folder and shows success “Resuming: … rows in files.”
  - Starts a background thread that runs the async scraper; main thread enters a **progress loop** (see below).

### 2.9 Progress and live activity (during scrape)

- **Progress bar:** Streamlit progress 0–1.
- **Status line:** Rotating fun message + “(done/total) — ETA: …” or “ETA: calculating…”.
- **Idle warning:** If no completion for 30+ s: “No completed URLs for Xs — N still in progress. Do not refresh.” or “No progress for Xs. App will wait for recovery…”
- **Elapsed:** “Elapsed: Xm Ys” when not idle.
- **Expander “Detailed logs”:** Code block with timestamped run logs (writer, workers, errors).
- **Dashboard (live activity):**
  - **Columns:** Progress (done/total, %), Rate (per min), Elapsed, Errors (count), Active (counts: scraping, AI, email).
  - **Sections:** “In progress” (last 10 URLs with status badge), “Recently completed” (last 8), “Issues” (last 6 errors with URL + message).
- **Loop:** Updates every ~1 s; thread runs scraper and writes part CSV/Excel; progress callback updates done count.

### 2.10 After scrape: downloads

- **Layout:** Three columns — Download ZIP | Excel Files | CSV Files.
- **ZIP:** Button “Download ZIP Archive” (all part CSVs + Excel + crash logs in one ZIP).
- **Excel:** “Create combined Excel” from part CSVs (source of truth); fallback to part Excel if no CSV. Download button “Download Combined Excel” + caption with row count. Combined has 4 columns: Website, ScrapedText, CompanySummary, EmailCopy.
- **CSV:** Same idea; “Download Combined CSV” with UTF-8 BOM; same 4 columns.
- **Expander “View Generated Files”:** Lists CSV file names, Excel file names, and output directory path.
- **Tip:** “CSV files use UTF-8 encoding with BOM for Excel compatibility. Excel files (.xlsx) are ready to open directly.”

### 2.11 Error handling and messaging

- **Scrape errors:** Rows get “❌ …” in ScrapedText or CompanySummary/EmailCopy; these are still written so the user sees which URL failed.
- **Run crash:** Error message at top; warning about partial results and “Resume or Download Partial Results”; info about “Download run logs” for bug reports.
- **Email copy disabled at run time:** If “Enable email copy” was checked but API key/provider/model missing, a warning is shown and the run uses 3 columns (no EmailCopy).

### 2.12 Backend behavior (for design context)

- **Fallbacks when a URL fails:** Direct fetch → Google cache → Archive.org → Playwright (headless) → Common Crawl. All automatic; no user toggles.
- **Output columns:** Always Website, ScrapedText, CompanySummary; EmailCopy only when Step 4 was enabled and API key valid at run start.
- **Resume:** Same CSV URL set → same run folder; writer continues from last checkpoint; part files appended.

---

## 3. Current UI design (visual and structure)

### 3.1 Global

- **Framework:** Streamlit; `layout="wide"`, `initial_sidebar_state="collapsed"`, `page_icon="🌐"`, `page_title="Web Scraper"`.
- **Custom CSS (injected):**
  - Main container: max-width 1200px, padding top/bottom 2rem.
  - Headers: h1 2.5rem, h3 1.5rem, h4 for subsections; colors #1f2937, #374151.
  - Info boxes: light blue background, left border 4px #0ea5e9, border-radius 8px.
  - Buttons: gradient #667eea → #764ba2, white text, rounded, shadow; hover lift + stronger shadow.
  - Inputs/selects/sliders: 8px border-radius, 2px border #e5e7eb; focus border #667eea and light shadow.
  - File uploader: dashed border, hover to #667eea and light background.
  - Success/Error: 8px radius, padding 1rem.
  - Expanders: bold header #374151.
  - Captions: gray #6b7280, 0.9rem.
  - Main menu, footer, header: hidden (no default Streamlit branding).

### 3.2 Hero

- Centered block:
  - **Title:** “Website Scraper” — gradient text (same purple gradient), 3rem, font-weight 800.
  - **Subtitle:** “Scrape websites from your CSV file and get clean, structured text content” — gray, 1.1rem.

### 3.3 Step cards (conceptual)

- Each step has:
  - **Markdown H3** with emoji (e.g. “Step 1: Upload CSV”, “Step 2: Settings”).
  - **Tip box:** Colored left border (blue Step 1, green Step 2, purple Step 3, green Step 4), rounded, padding 1rem.
  - Form controls and optional expanders; two-column layout in Step 2.

### 3.4 Buttons and CTAs

- **Primary CTA:** “Start Scraping” — full width, same gradient as global buttons.
- **Secondary:** “Test run”, “Check for results”, “Preview with sample lead”, “Insert variable”, “Download”, “Close” (in dialog). Standard Streamlit button styling with custom CSS override for primary gradient.

### 3.5 Dialogs

- **Sample prompt dialog:** Streamlit `@st.dialog` — title “Sample prompt — filled with one lead’s data”; caption; read-only filled prompt text area; “Close” button. No row selector inside dialog (row chosen on main page).

### 3.6 Typography and spacing

- Section dividers: `st.markdown("---")` (horizontal rule).
- Spacing between sections: markdown blocks and captions; expanders for optional/advanced content.
- No custom font stack specified (browser default).

---

## 4. User flows (for UX redesign)

1. **First-time run:** Upload CSV → configure URL column (and see variables) → set Step 2 (optional keywords, workers, depth) → optionally enable Step 3 and/or 4 and set API key/model/prompt → (optional) Test run → Start Scraping → wait on progress → download Combined Excel/CSV or ZIP.
2. **Resume after crash:** Open app → see “Resume or Download Partial Results” → re-upload same CSV → Start Scraping → app detects same URL set and resumes → progress continues → download when done.
3. **Preview prompts:** In Step 3 or 4, choose “Customize prompt” → edit template → choose “Preview row” → “Preview with sample lead” → dialog shows filled prompt → Close.
4. **Estimate cost:** Upload CSV (or set URL count) → enable Step 3 and/or 4 → open “Token usage estimator” → see tokens and cost.

---

## 5. What to improve (designer brief)

- **Hierarchy and scannability:** Long single page; consider clearer step numbering, collapsible steps, or a stepper so users know where they are.
- **Progressive disclosure:** Many options (advanced settings, User-Agent, token estimator) are in expanders; consider which should be more visible or grouped.
- **Feedback:** Progress is clear; test run “Check for results” is manual — consider optional auto-refresh or clearer “running” state.
- **Downloads:** Three columns (ZIP, Excel, CSV) after run; consider one “Download” area with format tabs or a single “Download results” with format choice.
- **Consistency:** Unify tip box colors and step treatment; ensure one primary CTA (Start Scraping) and consistent secondary actions.
- **Accessibility:** Ensure contrast for captions and errors; focus states are styled but could be checked for keyboard/screen-reader use.
- **Mobile:** Streamlit is responsive but the app is dense; consider which blocks should stack or simplify on small screens.

---

## 6. Technical constraints (for redesign)

- **Stack:** Streamlit only; no separate front-end. Any redesign should be achievable with Streamlit components, custom CSS, and optional `st.fragment` for partial reruns.
- **State:** Session state holds CSV config, API keys, prompts, and run state (test results, progress). No backend DB; file system for outputs and checkpoints.
- **Output:** Excel (openpyxl) and CSV (UTF-8 BOM); column set is fixed (Website, ScrapedText, CompanySummary, EmailCopy when email enabled).

This document is the single source of truth for features and current design for the UI designer.
