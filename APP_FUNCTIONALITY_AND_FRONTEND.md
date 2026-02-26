# Website Scraper — Functionality and Front-End Details

Single reference for app functionality and current UI/front-end. Built with Streamlit; single-page, vertical scroll, wide layout (max-width 1200px), sidebar collapsed by default.

---

## Functionality

### Resume and partial results
- If previous runs exist in `outputs/`, an expander **"Resume or Download Partial Results"** shows (up to 10 runs).
- Per run: run name, completed/total row count, **Download** button (ZIP of part CSVs, Excel, crash logs).
- Incomplete runs: caption says to re-upload CSV and click Start to resume (resume is automatic on next Start).

### Step 1: Upload CSV
- **File uploader:** CSV only, single file.
- **After upload — CSV configuration:**
  - Radio: "First row has column names?" (Yes / No).
  - Message: "Found N columns" with list preview.
  - Select: "Which column has URLs?" — options from column names or "Column 1", "Column 2" (no headers). Help: Column A in output = this column; pick the one with website URLs.
  - Conditional warning if selected column doesn’t look like URLs (e.g. names).
  - Expander **"Variables from your sheet"**: lists placeholders from other columns (e.g. `{company_name}`) for use in Steps 3 and 4.

### Step 2: Settings
- **Two columns:** Basic Settings | Advanced Settings.
- **Basic:** Keywords (text, comma-separated); Parallel workers (slider 1–max); Force maximum workers (checkbox); Pages to scrape per site (slider 0–5); Retries (number 0–5).
- **Advanced:** Wait time per site (seconds); expander "About Google cache & context"; Text limit per site (chars); Split files every X rows; expander "User-Agent" (text input).

### Step 3: Company Summary (optional)
- Checkbox: "Enable company summaries."
- When on: Provider (OpenAI / Gemini / OpenRouter); API key (password); Model (select, from API or defaults); Prompt radio (default / customize).
- If customize: "Available variables" caption; prompt text area; optional variable-completion select + "Insert variable" when prompt ends with `{`; read-only "Final prompt (preview)"; Preview row select + **"Preview with sample lead"** → dialog with filled prompt and Close.

### Step 4: Email Copy (optional)
- Checkbox: "Enable email copy generation."
- When on: "Use same API key and model as Step 3" or separate Provider / API key / Model; prompt text area; variable completion; read-only preview; Preview row + "Preview with sample lead" (same dialog pattern).

### Token usage estimator
- Expander (collapsed by default). Input: number of URLs (default from CSV). Output: total/per-lead tokens, estimated cost for Step 3 and Step 4. Info message if neither step enabled.

### Test run
- Number input "Test with X rows" (1–20); button **"Test run"** runs first X URLs (scrape + AI) in background.
- While running: info + "Check for results" button.
- When done: "Test results" — expandable Logs; one expander per URL (scraped content, company summary, email copy). Errors at top if any.

### Start Scraping
- Section "Ready to Start" + full-width button **"Start Scraping"**.
- Validates CSV and URL column; shows large-file tips if needed; auto-resumes same URL set if found; starts scraper in background and shows progress.

### Progress (during scrape)
- Progress bar; status line (done/total, ETA or "calculating"); idle message after 30s without completion; elapsed time; expander "Detailed logs"; live dashboard (Progress, Rate, Elapsed, Errors, Active; In progress / Recently completed / Issues lists). Updates ~1s.

### After scrape: downloads
- **Three columns:** ZIP | Excel | CSV.
  - ZIP: "Download ZIP Archive" (all part files + crash logs).
  - Excel: build combined from part CSVs; "Download Combined Excel" + row count. Columns: Website, ScrapedText, CompanySummary, EmailCopy.
  - CSV: same; "Download Combined CSV" (UTF-8 BOM, same columns).
- Expander "View Generated Files" (file names + output path). Tip about UTF-8 BOM and .xlsx.

### Error and state behavior
- Failed rows show "❌ …" in output columns; run crash shows error + partial-results note + "Download run logs." If email was enabled but API key missing, warning and run uses 3 columns (no EmailCopy).

---

## Front-End Details

### Global
- **Streamlit:** `layout="wide"`, `initial_sidebar_state="collapsed"`, `page_icon="🌐"`, `page_title="Web Scraper"`.
- **CSS:** Main container max-width 1200px, padding 2rem; h1 2.5rem, h3 1.5rem, h4 subsection; info boxes light blue, 4px left border #0ea5e9, 8px radius; buttons gradient #667eea → #764ba2, white text, 8px radius, shadow, hover lift; inputs/selects/sliders 8px radius, 2px border #e5e7eb, focus #667eea; file uploader dashed border, hover #667eea; success/error 8px radius; expander headers bold #374151; captions gray #6b7280 0.9rem; main menu, footer, header hidden.

### Hero
- Centered: title "Website Scraper" (gradient text, 3rem, 800 weight); subtitle gray 1.1rem.

### Steps
- Each step: H3 with emoji (e.g. "Step 1: Upload CSV"); tip box with colored left border (blue/green/purple by step), 8px radius, 1rem padding; form controls; Step 2 uses two columns.

### Buttons and CTAs
- Primary: "Start Scraping" full width, gradient. Secondary: "Test run", "Check for results", "Preview with sample lead", "Insert variable", "Download", "Close" (dialog).

### Dialog
- **Sample prompt:** `@st.dialog` title "Sample prompt — filled with one lead's data"; caption; read-only text area (filled prompt); "Close" button. Row chosen on main page.

### Layout and spacing
- Section dividers: horizontal rule (`---`). Expanders for optional/advanced. No custom font; browser default.

### Page order (top to bottom)
1. Resume/partial results expander (if any runs)
2. Step 1: Upload CSV + CSV configuration (after upload)
3. Step 2: Settings (two columns)
4. Step 3: Company Summary (optional)
5. Step 4: Email Copy (optional)
6. Token usage estimator expander
7. Test run (input + button) → then running state or test results
8. "Ready to Start" + Start Scraping button
9. (After start) Progress bar, status, logs expander, live dashboard
10. (After finish) Download section (ZIP | Excel | CSV), View Generated Files, tip
