import asyncio
import aiohttp
import pandas as pd
import re
import os
import time
import csv
import zipfile
from html import unescape, escape
import streamlit as st
from datetime import datetime
from io import BytesIO, StringIO, StringIO
import json
import random
import threading
from collections import deque

# AI imports (optional - only if API keys provided)
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# -------------------------
# Utilities
# -------------------------

def _is_low_resource_default() -> bool:
    """Auto-detect: use low-resource mode for slower machines (few CPUs)."""
    try:
        n = os.cpu_count() or 2
        return n <= 4
    except Exception:
        return True


def _get_concurrency_max() -> int:
    """Max parallel workers: liberal formula for I/O-bound scraping. 8 cores → 96, 16 cores → 192."""
    try:
        n = os.cpu_count() or 2
        return min(max(20, n * 12), 250)
    except Exception:
        return 80


def _should_use_fast_mode(total_urls: int) -> bool:
    """Auto-enable fast mode for large runs (500+ URLs)."""
    return total_urls >= 500


def normalize_url(url: str) -> str:
    """
    Normalize URL by adding https:// if protocol is missing.
    More lenient - handles various URL formats and edge cases.
    """
    if not url:
        return url
    
    url = str(url).strip()
    
    # Remove any leading/trailing whitespace, quotes, or brackets
    url = url.strip(' "\'[]()')
    
    # Skip empty URLs
    if not url:
        return url
    
    # If URL doesn't start with http:// or https://, add https://
    if not url.startswith(("http://", "https://")):
        # Remove any protocol-like prefixes that might be malformed
        url = url.lstrip("://")
        
        # If it starts with www., add https://
        if url.startswith("www."):
            url = "https://" + url
        else:
            # Otherwise, add https://
            url = "https://" + url
    
    # Clean up any double slashes (except after http:// or https://)
    url = url.replace(":///", "://")
    url = url.replace(":////", "://")
    
    # Basic validation - ensure it looks like a URL
    # But be lenient - let the actual HTTP request determine validity
    if "://" not in url:
        # Still try to fix it
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
    
    return url


def process_keywords(keyword_string: str) -> list:
    """
    Process keywords string into a clean list.
    Handles: commas with/without spaces, case-insensitive, hyphens, special chars.
    
    Examples:
    - "about,service,product" -> ["about", "service", "product"]
    - "about, service, product" -> ["about", "service", "product"]
    - "About, Service-Product" -> ["about", "service-product"]
    """
    if not keyword_string or not keyword_string.strip():
        return []
    
    # Split by comma and clean each keyword
    keywords = []
    for kw in keyword_string.split(","):
        kw = kw.strip()  # Remove leading/trailing spaces
        if kw:  # Only add non-empty keywords
            keywords.append(kw.lower())  # Normalize to lowercase for matching
    
    return keywords


def match_keyword_in_url(keyword: str, url: str) -> bool:
    """
    Check if keyword matches in URL (case-insensitive, handles hyphens/underscores).
    
    Features:
    - Case-insensitive matching
    - Handles hyphens, underscores, and spaces in URLs
    - Partial word matching (e.g., "about" matches "/about-us")
    - Handles URL encoding
    
    Examples:
    - keyword="about" matches "/about", "/about-us", "/About-Us", "/ABOUT"
    - keyword="service" matches "/services", "/our-services", "/SERVICE"
    """
    if not keyword or not url:
        return False
    
    # Normalize both to lowercase for comparison
    keyword_lower = keyword.lower().strip()
    url_lower = url.lower()
    
    # Replace common separators with the keyword separator for better matching
    # This helps match "about-us" with keyword "about"
    url_normalized = url_lower.replace("-", "").replace("_", "").replace("/", "/")
    
    # Check if keyword appears in URL
    # Also check with common separators
    keyword_variations = [
        keyword_lower,
        keyword_lower.replace("-", ""),
        keyword_lower.replace("_", ""),
    ]
    
    for kw_var in keyword_variations:
        if kw_var in url_lower or f"/{kw_var}" in url_lower or f"{kw_var}/" in url_lower:
            return True
    
    return False


# -------------------------
# Checkpoint for crash recovery and resume
# -------------------------

CHECKPOINT_FILENAME = "checkpoint.json"


def save_checkpoint(checkpoint_data: dict, checkpoint_path: str) -> None:
    """Save checkpoint to disk. Handles JSON serialization of completed_urls (set -> list)."""
    try:
        data = checkpoint_data.copy()
        # Convert set to list for JSON
        data["completed_urls"] = list(data.get("completed_urls", []))
        data["last_updated"] = datetime.now().isoformat()
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Checkpoint save failed: {e}")


def load_checkpoint(checkpoint_path: str) -> dict | None:
    """Load checkpoint from disk. Returns None if file doesn't exist or is invalid."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        completed = data.get("completed_urls", [])
        if isinstance(completed, list):
            data["completed_urls"] = set(completed)
        elif isinstance(completed, set):
            data["completed_urls"] = completed
        else:
            data["completed_urls"] = set()
        return data
    except Exception as e:
        print(f"⚠️ Checkpoint load failed: {e}")
        return None


def get_checkpoint_path(output_dir: str) -> str:
    """Return the checkpoint file path for a run."""
    return os.path.join(output_dir, CHECKPOINT_FILENAME)


def get_actual_completed_from_files(run_path: str) -> tuple[int, set]:
    """Count actual rows and extract URLs from output_part_*.csv files. Returns (row_count, set of urls)."""
    total_rows = 0
    urls = set()
    if not run_path or not os.path.isdir(run_path):
        return 0, urls
    try:
        for f in sorted(os.listdir(run_path)):
            if f.startswith("output_part_") and f.endswith(".csv") and "combined" not in f.lower():
                path = os.path.join(run_path, f)
                try:
                    df = pd.read_csv(path, encoding="utf-8-sig", usecols=[0], nrows=None)  # Website col
                    if len(df.columns) > 0:
                        col = df.columns[0]
                        for v in df[col].dropna().astype(str).str.strip():
                            if v and v.strip():
                                urls.add(v)
                        total_rows += len(df)
                except Exception:
                    pass
    except Exception:
        pass
    return total_rows, urls


def reconcile_checkpoint_with_files(output_dir: str) -> int:
    """If checkpoint has more completed_urls than actual rows in files, fix checkpoint. Returns actual count."""
    ck_path = get_checkpoint_path(output_dir)
    ck = load_checkpoint(ck_path)
    if not ck:
        return 0
    actual_rows, actual_urls = get_actual_completed_from_files(output_dir)
    completed = ck.get("completed_urls", set())
    if len(completed) > actual_rows and actual_rows >= 0:
        ck["completed_urls"] = actual_urls
        part_files = [f for f in os.listdir(output_dir) if f.startswith("output_part_") and (f.endswith(".csv") or f.endswith(".xlsx"))]
        part_nums = []
        for f in part_files:
            try:
                # output_part_1.csv -> 1
                n = int(f.replace("output_part_", "").replace(".csv", "").replace(".xlsx", ""))
                part_nums.append(n)
            except ValueError:
                pass
        ck["last_part"] = max(part_nums) if part_nums else 0
        save_checkpoint(ck, ck_path)
        return actual_rows
    return len(completed)


# -------------------------
# AI Summary Generation
# -------------------------

def _col_to_placeholder(col_name: str) -> str:
    """Convert column name to placeholder key: 'Company Name' -> 'company_name'."""
    if not col_name:
        return ""
    key = re.sub(r'[^\w\s]', '', str(col_name).strip())
    key = re.sub(r'\s+', '_', key).lower()
    return key or "col"


def _format_var_value(v) -> str:
    """Format any value for safe insertion into prompts. Handles None, NaN, control chars. Preserves newlines/formatting."""
    if v is None:
        return ""
    try:
        import math
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "n/a"):
        return ""
    s = s.replace("\x00", "")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", s)
    return s.strip()


def _replace_prompt_variables(template: str, data: dict) -> str:
    """
    Replace {key} and {{key}} placeholders with actual values. Both brace styles get the full value.
    Longest keys first so e.g. {company_name} is not broken by {company}. Formatting preserved.
    """
    if not template:
        return template
    keys_sorted = sorted(data.keys(), key=len, reverse=True)
    out = template
    for k in keys_sorted:
        if k == "scraped_content":
            sval = str(data.get(k, ""))[:15000].replace("\x00", "")
        else:
            sval = _format_var_value(data.get(k, "")) or ""
        out = out.replace("{{" + k + "}}", sval)
        out = out.replace("{" + k + "}", sval)
    return out


def _get_variable_definitions() -> list[tuple[str, str, str]]:
    """
    Return list of (display_label, placeholder, key) for prompt variables.
    Driven 100% by the uploaded sheet: URL column + all other columns + company_name + scraped_content.
    """
    result = []
    csv_cfg = st.session_state.get("csv_config") or {}
    url_col = csv_cfg.get("url_column")
    df = csv_cfg.get("df_preview")
    has_headers = csv_cfg.get("has_headers", True)
    col_list = list(df.columns) if df is not None and has_headers else ([f"Column {i+1}" for i in range(len(df.columns))] if df is not None else [])

    result.append(("URL", "{url}", "url"))
    result.append(("Company name", "{company_name}", "company_name"))
    for col in col_list:
        if col == url_col:
            continue
        key = _col_to_placeholder(col)
        if not key:
            continue
        result.append((str(col), "{" + key + "}", key))
    result.append(("Scraped content", "{scraped_content}", "scraped_content"))
    return result


def _get_available_placeholders() -> list[str]:
    """Return list of placeholder keys available for prompts."""
    return [r[2] for r in _get_variable_definitions()]


@st.dialog("Sample prompt — filled with one lead's data")
def _sample_prompt_dialog(prompt_key: str, sample_row_index: int | None = None):
    """Show the prompt with variables replaced by one lead's data. User can change the sample row."""
    prompt_text = st.session_state.get(prompt_key, "") or ""
    csv_cfg = st.session_state.get("csv_config") or {}
    df = csv_cfg.get("df_preview")
    n_rows = len(df) if df is not None and not df.empty else 1
    row_options = list(range(n_rows))
    prev_row = st.session_state.get("sample_dialog_row_select", 0)
    if prev_row not in row_options:
        prev_row = 0
    st.caption("Use this to verify structure. Variables are filled from one row of your uploaded sheet.")
    selected_row = st.selectbox(
        "Change sample lead (row from your sheet)",
        options=row_options,
        index=row_options.index(prev_row) if row_options else 0,
        format_func=lambda i: f"Row {i + 1}",
        key="sample_dialog_row_select"
    )
    sample = _get_lead_sample_from_row(selected_row)
    scraped_placeholder = "[Scraped content would appear here for this lead]"
    sample["scraped_content"] = scraped_placeholder
    if prompt_key == "master_prompt":
        filled = build_company_summary_prompt(prompt_text, sample, scraped_placeholder)
        filled = filled + COMPANY_SUMMARY_FINAL_REMINDER
    else:
        filled = build_email_copy_prompt(prompt_text, sample, scraped_placeholder)
    filled = filled.replace("{scraped_content}", scraped_placeholder).replace("{{scraped_content}}", scraped_placeholder)
    st.caption("This is the exact prompt (including formatting) that would be sent to the AI for this lead.")
    st.text_area("Filled prompt", value=filled, height=400, disabled=True, key="sample_dialog_ta", label_visibility="collapsed")
    if st.button("Close", key="sample_dialog_close"):
        st.session_state.pop("_sample_dialog", None)
        st.rerun()


def _get_sample_lead_data_for_preview() -> dict:
    """Get sample values from first CSV row for prompt preview. Returns dict of key -> value."""
    return _get_lead_sample_from_row(0)


def _get_lead_sample_from_row(row_index: int | None = None) -> dict:
    """Get sample values from one CSV row. If row_index is None, picks a random row. For scraped_content we use placeholder text."""
    out = {"url": "https://example.com", "company_name": "Example Company", "scraped_content": "[Scraped content would appear here for this lead]"}
    csv_cfg = st.session_state.get("csv_config") or {}
    df = csv_cfg.get("df_preview")
    if df is None or df.empty:
        return out
    has_headers = csv_cfg.get("has_headers", True)
    n = len(df)
    if n == 0:
        return out
    idx = row_index if row_index is not None and 0 <= row_index < n else random.randint(0, n - 1)
    row = df.iloc[idx]
    url_col = csv_cfg.get("url_column")
    if url_col:
        try:
            if has_headers and url_col in df.columns:
                v = row[url_col]
            else:
                ci = int(str(url_col).replace("Column ", "")) - 1
                v = row.iloc[ci]
            u = "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v).strip()
            if u:
                out["url"] = u
                out["company_name"] = u.replace("https://", "").replace("http://", "").split("/")[0]
        except Exception:
            pass
    lead_cols = csv_cfg.get("lead_data_columns") or {}
    for key, col_name in lead_cols.items():
        try:
            if has_headers and col_name in df.columns:
                val = row[col_name]
            else:
                ci = int(str(col_name).replace("Column ", "")) - 1
                val = row.iloc[ci]
            v = "" if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val).strip()
            out[key] = v
        except Exception:
            out[key] = ""
    return out


# Standard output column names (3 or 4 when email copy enabled)
EXCEL_COLS_3 = ["Website", "ScrapedText", "CompanySummary"]
EXCEL_COLS_4 = ["Website", "ScrapedText", "CompanySummary", "EmailCopy"]


def _clean_excel_value(val) -> str:
    """Single source of truth: clean any value for Excel (prevents corruption, formula injection)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    val = str(val).strip()
    if not val:
        return ""
    val = val.replace('\x00', '')
    val = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', val)
    val = ''.join(c for c in val if (ord(c) == 0x9 or ord(c) == 0xA or ord(c) == 0xD or
                                      0x20 <= ord(c) <= 0xD7FF or 0xE000 <= ord(c) <= 0xFFFD))
    val = re.sub(r'\s+', ' ', val.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')).strip()
    val = val[:32767]
    if val and val[0] in ('=', '+', '-', '@'):
        val = "'" + val
    return val


def _write_excel_sheet(ws, cols: list, df: "pd.DataFrame", sheet_title: str = "Scraped Data") -> None:
    """Write DataFrame to Excel sheet with proper formatting, column widths, frozen header."""
    ws.title = sheet_title
    ws.append(cols)
    for idx in range(len(df)):
        row = df.iloc[idx]
        row_data = [_clean_excel_value(row[c] if c in df.columns else "") for c in cols]
        row_num = idx + 2
        for col_idx, val in enumerate(row_data, start=1):
            ws.cell(row=row_num, column=col_idx, value=str(val) if val is not None else "")
    # Column widths (Website: 45, others: 80)
    widths = [45, 80, 80, 80][:len(cols)]
    for i, w in enumerate(widths, start=1):
        from openpyxl.utils import get_column_letter
        ws.column_dimensions[get_column_letter(i)].width = min(w, 100)
    ws.freeze_panes = "A2"


def _normalize_scrape_error_for_display(msg: str) -> str:
    """Convert emoji-prefixed scrape errors so narrow columns show 'Scrape timeout' not 'Timec'."""
    if not msg or not isinstance(msg, str):
        return ""
    s = str(msg).strip()
    if not s.startswith("❌"):
        return s
    rest = s[1:].lstrip()
    if "Timeout" in rest or "timeout" in rest:
        import re
        m = re.search(r'\((\d+)s?\)', rest)
        secs = m.group(1) if m else "?"
        return f"Scrape timeout ({secs}s exceeded)"
    if ":" in rest:
        _, detail = rest.split(":", 1)
        return f"Scrape error: {detail.strip()}"
    return f"Scrape error: {rest}"


def _normalize_ai_error_for_display(msg: str, prefix: str = "Summary") -> str:
    """Convert emoji-prefixed errors to clearer format for CSV/Excel.
    Uses 'Summary failed:' prefix so narrow columns never truncate to 'AI Sum'."""
    if not msg or not isinstance(msg, str):
        return ""
    s = str(msg).strip()
    if not s.startswith("❌"):
        return s
    rest = s[1:].lstrip()
    if ":" in rest:
        lead, detail = rest.split(":", 1)
        detail = detail.strip()
        if "skipped" in lead.lower():
            return f"{prefix} skipped: {detail}" if detail else f"{prefix} skipped"
        return f"{prefix} failed: {detail}" if detail else f"{prefix} failed"
    return f"{prefix} failed: {rest}" if rest else f"{prefix} failed"


# Appended to company summary prompt when sending to AI (so sample preview matches exactly)
COMPANY_SUMMARY_FINAL_REMINDER = """

═══════════════════════════════════════════════════════════
CRITICAL OUTPUT FORMAT REQUIREMENTS:
═══════════════════════════════════════════════════════════
1. Use EXACT section headers: ===SUMMARY===, ===FACTS===, ===HYPOTHESES===
2. Do NOT use markdown formatting (**, __, #). Use plain text only.
3. Evidence quotes must be COMPLETE sentences, not truncated with "..."
4. Do NOT truncate words mid-word. Always end at sentence boundaries.
5. Keep sections separate. Do NOT mix facts into hypotheses or vice versa.
6. Facts must be grounded (from webcopy). Hypotheses must be inferred (marked with (obs)).
7. Use consistent format:
   - Fact: [statement]
   - Evidence Quote: "[complete quote from webcopy]"
   - Source: [page name or Webcopy]
   - Hypothesis: [inference] (obs)
   - Signal: "[specific wording from webcopy]"
   - Commercial implication: [why it matters]
   - Confidence: High/Medium/Low
8. Do NOT include inline labels like "Confidence: Medium Signal: ..." - use structured format above.
═══════════════════════════════════════════════════════════
"""


def build_company_summary_prompt(base_prompt: str, lead_data: dict | None, scraped_content: str | None) -> str:
    """
    Build the complete prompt for AI company summary generation.
    Scraped content is inserted only where {scraped_content} or {{scraped_content}} appears;
    it is never injected elsewhere, so no double-adding.
    
    Args:
        base_prompt: User-provided base prompt (with placeholders)
        lead_data: Dictionary with lead information (URL, company name, etc.)
        scraped_content: Scraped website content (replaces {scraped_content} in prompt)
    
    Returns:
        Complete prompt string ready for AI
    """
    # Default prompt structure - Commercial analysis focused
    default_base = """You are Hypothesis Bot… an advanced commercial analysis agent.

INPUT
You will receive ONLY one input: raw website copy scraped from a company's website (may include multiple pages).

CORE JOB
Turn messy webcopy into:
1) a sharp company intelligence SUMMARY
2) a clean set of FACTS grounded in the text
3) commercially relevant HYPOTHESES inferred from signals, gaps, tone, and structure

You are not writing outreach. You are building the intelligence layer that enables personalization later.

NON NEGOTIABLE RULES
1) Do not invent facts. If it is not explicitly supported by the webcopy, it is not a fact.
2) Facts must be short and must include Evidence Quote… an exact snippet from the webcopy.
3) Hypotheses must be explicitly labeled as hypotheses and must include:
   • Signal… the specific wording or structural clue that triggered the inference
   • Commercial implication… why it matters in a sales or growth context
   • Confidence: High, Medium, or Low
4) Never mention any external tools or data sources (Apollo, LinkedIn, Crunchbase, funding, headcount, etc.). You only have webcopy.
5) If the company name is unclear, write "Company: Not explicitly stated" and proceed.
6) Avoid cringe adjectives like great, amazing, innovative. Be surgical.
7) Avoid criticism. Frame gaps neutrally as "signals" or "absence suggests".
8) Keep it concise. No fluff. No extra sections.

ANALYSIS GUIDELINES
When extracting facts, prioritize:
• What they do (offer categories, deliverables)
• Who they serve (industries, segments, buyer language)
• How they sell (engagement models, pricing mentions, process)
• Proof (case studies, client names, testimonials, quantified claims)
• Differentiators (positioning phrases, guarantees, compliance, security)
• Operational signals (hiring, support hours, global language, locations)
• Technology signals (stacks, platforms, integrations) only if stated

When generating hypotheses, prioritize:
• Likely buying triggers (growth, hiring, new initiatives, modernization)
• Likely pain points (capacity, speed, differentiation, trust, compliance, delivery)
• Likely maturity level (specialist vs generalist, product vs services)
• Likely stakeholder priorities (risk reduction, outcomes, speed, cost certainty)
• Contradictions or gaps between claims and proof
Each hypothesis must connect to a commercial implication.

LEAD INFORMATION:
- Website URL: {url}
- Company Name: {company_name}

WEBSITE CONTENT (ONLY SOURCE OF INFORMATION):
{scraped_content}

OUTPUT FORMAT (STRICT - NO MARKDOWN, NO TRUNCATION)
Return exactly these 3 sections with EXACT headers:

===SUMMARY===
Write 3 to 5 sentences in one paragraph describing:
• what the company appears to do
• who it appears to serve
• how it positions itself
• one notable proof element (only if present)
Mark inferences with (obs).
Do NOT use markdown (**, __, #). Use plain text only.

===FACTS===
Provide 10 to 18 facts.
Format each EXACTLY as:
Fact: [statement]
Evidence Quote: "[COMPLETE quote from webcopy - do NOT truncate with ...]"
Source: [Page name if present, otherwise "Webcopy"]

CRITICAL: Evidence quotes must be COMPLETE sentences from the webcopy. Do NOT truncate mid-word or mid-sentence.

===HYPOTHESES===
Provide 6 to 12 hypotheses.
Format each EXACTLY as:
Hypothesis: [inference statement] (obs)
Signal: "[specific wording or structural clue from webcopy]"
Commercial implication: [why it matters in sales/growth context]
Confidence: High or Medium or Low

CRITICAL: 
- Do NOT use markdown formatting (**, __, #)
- Do NOT truncate words mid-word
- Do NOT mix facts into hypotheses section
- Keep facts grounded (from webcopy), hypotheses inferred (from signals)"""
    
    # Use user prompt if provided, otherwise use default
    prompt_template = base_prompt if base_prompt.strip() else default_base
    
    lead_data = dict(lead_data or {})
    # Ensure company_name (default from URL if missing)
    company_name = _format_var_value(lead_data.get('company_name')) or ''
    if not company_name or company_name == 'Unknown':
        url_str = lead_data.get('url', '')
        company_name = url_str.replace('https://', '').replace('http://', '').split('/')[0] if url_str else 'Unknown'
    lead_data['company_name'] = company_name
    lead_data['url'] = _format_var_value(lead_data.get('url')) or 'N/A'
    lead_data['scraped_content'] = (scraped_content or "")[:15000]

    def _humanize_key(k):
        return k.replace('_', ' ').title()

    # Build lead information section from lead_data only (exclude scraped_content so it appears only via {scraped_content})
    lead_info_parts = [f"- {_humanize_key(k)}: {_format_var_value(v)}" for k, v in sorted(lead_data.items()) if k != 'scraped_content' and _format_var_value(v)]
    lead_info_section = "LEAD INFORMATION:\n" + "\n".join(lead_info_parts) if lead_info_parts else "LEAD INFORMATION:\n(no additional lead data)"

    # Replace all placeholders including {scraped_content}; scraped content is never injected elsewhere
    prompt = _replace_prompt_variables(prompt_template, lead_data)

    if 'LEAD INFORMATION:' in prompt:
        prompt = re.sub(
            r'LEAD INFORMATION:.*?(?=WEBSITE CONTENT:|$)',
            lead_info_section + '\n\n',
            prompt,
            flags=re.DOTALL
        )
    else:
        if 'WEBSITE CONTENT:' in prompt:
            prompt = prompt.replace('WEBSITE CONTENT:', lead_info_section + '\n\nWEBSITE CONTENT:')
        else:
            prompt = lead_info_section + '\n\n' + prompt

    return prompt


async def fetch_openai_models(api_key: str) -> list:
    """Fetch ALL available OpenAI models from API in real-time."""
    if not OPENAI_AVAILABLE:
        return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]  # Fallback
    
    try:
        client = AsyncOpenAI(api_key=api_key)
        models_response = await client.models.list()
        
        # Get ALL GPT models (chat completion capable)
        all_models = [model.id for model in models_response.data]
        
        # Filter for GPT models - Get ALL GPT models that can be used for chat completions
        # Only exclude: embeddings, deprecated models, and base/completion models (ada, babbage, curie, davinci)
        # Include ALL GPT models including variants like gpt-4o, gpt-4-turbo, gpt-3.5-turbo, gpt-4o-mini, etc.
        # IMPORTANT: Include ALL GPT models, don't filter by 'chat' or 'instruct' as those are valid chat models
        gpt_models = []
        skip_patterns = ['embedding', 'ada', 'babbage', 'curie', 'davinci', 'deprecated']
        
        for model_id in all_models:
            model_lower = model_id.lower()
            # Include if it's a GPT model
            if 'gpt' in model_lower:
                # Skip only if it's clearly an embedding or base model (not chat models)
                # Include ALL GPT models including instruct, turbo, o, etc.
                if not any(skip in model_lower for skip in skip_patterns):
                    gpt_models.append(model_id)
        
        # Debug: Log how many models we found (remove in production)
        # print(f"Found {len(gpt_models)} GPT models: {gpt_models[:10]}...")
        
        # Remove duplicates and sort
        unique_models = sorted(list(set(gpt_models)), reverse=True)
        
        # Prioritize common/recommended models at the top
        priority_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        ordered = []
        
        # Add priority models first (if they exist)
        for pm in priority_models:
            if pm in unique_models:
                ordered.append(pm)
        
        # Add ALL remaining models (not just filtered ones)
        for model in unique_models:
            if model not in ordered:
                ordered.append(model)
        
        return ordered if ordered else priority_models
    except Exception as e:
        # Return fallback models on error
        return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]


async def fetch_gemini_models(api_key: str) -> list:
    """Fetch available Gemini models from API."""
    if not GEMINI_AVAILABLE:
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]  # Fallback
    
    try:
        genai.configure(api_key=api_key)
        # Gemini models are typically fixed, but we can verify
        # Common Gemini models
        common_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
        
        # Try to get list of models (if API supports it)
        try:
            # Note: Gemini API doesn't have a direct models.list() endpoint
            # So we'll use common models and verify they work
            return common_models
        except:
            return common_models
    except Exception as e:
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]


async def fetch_openrouter_models(api_key: str = "") -> list:
    """Fetch available models from OpenRouter API. No API key needed for listing."""
    fallback = [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "google/gemini-2.0-flash-exp:free",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-1.5-flash",
    ]
    try:
        async with aiohttp.ClientSession() as session:
            headers = {}
            if api_key and api_key.strip():
                headers["Authorization"] = f"Bearer {api_key.strip()}"
            async with session.get("https://openrouter.ai/api/v1/models", headers=headers) as resp:
                if resp.status != 200:
                    return fallback
                data = await resp.json()
        models_data = data.get("data", [])
        if not models_data:
            return fallback
        model_ids = [m.get("id") for m in models_data if m.get("id")]
        if not model_ids:
            return fallback
        priority = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-2.0-flash",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-1.5-flash",
        ]
        ordered = [m for m in priority if m in model_ids]
        for m in model_ids:
            if m not in ordered:
                ordered.append(m)
        return ordered
    except Exception:
        return fallback


async def generate_openrouter_summary(api_key: str, model: str, prompt: str, max_retries: int = 5, status_callback=None) -> str:
    """Generate company summary using OpenRouter API (OpenAI-compatible)."""
    if not OPENAI_AVAILABLE:
        return "❌ OpenAI library not installed. Install with: pip install openai"
    
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key.strip(),
    )
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are Hypothesis Bot, an advanced commercial analysis agent. Your job is to turn messy webcopy into structured intelligence: ===SUMMARY===, ===FACTS=== (with complete evidence quotes), and ===HYPOTHESES=== (with signals, commercial implications, and confidence levels). CRITICAL: Use EXACT section headers (===SUMMARY===, ===FACTS===, ===HYPOTHESES===). Do NOT use markdown formatting (**, __, #). Evidence quotes must be COMPLETE sentences, not truncated. Do NOT truncate words mid-word. Never invent facts. Only use what's explicitly in the webcopy. Be surgical and concise. Follow the output format STRICTLY."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            
            is_rate_limit = (
                "rate limit" in error_str or
                "429" in error_str or
                "quota" in error_str or
                "too many requests" in error_str or
                "credit" in error_str
            )
            
            if is_rate_limit:
                wait_time = min(2 ** attempt, 60)
                if status_callback:
                    status_callback(f"⏳ Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                await asyncio.sleep(wait_time)
                continue
            
            is_retryable = (
                "timeout" in error_str or
                "connection" in error_str or
                "503" in error_str or
                "502" in error_str or
                "500" in error_str
            )
            
            if is_retryable and attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)
                if status_callback:
                    status_callback(f"⚠️ API error. Retrying in {wait_time}s ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(wait_time)
                continue
            
            if attempt == max_retries - 1:
                if is_rate_limit:
                    return f"❌ OpenRouter Rate Limit: Exceeded after {max_retries} retries."
                return f"❌ OpenRouter API Error: {error_msg}"
            
            await asyncio.sleep(1 + attempt)
    
    return "❌ OpenRouter API failed after retries"


async def generate_openai_summary(api_key: str, model: str, prompt: str, max_retries: int = 5, status_callback=None) -> str:
    """Generate company summary using OpenAI API with automatic rate limit handling."""
    if not OPENAI_AVAILABLE:
        return "❌ OpenAI library not installed. Install with: pip install openai"
    
    client = AsyncOpenAI(api_key=api_key, timeout=120.0)
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are Hypothesis Bot, an advanced commercial analysis agent. Your job is to turn messy webcopy into structured intelligence: ===SUMMARY===, ===FACTS=== (with complete evidence quotes), and ===HYPOTHESES=== (with signals, commercial implications, and confidence levels). CRITICAL: Use EXACT section headers (===SUMMARY===, ===FACTS===, ===HYPOTHESES===). Do NOT use markdown formatting (**, __, #). Evidence quotes must be COMPLETE sentences, not truncated. Do NOT truncate words mid-word. Never invent facts. Only use what's explicitly in the webcopy. Be surgical and concise. Follow the output format STRICTLY."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent formatting
                max_tokens=4000,  # Increased for complete quotes
                top_p=0.9  # More focused responses
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            
            # Check for rate limit errors
            is_rate_limit = (
                "rate limit" in error_str or
                "429" in error_str or
                "quota" in error_str or
                "too many requests" in error_str or
                "requests per minute" in error_str or
                "rate_limit_exceeded" in error_str
            )
            
            if is_rate_limit:
                # Calculate exponential backoff: 2^attempt seconds, max 60 seconds
                wait_time = min(2 ** attempt, 60)
                
                if status_callback:
                    status_callback(f"⏳ Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                
                await asyncio.sleep(wait_time)
                continue  # Retry after waiting
            
            # Check for other retryable errors
            is_retryable = (
                "timeout" in error_str or
                "connection" in error_str or
                "503" in error_str or
                "502" in error_str or
                "500" in error_str
            )
            
            if is_retryable and attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)
                
                if status_callback:
                    status_callback(f"⚠️ API error. Retrying in {wait_time}s ({attempt + 1}/{max_retries})...")
                
                await asyncio.sleep(wait_time)
                continue
            
            # Final attempt or non-retryable error
            if attempt == max_retries - 1:
                if is_rate_limit:
                    return f"❌ OpenAI Rate Limit: Exceeded after {max_retries} retries. Please wait a few minutes and try again, or upgrade your API plan."
                return f"❌ OpenAI API Error: {error_msg}"
            
            # Default wait for other errors
            await asyncio.sleep(1 + attempt)


async def generate_gemini_summary(api_key: str, model: str, prompt: str, max_retries: int = 5, status_callback=None) -> str:
    """Generate company summary using Google Gemini API with automatic rate limit handling."""
    if not GEMINI_AVAILABLE:
        return "❌ Gemini library not installed. Install with: pip install google-generativeai"
    
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return f"❌ Gemini API Configuration Error: {str(e)}"
    
    for attempt in range(max_retries):
        try:
            # Configure generation parameters for analytical but factual output
            generation_config = {
                "temperature": 0.2,  # Lower temperature for more consistent formatting
                "max_output_tokens": 4000,  # Increased for complete quotes
                "top_p": 0.9,
                "top_k": 40,  # More focused
            }
            
            gemini_model = genai.GenerativeModel(
                model,
                generation_config=generation_config
            )
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: gemini_model.generate_content(prompt)
            )
            if hasattr(response, 'text'):
                return response.text.strip()
            else:
                return f"❌ Gemini API returned unexpected response format"
        except Exception as e:
            error_str = str(e).lower()
            error_msg = str(e)
            
            # Check for rate limit errors
            is_rate_limit = (
                "rate limit" in error_str or
                "429" in error_str or
                "quota" in error_str or
                "resource_exhausted" in error_str or
                "too many requests" in error_str or
                "requests per minute" in error_str or
                "per_minute" in error_str
            )
            
            if is_rate_limit:
                # Calculate exponential backoff: 2^attempt seconds, max 60 seconds
                wait_time = min(2 ** attempt, 60)
                
                if status_callback:
                    status_callback(f"⏳ Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                
                await asyncio.sleep(wait_time)
                continue  # Retry after waiting
            
            # Check for other retryable errors
            is_retryable = (
                "timeout" in error_str or
                "connection" in error_str or
                "503" in error_str or
                "502" in error_str or
                "500" in error_str or
                "unavailable" in error_str
            )
            
            if is_retryable and attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 30)
                
                if status_callback:
                    status_callback(f"⚠️ API error. Retrying in {wait_time}s ({attempt + 1}/{max_retries})...")
                
                await asyncio.sleep(wait_time)
                continue
            
            # Final attempt or non-retryable error
            if attempt == max_retries - 1:
                if is_rate_limit:
                    return f"❌ Gemini Rate Limit: Exceeded after {max_retries} retries. Please wait a few minutes and try again, or check your API quota."
                return f"❌ Gemini API Error: {error_msg}"
            
            # Default wait for other errors
            await asyncio.sleep(1 + attempt)
    
    return "❌ Gemini API failed after retries"


def clean_and_structure_ai_output(raw_output: str) -> str:
    """
    Post-process AI output to fix all formatting issues:
    - Remove markdown formatting
    - Enforce strict section boundaries
    - Fix quote truncation
    - Ensure sentence-aware truncation
    - Standardize section headers
    - Clean up semantic noise
    """
    if not raw_output or raw_output.startswith("❌"):
        return raw_output
    
    import re
    
    # Step 1: Remove all markdown formatting (**, __, #, etc.)
    cleaned = raw_output
    # Remove bold/italic markdown
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)  # Remove **text**
    cleaned = re.sub(r'__([^_]+)__', r'\1', cleaned)  # Remove __text__
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)  # Remove *text*
    cleaned = re.sub(r'_([^_]+)_', r'\1', cleaned)  # Remove _text_
    # Remove headers
    cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
    
    # Step 2: Normalize section headers to consistent format
    # Find and standardize section headers
    section_patterns = [
        (r'^#*\s*\*?\*?SUMMARY\*?\*?\s*#*:?\s*$', '===SUMMARY==='),
        (r'^#*\s*\*?\*?FACTS\*?\*?\s*#*:?\s*$', '===FACTS==='),
        (r'^#*\s*\*?\*?HYPOTHESES\*?\*?\s*#*:?\s*$', '===HYPOTHESES==='),
        (r'^SUMMARY\s*:?\s*$', '===SUMMARY==='),
        (r'^FACTS\s*:?\s*$', '===FACTS==='),
        (r'^HYPOTHESES\s*:?\s*$', '===HYPOTHESES==='),
    ]
    
    for pattern, replacement in section_patterns:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    # Step 3: Split into sections
    sections = {}
    current_section = None
    current_content = []
    
    lines = cleaned.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('===') and line.endswith('==='):
            # Save previous section
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            # Start new section
            current_section = line.replace('===', '').upper()
            current_content = []
        elif current_section:
            current_content.append(line)
        elif not current_section and line:
            # Content before first section - assume it's SUMMARY
            if 'SUMMARY' not in sections:
                current_section = 'SUMMARY'
                current_content = [line]
    
    # Save last section
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    # Step 4: Clean each section
    cleaned_sections = {}
    
    for section_name, content in sections.items():
        if not content:
            continue
        
        # Clean SUMMARY section
        if section_name == 'SUMMARY':
            # Remove any remaining markdown, normalize whitespace
            cleaned_content = re.sub(r'\s+', ' ', content).strip()
            # Ensure it's a paragraph (no line breaks unless intentional)
            cleaned_content = cleaned_content.replace('\n', ' ')
            cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
            # Limit length to reasonable size (sentence-aware)
            if len(cleaned_content) > 1000:
                # Truncate at sentence boundary
                sentences = re.split(r'([.!?]\s+)', cleaned_content[:1100])
                if len(sentences) > 1:
                    cleaned_content = ''.join(sentences[:-1]).strip()
                else:
                    cleaned_content = cleaned_content[:1000] + '...'
            cleaned_sections[section_name] = cleaned_content
        
        # Clean FACTS section
        elif section_name == 'FACTS':
            facts = []
            # Split by "Fact:" pattern
            fact_blocks = re.split(r'(?=Fact\s*:)', content, flags=re.IGNORECASE)
            
            for block in fact_blocks:
                if not block.strip() or 'Fact:' not in block[:20]:
                    continue
                
                # Extract fact
                fact_match = re.search(r'Fact\s*:\s*(.+?)(?=Evidence Quote:|Source:|$)', block, re.DOTALL | re.IGNORECASE)
                fact_text = fact_match.group(1).strip() if fact_match else ''
                
                # Extract evidence quote (must be complete, not truncated)
                quote_match = re.search(r'Evidence Quote\s*:\s*"([^"]+)"', block, re.DOTALL | re.IGNORECASE)
                if not quote_match:
                    # Try without quotes
                    quote_match = re.search(r'Evidence Quote\s*:\s*(.+?)(?=Source:|$)', block, re.DOTALL | re.IGNORECASE)
                
                quote_text = ''
                if quote_match:
                    quote_text = quote_match.group(1).strip().strip('"\'')
                    # Remove ellipses that indicate truncation (we want complete quotes)
                    quote_text = re.sub(r'\s*\.\.\.\s*$', '', quote_text)
                    # Ensure quote is not mid-word truncated
                    if quote_text and not quote_text[-1].isalnum() and quote_text[-1] not in '.!?"\'':
                        # Might be truncated, try to find sentence end
                        pass
                
                # Extract source
                source_match = re.search(r'Source\s*:\s*(.+?)(?=Fact:|$)', block, re.DOTALL | re.IGNORECASE)
                source_text = source_match.group(1).strip() if source_match else 'Webcopy'
                
                if fact_text:
                    fact_entry = f"Fact: {fact_text}"
                    if quote_text:
                        fact_entry += f'\nEvidence Quote: "{quote_text}"'
                    fact_entry += f'\nSource: {source_text}'
                    facts.append(fact_entry)
            
            cleaned_sections[section_name] = '\n\n'.join(facts[:18])  # Limit to 18 facts
        
        # Clean HYPOTHESES section
        elif section_name == 'HYPOTHESES':
            hypotheses = []
            # Split by "Hypothesis:" pattern
            hyp_blocks = re.split(r'(?=Hypothesis\s*:)', content, flags=re.IGNORECASE)
            
            for block in hyp_blocks:
                if not block.strip() or 'Hypothesis:' not in block[:30]:
                    continue
                
                # Extract hypothesis
                hyp_match = re.search(r'Hypothesis\s*:\s*(.+?)(?=Signal:|Commercial implication:|Confidence:|$)', block, re.DOTALL | re.IGNORECASE)
                hyp_text = hyp_match.group(1).strip() if hyp_match else ''
                hyp_text = re.sub(r'\s*\(obs\)\s*', ' (obs)', hyp_text)  # Normalize (obs) marker
                
                # Extract signal
                signal_match = re.search(r'Signal\s*:\s*"([^"]+)"', block, re.DOTALL | re.IGNORECASE)
                if not signal_match:
                    signal_match = re.search(r'Signal\s*:\s*(.+?)(?=Commercial implication:|Confidence:|$)', block, re.DOTALL | re.IGNORECASE)
                signal_text = signal_match.group(1).strip().strip('"\'') if signal_match else ''
                
                # Extract commercial implication
                impl_match = re.search(r'Commercial implication\s*:\s*(.+?)(?=Confidence:|Signal:|$)', block, re.DOTALL | re.IGNORECASE)
                impl_text = impl_match.group(1).strip() if impl_match else ''
                
                # Extract confidence
                conf_match = re.search(r'Confidence\s*:\s*(High|Medium|Low)', block, re.IGNORECASE)
                conf_text = conf_match.group(1) if conf_match else 'Medium'
                
                if hyp_text:
                    hyp_entry = f"Hypothesis: {hyp_text}"
                    if signal_text:
                        hyp_entry += f'\nSignal: "{signal_text}"'
                    if impl_text:
                        hyp_entry += f'\nCommercial implication: {impl_text}'
                    hyp_entry += f'\nConfidence: {conf_text}'
                    hypotheses.append(hyp_entry)
            
            cleaned_sections[section_name] = '\n\n'.join(hypotheses[:12])  # Limit to 12 hypotheses
    
    # Step 5: Reconstruct output with strict separators
    output_parts = []
    
    if 'SUMMARY' in cleaned_sections:
        output_parts.append('===SUMMARY===')
        output_parts.append(cleaned_sections['SUMMARY'])
        output_parts.append('')
    
    if 'FACTS' in cleaned_sections:
        output_parts.append('===FACTS===')
        output_parts.append(cleaned_sections['FACTS'])
        output_parts.append('')
    
    if 'HYPOTHESES' in cleaned_sections:
        output_parts.append('===HYPOTHESES===')
        output_parts.append(cleaned_sections['HYPOTHESES'])
    
    result = '\n'.join(output_parts).strip()
    
    # Step 6: Final cleanup - remove any remaining markdown artifacts
    result = re.sub(r'\*\*', '', result)
    result = re.sub(r'__', '', result)
    result = re.sub(r'#{1,6}\s+', '', result)
    
    # Ensure no mid-word truncation (check for patterns like "word...word")
    result = re.sub(r'(\w+)\.\.\.(\w+)', r'\1 \2', result)
    
    return result if result else raw_output  # Return original if cleaning failed


async def generate_ai_summary(
    api_key: str,
    provider: str,
    model: str,
    prompt: str,
    lead_data: dict,
    scraped_content: str,
    status_callback=None
) -> str:
    """
    Generate AI summary for a company.
    
    Args:
        api_key: API key for the AI provider
        provider: 'openai' or 'gemini'
        model: Model name (e.g., 'gpt-4', 'gemini-pro')
        prompt: Base prompt template
        lead_data: Lead information dictionary
        scraped_content: Scraped website content
    
    Returns:
        Generated summary string (cleaned and structured)
    """
    if not api_key or not api_key.strip():
        return "❌ No API key provided"
    
    if not scraped_content or scraped_content.startswith("❌"):
        return "❌ No valid scraped content available"
    
    # Validate scraped content has meaningful content before sending to AI
    if len(scraped_content.strip()) < 50:
        return "❌ Insufficient content scraped. Content too short to generate meaningful summary."
    
    # Build complete prompt with STRICT formatting requirements
    full_prompt = build_company_summary_prompt(prompt, lead_data, scraped_content)
    
    full_prompt = full_prompt + COMPANY_SUMMARY_FINAL_REMINDER
    
    # Generate summary based on provider
    raw_output = ""
    if provider.lower() == 'openai':
        raw_output = await generate_openai_summary(api_key, model, full_prompt, status_callback=status_callback)
    elif provider.lower() == 'gemini':
        raw_output = await generate_gemini_summary(api_key, model, full_prompt, status_callback=status_callback)
    elif provider.lower() == 'openrouter':
        raw_output = await generate_openrouter_summary(api_key, model, full_prompt, status_callback=status_callback)
    else:
        return f"❌ Unknown provider: {provider}"
    
    # Post-process to fix all formatting issues
    cleaned_output = clean_and_structure_ai_output(raw_output)
    
    return cleaned_output


def build_email_copy_prompt(prompt_template: str, lead_data: dict | None, scraped_content: str | None) -> str:
    """Build the complete prompt for email copy. Replaces {key} and {{key}} with formatted values."""
    lead_data = dict(lead_data or {})
    url = _format_var_value(lead_data.get("url")) or ""
    company_name = _format_var_value(lead_data.get("company_name")) or ""
    if not company_name and url:
        company_name = url.replace("https://", "").replace("http://", "").split("/")[0]
    lead_data["url"] = url
    lead_data["company_name"] = company_name
    lead_data["scraped_content"] = str(scraped_content or "")
    return _replace_prompt_variables(prompt_template, lead_data)


async def generate_email_copy(
    api_key: str, provider: str, model: str, prompt_template: str,
    lead_data: dict, scraped_content: str, status_callback=None
) -> str:
    """Generate email copy for a lead. Works independently of company summary."""
    if not api_key or not api_key.strip():
        return "❌ No API key provided"
    if not scraped_content or scraped_content.startswith("❌"):
        return "❌ No valid scraped content available"
    if len(scraped_content.strip()) < 50:
        return "❌ Insufficient content scraped. Content too short to generate email copy."
    full_prompt = build_email_copy_prompt(prompt_template, lead_data, scraped_content)
    raw_output = ""
    if provider.lower() == "openai":
        raw_output = await generate_openai_summary(api_key, model, full_prompt, status_callback=status_callback)
    elif provider.lower() == "gemini":
        raw_output = await generate_gemini_summary(api_key, model, full_prompt, status_callback=status_callback)
    elif provider.lower() == "openrouter":
        raw_output = await generate_openrouter_summary(api_key, model, full_prompt, status_callback=status_callback)
    else:
        return f"❌ Unknown provider: {provider}"
    # Light cleanup for email copy (preserve line breaks)
    if raw_output and not raw_output.startswith("❌"):
        raw_output = raw_output.strip()
    return raw_output


def cleanup_html(html: str) -> str:
    """Clean and organize HTML content into well-structured text."""
    # Remove unwanted elements first
    html = re.sub(
        r"<(style|script|nav|footer|header|aside|noscript|iframe|svg|canvas|form|button)[\s\S]*?</\1>", 
        "", html, flags=re.IGNORECASE)
    
    # Remove common noise patterns (cookies, ads, etc.)
    html = re.sub(
        r'<(div|section|span)[^>]*(?:class|id)=["\'](?:cookie|ad|advertisement|popup|modal|overlay|banner|promo|marketing|tracking|analytics|social-share|share-buttons|newsletter|subscribe|signup)[^"\']*["\'][^>]*>[\s\S]*?</\1>',
        "", html, flags=re.IGNORECASE)
    
    # Convert headings to markdown-style headers
    html = re.sub(r'<h1[^>]*>(.*?)</h1>', r'\n\n# \1\n', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\n\n## \1\n', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n\n### \1\n', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<h4[^>]*>(.*?)</h4>', r'\n\n#### \1\n', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<h5[^>]*>(.*?)</h5>', r'\n\n##### \1\n', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<h6[^>]*>(.*?)</h6>', r'\n\n###### \1\n', html, flags=re.IGNORECASE | re.DOTALL)
    
    # Convert lists to markdown-style
    html = re.sub(r'<li[^>]*>(.*?)</li>', r'\n- \1', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'<(ul|ol)[^>]*>', '\n', html, flags=re.IGNORECASE)
    html = re.sub(r'</(ul|ol)>', '\n', html, flags=re.IGNORECASE)
    
    # Convert paragraphs to double newlines
    html = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\n\1\n\n', html, flags=re.IGNORECASE | re.DOTALL)
    
    # Convert line breaks
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    
    # Convert strong/bold to markdown
    html = re.sub(r'<(strong|b)[^>]*>(.*?)</\1>', r'**\2**', html, flags=re.IGNORECASE | re.DOTALL)
    
    # Convert emphasis/italic to markdown
    html = re.sub(r'<(em|i)[^>]*>(.*?)</\1>', r'*\2*', html, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove all remaining HTML tags
    html = re.sub(r'<[^>]+>', '', html)
    
    # Decode HTML entities
    html = unescape(html)
    
    # Clean up whitespace
    html = re.sub(r'&nbsp;', ' ', html)
    html = re.sub(r'\s+', ' ', html)  # Multiple spaces to single
    html = re.sub(r' \n', '\n', html)  # Space before newline
    html = re.sub(r'\n{3,}', '\n\n', html)  # Multiple newlines to double
    html = re.sub(r'^\s+', '', html, flags=re.MULTILINE)  # Leading whitespace
    html = re.sub(r'\s+$', '', html, flags=re.MULTILINE)  # Trailing whitespace
    
    # Remove common noise phrases (more conservative - only remove standalone phrases)
    # This prevents removing legitimate content that might contain these words
    noise_patterns = [
        r'^\s*cookie\s+policy\s*$',
        r'^\s*privacy\s+policy\s*$',
        r'^\s*terms\s+of\s+service\s*$',
        r'^\s*accept\s+cookies\s*$',
        r'^\s*we\s+use\s+cookies\s*$',
        r'^\s*skip\s+to\s+content\s*$',
        r'^\s*jump\s+to\s+content\s*$',
        r'^\s*menu\s*$',
        r'^\s*close\s*$',
        r'^\s*share\s+on\s*$',
        r'^\s*follow\s+us\s*$',
        r'^\s*subscribe\s+to\s*$',
    ]
    for pattern in noise_patterns:
        html = re.sub(pattern, '', html, flags=re.IGNORECASE | re.MULTILINE)
    
    # Final cleanup
    html = html.strip()
    
    # Split into lines and clean each line (more conservative filtering)
    lines = html.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Skip very short lines that are likely noise (but keep single characters if they're meaningful)
        if len(line) < 2:
            continue
        # Skip lines that are ONLY punctuation or symbols (but allow lines with some punctuation)
        if re.match(r'^[^\w\s]+$', line) and len(line) < 5:
            continue
        # Skip lines that are just numbers (likely page numbers or IDs)
        if re.match(r'^\d+$', line) and len(line) < 10:
            continue
        cleaned_lines.append(line)
    
    # Join back with proper spacing
    result = '\n'.join(cleaned_lines)
    
    # Final pass: ensure proper spacing around headers
    result = re.sub(r'\n(#{1,6}\s+[^\n]+)\n+', r'\n\1\n\n', result)
    
    return result.strip()


def extract_links(html: str, base_url: str):
    """Extract links from HTML with improved accuracy and URL resolution."""
    from urllib.parse import urljoin, urlparse
    
    # More robust regex pattern for href extraction
    matches = re.findall(
        r'<a\s+(?:[^>]*?\s+)?href\s*=\s*(["\']?)([^"\'>\s]+)\1', html, flags=re.IGNORECASE)
    
    base_domain = "/".join(base_url.split("/")[:3])
    parsed_base = urlparse(base_url)
    base_scheme = parsed_base.scheme or "https"
    base_netloc = parsed_base.netloc or parsed_base.path.split("/")[0] if "/" in parsed_base.path else ""
    
    urls = []
    seen = set()
    
    for quote_char, href in matches:
        href = href.strip()
        if not href:
            continue
            
        # Skip anchors, mailto, javascript, and other non-HTTP links
        if href.startswith(("#", "mailto:", "javascript:", "tel:", "sms:", "data:")):
            continue
        
        # Resolve relative URLs properly
        try:
            if href.startswith("//"):
                # Protocol-relative URL
                resolved_url = base_scheme + ":" + href
            elif href.startswith("/"):
                # Absolute path
                resolved_url = base_scheme + "://" + base_netloc + href
            elif href.startswith("http://") or href.startswith("https://"):
                # Absolute URL
                resolved_url = href
            else:
                # Relative URL - use urljoin for proper resolution
                resolved_url = urljoin(base_url, href)
            
            # Parse to validate and normalize
            parsed = urlparse(resolved_url)
            if not parsed.netloc:
                continue
            
            # Only include URLs from the same domain
            if parsed.netloc == base_netloc or parsed.netloc.endswith("." + base_netloc):
                # Normalize URL (remove fragment, normalize path)
                normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    normalized += "?" + parsed.query
                
                # Avoid duplicates
                if normalized not in seen:
                    seen.add(normalized)
                    urls.append(normalized)
        except Exception:
            # Skip invalid URLs silently
            continue
    
    return urls

# -------------------------
# Network fetch + scraping
# -------------------------


def get_random_user_agent():
    """Generate a random realistic user agent."""
    import random
    user_agents = [
        # Chrome on Windows (most common)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        # Chrome on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
        # Firefox on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:132.0) Gecko/20100101 Firefox/132.0",
        # Safari on Mac
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.6 Safari/605.1.15",
        # Edge on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    ]
    return random.choice(user_agents)

def get_realistic_headers(user_agent=None, target_url=None):
    """Generate realistic browser headers based on user agent - EPIC version."""
    import random
    if user_agent is None:
        user_agent = get_random_user_agent()
    
    # Detect browser type from user agent
    is_chrome = "Chrome" in user_agent and "Edg" not in user_agent
    is_firefox = "Firefox" in user_agent
    is_safari = "Safari" in user_agent and "Chrome" not in user_agent
    is_edge = "Edg" in user_agent
    
    # EPIC referer strategy - sometimes no referer (direct navigation), sometimes search engines
    # 30% chance of no referer (direct navigation), 70% chance of search engine referer
    referers = [
        None,  # Direct navigation - no referer
        "https://www.google.com/",
        "https://www.google.com/search?q=example",
        "https://www.google.com/search?q=site",
        "https://www.bing.com/",
        "https://duckduckgo.com/",
        "https://www.yahoo.com/",
        "https://www.google.com/search?client=firefox-b-d&q=example",
    ]
    
    # If target_url provided, sometimes use it as referer (internal navigation)
    if target_url and random.random() < 0.1:
        referer = target_url.rsplit('/', 1)[0] + '/'  # Parent directory
    else:
        referer = random.choice(referers)
    
    # Random viewport sizes (common resolutions)
    viewports = [
        (1920, 1080),
        (1366, 768),
        (1536, 864),
        (1440, 900),
        (1280, 720),
    ]
    viewport_width, viewport_height = random.choice(viewports)
    
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": random.choice([
            "en-US,en;q=0.9",
            "en-US,en;q=0.9,es;q=0.8",
            "en-US,en;q=0.9,fr;q=0.8",
            "en-US,en;q=0.9,de;q=0.8",
            "en-GB,en;q=0.9",
        ]),
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "DNT": random.choice(["1", "0"]),  # Sometimes DNT, sometimes not
    }
    
    # Only add referer if we have one (30% chance of None for direct navigation)
    if referer:
        headers["Referer"] = referer
    
    if is_chrome or is_edge:
        headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": random.choice(["none", "cross-site", "same-origin"]),
            "Sec-Fetch-User": "?1",
            "Sec-CH-UA": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Windows"',
            "Cache-Control": random.choice(["max-age=0", "no-cache"]),
            "Viewport-Width": str(viewport_width),
        })
    elif is_firefox:
        headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        })
    elif is_safari:
        headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
    
    return headers

# Global domain-based rate limiting - EPIC anti-bot feature
_domain_request_times = {}  # Track last request time per domain
_domain_lock = asyncio.Lock()  # Lock for thread-safe access to domain timing


async def _fetch_from_cache_fallbacks(session: aiohttp.ClientSession, url: str, timeout: int) -> tuple | str:
    """
    When direct fetch fails, try Google cache or archive.org for meta/content.
    Returns (page_url, html) on success, or error string on failure.
    """
    from urllib.parse import quote
    cache_timeout = aiohttp.ClientTimeout(total=timeout + 15, connect=15, sock_read=60)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    # 1. Try Google cache
    try:
        cache_url = f"https://webcache.googleusercontent.com/search?q=cache:{quote(url, safe='')}"
        async with session.get(cache_url, timeout=cache_timeout, headers=headers) as resp:
            if resp.status < 400:
                html = await resp.text(errors="replace")
                if html and len(html.strip()) > 200 and ("<html" in html.lower() or "<!doctype" in html.lower() or "<meta" in html.lower()):
                    return (url, html)
    except Exception:
        pass
    # 2. Try archive.org snapshot (Wayback Machine)
    try:
        wayback_url = f"https://web.archive.org/web/{url}"
        async with session.get(wayback_url, timeout=cache_timeout, headers=headers, allow_redirects=True) as resp:
            if resp.status < 400:
                html = await resp.text(errors="replace")
                if html and len(html.strip()) > 200:
                    return (url, html)
    except Exception:
        pass
    return ""


def _process_cached_html_to_text(url: str, html: str, max_chars: int = 50000) -> str:
    """Convert cached HTML to scraped-text format (same as scrape_site output)."""
    if not html or len(html.strip()) < 50:
        return ""
    cleaned = cleanup_html(html)
    if not cleaned or len(cleaned.strip()) < 10:
        return ""
    page_header = f"\n{'='*80}\nPAGE: {url}\n{'='*80}\n\n"
    content = page_header + cleaned[:max_chars]
    return content


async def fetch(session: aiohttp.ClientSession, url: str, timeout: int, retries: int, fast_mode: bool = False):
    """Fetch URL with EPIC anti-bot detection avoidance. fast_mode reduces delays for 4k+ runs."""
    from urllib.parse import urlparse
    import random
    
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '').lower()
    
    # Domain rate limit - much shorter in fast mode
    if fast_mode:
        min_delay = random.uniform(0.15, 0.4)
        pre_request_delay = random.uniform(0.02, 0.1)
    else:
        min_delay = random.uniform(2.0, 5.0)
        pre_request_delay = random.uniform(1.0, 3.0)
    
    async with _domain_lock:
        if domain in _domain_request_times:
            last_request_time = _domain_request_times[domain]
            time_since_last = time.time() - last_request_time
            if time_since_last < min_delay:
                wait_time = min_delay - time_since_last
                await asyncio.sleep(wait_time)
        _domain_request_times[domain] = time.time()
    
    await asyncio.sleep(pre_request_delay)
    
    # Try different URL variations and header combinations for better success rate
    url_variations = [url]
    parsed = urlparse(url)
    
    # Generate URL variations to try
    if parsed.netloc:
        domain = parsed.netloc
        path = parsed.path or '/'
        # Try with/without www
        if domain.startswith('www.'):
            url_variations.append(url.replace('www.', ''))
        else:
            url_variations.append(url.replace(domain, 'www.' + domain))
        # Try http if https
        if url.startswith('https://'):
            url_variations.append(url.replace('https://', 'http://'))
    
    # Generate MANY realistic header sets - more variations = better success rate
    header_variations = []
    for _ in range(15):  # Generate 15 different header combinations
        headers = get_realistic_headers(target_url=url)
        header_variations.append(headers)
    
    # Add variations without brotli (for compatibility)
    for _ in range(5):
        headers = get_realistic_headers(target_url=url)
        headers["Accept-Encoding"] = "gzip, deflate"
        header_variations.append(headers)
    
    # Shuffle header variations for maximum randomness
    random.shuffle(header_variations)
    
    last_error = None
    
    # Try URL variations and header combinations
    for url_to_try in url_variations:
        for headers in header_variations:
            # Override with session's user agent if provided, but regenerate headers to maintain consistency
            if session.headers.get("User-Agent"):
                # Regenerate headers with the session's user agent to maintain consistency
                headers = get_realistic_headers(session.headers.get("User-Agent"), target_url=url_to_try)
            
            for attempt in range(retries + 1):
                # Increase timeout for retries (some sites are slow) - more generous on later attempts
                retry_timeout = timeout + (attempt * 10)
                sock_read_cap = min(retry_timeout, 90)  # Allow up to 90s for slow sites to send data
                req_timeout = aiohttp.ClientTimeout(total=retry_timeout, connect=15, sock_read=sock_read_cap)
                
                try:
                    if attempt > 0:
                        delay = (random.uniform(0.3, 0.8) + attempt * 0.2) if fast_mode else (random.uniform(3.0, 6.0) + attempt * 1.5)
                        await asyncio.sleep(delay)
                    
                    # Parse URL
                    parsed_original = urlparse(url_to_try)
                    if not parsed_original.netloc:
                        continue
                    
                    original_domain = parsed_original.netloc.replace('www.', '').lower()
                    
                    await asyncio.sleep(random.uniform(0.02, 0.15) if fast_mode else random.uniform(0.5, 2.0))
                    
                    async with session.get(url_to_try, timeout=req_timeout, allow_redirects=True, headers=headers) as resp:
                        # Handle redirects properly - use the final URL after redirects
                        final_url = str(resp.url)
                        parsed_final = urlparse(final_url)
                        final_domain = parsed_final.netloc.replace('www.', '').lower()
                        
                        # More lenient domain validation - allow common redirect patterns
                        # Extract base domains for comparison
                        original_base = '.'.join(original_domain.split('.')[-2:]) if '.' in original_domain else original_domain
                        final_base = '.'.join(final_domain.split('.')[-2:]) if '.' in final_domain else final_domain
                        
                        # Only reject if base domains are completely different
                        if final_domain != original_domain:
                            # Check if it's a subdomain variation
                            if not (final_domain.endswith('.' + original_domain) or original_domain.endswith('.' + final_domain)):
                                # Check if base domains match (e.g., www.example.com -> example.com)
                                if original_base != final_base:
                                    # Completely different domain - but be lenient, allow it
                                    # Many sites redirect to different domains (e.g., country-specific domains)
                                    # Only log for debugging, don't block
                                    pass
                        
                        # Check status code - handle 403 and 429 specially
                        if resp.status == 403:
                            last_error = f"HTTP 403 at {final_url}"
                            backoff_delay = (random.uniform(1.0, 3.0) * (1.5 ** attempt)) if fast_mode else (random.uniform(5.0, 10.0) * (2 ** attempt))
                            backoff_delay = min(backoff_delay, 15.0 if fast_mode else 60.0)
                            await asyncio.sleep(backoff_delay)
                            async with _domain_lock:
                                _domain_request_times[domain] = time.time() + (random.uniform(2.0, 5.0) if fast_mode else random.uniform(10.0, 20.0))
                            
                            break  # Try next variation
                        elif resp.status == 429:
                            retry_after = int(resp.headers.get('Retry-After', 3 + attempt if fast_mode else 5 + attempt * 2))
                            last_error = f"HTTP 429 at {final_url} (rate limited)"
                            await asyncio.sleep(min(retry_after, 15 if fast_mode else 30))
                            continue  # Retry same URL/headers
                        elif resp.status >= 400:
                            # Other 4xx/5xx errors
                            last_error = f"HTTP {resp.status} at {final_url}"
                            if resp.status == 404:
                                # 404 is final, don't retry
                                return last_error
                            break  # Try next variation
                        
                        # Success! Read content
                        try:
                            html = await resp.text(errors="replace")
                        except Exception as decode_error:
                            error_str = str(decode_error).lower()
                            # Handle brotli encoding error
                            if 'brotli' in error_str or 'br' in error_str or 'content-encoding' in error_str:
                                # Retry without brotli
                                headers_no_br = headers.copy()
                                headers_no_br["Accept-Encoding"] = "gzip, deflate"
                                # Try once more with different encoding
                                async with session.get(url_to_try, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True, headers=headers_no_br) as resp2:
                                    if resp2.status < 400:
                                        html = await resp2.text(errors="replace")
                                    else:
                                        last_error = f"HTTP {resp2.status} at {str(resp2.url)}"
                                        break
                            else:
                                last_error = f"Error decoding content: {str(decode_error)}"
                                break
                        
                        # Validate it's actually HTML
                        html_lower = html.lower()
                        if '<html' not in html_lower and '<!doctype' not in html_lower:
                            # Check if it's a redirect page or error
                            if 'redirect' in html_lower[:500] or 'location.replace' in html_lower[:500] or 'window.location' in html_lower[:500]:
                                last_error = f"JavaScript redirect detected at {final_url}"
                                break
                            # Might be valid content without HTML tags (rare), but check length
                            if len(html.strip()) < 100:
                                last_error = f"Non-HTML or empty content at {final_url}"
                                break
                        
                        # Success! Return the content
                        return (final_url, html)
                        
                except asyncio.TimeoutError:
                    last_error = f"Timeout fetching {url_to_try} (exceeded {retry_timeout}s)"
                    if attempt < retries:
                        await asyncio.sleep(0.5 + attempt * 0.3 if fast_mode else 2 + attempt * 1)
                    continue
                except (aiohttp.ClientError, ConnectionResetError, OSError) as e:
                    error_str = str(e).lower()
                    # Handle brotli encoding error specifically
                    if 'brotli' in error_str or 'br' in error_str or 'content-encoding' in error_str:
                        last_error = f"Brotli encoding error: {str(e)}"
                        if attempt < retries:
                            await asyncio.sleep(0.3 + attempt * 0.2 if fast_mode else 1 + attempt * 0.5)
                        continue
                    if 'connection' in error_str or 'disconnected' in error_str or 'getaddrinfo' in error_str:
                        last_error = f"Error fetching {url_to_try}: {str(e)}"
                        if attempt < retries:
                            await asyncio.sleep(0.5 + attempt * 0.3 if fast_mode else 2 + attempt * 1)
                        continue
                    last_error = f"Error fetching {url_to_try}: {str(e)}"
                    if attempt < retries:
                        await asyncio.sleep(0.3 + attempt * 0.2 if fast_mode else 1 + attempt * 0.5)
                    continue
                except UnicodeDecodeError as e:
                    last_error = f"Encoding error: {str(e)}"
                    if attempt < retries:
                        await asyncio.sleep(0.3 + attempt * 0.2 if fast_mode else 1 + attempt * 0.5)
                    continue
                except Exception as e:
                    last_error = f"Unexpected error fetching {url_to_try}: {str(e)}"
                    if attempt < retries:
                        await asyncio.sleep(0.3 + attempt * 0.2 if fast_mode else 1 + attempt * 0.5)
                    continue
    
    # If we get here, all variations failed
    return last_error or f"Failed to fetch {url} after trying all variations"


async def scrape_site(session, url: str, depth: int, keywords, max_chars: int, retries: int, timeout: int, fast_mode: bool = False):
    depth = max(1, int(depth) if depth is not None else 1)
    max_chars = max(100, min(int(max_chars) if max_chars is not None else 10000, 50000))
    visited, results, errors = set(), [], []
    total_chars = 0
    separator = "\n\n" + "─" * 80 + "\n\n"  # Better visual separator between pages
    separator_len = len(separator)

    # Normalize URL before fetching
    normalized_url = normalize_url(url)

    homepage = await fetch(session, normalized_url, timeout, retries, fast_mode)
    if isinstance(homepage, str):
        # Direct fetch failed: try cache fallbacks (Google cache, archive.org)
        cache_result = await _fetch_from_cache_fallbacks(session, normalized_url, timeout)
        if isinstance(cache_result, tuple):
            homepage = cache_result
        else:
            return f"❌ {homepage}"

    page_url, html = homepage
    
    # STRICT validation: Verify we're on the correct domain
    from urllib.parse import urlparse
    parsed_page = urlparse(page_url)
    parsed_original = urlparse(normalized_url)
    
    page_domain = parsed_page.netloc.replace('www.', '').lower()
    original_domain = parsed_original.netloc.replace('www.', '').lower()
    
    # Log the actual URL we fetched (useful for debugging redirects)
    if page_url != normalized_url:
        errors.append(f"ℹ️ Redirected from {normalized_url} to {page_url}")
    
    # More lenient domain validation - allow common redirect patterns
    # Extract base domains for comparison
    original_base = '.'.join(original_domain.split('.')[-2:]) if '.' in original_domain else original_domain
    page_base = '.'.join(page_domain.split('.')[-2:]) if '.' in page_domain else page_domain
    
    # Only reject if base domains are completely different
    if page_domain != original_domain:
        # Check if it's a subdomain variation
        if not (page_domain.endswith('.' + original_domain) or original_domain.endswith('.' + page_domain)):
            # Check if base domains match (e.g., www.example.com -> example.com)
            if original_base != page_base:
                # Completely different domain - but be lenient, allow it
                # Many sites redirect to different domains (e.g., country-specific domains)
                # Only log for debugging, don't block
                pass
    
    # Validate HTML content before processing
    if not html or len(html.strip()) < 50:
        return f"❌ Empty or too short HTML content from {page_url}"
    
    # Check if HTML looks valid (has some HTML structure)
    html_lower = html.lower()
    
    # More robust validation - check for common content indicators
    has_content = (
        '<html' in html_lower or 
        '<!doctype' in html_lower or 
        '<body' in html_lower or
        '<main' in html_lower or
        '<article' in html_lower or
        '<div' in html_lower[:2000]  # Check first 2000 chars for div tags
    )
    
    if not has_content:
        # Check if it's a redirect page
        if 'location.href' in html_lower or 'window.location' in html_lower or 'meta http-equiv="refresh"' in html_lower:
            return f"❌ JavaScript/Meta redirect detected at {page_url}. Content may not be accessible."
        # Might still have content, continue
    
    # ADDITIONAL SAFETY CHECK: Detect if content seems wrong for the domain
    # Check for common e-commerce patterns that shouldn't be on a business consultant site
    if len(html) > 500:
        html_sample = html_lower[:10000]  # Check first 10k chars
        
        # E-commerce indicators
        ecommerce_patterns = [
            'add to cart', 'shopping cart', 'buy now', 'add to bag',
            'product reviews', 'customer reviews', 'star rating',
            'amazon.com', 'ebay.com', 'etsy.com', 'shopify store',
            'price:', '$', 'usd', 'eur', 'gbp',
            'remote helicopter', 'rc helicopter', 'replacement parts'
        ]
        
        ecommerce_count = sum(1 for pattern in ecommerce_patterns if pattern in html_sample)
        
        # If we find many e-commerce patterns, it's likely wrong content
        if ecommerce_count >= 5:
            # Check if domain name appears in content (if not, it's definitely wrong)
            domain_in_content = original_domain.split('.')[0] in html_sample
            if not domain_in_content:
                return f"❌ Content mismatch detected! Page at {page_url} contains e-commerce content that doesn't match expected domain {original_domain}. This might be a redirect or wrong content."
    
    cleaned = cleanup_html(html)
    
    # Validate cleaned content
    if cleaned and len(cleaned.strip()) > 10:  # Ensure meaningful content
        # Better page header with clear separation
        page_header = f"\n{'='*80}\nPAGE: {page_url}\n{'='*80}\n\n"
        page_content = page_header + cleaned
        # Check if adding this page would exceed max_chars
        if total_chars + len(page_content) + separator_len <= max_chars:
            results.append(page_content)
            total_chars += len(page_content) + separator_len
        elif total_chars == 0:  # At least include homepage even if it's large
            # Truncate homepage to fit
            available_chars = max_chars - len(page_header) - separator_len
            if available_chars > 0:
                truncated_content = cleaned[:available_chars]
                results.append(page_header + truncated_content)
                total_chars = max_chars
        # If total_chars > 0 and adding would exceed, skip
    else:
        # Check if it's an error page or redirect
        if 'error' in html_lower[:500] or '404' in html_lower[:500] or 'not found' in html_lower[:500]:
            errors.append(f"❌ Error page detected: {page_url}")
        else:
            errors.append(f"❌ No extractable text content on homepage: {page_url}")

    # Use the actual fetched URL (after redirects) for link extraction
    links = extract_links(html, page_url)
    # Improved keyword matching - case-insensitive, handles hyphens/underscores
    if keywords:  # Only process if keywords are provided
        for kw in keywords:
            if not kw or not kw.strip():
                continue
            kw_clean = kw.strip()
            match = next((l for l in links if match_keyword_in_url(kw_clean, l)), None)
            if match:
                visited.add(match)
    for l in links[: depth * 2]:
        visited.add(l)

    # Process additional pages while respecting max_chars limit
    for link in visited:
        if total_chars >= max_chars:
            break  # Stop if we've reached the limit
        
        res = await fetch(session, link, timeout, retries, fast_mode)
        if isinstance(res, str):
            errors.append(f"❌ {res}")
        elif isinstance(res, tuple):
            link_url, html2 = res
            
            # Validate HTML content
            if not html2 or len(html2.strip()) < 50:
                errors.append(f"❌ Empty or invalid HTML from: {link_url}")
                continue
            
            # Check for error pages
            html2_lower = html2.lower()
            if 'error' in html2_lower[:500] or '404' in html2_lower[:500] or 'not found' in html2_lower[:500]:
                errors.append(f"❌ Error page detected: {link_url}")
                continue
            
            cleaned2 = cleanup_html(html2)
            
            # Validate cleaned content (must have meaningful text)
            if cleaned2 and len(cleaned2.strip()) > 10:
                # Better page header with clear separation
                page_header = f"\n{'='*80}\nPAGE: {link_url}\n{'='*80}\n\n"
                page_content = page_header + cleaned2
                # Check if adding this page would exceed max_chars
                if total_chars + len(page_content) + separator_len <= max_chars:
                    results.append(page_content)
                    total_chars += len(page_content) + separator_len
                else:
                    # Add partial content if there's space
                    available_chars = max_chars - total_chars - separator_len - len(page_header)
                    if available_chars > 100:  # Only add if meaningful space remains
                        truncated_content = cleaned2[:available_chars]
                        results.append(page_header + truncated_content)
                        total_chars = max_chars
                    break  # No more space
            else:
                errors.append(f"❌ No extractable text content on page: {link_url}")

    if not results:
        return errors[0] if errors else f"❌ Unknown error on site: {url}"
    
    # Join results with separator
    final_text = separator.join(results)
    
    # Add errors at the end if there's space (errors are usually short)
    if errors and len(final_text) + len("\n".join(errors)) + 20 <= max_chars:
        final_text += "\n\n" + "\n".join(errors)
    
    # Final safety check - ensure we don't exceed max_chars (improved truncation)
    if len(final_text) > max_chars:
        # Try to cut at a reasonable point (prefer sentence boundaries)
        truncated = final_text[:max_chars]
        
        # Look for sentence endings (period, exclamation, question mark followed by space/newline)
        sentence_endings = []
        for i in range(len(truncated) - 1, max(0, len(truncated) - 200), -1):
            if truncated[i] in '.!?' and i < len(truncated) - 1:
                next_char = truncated[i + 1]
                if next_char in ' \n\t':
                    sentence_endings.append(i + 1)
                    if len(sentence_endings) >= 3:  # Get a few options
                        break
        
        # Also look for paragraph boundaries
        paragraph_endings = []
        for i in range(len(truncated) - 1, max(0, len(truncated) - 200), -1):
            if truncated[i] == '\n' and i < len(truncated) - 1:
                if truncated[i + 1] == '\n' or (i > 0 and truncated[i - 1] == '\n'):
                    paragraph_endings.append(i + 1)
                    if len(paragraph_endings) >= 2:
                        break
        
        # Choose best cut point
        best_cut = len(truncated)
        if sentence_endings:
            # Use the last sentence ending that's not too early
            for end_pos in reversed(sentence_endings):
                if end_pos >= max_chars * 0.85:  # At least 85% of max_chars
                    best_cut = end_pos
                    break
        elif paragraph_endings:
            # Fall back to paragraph boundary
            for end_pos in reversed(paragraph_endings):
                if end_pos >= max_chars * 0.85:
                    best_cut = end_pos
                    break
        
        final_text = truncated[:best_cut].rstrip() + "\n\n[... content truncated to fit character limit ...]"
    
    return final_text

# -------------------------
# Worker + Writer
# -------------------------


async def worker_coroutine(name, session, url_queue: asyncio.Queue, result_queue: asyncio.Queue,
                           depth, keywords, max_chars, retries, timeout,
                           ai_enabled=False, ai_api_key=None, ai_provider=None, ai_model=None, ai_prompt=None,
                           email_copy_enabled=False, email_copy_api_key=None, email_copy_provider=None,
                           email_copy_model=None, email_copy_prompt=None,
                           lead_data_map=None, ai_status_callback=None, scrape_status_callback=None, fast_mode: bool = False):
    from urllib.parse import urlparse
    import random
    
    worker_id = int(name.split('-')[-1]) if '-' in name else 0
    initial_delay = worker_id * (random.uniform(0.03, 0.12) if fast_mode else random.uniform(0.5, 1.5))
    await asyncio.sleep(initial_delay)
    
    while True:
        item = await url_queue.get()
        if item is None:
            url_queue.task_done()
            break
        
        # CRITICAL: Wrap entire processing in try-finally to ensure task_done() is always called
        try:
            await asyncio.sleep(random.uniform(0.03, 0.15) if fast_mode else random.uniform(0.5, 2.0))
            # Handle both old format (just URL) and new format (URL, index)
            if isinstance(item, tuple):
                original_url, url_index = item
            else:
                original_url = item
                url_index = None
            
            # Normalize URL (add https:// if missing)
            normalized_url = normalize_url(original_url)
            
            # Get lead data from map if available
            lead_data = None
            if lead_data_map and url_index is not None and url_index in lead_data_map:
                try:
                    ld = lead_data_map[url_index]
                    lead_data = dict(ld) if ld is not None and hasattr(ld, 'keys') else {}
                except (TypeError, AttributeError, ValueError):
                    lead_data = {}
                lead_data['url'] = original_url  # Ensure URL is set
            else:
                # Fallback: extract company name from URL
                lead_data = {
                    'url': original_url,
                    'company_name': original_url.replace('https://', '').replace('http://', '').split('/')[0]
                }
            
            def _scrape_status(status: str, msg: str):
                if scrape_status_callback:
                    try:
                        scrape_status_callback(original_url, status, msg)
                    except Exception:
                        pass

            _scrape_status("scraping", "Fetching...")
            # Be lenient with URL validation - let the actual HTTP request determine validity
            # Only reject obviously invalid URLs (empty or completely malformed)
            if not normalized_url or len(normalized_url.strip()) < 4:
                scraped_text = "❌ Invalid URL: URL is empty or too short"
                ai_summary = "❌ Invalid URL: URL is empty or too short"
                email_copy = ""
                _scrape_status("error", scraped_text)
            else:
                # Global timeout per URL - shorter in fast mode, generous for slow sites
                max_total_time = min((timeout * (retries + 1) * (depth + 1) * 2) + (30 if fast_mode else 60), 90 if fast_mode else 300)
                
                try:
                    # Wrap scraping in a timeout to prevent infinite hangs
                    scraped_text = await asyncio.wait_for(
                        scrape_site(session, normalized_url, depth, keywords, max_chars, retries, timeout, fast_mode),
                        timeout=max_total_time
                    )
                except asyncio.TimeoutError:
                    err_msg = f"❌ Timeout: Scraping {normalized_url} exceeded maximum time limit ({max_total_time}s)"
                    _scrape_status("error", "Timeout, trying cache fallback...")
                    cache_result = await _fetch_from_cache_fallbacks(session, normalized_url, timeout)
                    if isinstance(cache_result, tuple):
                        page_url, html = cache_result
                        scraped_text = _process_cached_html_to_text(page_url, html, max_chars)
                        if scraped_text:
                            _scrape_status("scraped", f"Recovered from cache ({len(scraped_text):,} chars)")
                        else:
                            scraped_text = err_msg
                            _scrape_status("error", scraped_text[:120])
                    else:
                        scraped_text = err_msg
                        _scrape_status("error", scraped_text[:120])
                except Exception as e:
                    # If scraping fails with an exception, try once more with a more lenient approach
                    try:
                        # Try with http:// if https:// failed, but also wrap in timeout
                        if normalized_url.startswith("https://"):
                            fallback_url = normalized_url.replace("https://", "http://", 1)
                            try:
                                scraped_text = await asyncio.wait_for(
                                    scrape_site(session, fallback_url, depth, keywords, max_chars, retries, timeout, fast_mode),
                                    timeout=max_total_time
                                )
                            except asyncio.TimeoutError:
                                scraped_text = f"❌ Timeout: Scraping {fallback_url} exceeded maximum time limit ({max_total_time}s)"
                        else:
                            scraped_text = f"❌ Error scraping {normalized_url}: {str(e)}"
                    except Exception as e2:
                        scraped_text = f"❌ Error scraping {normalized_url}: {str(e2)}"
                    if scraped_text and scraped_text.startswith("❌"):
                        _scrape_status("error", "Scrape failed, trying cache fallback...")
                        cache_result = await _fetch_from_cache_fallbacks(session, normalized_url, timeout)
                        if isinstance(cache_result, tuple):
                            page_url, html = cache_result
                            cached_text = _process_cached_html_to_text(page_url, html, max_chars)
                            if cached_text:
                                scraped_text = cached_text
                                _scrape_status("scraped", f"Recovered from cache ({len(scraped_text):,} chars)")
                    if scraped_text and scraped_text.startswith("❌"):
                        _scrape_status("error", scraped_text[:120])
                
                # RELAXED VALIDATION: Only flag obvious mismatches
                # Many sites legitimately redirect to different domains (same organization)
                # Only check for obvious content mismatches, not domain redirects
                if scraped_text and not scraped_text.startswith("❌"):
                    # Check if scraped content contains PAGE: header
                    page_header_match = re.search(r'PAGE:\s*(https?://[^\s\n]+)', scraped_text[:5000])
                    if page_header_match:
                        actual_scraped_url = page_header_match.group(1)
                        # Normalize both URLs for comparison
                        actual_domain = urlparse(actual_scraped_url).netloc.replace('www.', '').lower()
                        expected_domain = urlparse(normalized_url).netloc.replace('www.', '').lower()
                        
                        # Extract base domains (last two parts)
                        actual_base = '.'.join(actual_domain.split('.')[-2:]) if '.' in actual_domain else actual_domain
                        expected_base = '.'.join(expected_domain.split('.')[-2:]) if '.' in expected_domain else expected_domain
                        
                        # Only flag if:
                        # 1. Base domains are completely different (not related)
                        # 2. AND it's not a subdomain variation
                        # 3. AND content is suspiciously short (might be wrong page)
                        if actual_base != expected_base:
                            if not (actual_domain.endswith('.' + expected_domain) or expected_domain.endswith('.' + actual_domain)):
                                # Check if content is suspiciously short (might be wrong page)
                                if len(scraped_text.strip()) < 500:
                                    # Very short content from different domain - likely wrong
                                    scraped_text = f"❌ CONTENT MISMATCH: Scraped content is from {actual_domain} but expected {expected_domain}. Original URL: {original_url}"
                                    _scrape_status("error", scraped_text[:120])
                                # Otherwise, allow it - many sites redirect to related domains
                if scraped_text and not scraped_text.startswith("❌"):
                    _scrape_status("scraped", f"{len(scraped_text):,} chars scraped")
                
                # Generate AI summary if enabled (skip when scraping already failed)
                if ai_enabled and ai_api_key and ai_provider and ai_model:
                    if not scraped_text or scraped_text.startswith("❌") or len(scraped_text.strip()) < 50:
                        ai_summary = "❌ Summary skipped: site unreachable or insufficient content"
                    else:
                        _scrape_status("ai_summarizing", "Generating AI summary...")
                        def url_status_callback(msg):
                            if scrape_status_callback:
                                try:
                                    scrape_status_callback(original_url, "ai_summarizing", msg)
                                except Exception:
                                    pass
                            if ai_status_callback:
                                ai_status_callback(original_url, msg)
                        try:
                            ai_summary = await asyncio.wait_for(
                                generate_ai_summary(
                                    ai_api_key, ai_provider, ai_model, ai_prompt or "",
                                    lead_data, scraped_text, status_callback=url_status_callback
                                ),
                                timeout=120
                            )
                        except asyncio.TimeoutError:
                            ai_summary = "❌ AI Summary timeout: Generation exceeded 2 minutes"
                            _scrape_status("error", "Company summary: timeout (2 min)")
                        except Exception as e:
                            ai_summary = f"❌ AI Summary error: {str(e)}"
                            err_short = str(e)[:80] if e else "unknown"
                            _scrape_status("error", f"Company summary failed: {err_short}")
                        else:
                            if ai_summary and ai_summary.startswith("❌"):
                                _scrape_status("error", "Company summary failed")
                else:
                    ai_summary = ""
                # Generate email copy if enabled (skip when scraping already failed)
                if email_copy_enabled and email_copy_api_key and email_copy_provider and email_copy_model:
                    if not scraped_text or scraped_text.startswith("❌") or len(scraped_text.strip()) < 50:
                        email_copy = "❌ Email skipped: site unreachable or insufficient content"
                    else:
                        _scrape_status("email_copy", "Generating email copy...")
                        def email_status_callback(msg):
                            if scrape_status_callback:
                                try:
                                    scrape_status_callback(original_url, "email_copy", msg)
                                except Exception:
                                    pass
                        try:
                            email_copy = await asyncio.wait_for(
                                generate_email_copy(
                                    email_copy_api_key, email_copy_provider, email_copy_model,
                                    email_copy_prompt or "", lead_data, scraped_text,
                                    status_callback=email_status_callback
                                ),
                                timeout=120
                            )
                        except asyncio.TimeoutError:
                            email_copy = "❌ Email copy timeout: Generation exceeded 2 minutes"
                            _scrape_status("error", "Email copy: timeout (2 min)")
                        except Exception as e:
                            email_copy = f"❌ Email copy error: {str(e)}"
                            err_short = str(e)[:80] if e else "unknown"
                            _scrape_status("error", f"Email copy failed: {err_short}")
                        else:
                            if email_copy and email_copy.startswith("❌"):
                                _scrape_status("error", "Email copy failed")
                else:
                    email_copy = ""
            
            # PERFECT CSV CLEANING - Zero tolerance for formatting errors
            def clean_csv_field(field_value):
                """
                Perfect CSV field cleaning:
                - Removes all newlines, carriage returns, tabs
                - Removes null bytes and control characters
                - Normalizes whitespace
                - Ensures valid UTF-8 encoding
                - Returns empty string for None/invalid values
                """
                if field_value is None:
                    return ""
                
                # Convert to string
                field_str = str(field_value)
                
                # Remove null bytes and control characters (except space, tab)
                # Keep only printable characters and whitespace
                import re
                # Remove null bytes
                field_str = field_str.replace('\x00', '')
                # Replace all newlines, carriage returns, form feeds, vertical tabs with space
                field_str = re.sub(r'[\n\r\f\v]', ' ', field_str)
                # Replace tabs with space
                field_str = field_str.replace('\t', ' ')
                # Remove other control characters (0x00-0x1F except space)
                field_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', field_str)
                
                # Normalize whitespace: collapse multiple spaces/tabs into single space
                field_str = re.sub(r'\s+', ' ', field_str)
                
                # Strip leading/trailing whitespace
                field_str = field_str.strip()
                
                # Ensure valid UTF-8 encoding (replace invalid sequences)
                try:
                    field_str = field_str.encode('utf-8', errors='replace').decode('utf-8')
                except:
                    field_str = ""
                
                return field_str
            
            # Clean all fields perfectly; normalize AI/email errors for clearer CSV display
            ai_for_csv = _normalize_ai_error_for_display(ai_summary, "Summary")
            email_for_csv = _normalize_ai_error_for_display(email_copy, "Email") if email_copy else ""
            cleaned_url = clean_csv_field(original_url)
            cleaned_scraped_text = clean_csv_field(scraped_text)
            cleaned_ai_summary = clean_csv_field(ai_for_csv)
            cleaned_email_copy = clean_csv_field(email_for_csv) if email_copy_enabled else ""
            # Output: 4 columns when email copy enabled, else 3
            if email_copy_enabled:
                out = (cleaned_url, cleaned_scraped_text, cleaned_ai_summary, cleaned_email_copy)
            else:
                out = (cleaned_url, cleaned_scraped_text, cleaned_ai_summary)
            await result_queue.put(out)
        except Exception as e:
            # If anything goes wrong, still output an error row
            try:
                # Helper function for cleaning (defined here since it might not be in scope)
                def clean_csv_field(field_value):
                    if field_value is None:
                        return ""
                    field_str = str(field_value)
                    import re
                    field_str = field_str.replace('\x00', '')
                    field_str = re.sub(r'[\n\r\f\v]', ' ', field_str)
                    field_str = field_str.replace('\t', ' ')
                    field_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', field_str)
                    field_str = re.sub(r'\s+', ' ', field_str)
                    field_str = field_str.strip()
                    try:
                        field_str = field_str.encode('utf-8', errors='replace').decode('utf-8')
                    except:
                        field_str = ""
                    return field_str
                
                original_url_val = original_url if 'original_url' in locals() else "unknown URL"
                error_msg = f"❌ Worker error processing {original_url_val}: {str(e)}"
                cleaned_url = clean_csv_field(original_url_val)
                cleaned_scraped_text = clean_csv_field(error_msg)
                cleaned_ai_summary = clean_csv_field("")
                cleaned_email_copy = clean_csv_field("")
                if email_copy_enabled:
                    out = (cleaned_url, cleaned_scraped_text, cleaned_ai_summary, cleaned_email_copy)
                else:
                    out = (cleaned_url, cleaned_scraped_text, cleaned_ai_summary)
                await result_queue.put(out)
            except:
                try:
                    if email_copy_enabled:
                        await result_queue.put(("", f"❌ Critical worker error: {str(e)}", "", ""))
                    else:
                        await result_queue.put(("", f"❌ Critical worker error: {str(e)}", ""))
                except:
                    pass  # If even this fails, just continue
        finally:
            # CRITICAL: Always mark task as done, even if there was an error
            url_queue.task_done()


async def writer_coroutine(result_queue: asyncio.Queue, rows_per_file: int, output_dir: str,
                           total_urls: int, progress_callback,
                           start_part: int = 0, checkpoint_data: dict | None = None,
                           checkpoint_path: str | None = None,
                           log_callback=None, include_email_copy: bool = False):
    """
    Write results to disk. If checkpoint_data and checkpoint_path are provided,
    saves progress after each file write for crash recovery and resume.
    """
    def emit(msg: str):
        print(msg)
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass

    # CRITICAL: Create output directory first
    try:
        os.makedirs(output_dir, exist_ok=True)
        emit(f"📁 Writer: Created output directory: {output_dir}")
    except Exception as e:
        emit(f"❌ Writer: Failed to create output directory: {e}")
        import traceback
        traceback.print_exc()
        return  # Exit if we can't create directory
    
    loop = asyncio.get_running_loop()
    buffer = []
    part = start_part
    processed = 0
    files_written = []
    last_flush_ts = time.time()
    emergency_flush_every_s = 10  # Flush more often to reduce data loss on crash
    emergency_flush_min_rows = max(25, min(100, rows_per_file // 10))

    while True:
        item = await result_queue.get()
        if item is None:
            emit(f"📝 Writer: Received stop signal. Processed {processed}/{total_urls}. Buffer has {len(buffer)}.")
            result_queue.task_done()
            break
        buffer.append(item)
        processed += 1
        result_queue.task_done()  # CRITICAL: one per item (join() counts these)
        if progress_callback:
            progress_callback(processed, total_urls)

        # Write chunks to disk when:
        # 1) regular threshold is reached, or
        # 2) enough time passed with partial buffer (prevents progress loss on sudden refresh/crash)
        while len(buffer) >= rows_per_file or (
            len(buffer) >= emergency_flush_min_rows and (time.time() - last_flush_ts) >= emergency_flush_every_s
        ):
            part += 1
            chunk_size = rows_per_file if len(buffer) >= rows_per_file else len(buffer)
            chunk_rows = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            last_flush_ts = time.time()
            if chunk_size < rows_per_file:
                emit(f"💾 Early flush: writing {chunk_size} buffered rows to reduce loss risk")
            
            try:
                # Filter out ONLY truly empty rows (keep error messages as they are valid data)
                filtered_rows = []
                for row in chunk_rows:
                    if len(row) >= 2:
                        url = str(row[0]).strip() if row[0] else ""
                        scraped_text = str(row[1]).strip() if row[1] else ""
                        # Include rows with valid URL - keep error messages (they start with ❌)
                        # Only exclude rows where URL is empty or scraped_text is completely empty
                        if url and url != "":
                            # Include even if scraped_text is empty or is an error message
                            # Error messages are valid data that should be preserved
                            filtered_rows.append(row)
                
                if not filtered_rows:
                    # Skip writing if no valid rows
                    continue
                
                # Normalize rows: 4 columns when include_email_copy, else 3
                n_cols = 4 if include_email_copy else 3
                cols = ["Website", "ScrapedText", "CompanySummary", "EmailCopy"] if include_email_copy else ["Website", "ScrapedText", "CompanySummary"]
                normalized_rows = []
                for row in filtered_rows:
                    normalized_row = list(row[:n_cols])
                    while len(normalized_row) < n_cols:
                        normalized_row.append("")
                    normalized_rows.append(tuple(normalized_row[:n_cols]))
                
                df = pd.DataFrame(normalized_rows, columns=cols)
                
                # Normalize scrape/AI/email error strings so narrow columns show clear messages
                if "ScrapedText" in df.columns:
                    df["ScrapedText"] = df["ScrapedText"].apply(
                        lambda x: _normalize_scrape_error_for_display(str(x) if pd.notna(x) else "")
                    )
                if "CompanySummary" in df.columns:
                    df["CompanySummary"] = df["CompanySummary"].apply(
                        lambda x: _normalize_ai_error_for_display(str(x) if pd.notna(x) else "", "Summary")
                    )
                if include_email_copy and "EmailCopy" in df.columns:
                    df["EmailCopy"] = df["EmailCopy"].apply(
                        lambda x: _normalize_ai_error_for_display(str(x) if pd.notna(x) else "", "Email")
                    )
                
                # PERFECT CSV CLEANING - Clean all DataFrame columns before writing
                def clean_dataframe_for_csv(df):
                    """Clean all string columns in DataFrame for perfect CSV formatting"""
                    import re
                    for col in df.columns:
                        # Convert to string, handle None/NaN
                        df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
                        # Remove null bytes
                        df[col] = df[col].str.replace('\x00', '', regex=False)
                        # Replace newlines, carriage returns, tabs with space
                        df[col] = df[col].str.replace(r'[\n\r\f\v\t]', ' ', regex=True)
                        # Remove other control characters
                        df[col] = df[col].str.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', regex=True)
                        # Normalize whitespace
                        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                        # Strip whitespace
                        df[col] = df[col].str.strip()
                        # Replace empty strings with empty string (not NaN)
                        df[col] = df[col].replace('', '')
                    return df
                
                # Clean DataFrame
                df = clean_dataframe_for_csv(df.copy())
                
                # Remove ONLY rows with empty URLs (keep error messages)
                df = df[df["Website"].astype(str).str.strip() != ""]
                
                # Enforce column structure
                for c in cols:
                    if c not in df.columns:
                        df[c] = ""
                df = df[cols]
                
                if len(df) == 0:
                    # Skip writing if DataFrame is empty after filtering
                    continue
                
                # CRITICAL: Write Excel file FIRST (more reliable than CSV)
                excel_path = os.path.join(output_dir, f"output_part_{part}.xlsx")
                excel_written = False
                try:
                    df_excel = df[cols].copy()
                    from openpyxl import Workbook
                    wb = Workbook()
                    ws = wb.active
                    _write_excel_sheet(ws, cols, df_excel)
                    
                    # Run heavy I/O in executor to avoid blocking event loop (fixes progress freeze)
                    def _save_excel():
                        wb.save(excel_path)
                    try:
                        await loop.run_in_executor(None, _save_excel)
                    except Exception as save_error:
                        # If save fails, try to save with minimal data
                        print(f"⚠️ Excel save failed, attempting recovery: {save_error}")
                        # Create a fresh workbook with just headers
                        wb_recovery = Workbook()
                        ws_recovery = wb_recovery.active
                        ws_recovery.title = "Scraped Data"
                        ws_recovery.append(cols)
                        wb_recovery.save(excel_path)
                        raise save_error
                    
                    # Verify Excel file was written
                    if os.path.exists(excel_path) and os.path.getsize(excel_path) > 0:
                        files_written.append(excel_path)
                        excel_written = True
                        emit(f"✅ Writer: Wrote Excel file: {os.path.basename(excel_path)} ({len(df_excel)} rows)")
                    else:
                        emit(f"❌ Writer: Excel file missing/empty: {os.path.basename(excel_path)}")
                except Exception as e:
                    # If Excel fails, log error but continue
                    emit(f"⚠️ Writer: Excel export failed for part {part}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # PERFECT CSV WRITING - Zero tolerance for errors
                csv_path = os.path.join(output_dir, f"output_part_{part}.csv")
                
                # Write CSV with perfect formatting:
                # - UTF-8 with BOM for Excel compatibility
                # - QUOTE_ALL: All fields quoted to handle commas, quotes, special chars
                # - lineterminator='\n': Standard Unix line endings
                # - doublequote=True: Escape quotes by doubling them (CSV standard)
                try:
                    df = df[cols]
                    def _write_csv():
                        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                            writer = csv.writer(f, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n', quotechar='"')
                            writer.writerow(cols)
                            for idx in range(len(df)):
                                row = df.iloc[idx]
                                def clean_val(v):
                                    if not v: return ""
                                    v = str(v).replace('\x00', '')
                                    v = re.sub(r'[\n\r\f\v\t]', ' ', v)
                                    v = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', v)
                                    return re.sub(r'\s+', ' ', v).strip()
                                row_values = [clean_val(row[c]) for c in cols]
                                row_values = (row_values + [""] * len(cols))[:len(cols)]
                                writer.writerow(row_values)
                    await loop.run_in_executor(None, _write_csv)
                    
                    # Verify file was written
                    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                        files_written.append(csv_path)
                        emit(f"✅ Writer: Wrote CSV file: {os.path.basename(csv_path)} ({len(df)} rows)")
                    else:
                        emit(f"❌ Writer: CSV file missing/empty: {os.path.basename(csv_path)}")
                    
                    # VALIDATION: Verify CSV file is valid by reading it back
                    try:
                        test_df = pd.read_csv(csv_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL, engine='python')
                        # Check row count matches
                        if len(test_df) != len(df):
                            raise ValueError(f"CSV validation failed: row count mismatch (expected {len(df)}, got {len(test_df)})")
                        if len(test_df.columns) != len(cols):
                            raise ValueError(f"CSV validation failed: column count is {len(test_df.columns)}, expected {len(cols)}. Columns: {list(test_df.columns)}")
                        if list(test_df.columns) != cols:
                            raise ValueError(f"CSV validation failed: column names don't match. Expected {cols}, got {list(test_df.columns)}")
                    except Exception as e:
                        # If validation fails, log but don't rewrite (already used manual writer)
                        import logging
                        logging.warning(f"CSV validation warning: {e}")
                        # Also print to console for debugging
                        emit(f"⚠️ CSV validation warning for {os.path.basename(csv_path)}: {e}")
                    # CRASH RECOVERY: Update checkpoint after each successful write
                    if checkpoint_data is not None and checkpoint_path and len(df) > 0:
                        try:
                            urls_in_chunk = df["Website"].astype(str).str.strip().tolist()
                            checkpoint_data.setdefault("completed_urls", set()).update(u for u in urls_in_chunk if u)
                            checkpoint_data["last_part"] = part
                            save_checkpoint(checkpoint_data, checkpoint_path)
                        except Exception as cp_err:
                            print(f"⚠️ Checkpoint update failed: {cp_err}")
                except Exception as e:
                    import logging
                    logging.error(f"CSV write failed: {e}. Using manual writer...")
                    df = df[cols]
                    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n')
                        writer.writerow(cols)
                        for idx in range(len(df)):
                            row = df.iloc[idx]
                            row_values = ["" if pd.isna(row[c]) else str(row[c]) for c in cols]
                            writer.writerow(row_values)
                
                # Excel file already written above, skip duplicate writing
                pass
            except Exception as e:
                # If Excel fails for large file, at least save CSV with perfect formatting
                try:
                    # Clean DataFrame first
                    def clean_dataframe_for_csv(df):
                        import re
                        for col in df.columns:
                            df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
                            df[col] = df[col].str.replace('\x00', '', regex=False)
                            df[col] = df[col].str.replace(r'[\n\r\f\v\t]', ' ', regex=True)
                            df[col] = df[col].str.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', regex=True)
                            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                            df[col] = df[col].str.strip()
                            df[col] = df[col].replace('', '')
                        return df
                    df = clean_dataframe_for_csv(df.copy())
                    
                    # CRITICAL: Ensure CompanySummary exists before writing
                    if "CompanySummary" not in df.columns:
                        df["CompanySummary"] = ""
                    df = df[["Website", "ScrapedText", "CompanySummary"]]
                    
                    # Use manual CSV writer for perfect quoting
                    # CRITICAL: Use iloc instead of iterrows() to prevent misalignment
                    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n', quotechar='"')
                        # ALWAYS write 3-column header
                        writer.writerow(["Website", "ScrapedText", "CompanySummary"])
                        
                        # CRITICAL: Use iloc to prevent iterrows() misalignment issues
                        for idx in range(len(df)):
                            row = df.iloc[idx]
                            row_values = []
                            
                            # Extract by column name to ensure correct order
                            website = "" if pd.isna(row["Website"]) else str(row["Website"])
                            scraped_text = "" if pd.isna(row["ScrapedText"]) else str(row["ScrapedText"])
                            company_summary = "" if pd.isna(row["CompanySummary"]) else str(row["CompanySummary"])
                            
                            # Clean values - CRITICAL: Remove newlines and normalize whitespace
                            def clean_csv_val(v):
                                if not v:
                                    return ''
                                import re
                                v = str(v)
                                v = v.replace('\x00', '')  # Remove null bytes
                                v = v.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')  # Replace newlines/tabs with space
                                v = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', v)  # Remove control chars
                                v = re.sub(r'\s+', ' ', v).strip()  # Normalize whitespace
                                return v
                            
                            row_values.append(clean_csv_val(website))
                            row_values.append(clean_csv_val(scraped_text))
                            row_values.append(clean_csv_val(company_summary))
                            
                            # Ensure exactly 3 values
                            while len(row_values) < 3:
                                row_values.append('')
                            row_values = row_values[:3]
                            
                            # CRITICAL: csv.writer with QUOTE_ALL will automatically quote all fields
                            # and escape internal quotes by doubling them (Excel standard)
                            writer.writerow(row_values)
                except:
                    # Ultimate fallback: manual writer - CRITICAL: Always write 3 columns
                    if "CompanySummary" not in df.columns:
                        df["CompanySummary"] = ""
                    df = df[["Website", "ScrapedText", "CompanySummary"]]
                    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n')
                        # ALWAYS write 3-column header
                        writer.writerow(["Website", "ScrapedText", "CompanySummary"])
                        # CRITICAL: Use iloc instead of iterrows() to prevent misalignment
                        for idx in range(len(df)):
                            row = df.iloc[idx]
                            # Extract by column name to ensure correct order, always 3 values
                            website = "" if pd.isna(row["Website"]) else str(row["Website"])
                            scraped_text = "" if pd.isna(row["ScrapedText"]) else str(row["ScrapedText"])
                            company_summary = "" if pd.isna(row["CompanySummary"]) else str(row["CompanySummary"])
                            writer.writerow([website, scraped_text, company_summary])

    # Write remaining buffer
    if buffer:
        # Filter out ONLY rows with empty URLs (keep error messages as they are valid data)
        filtered_buffer = []
        for row in buffer:
            if len(row) >= 2:
                url = str(row[0]).strip() if row[0] else ""
                scraped_text = str(row[1]).strip() if row[1] else ""
                # Include rows with valid URL - keep error messages (they start with ❌)
                # Only exclude rows where URL is empty
                if url and url != "":
                    n_c = 4 if include_email_copy else 3
                    norm = list(row[:n_c])
                    while len(norm) < n_c:
                        norm.append("")
                    filtered_buffer.append(tuple(norm[:n_c]))
        
        if filtered_buffer:
            part += 1
            cols_buf = ["Website", "ScrapedText", "CompanySummary", "EmailCopy"] if include_email_copy else ["Website", "ScrapedText", "CompanySummary"]
            try:
                df = pd.DataFrame(filtered_buffer, columns=cols_buf)
                
                # Remove ONLY rows with empty URLs (keep error messages)
                df = df[df["Website"].astype(str).str.strip() != ""]
                
                if len(df) == 0:
                    # Skip writing if DataFrame is empty after filtering
                    pass
                else:
                    csv_path = os.path.join(output_dir, f"output_part_{part}.csv")
                    df = df[cols_buf]
                    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n', quotechar='"')
                        writer.writerow(cols_buf)
                        
                        for idx in range(len(df)):
                            row = df.iloc[idx]
                            def clean_val(v):
                                if not v: return ""
                                import re
                                v = str(v).replace('\x00', '')
                                v = v.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                                v = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', v)
                                return re.sub(r'\s+', ' ', v).strip()
                            row_values = [clean_val(row[c]) for c in cols_buf]
                            row_values = (row_values + [""] * len(cols_buf))[:len(cols_buf)]
                            writer.writerow(row_values)
                    
                    excel_path = os.path.join(output_dir, f"output_part_{part}.xlsx")
                    try:
                        df_excel_buf = df[cols_buf].copy()
                        from openpyxl import Workbook
                        wb = Workbook()
                        ws = wb.active
                        _write_excel_sheet(ws, cols_buf, df_excel_buf)
                        
                        # CRITICAL: Save with explicit error handling
                        try:
                            wb.save(excel_path)
                        except Exception as save_error:
                            print(f"⚠️ Excel save failed, attempting recovery: {save_error}")
                            wb_recovery = Workbook()
                            ws_recovery = wb_recovery.active
                            ws_recovery.title = "Scraped Data"
                            ws_recovery.append(cols_buf)
                            wb_recovery.save(excel_path)
                            raise save_error
                        
                        # Verify Excel file was written
                        if os.path.exists(excel_path) and os.path.getsize(excel_path) > 0:
                            files_written.append(excel_path)
                            print(f"✅ Writer: Successfully wrote Excel file: {excel_path} ({len(df)} rows)")
                        else:
                            print(f"❌ Writer: Excel file was not created or is empty: {excel_path}")
                    except Exception as e:
                        # If Excel fails, log but continue (CSV is more important)
                        print(f"⚠️ Writer: Excel export failed for part {part}: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                emit(f"❌ Writer: Failed to write output part {part}: {type(e).__name__}: {e}")
                for line in tb.strip().split("\n"):
                    emit(line)
    
    # Final summary
    emit(f"📊 Writer: Finished. Wrote {len(files_written)} files total.")
    if files_written:
        emit(f"📁 Writer: Files written: {', '.join([os.path.basename(f) for f in files_written])}")

# -------------------------
# Test preview (no file write)
# -------------------------


async def _collector_coroutine(result_queue: asyncio.Queue, results_list: list):
    """Collect results into a list instead of writing to disk."""
    while True:
        item = await result_queue.get()
        result_queue.task_done()
        if item is None:
            break
        results_list.append(item)


async def run_test_preview(urls: list, n: int, retries, timeout, depth, keywords, max_chars,
                           user_agent: str, ai_enabled: bool, ai_api_key, ai_provider, ai_model, ai_prompt,
                           email_copy_enabled: bool, email_copy_api_key, email_copy_provider,
                           email_copy_model, email_copy_prompt,
                           lead_data_map: dict, log_callback=None) -> list:
    """Run scraping + AI on first n URLs, return results for in-browser preview. No file output."""
    def emit(msg: str):
        print(msg)
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass

    urls_subset = urls[:n]
    remaining = [(u, idx) for idx, u in enumerate(urls_subset)]
    results_list = []

    url_queue = asyncio.Queue()
    result_queue = asyncio.Queue(maxsize=n + 5)
    for u, idx in remaining:
        await url_queue.put((u, idx))

    connector = aiohttp.TCPConnector(limit=10, limit_per_host=3, ssl=False)
    session = aiohttp.ClientSession(
        headers={"User-Agent": user_agent},
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=None),
        trust_env=True,
    )

    collector_task = asyncio.create_task(_collector_coroutine(result_queue, results_list))
    workers = [
        asyncio.create_task(worker_coroutine(
            f"test-worker-{i+1}", session, url_queue, result_queue, depth, keywords, max_chars,
            retries, timeout, ai_enabled, ai_api_key, ai_provider, ai_model, ai_prompt,
            email_copy_enabled, email_copy_api_key, email_copy_provider, email_copy_model, email_copy_prompt,
            lead_data_map, None, None, fast_mode=True))
        for i in range(min(2, n))
    ]

    emit(f"🧪 Test run: scraping + AI on {n} URL(s)...")
    try:
        await url_queue.join()
        for _ in workers:
            try:
                await url_queue.put(None)
            except Exception:
                pass
        await asyncio.gather(*workers)
        await result_queue.put(None)
        await asyncio.gather(collector_task)
    except Exception as e:
        emit(f"⚠️ Test error: {e}")
    finally:
        await session.close()

    return results_list


# -------------------------
# Runner
# -------------------------


async def run_scraper(urls, concurrency, retries, timeout, depth, keywords, max_chars,
                      user_agent, rows_per_file, output_dir, progress_callback,
                      ai_enabled=False, ai_api_key=None, ai_provider=None, ai_model=None, ai_prompt=None,
                      email_copy_enabled=False, email_copy_api_key=None, email_copy_provider=None,
                      email_copy_model=None, email_copy_prompt=None,
                      lead_data_map=None, ai_status_callback=None, scrape_status_callback=None,
                      run_folder: str = "", fast_mode: bool = False,
                      low_resource: bool = False, log_callback=None):
    """
    Run the scraper. Supports resume from checkpoint if output_dir contains a checkpoint file.
    Uses backpressure on result_queue to prevent memory exhaustion with large datasets.
    Saves checkpoint on every file write and in finally block for crash recovery.
    """
    def emit(msg: str):
        print(msg)
        if log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass

    if not urls:
        emit("❌ No URLs to process. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = get_checkpoint_path(output_dir)
    checkpoint_data = load_checkpoint(checkpoint_path)
    
    if checkpoint_data is not None:
        completed_urls = checkpoint_data.get("completed_urls", set())
        start_part = checkpoint_data.get("last_part", 0)
        stored_urls = checkpoint_data.get("urls", [])
        # Only resume if stored URLs match (same run)
        if set(stored_urls) == set(urls):
            remaining_with_idx = [(u, idx) for idx, u in enumerate(urls) if u not in completed_urls]
            if not remaining_with_idx:
                emit("✅ All URLs already completed (resume found nothing to do)")
                return
            emit(f"📂 Resuming: {len(completed_urls)} already done, {len(remaining_with_idx)} remaining")
        else:
            completed_urls = set()
            start_part = 0
            remaining_with_idx = [(u, idx) for idx, u in enumerate(urls)]
    else:
        completed_urls = set()
        start_part = 0
        remaining_with_idx = [(u, idx) for idx, u in enumerate(urls)]
    
    total_overall = len(urls)
    total_this_run = len(remaining_with_idx)
    completed_before = len(completed_urls)
    
    checkpoint_data = {
        "urls": urls,
        "completed_urls": completed_urls,
        "last_part": start_part,
        "run_folder": run_folder or os.path.basename(output_dir),
        "started_at": datetime.now().isoformat(),
    }
    save_checkpoint(checkpoint_data, checkpoint_path)
    
    progress_state = {"last_count": completed_before, "last_time": time.time()}
    
    def adapted_progress(done, total_queue):
        progress_state["last_count"] = completed_before + done
        progress_state["last_time"] = time.time()
        if progress_callback:
            progress_callback(completed_before + done, total_overall)
    
    # Backpressure: smaller queue for low-resource machines
    queue_max = min(200 if low_resource else 500, total_this_run + 10)
    result_queue = asyncio.Queue(maxsize=queue_max)
    url_queue = asyncio.Queue()

    for u, idx in remaining_with_idx:
        await url_queue.put((u, idx))

    timeout_obj = aiohttp.ClientTimeout(total=None)
    # Use cookie jar for session persistence (makes requests look more legitimate)
    cookie_jar = aiohttp.CookieJar(unsafe=True)  # Allow cross-domain cookies
    # Connector - tune for low-resource vs fast mode
    if low_resource:
        conn_limit, conn_per_host = 30, 3
    elif fast_mode:
        conn_limit, conn_per_host = 200, 8
    else:
        conn_limit, conn_per_host = 100, 5
    connector = aiohttp.TCPConnector(
        limit=conn_limit,
        limit_per_host=conn_per_host,
        ssl=False,
        enable_cleanup_closed=True,
        keepalive_timeout=30,
        ttl_dns_cache=300,
        use_dns_cache=True,
    )
    # Store user agent in session headers for fetch() to use when generating realistic headers
    session = aiohttp.ClientSession(
        headers={"User-Agent": user_agent}, 
        connector=connector, 
        timeout=timeout_obj, 
        cookie_jar=cookie_jar,
        trust_env=True  # Use system proxy settings if available
    )

    writer_task = asyncio.create_task(writer_coroutine(
        result_queue, rows_per_file, output_dir, total_this_run, adapted_progress,
        start_part=start_part, checkpoint_data=checkpoint_data, checkpoint_path=checkpoint_path,
        log_callback=log_callback, include_email_copy=email_copy_enabled))

    workers = [asyncio.create_task(worker_coroutine(
        f"worker-{i+1}", session, url_queue, result_queue, depth, keywords, max_chars, retries, timeout,
        ai_enabled, ai_api_key, ai_provider, ai_model, ai_prompt,
        email_copy_enabled, email_copy_api_key, email_copy_provider, email_copy_model, email_copy_prompt,
        lead_data_map, ai_status_callback, scrape_status_callback, fast_mode)) for i in range(concurrency)]

    # Timeout: cap at 3 hours, plus 10 min headroom for pause/recovery phases
    base_time = (timeout * (retries + 1) * (depth + 1) * 2 * total_this_run) + (30 * total_this_run)
    max_queue_time = min(base_time + 600, 3 * 3600 + 600)  # +10 min for pause-and-wait recovery
    emit(f"⚙️ Run config: total={total_overall}, remaining={total_this_run}, queue_max={queue_max}, max_queue_time={int(max_queue_time)}s, fast_mode={fast_mode}, low_resource={low_resource}")
    
    recovery_wait_s = 300  # 5 minutes: wait for slow network/system before giving up
    recovery_chunk_s = 30  # Check progress every 30s during recovery
    stall_complete_event = asyncio.Event()  # Set when stall confirmed and we should force shutdown
    near_completion_recovery = 90  # When 99%+ done and stuck, use shorter recovery

    async def force_completion_on_stall():
        """Detect stall and PAUSE for recovery (slow network/disconnect) before draining."""
        stall_threshold = 240 if low_resource else 300  # 4–5 min before considering stall
        while True:
            await asyncio.sleep(15)
            elapsed = time.time() - progress_state["last_time"]
            last_count = progress_state["last_count"]
            remaining = total_overall - last_count
            if remaining <= 0:
                break
            # Near completion (1–5 URLs left): use shorter threshold so we don't wait hours for 1 stuck worker
            pct_done = last_count / max(total_overall, 1)
            use_shorter = remaining <= 5 and pct_done >= 0.99
            effective_threshold = min(stall_threshold, near_completion_recovery) if use_shorter else stall_threshold
            if use_shorter:
                effective_recovery = 60  # 1 min for near-completion stall
            else:
                effective_recovery = recovery_wait_s
            if elapsed > effective_threshold:
                emit(f"⏸️ Pausing: no progress for {int(elapsed)}s ({last_count}/{total_overall} done). Waiting up to {effective_recovery // 60} min for recovery...")
                waited = 0
                recovered = False
                while waited < effective_recovery:
                    await asyncio.sleep(recovery_chunk_s)
                    waited += recovery_chunk_s
                    new_count = progress_state["last_count"]
                    if new_count > last_count:
                        recovered = True
                        emit(f"✅ Recovery: progress resumed ({new_count - last_count} completed). Continuing.")
                        break
                    emit(f"⏳ Still waiting... {waited}s / {effective_recovery}s (no progress yet)")
                if not recovered:
                    emit(f"⚠️ No recovery. Stopping stuck workers (1 URL may be missing; will retry on resume).")
                    drained = 0
                    while drained < remaining + 100:
                        try:
                            item = url_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if item is None or item == (None, None):
                            url_queue.task_done()
                            continue
                        url, idx = (item[0], item[1]) if isinstance(item, tuple) else (item, None)
                        err_result = (url, "❌ Stall: worker stopped after recovery wait", "", "") if email_copy_enabled else (url, "❌ Stall: worker stopped after recovery wait", "")
                        for _ in range(20):
                            try:
                                result_queue.put_nowait(err_result)
                                break
                            except asyncio.QueueFull:
                                await asyncio.sleep(0.3)
                        else:
                            emit("⚠️ Result queue full during stall drain")
                        url_queue.task_done()
                        drained += 1
                    stall_complete_event.set()
                break
    
    stall_monitor = asyncio.create_task(force_completion_on_stall())
    
    async def heartbeat():
        while True:
            await asyncio.sleep(20)
            done = progress_state["last_count"]
            elapsed = int(time.time() - progress_state["last_time"])
            emit(f"💓 Heartbeat: {done}/{total_overall} done, idle_for={elapsed}s, url_q={url_queue.qsize()}, result_q={result_queue.qsize()}")
    heartbeat_task = asyncio.create_task(heartbeat())
    
    try:
        join_task = asyncio.create_task(url_queue.join())
        stall_wait_task = asyncio.create_task(stall_complete_event.wait())
        done, pending = await asyncio.wait(
            [join_task, stall_wait_task],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=max_queue_time
        )
        stall_wait_task.cancel()
        try:
            await stall_wait_task
        except asyncio.CancelledError:
            pass
        join_completed = join_task.done() and not join_task.cancelled() and join_task.exception() is None
        stall_triggered = stall_complete_event.is_set()
        if not join_completed:
            join_task.cancel()
            try:
                await join_task
            except asyncio.CancelledError:
                pass
            if not stall_triggered:
                emit(f"⏸️ Queue join timed out. Pausing 3 min for recovery...")
                await asyncio.sleep(180)
                emit(f"⚠️ Draining remaining URLs...")
            drained = 0
            while drained < total_this_run + 50:
                try:
                    item = url_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None or item == (None, None):
                    url_queue.task_done()
                    continue
                url = item[0] if isinstance(item, tuple) else item
                err_msg = "❌ Stall: worker stopped" if stall_triggered else f"❌ Timeout: exceeded {max_queue_time}s"
                err_result = (url, err_msg, "", "") if email_copy_enabled else (url, err_msg, "")
                for _ in range(30):
                    try:
                        result_queue.put_nowait(err_result)
                        break
                    except asyncio.QueueFull:
                        await asyncio.sleep(0.3)
                else:
                    emit("⚠️ Result queue full during drain")
                url_queue.task_done()
                drained += 1

        stall_monitor.cancel()
        heartbeat_task.cancel()
        try:
            await stall_monitor
        except asyncio.CancelledError:
            pass
        except Exception as e:
            emit(f"⚠️ Stall monitor error: {e}")
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            emit(f"⚠️ Heartbeat monitor error: {e}")

        for _ in workers:
            try:
                await url_queue.put(None)
            except Exception:
                pass
        
        try:
            await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=30)
        except asyncio.TimeoutError:
            for worker in workers:
                if not worker.done():
                    worker.cancel()

        try:
            await result_queue.put(None)
        except Exception:
            pass
        
        try:
            await asyncio.wait_for(result_queue.join(), timeout=60)
        except asyncio.TimeoutError:
            emit("⚠️ Result queue join timed out. Proceeding anyway...")
        
        try:
            await asyncio.wait_for(writer_task, timeout=120)
        except asyncio.TimeoutError:
            emit("⚠️ Writer task timed out. Proceeding anyway...")
            writer_task.cancel()
        except Exception as e:
            emit(f"⚠️ Error waiting for writer: {e}")

        await session.close()
        await asyncio.sleep(2)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        emit(f"⚠️ Scraper error (progress saved): {type(e).__name__}: {e}")
        emit("--- Full traceback ---")
        for line in tb.strip().split("\n"):
            emit(line)
        emit("--- End traceback ---")
    finally:
        # Always clean up running tasks/session to avoid hidden hangs between runs
        for task in [stall_monitor, heartbeat_task, writer_task, *workers]:
            try:
                if task and not task.done():
                    task.cancel()
            except Exception:
                pass
        try:
            if not session.closed:
                await session.close()
        except Exception:
            pass
        save_checkpoint(checkpoint_data, checkpoint_path)
        emit("💾 Checkpoint saved.")

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(
    page_title="Web Scraper", 
    layout="wide", 
    page_icon="🌐",
    initial_sidebar_state="collapsed"
)

# Comprehensive modern UI styling
st.markdown("""
<style>
    /* Main container improvements */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    h1 {
        color: #1f2937;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }
    
    /* Section headers */
    h3 {
        color: #1f2937;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    
    h4 {
        color: #374151;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Select boxes */
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
    }
    
    /* Sliders */
    .stSlider > div > div {
        border-radius: 8px;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border-radius: 8px;
        border: 2px dashed #cbd5e1;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background-color: #f8fafc;
    }
    
    /* Success/Error messages */
    .stSuccess {
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #374151;
    }
    
    /* Captions */
    .stCaption {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e5e7eb;
    }
    
    /* Radio buttons */
    .stRadio > div {
        gap: 1rem;
    }
    
    .stRadio > div > label {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: background-color 0.2s ease;
    }
    
    /* Number inputs */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
    }
    
    /* Remove default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Modern header with better styling
st.markdown("""
<div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
    <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               margin-bottom: 0.5rem;
               font-size: 3rem;
               font-weight: 800;">
        🌐 Website Scraper
    </h1>
    <p style="color: #6b7280; font-size: 1.1rem; margin-top: 0.5rem;">
        Scrape websites from your CSV file and get clean, structured text content
    </p>
</div>
""", unsafe_allow_html=True)

# -------- RESUME / PARTIAL RESULTS --------
# Scan outputs/ for checkpoints - show runs that can be resumed or have partial data
outputs_dir = "outputs"
if os.path.isdir(outputs_dir):
    incomplete_runs = []
    for run_folder in sorted(os.listdir(outputs_dir), reverse=True)[:10]:
        run_path = os.path.join(outputs_dir, run_folder)
        if not os.path.isdir(run_path):
            continue
        ck_path = get_checkpoint_path(run_path)
        ck = load_checkpoint(ck_path)
        if not ck:
            continue
        total_urls = len(ck.get("urls", []))
        # Use actual row count from files (source of truth) — checkpoint can be wrong after crash
        actual_rows, _ = get_actual_completed_from_files(run_path)
        completed = actual_rows if actual_rows > 0 else len(ck.get("completed_urls", []))
        if completed > 0 and completed < total_urls:
            incomplete_runs.append({"folder": run_folder, "completed": completed, "total": total_urls, "path": run_path})
        elif completed >= total_urls and total_urls > 0:
            incomplete_runs.append({"folder": run_folder, "completed": completed, "total": total_urls, "path": run_path, "done": True})

    if incomplete_runs:
        with st.expander("📂 Resume or Download Partial Results", expanded=True):
            st.markdown("**If the app crashed or stopped, you can:**")
            for run in incomplete_runs:
                is_done = run.get("done", False)
                label = f"✅ {run['folder']}: {run['completed']:,}/{run['total']:,} rows" if is_done else f"⏸️ {run['folder']}: {run['completed']:,}/{run['total']:,} rows in files (partial)"
                with st.container():
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"**{label}**")
                        if not is_done:
                            st.caption("To resume: Re-upload your CSV and click Start — the app will auto-detect and continue.")
                    with col_b:
                        zip_path = os.path.join(run["path"], f"{run['folder']}.zip")
                        if not os.path.exists(zip_path):
                            csv_files = [f for f in os.listdir(run["path"]) if f.endswith(".csv") and "combined" not in f]
                            if csv_files:
                                try:
                                    import zipfile
                                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                                        for f in csv_files:
                                            zf.write(os.path.join(run["path"], f), arcname=f)
                                        for f in os.listdir(run["path"]):
                                            if (f.endswith(".xlsx") or (f.endswith(".txt") and f.startswith("crash_log_"))) and "combined" not in f.lower():
                                                zf.write(os.path.join(run["path"], f), arcname=f)
                                except Exception:
                                    pass
                        if os.path.exists(zip_path):
                            with open(zip_path, "rb") as f:
                                st.download_button("⬇️ Download", f.read(), file_name=f"{run['folder']}.zip", mime="application/zip", key=f"resume_dl_{run['folder']}")
                    st.markdown("---")

# Step 1: Upload CSV
st.markdown("### 📁 Step 1: Upload CSV")
st.markdown("""
<div style="background-color: #f0f9ff; border-left: 4px solid #0ea5e9; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
    <strong>💡 Tip:</strong> Your CSV should have a column with website URLs (one per row)
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose CSV file", 
    type=["csv"],
    help="Upload a CSV file with website URLs"
)

# CSV Configuration (shown after file upload)
csv_has_headers = None
url_column = None
lead_data_columns = {}
df_preview = None

if uploaded_file is not None:
    # CSV Configuration - Cleaner layout
    st.markdown("---")
    st.markdown("### 📋 CSV Configuration")

    # Preview first few rows
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Read a sample to detect structure
        sample_df = pd.read_csv(uploaded_file, nrows=5)
        uploaded_file.seek(0)  # Reset again
        
        col_csv1, col_csv2 = st.columns(2)
        
        with col_csv1:
            csv_has_headers = st.radio(
                "First row has column names?",
                ["Yes", "No"],
                help="Does the first row contain column names like 'URL', 'Website', etc?",
                key="csv_headers_radio"
            )
        
        with col_csv2:
            st.caption("💡 **Tip:** If yes, you'll see column names to choose from")
        
        # Read CSV based on header selection
        uploaded_file.seek(0)  # Reset file pointer
        if csv_has_headers == "Yes":
            df_preview = pd.read_csv(uploaded_file, nrows=10)
            st.success(f"✅ Found {len(df_preview.columns)} columns: {', '.join(df_preview.columns.tolist()[:5])}{'...' if len(df_preview.columns) > 5 else ''}")
        else:
            df_preview = pd.read_csv(uploaded_file, header=None, nrows=10)
            st.info(f"ℹ️ Found {len(df_preview.columns)} columns (no headers)")
        
        # Column selection
        st.markdown("#### Select Columns")
        
        col_sel1, col_sel2 = st.columns(2)
        
        with col_sel1:
            # URL column selection
            if csv_has_headers == "Yes":
                url_column = st.selectbox(
                    "Which column has URLs?",
                    options=list(df_preview.columns),
                    index=0,
                    help="Select the column that contains website URLs"
                )
            else:
                url_column = st.selectbox(
                    "Which column has URLs?",
                    options=[f"Column {i+1}" for i in range(len(df_preview.columns))],
                    index=0,
                    help="Select the column that contains website URLs"
                )
        
        with col_sel2:
            st.caption("📌 **Required** - Must select this")
        
        # All non-URL columns become variables for AI prompts (Step 3 & 4)
        st.markdown("---")
        col_options = list(df_preview.columns) if csv_has_headers == "Yes" else [f"Column {i+1}" for i in range(len(df_preview.columns))]
        lead_data_columns = {}
        for col in col_options:
            if col == url_column:
                continue
            key = _col_to_placeholder(col)
            if key:
                lead_data_columns[key] = col
        with st.expander("📊 Variables from your sheet", expanded=True):
            st.caption("Every column (except the URL column) is available as a variable in Steps 3 & 4. E.g. column \"Website Name\" → {website_name} in your prompt.")
            if lead_data_columns:
                st.code(" • ".join(f"{{{k}}}" for k in sorted(lead_data_columns.keys())[:20]) + (" …" if len(lead_data_columns) > 20 else ""), language=None)
        
        # Store in session state for use during scraping and token estimator
        url_col_idx = list(df_preview.columns).index(url_column) if csv_has_headers == "Yes" else int(url_column.replace("Column ", "")) - 1
        url_count = sum(1 for u in df_preview.iloc[:, url_col_idx].fillna("").astype(str) if str(u).strip())
        st.session_state['csv_config'] = {
            'has_headers': csv_has_headers == "Yes",
            'url_column': url_column,
            'lead_data_columns': lead_data_columns,
            'df_preview': df_preview,
            'url_count': url_count
        }
        st.session_state['_csv_url_count'] = url_count
        
    except Exception as e:
        st.error(f"❌ Error reading CSV: {str(e)}")
        st.info("Please check your CSV file format and try again.")

# Step 2: Settings
st.markdown("### ⚙️ Step 2: Settings")
st.markdown("""
<div style="background-color: #f0fdf4; border-left: 4px solid #22c55e; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
    <strong>💡 Tip:</strong> Most settings have good defaults. The app auto-adapts (fewer workers on slow PCs, cache fallback for unreachable sites).
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Basic Settings")
    
    # Keywords - Simplified
    keywords_input = st.text_input(
        "Keywords to find (optional)",
        value=st.session_state.get('keywords_input', "about,service,product"),
        help="Comma-separated keywords. The scraper will look for pages with these in the URL (e.g., 'about', 'service', 'product')",
        key="keywords_input"
    )
    keywords = process_keywords(keywords_input)
    st.session_state['keywords'] = keywords
    
    if keywords:
        st.caption(f"✅ Looking for: {', '.join(keywords)}")
    else:
        st.caption("💡 Leave empty to scrape homepage only")
    
    concurrency_max = _get_concurrency_max()
    concurrency_default = min(st.session_state.get('concurrency', min(20, concurrency_max)), concurrency_max)
    concurrency = st.slider(
        "Parallel workers",
        1, concurrency_max, max(1, concurrency_default),
        help=f"Sites scraped at once. Max {concurrency_max} (CPU-based). Lower = gentler on slow PCs.",
        key="concurrency"
    )
    force_max_workers = st.checkbox(
        "Force maximum workers",
        value=st.session_state.get('force_max_workers', False),
        help=f"Override to {concurrency_max} workers for testing. Use with caution on weaker machines.",
        key="force_max_workers"
    )
    if force_max_workers:
        concurrency = concurrency_max
    
    # Pages to scrape per site
    depth = st.slider(
        "Pages to scrape per site",
        0, 5, st.session_state.get('depth', 3),
        help="0 = homepage only. 3 = homepage + 3 more pages.",
        key="depth"
    )
    
    # Retries
    retries = st.number_input(
        "Retries if failed", 
        0, 5, st.session_state.get('retries', 2),
        help="If a website fails, how many times to try again. 2 is usually enough.",
        key="retries"
    )

with col2:
    st.markdown("#### Advanced Settings")
    
    # Timeout - generous defaults for unreliable sites (auto-applied)
    timeout = st.number_input(
        "Wait time per site (seconds)",
        5, 120, st.session_state.get('timeout', 30),
        help="How long to wait per site. Unreachable sites fall back to Google cache / archive.org.",
        key="timeout"
    )

    # Max chars: default 10k, max 50k
    max_chars = st.number_input(
        "Text limit per site",
        1000, 50000, st.session_state.get('max_chars', 10000),
        step=1000,
        help="Characters per site (default 10,000, max 50,000).",
        key="max_chars"
    )

    # Rows per file - auto-tuned for system
    rows_per_file_default = 1000 if _is_low_resource_default() else st.session_state.get('rows_per_file', 2000)
    rows_per_file = st.number_input(
        "Split files every X rows", 
        500, 50000, min(rows_per_file_default, 50000), step=500,
        help="Lower = less memory per write. Use 500-1000 on slow computers.",
        key="rows_per_file"
    )
    
    # User Agent - Hidden in expander
    with st.expander("🔧 User-Agent (Advanced)", expanded=False):
        user_agent = st.text_input(
            "User-Agent", 
            value=st.session_state.get('user_agent', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"),
            help="Browser identifier sent to websites. Usually don't need to change this.",
            key="user_agent"
        )

# Step 3: Company Summary Generator (optional)
st.markdown("### 📋 Step 3: Company Summary Generator (optional)")
st.markdown("""
<div style="background-color: #faf5ff; border-left: 4px solid #a855f7; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
    Turn your web scrapes into structured company summaries — clear summaries, grounded facts, and commercial hypotheses. Helps you quickly understand what each company does. Requires an API key.
</div>
""", unsafe_allow_html=True)

ai_enabled = st.checkbox(
    "Enable company summaries",
    value=st.session_state.get('ai_enabled', False),
    help="Generate structured summaries for each scraped website using AI",
    key="ai_enabled_checkbox"
)
st.session_state['ai_enabled'] = ai_enabled

if ai_enabled:
    st.markdown("---")
    st.markdown("#### Provider")
    ai_provider = st.selectbox(
        "AI provider",
        ["OpenAI", "Gemini", "OpenRouter"],
        help="OpenAI (GPT models), Google Gemini, or OpenRouter (one key for 600+ models)",
        key="ai_provider_select"
    )
    st.session_state['ai_provider'] = ai_provider
    
    st.markdown("#### API key")
    api_key_key = f"{ai_provider.lower()}_api_key"
    stored_api_key = st.session_state.get(api_key_key, "")
    
    if ai_provider == "OpenAI":
        st.caption("Create a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)")
    elif ai_provider == "Gemini":
        st.caption("Create a key at [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)")
    else:
        st.caption("Create a key at [openrouter.ai/keys](https://openrouter.ai/keys) — one key for GPT, Claude, Gemini, and more")
    
    ai_api_key = st.text_input(
        f"{ai_provider} API key",
        value=stored_api_key,
        type="password",
        help="Paste your API key. It’s stored in your session and never saved to disk.",
    )
    
    # Save API key
    if ai_api_key:
        if ai_api_key != stored_api_key:
            st.session_state[api_key_key] = ai_api_key
            st.session_state['ai_api_key'] = ai_api_key
            st.session_state[f"{ai_provider.lower()}_models"] = None
        st.success("API key saved")
    elif stored_api_key and not ai_api_key:
        st.session_state[api_key_key] = ""
        st.session_state['ai_api_key'] = ""
        st.session_state[f"{ai_provider.lower()}_models"] = None
    
    # Always sync generic key
    st.session_state['ai_api_key'] = st.session_state.get(api_key_key, "")
    
    st.markdown("#### Model")
    models_cache_key = f"{ai_provider.lower()}_models"
    cached_models = st.session_state.get(models_cache_key)
    
    if ai_api_key and ai_api_key.strip():
        if cached_models is None:
            with st.spinner(f"Loading available models..."):
                try:
                    if os.name == "nt":
                        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                    
                    if ai_provider == "OpenAI":
                        models = asyncio.run(fetch_openai_models(ai_api_key))
                    elif ai_provider == "Gemini":
                        models = asyncio.run(fetch_gemini_models(ai_api_key))
                    else:
                        models = asyncio.run(fetch_openrouter_models(ai_api_key))
                    
                    if models:
                        st.session_state[models_cache_key] = models
                        cached_models = models
                    else:
                        cached_models = (
                            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] if ai_provider == "OpenAI"
                            else ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"] if ai_provider == "Gemini"
                            else ["openai/gpt-4o-mini", "openai/gpt-4o", "google/gemini-2.0-flash-exp:free"]
                        )
                        st.session_state[models_cache_key] = cached_models
                except Exception as e:
                    st.warning("⚠️ Could not load models. Using defaults.")
                    cached_models = (
                        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] if ai_provider == "OpenAI"
                        else ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"] if ai_provider == "Gemini"
                        else ["openai/gpt-4o-mini", "openai/gpt-4o", "google/gemini-2.0-flash-exp:free"]
                    )
                    st.session_state[models_cache_key] = cached_models
        
        if cached_models:
            default_index = 0
            if ai_provider == "OpenAI":
                if "gpt-4o-mini" in cached_models:
                    default_index = cached_models.index("gpt-4o-mini")
                elif "gpt-4o" in cached_models:
                    default_index = cached_models.index("gpt-4o")
            elif ai_provider == "Gemini":
                if "gemini-1.5-flash" in cached_models:
                    default_index = cached_models.index("gemini-1.5-flash")
                elif "gemini-1.5-pro" in cached_models:
                    default_index = cached_models.index("gemini-1.5-pro")
            else:
                for preferred in ["openai/gpt-4o-mini", "openai/gpt-4o", "google/gemini-2.0-flash-exp:free"]:
                    if preferred in cached_models:
                        default_index = cached_models.index(preferred)
                        break
            
            ai_model = st.selectbox(
                "Model",
                options=cached_models,
                index=default_index,
                help="Recommended: gpt-4o-mini (fast) or gpt-4o (best).",
                key=f"{ai_provider}_model_select"
            )
            st.session_state['ai_model'] = ai_model
            st.session_state['ai_provider'] = ai_provider
            
            st.caption(f"{len(cached_models)} models available")
        else:
            ai_model = None
    else:
        st.info("Enter your API key above to see available models.")
        ai_model = None
    
    st.markdown("#### Prompt")
    prompt_mode = st.radio(
        "Prompt",
        ["Use default prompt", "Customize prompt"],
        horizontal=True,
        help="The default prompt is tuned for company summaries, facts, and hypotheses."
    )
    
    default_prompt_template = """You are Hypothesis Bot… an advanced commercial analysis agent.

INPUT
You will receive ONLY one input: raw website copy scraped from a company's website (may include multiple pages).

CORE JOB
Turn messy webcopy into:
1) a sharp company intelligence SUMMARY
2) a clean set of FACTS grounded in the text
3) commercially relevant HYPOTHESES inferred from signals, gaps, tone, and structure

You are not writing outreach. You are building the intelligence layer that enables personalization later.

NON NEGOTIABLE RULES
1) Do not invent facts. If it is not explicitly supported by the webcopy, it is not a fact.
2) Facts must be short and must include Evidence Quote… an exact snippet from the webcopy.
3) Hypotheses must be explicitly labeled as hypotheses and must include:
   • Signal… the specific wording or structural clue that triggered the inference
   • Commercial implication… why it matters in a sales or growth context
   • Confidence: High, Medium, or Low
4) Never mention any external tools or data sources (Apollo, LinkedIn, Crunchbase, funding, headcount, etc.). You only have webcopy.
5) If the company name is unclear, write "Company: Not explicitly stated" and proceed.
6) Avoid cringe adjectives like great, amazing, innovative. Be surgical.
7) Avoid criticism. Frame gaps neutrally as "signals" or "absence suggests".
8) Keep it concise. No fluff. No extra sections.

ANALYSIS GUIDELINES
When extracting facts, prioritize:
• What they do (offer categories, deliverables)
• Who they serve (industries, segments, buyer language)
• How they sell (engagement models, pricing mentions, process)
• Proof (case studies, client names, testimonials, quantified claims)
• Differentiators (positioning phrases, guarantees, compliance, security)
• Operational signals (hiring, support hours, global language, locations)
• Technology signals (stacks, platforms, integrations) only if stated

When generating hypotheses, prioritize:
• Likely buying triggers (growth, hiring, new initiatives, modernization)
• Likely pain points (capacity, speed, differentiation, trust, compliance, delivery)
• Likely maturity level (specialist vs generalist, product vs services)
• Likely stakeholder priorities (risk reduction, outcomes, speed, cost certainty)
• Contradictions or gaps between claims and proof
Each hypothesis must connect to a commercial implication.

LEAD INFORMATION:
- Website URL: {url}
- Company Name: {company_name}

WEBSITE CONTENT (ONLY SOURCE OF INFORMATION):
{scraped_content}

OUTPUT FORMAT (STRICT - NO MARKDOWN, NO TRUNCATION)
Return exactly these 3 sections with EXACT headers:

===SUMMARY===
Write 3 to 5 sentences in one paragraph describing:
• what the company appears to do
• who it appears to serve
• how it positions itself
• one notable proof element (only if present)
Mark inferences with (obs).
Do NOT use markdown (**, __, #). Use plain text only.

===FACTS===
Provide 10 to 18 facts.
Format each EXACTLY as:
Fact: [statement]
Evidence Quote: "[COMPLETE quote from webcopy - do NOT truncate with ...]"
Source: [Page name if present, otherwise "Webcopy"]

CRITICAL: Evidence quotes must be COMPLETE sentences from the webcopy. Do NOT truncate mid-word or mid-sentence.

===HYPOTHESES===
Provide 6 to 12 hypotheses.
Format each EXACTLY as:
Hypothesis: [inference statement] (obs)
Signal: "[specific wording or structural clue from webcopy]"
Commercial implication: [why it matters in sales/growth context]
Confidence: High or Medium or Low

CRITICAL: 
- Do NOT use markdown formatting (**, __, #)
- Do NOT truncate words mid-word
- Do NOT mix facts into hypotheses section
- Keep facts grounded (from webcopy), hypotheses inferred (from signals)"""
    
    if 'master_prompt' not in st.session_state:
        st.session_state['master_prompt'] = default_prompt_template
    
    if prompt_mode == "Customize prompt":
        st.markdown("#### Select variables (from your sheet)")
        vars_def = _get_variable_definitions()
        label_to_placeholder = {label: placeholder for (label, placeholder, _) in vars_def}
        var_options = [label for (label, _, _) in vars_def]
        prev_selection = set(st.session_state.get("step3_var_selection", []))
        selected_labels = st.multiselect(
            "Add variables to prompt",
            options=var_options,
            default=st.session_state.get("step3_var_selection", []),
            help="Select one or more; they will be inserted at the end of the prompt. Same headers as your uploaded CSV.",
            key="step3_var_multiselect"
        )
        st.session_state["step3_var_selection"] = selected_labels
        new_added = set(selected_labels) - prev_selection
        if new_added:
            cur = st.session_state.get("master_prompt", default_prompt_template)
            to_append = "".join(label_to_placeholder[l] for l in selected_labels if l in new_added and l in label_to_placeholder)
            if to_append:
                st.session_state["master_prompt"] = cur + to_append
                st.rerun()
        ai_prompt = st.text_area(
            "Custom prompt",
            value=st.session_state.get("master_prompt", default_prompt_template),
            height=300,
            help="You can also type variables manually with curly brackets, e.g. {company_name} or {revenue_m}.",
            key="ai_prompt_edit"
        )
        st.session_state["master_prompt"] = ai_prompt
        prompt_cur = st.session_state.get("master_prompt", "")
        ends_with_brace = prompt_cur.rstrip().endswith("{{") or prompt_cur.rstrip().endswith("{")
        if ends_with_brace:
            vars_def_comp = _get_variable_definitions()
            placeholders_comp = [p for (_, p, _) in vars_def_comp]
            st.caption("Complete your variable — select one to insert:")
            complete_choice = st.selectbox(
                "Variable to insert",
                options=placeholders_comp,
                key="step3_complete_var",
                label_visibility="collapsed"
            )
            if st.button("Insert variable", key="step3_insert_var_btn"):
                base = prompt_cur.rstrip()
                if base.endswith("{{"):
                    base = base[:-2]
                    key = complete_choice.strip("{}")
                    insert = "{{" + key + "}}"
                else:
                    base = base[:-1]
                    insert = complete_choice if complete_choice.startswith("{") else "{" + complete_choice + "}"
                st.session_state["master_prompt"] = base + insert
                st.rerun()
        st.caption("You can type variables manually using curly brackets, e.g. {company_name}. Use {var} or {{var}} — both work.")
        if st.button("Preview with sample lead", key="step3_sample_btn"):
            st.session_state["_sample_dialog"] = "master_prompt"
            st.rerun()
    else:
        ai_prompt = st.session_state.get("master_prompt", default_prompt_template)
    
    st.session_state["ai_prompt"] = ai_prompt

else:
    ai_api_key = None
    ai_provider = None
    ai_model = None
    ai_prompt = None

# Step 4: Email Copy Writer (optional, works independently)
st.markdown("---")
st.markdown("### ✉️ Step 4: Email Copy Writer (optional)")
st.markdown("""
<div style="background-color: #f0fdf4; border-left: 4px solid #22c55e; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
    Generate personalized email copy for each lead based on scraped content. Runs independently from company summaries. Same AI providers supported.
</div>
""", unsafe_allow_html=True)

email_copy_enabled = st.checkbox(
    "Enable email copy generation",
    value=st.session_state.get('email_copy_enabled', False),
    help="Generate email copy for each scraped lead using AI",
    key="email_copy_enabled_checkbox"
)
st.session_state['email_copy_enabled'] = email_copy_enabled

if email_copy_enabled:
    st.markdown("---")
    use_step3_credentials = st.checkbox(
        "Use same API key and model as Step 3",
        value=st.session_state.get('email_copy_use_step3', False),
        help="Reuse the provider, API key, and model from Company Summary (Step 3). Requires Step 3 to be enabled.",
        key="email_copy_use_step3_checkbox"
    )
    st.session_state['email_copy_use_step3'] = use_step3_credentials
    if use_step3_credentials and ai_enabled:
        email_copy_provider = ai_provider
        email_copy_api_key = ai_api_key
        email_copy_model = ai_model
        st.session_state['email_copy_provider'] = email_copy_provider
        st.session_state['email_copy_model'] = email_copy_model
        st.caption(f"Using Step 3: {ai_provider} / {ai_model}")
    else:
        if use_step3_credentials and not ai_enabled:
            st.warning("Enable Step 3 (Company Summary) to use its API key and model.")
        st.markdown("#### Provider")
        email_copy_provider = st.selectbox(
            "AI provider",
            ["OpenAI", "Gemini", "OpenRouter"],
            key="email_copy_provider_select"
        )
        st.session_state['email_copy_provider'] = email_copy_provider
        st.markdown("#### API key")
        ec_api_key_name = f"{email_copy_provider.lower()}_api_key"
        email_copy_api_key = st.text_input(
            "API key",
            value=st.session_state.get(ec_api_key_name, ""),
            type="password",
            key="email_copy_api_key_input"
        )
        st.session_state[ec_api_key_name] = email_copy_api_key
        st.markdown("#### Model")
        if email_copy_provider == "OpenAI":
            email_copy_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], key="email_copy_model_select")
        elif email_copy_provider == "Gemini":
            email_copy_model = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"], key="email_copy_model_select")
        else:
            email_copy_model = st.text_input("OpenRouter model (e.g. openai/gpt-4o-mini)", value="openai/gpt-4o-mini", key="email_copy_model_input")
        st.session_state['email_copy_model'] = email_copy_model
    st.markdown("#### Prompt")
    default_email_prompt = """You are an expert B2B sales copywriter.

Write a personalized outreach email for this company based on their website content.

LEAD INFO:
- URL: {url}
- Company: {company_name}

WEBSITE CONTENT:
{scraped_content}

Requirements:
- Keep the email under 150 words
- Be specific to their business (reference facts from their site)
- Professional, conversational tone
- Clear call-to-action
- No generic fluff"""
    if 'email_copy_prompt' not in st.session_state:
        st.session_state['email_copy_prompt'] = default_email_prompt
    st.markdown("#### Select variables (from your sheet)")
    vars_def_ec = _get_variable_definitions()
    label_to_placeholder_ec = {label: placeholder for (label, placeholder, _) in vars_def_ec}
    var_options_ec = [label for (label, _, _) in vars_def_ec]
    prev_selection_ec = set(st.session_state.get("step4_var_selection", []))
    selected_labels_ec = st.multiselect(
        "Add variables to prompt",
        options=var_options_ec,
        default=st.session_state.get("step4_var_selection", []),
        help="Select one or more; they will be inserted at the end of the prompt. Same headers as your uploaded CSV.",
        key="step4_var_multiselect"
    )
    st.session_state["step4_var_selection"] = selected_labels_ec
    new_added_ec = set(selected_labels_ec) - prev_selection_ec
    if new_added_ec:
        cur_ec = st.session_state.get("email_copy_prompt", default_email_prompt)
        to_append_ec = "".join(label_to_placeholder_ec[l] for l in selected_labels_ec if l in new_added_ec and l in label_to_placeholder_ec)
        if to_append_ec:
            st.session_state["email_copy_prompt"] = cur_ec + to_append_ec
            st.rerun()
    email_copy_prompt = st.text_area(
        "Email copy prompt",
        value=st.session_state.get('email_copy_prompt', default_email_prompt),
        height=180,
        help="You can also type variables manually with curly brackets, e.g. {company_name}.",
        key="email_copy_prompt_edit"
    )
    st.session_state['email_copy_prompt'] = email_copy_prompt
    prompt_cur_ec = st.session_state.get("email_copy_prompt", "")
    ends_with_brace_ec = prompt_cur_ec.rstrip().endswith("{{") or prompt_cur_ec.rstrip().endswith("{")
    if ends_with_brace_ec:
        vars_def_comp_ec = _get_variable_definitions()
        placeholders_comp_ec = [p for (_, p, _) in vars_def_comp_ec]
        st.caption("Complete your variable — select one to insert:")
        complete_choice_ec = st.selectbox(
            "Variable to insert",
            options=placeholders_comp_ec,
            key="step4_complete_var",
            label_visibility="collapsed"
        )
        if st.button("Insert variable", key="step4_insert_var_btn"):
            base_ec = prompt_cur_ec.rstrip()
            if base_ec.endswith("{{"):
                base_ec = base_ec[:-2]
                key_ec = complete_choice_ec.strip("{}")
                insert_ec = "{{" + key_ec + "}}"
            else:
                base_ec = base_ec[:-1]
                insert_ec = complete_choice_ec if complete_choice_ec.startswith("{") else "{" + complete_choice_ec + "}"
            st.session_state["email_copy_prompt"] = base_ec + insert_ec
            st.rerun()
    st.caption("You can type variables manually using curly brackets, e.g. {company_name}. Use {var} or {{var}} — both work.")
    if st.button("Preview with sample lead", key="step4_sample_btn"):
        st.session_state["_sample_dialog"] = "email_copy_prompt"
        st.rerun()
else:
    email_copy_api_key = None
    email_copy_provider = None
    email_copy_model = None
    email_copy_prompt = None

if st.session_state.get("_sample_dialog"):
    _sample_prompt_dialog(st.session_state["_sample_dialog"])

# Token usage estimator (self-calculating from scraper settings + prompts)
st.markdown("---")
with st.expander("💰 Token usage estimator", expanded=False):
    MODEL_PRICING = {
        "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-4-turbo": (10.00, 30.00),
        "gpt-3.5-turbo": (0.50, 1.50), "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.5-pro": (1.25, 5.00), "gemini-pro": (0.50, 1.50),
    }
    def get_model_pricing(model):
        for k, v in MODEL_PRICING.items():
            if k in (model or "").lower():
                return v
        return (1.0, 3.0)
    def chars_to_tokens(chars):
        return max(0, int(chars / 4))
    # URLs: use CSV count if available, else number input
    num_urls_csv = st.session_state.get("_csv_url_count") or st.session_state.get("csv_config", {}).get("url_count", 0) or 0
    default_urls = num_urls_csv if num_urls_csv > 0 else 100
    est_urls = st.number_input("Number of URLs", min_value=1, max_value=100000, value=default_urls, key="est_urls", help="Auto-filled from CSV if uploaded")
    # Scraper settings - derive avg scraped content per site
    max_chars_est = st.session_state.get("max_chars", 10000)
    depth_est = max(1, int(st.session_state.get("depth", 3) or 3))
    retries_est = max(0, int(st.session_state.get("retries", 2) or 2))
    max_chars_est = max(100, min(int(max_chars_est or 10000), 50000))
    base_chars = 4000 + (depth_est - 1) * 3500
    avg_chars_per_site = min(max_chars_est, max(100, int(base_chars)))
    # Error factor: ~5-12% scrape/AI failures (invalid URLs, timeouts, rate limits)
    success_rate = 0.90 - (retries_est * 0.01)
    effective_leads = max(1, int(est_urls * success_rate))
    # Per-call overhead: system msg + template structure
    base_overhead = 150
    ai_prompt_est = st.session_state.get("ai_prompt", "") or ""
    email_prompt_est = st.session_state.get("email_copy_prompt", "") or ""
    ai_enabled_est = st.session_state.get("ai_enabled_checkbox", False)
    email_enabled_est = st.session_state.get("email_copy_enabled_checkbox", False)
    ai_model_est = st.session_state.get("ai_model", "gpt-4o-mini")
    email_model_est = st.session_state.get("email_copy_model", "gpt-4o-mini")
    total_input = 0
    total_output = 0
    costs = []
    per_lead_input = 0
    per_lead_output = 0
    if ai_enabled_est:
        prompt_len = len(ai_prompt_est) + 400  # template + url/company placeholders
        prompt_tokens = chars_to_tokens(prompt_len) + base_overhead
        content_tokens = chars_to_tokens(avg_chars_per_site)
        inp_per = prompt_tokens + content_tokens
        out_per = chars_to_tokens(3500)  # ~SUMMARY + ~15 facts + ~8 hypotheses
        total_input += inp_per * effective_leads
        total_output += out_per * effective_leads
        per_lead_input += inp_per
        per_lead_output += out_per
        in_p, out_p = get_model_pricing(ai_model_est)
        cost = (inp_per * effective_leads / 1e6 * in_p) + (out_per * effective_leads / 1e6 * out_p)
        costs.append(("Company summary", ai_model_est, cost))
    if email_enabled_est:
        prompt_len = len(email_prompt_est) + 250
        prompt_tokens = chars_to_tokens(prompt_len) + base_overhead
        content_tokens = chars_to_tokens(avg_chars_per_site)
        inp_per = prompt_tokens + content_tokens
        out_per = chars_to_tokens(450)  # ~150 words
        total_input += inp_per * effective_leads
        total_output += out_per * effective_leads
        per_lead_input += inp_per
        per_lead_output += out_per
        in_p, out_p = get_model_pricing(email_model_est)
        cost = (inp_per * effective_leads / 1e6 * in_p) + (out_per * effective_leads / 1e6 * out_p)
        costs.append(("Email copy", email_model_est, cost))
    if total_input == 0 and total_output == 0:
        st.info("Enable Step 3 (Company Summary) or Step 4 (Email Copy) to see estimates.")
    else:
        st.markdown("**Totals**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Input tokens (total)", f"{total_input:,}")
        with c2:
            st.metric("Output tokens (total)", f"{total_output:,}")
        with c3:
            st.metric("Total tokens", f"{total_input + total_output:,}")
        st.markdown(f"**Per lead** (effective leads: ~{effective_leads:,}, ~{success_rate * 100:.0f}% success rate)")
        c4, c5 = st.columns(2)
        with c4:
            st.metric("Input tokens per lead", f"{per_lead_input:,}")
        with c5:
            st.metric("Output tokens per lead", f"{per_lead_output:,}")
        st.markdown("**Estimated cost**")
        for name, model, c in costs:
            st.markdown(f"- {name} ({model}): ~${c:.3f}")
        if costs:
            st.markdown(f"- **Total: ~${sum(x[2] for x in costs):.3f}**")
        st.caption("Based on max_chars={:,}, depth={}, retries={}. ~{:.0f}% of URLs assumed to succeed.".format(max_chars_est, depth_est, retries_est, success_rate * 100))

# Test run option
st.markdown("---")
test_col1, test_col2 = st.columns([1, 2])
with test_col1:
    test_rows = st.number_input("Test with X rows", min_value=1, max_value=20, value=5, help="Preview scraping + AI on first X URLs before full run", key="test_rows_input")
with test_col2:
    st.caption("Run a quick test to see scraped content and AI summary in the browser (no files created)")
    test_clicked = st.button("🧪 Test run", key="test_run_btn", help=f"Test scraping + AI on first {test_rows} row(s)")

# Main action button with better styling
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h2 style="color: #1f2937; margin-bottom: 1rem;">🚀 Ready to Start</h2>
    <p style="color: #6b7280; font-size: 1rem; margin-bottom: 1.5rem;">
        Make sure you've uploaded your CSV and configured settings above.<br>
        Then click the button below to start scraping!
    </p>
</div>
""", unsafe_allow_html=True)

# Get variables from tabs - Use session_state (proper Streamlit way)
keywords = st.session_state.get('keywords', [])
concurrency = st.session_state.get('concurrency', 20)
if st.session_state.get('force_max_workers', False):
    concurrency = _get_concurrency_max()
retries = st.session_state.get('retries', 2)
depth = st.session_state.get('depth', 3)
timeout = st.session_state.get('timeout', 30)
max_chars = st.session_state.get('max_chars', 10000)
rows_per_file = st.session_state.get('rows_per_file', 2000)
user_agent = st.session_state.get('user_agent', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
run_name = st.session_state.get('run_name', "")
# Auto-detect: no user checkboxes - app adapts to system and run size
low_resource = _is_low_resource_default()
# Use checkbox key as source of truth (widget state); fallback to our sync key
ai_enabled = st.session_state.get("ai_enabled_checkbox", st.session_state.get('ai_enabled', False))
if not ai_enabled:
    st.session_state['ai_enabled'] = False  # Ensure sync when disabled
ai_provider = st.session_state.get('ai_provider', None) if ai_enabled else None
# Get API key - check provider-specific key first, then generic
if ai_enabled and ai_provider:
    api_key_key = f"{ai_provider.lower()}_api_key"
    ai_api_key = st.session_state.get(api_key_key) or st.session_state.get('ai_api_key', None)
else:
    ai_api_key = None
ai_model = st.session_state.get('ai_model', None) if ai_enabled else None
ai_prompt = st.session_state.get('ai_prompt', None) if ai_enabled else None

email_copy_enabled_ui = st.session_state.get("email_copy_enabled_checkbox", st.session_state.get('email_copy_enabled', False))
if not email_copy_enabled_ui:
    st.session_state['email_copy_enabled'] = False
email_copy_provider_ui = st.session_state.get('email_copy_provider', None) if email_copy_enabled_ui else None
if email_copy_enabled_ui and email_copy_provider_ui:
    ec_key_name = f"{email_copy_provider_ui.lower()}_api_key"
    email_copy_api_key = st.session_state.get(ec_key_name) or st.session_state.get('email_copy_api_key', None)
else:
    email_copy_api_key = None
email_copy_model_ui = st.session_state.get('email_copy_model', None) if email_copy_enabled_ui else None
email_copy_prompt_ui = st.session_state.get('email_copy_prompt', None) if email_copy_enabled_ui else None

# -------- TEST RUN --------
def _run_test_in_thread():
    """Run test in a background thread to avoid blocking the UI."""
    import traceback
    params = st.session_state.get("_test_params", {})
    try:
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        results = asyncio.run(run_test_preview(
            params["urls"], params["test_rows"], params["retries"], params["timeout"],
            params["depth"], params["keywords"], params["max_chars"], params["user_agent"],
            params["ai_enabled"], params["ai_api_key"], params["ai_provider"],
            params["ai_model"], params["ai_prompt"],
            params["email_copy_enabled"], params["email_copy_api_key"], params["email_copy_provider"],
            params["email_copy_model"], params["email_copy_prompt"],
            params["lead_data_map"], log_callback=params.get("log_callback")))
        st.session_state["_test_results"] = results
        st.session_state["_test_logs"] = params.get("_logs", [])
    except Exception as e:
        st.session_state["_test_results"] = []
        st.session_state["_test_error"] = str(e)
        st.session_state["_test_tb"] = traceback.format_exc()
        st.session_state["_test_logs"] = params.get("_logs", []) if params else []
    finally:
        st.session_state["_test_running"] = False

if uploaded_file and test_clicked:
    csv_config = st.session_state.get('csv_config', {}) or {}
    if not csv_config:
        st.error("❌ Please configure your CSV above (select URL column) first.")
    else:
        uploaded_file.seek(0)
        has_headers = csv_config.get('has_headers', True)
        url_col = csv_config.get('url_column')
        lead_cols = csv_config.get('lead_data_columns', {}) or {}
        if not url_col:
            st.error("❌ URL column not configured. Select URL column in Step 1.")
        else:
            try:
                if has_headers:
                    df_in = pd.read_csv(uploaded_file)
                else:
                    df_in = pd.read_csv(uploaded_file, header=None)
            except Exception as e:
                st.error(f"❌ Failed to read CSV: {e}")
                df_in = pd.DataFrame()
            if df_in.empty:
                st.error("❌ CSV is empty or unreadable.")
            else:
                try:
                    url_col_idx = list(df_in.columns).index(url_col) if has_headers else int(str(url_col).replace("Column ", "")) - 1
                except (ValueError, TypeError):
                    st.error("❌ Invalid URL column. Reconfigure CSV in Step 1.")
                    url_col_idx = 0
                url_series = df_in.iloc[:, url_col_idx].fillna("").astype(str)
                url_list_with_idx = [(u.strip(), i) for i, u in enumerate(url_series) if u and str(u).strip()]
                urls = [u for u, _ in url_list_with_idx]
                lead_data_map = {}
                for idx, (url, orig_row) in enumerate(url_list_with_idx):
                    lead_data = {'url': url}
                    default_company = url.replace('https://', '').replace('http://', '').split('/')[0] if url else ''
                    for key, col_name in (lead_cols or {}).items():
                        try:
                            ci = list(df_in.columns).index(col_name) if has_headers else int(str(col_name).replace("Column ", "")) - 1
                            val = df_in.iloc[orig_row, ci] if orig_row < len(df_in) else None
                            v = "" if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val).strip()
                            lead_data[key] = v
                        except (ValueError, TypeError, IndexError):
                            lead_data[key] = ""
                    if not lead_data.get('company_name'):
                        lead_data['company_name'] = default_company
                    lead_data_map[idx] = lead_data

                if not urls:
                    st.error("❌ No valid URLs found in CSV. Check your URL column.")
                else:
                    ai_enabled_test = st.session_state.get("ai_enabled_checkbox", False)
                    ai_provider_test = st.session_state.get('ai_provider', None) if ai_enabled_test else None
                    ai_api_key_test = st.session_state.get(f"{ai_provider_test.lower()}_api_key") or st.session_state.get('ai_api_key') if (ai_enabled_test and ai_provider_test) else None
                    ai_model_test = st.session_state.get('ai_model') if ai_enabled_test else None
                    ai_prompt_test = st.session_state.get('ai_prompt') or ""
                    email_copy_enabled_test = st.session_state.get("email_copy_enabled_checkbox", False)
                    email_copy_provider_test = st.session_state.get('email_copy_provider', None) if email_copy_enabled_test else None
                    email_copy_api_key_test = st.session_state.get(f"{email_copy_provider_test.lower()}_api_key") or st.session_state.get('email_copy_api_key') if (email_copy_enabled_test and email_copy_provider_test) else None
                    email_copy_model_test = st.session_state.get('email_copy_model') if email_copy_enabled_test else None
                    email_copy_prompt_test = st.session_state.get('email_copy_prompt') or ""

                    test_logs = []
                    def test_log(msg):
                        test_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

                    _test_timeout = timeout
                    _test_retries = min(retries + 2, 8)
                    st.session_state["_test_params"] = {
                        "urls": urls, "test_rows": test_rows, "retries": _test_retries, "timeout": _test_timeout,
                        "depth": depth, "keywords": keywords, "max_chars": max_chars, "user_agent": user_agent,
                        "ai_enabled": ai_enabled_test, "ai_api_key": ai_api_key_test, "ai_provider": ai_provider_test,
                        "ai_model": ai_model_test, "ai_prompt": ai_prompt_test,
                        "email_copy_enabled": email_copy_enabled_test, "email_copy_api_key": email_copy_api_key_test,
                        "email_copy_provider": email_copy_provider_test, "email_copy_model": email_copy_model_test,
                        "email_copy_prompt": email_copy_prompt_test,
                        "lead_data_map": lead_data_map, "log_callback": test_log, "_logs": test_logs
                    }
                    st.session_state["_test_running"] = True
                    st.session_state["_test_error"] = None
                    st.session_state["_test_results"] = None  # Clear previous
                    t = threading.Thread(target=_run_test_in_thread, daemon=True)
                    t.start()
                    st.rerun()

if st.session_state.get("_test_running"):
    with st.spinner("Running test... (scraping + AI may take 1–3 min, please wait)"):
        time.sleep(1)
        st.rerun()

if st.session_state.get("_test_results") is not None and not st.session_state.get("_test_running"):
    results = st.session_state["_test_results"]
    test_logs = st.session_state.get("_test_logs", [])
    test_error = st.session_state.pop("_test_error", None)
    test_tb = st.session_state.pop("_test_tb", None)
    if test_error:
        st.error(f"❌ Test failed: {test_error}")
        if test_tb:
            st.code(test_tb, language=None)
    st.session_state["_test_results"] = None

    st.markdown("### 🧪 Test results")
    if test_logs:
        with st.expander("Logs", expanded=False):
            st.code("\n".join(test_logs), language=None)
    email_copy_enabled_test = st.session_state.get("email_copy_enabled_checkbox", False)
    for i, row in enumerate(results):
        url = row[0]
        scraped_text = row[1] if len(row) >= 2 else ""
        ai_summary = row[2] if len(row) >= 3 else ""
        email_copy_val = row[3] if len(row) >= 4 else ""
        with st.expander(f"**{i+1}. {url[:60]}{'…' if len(url) > 60 else ''}**", expanded=True):
            st.markdown("#### Scraped content")
            scraped_display = scraped_text[:8000] + ("…" if len(scraped_text) > 8000 else "") if scraped_text else "(empty)"
            st.text_area("", value=scraped_display, height=200, key=f"test_scraped_{i}", disabled=True, label_visibility="collapsed")
            if ai_summary:
                st.markdown("#### Company summary")
                formatted = ai_summary.replace("===SUMMARY===", "\n### Summary\n").replace("===FACTS===", "\n### Facts\n").replace("===HYPOTHESES===", "\n### Hypotheses\n")
                st.markdown(formatted)
            else:
                st.caption("_No company summary (Step 3 disabled or error)_")
            if email_copy_val:
                st.markdown("#### Email copy")
                st.markdown(email_copy_val)
            elif email_copy_enabled_test:
                st.caption("_No email copy (error or insufficient content)_")
    if not results:
        st.info("No results. Check logs above.")
    # Clear results when user starts a new test (handled by test_clicked block)

# -------- SCRAPE BUTTON --------
if uploaded_file and st.button("🚀 Start Scraping", use_container_width=True):
    # Get CSV configuration from session state
    csv_config = st.session_state.get('csv_config', {})
    
    if not csv_config:
        st.error("❌ Please configure your CSV file above (select columns)")
        st.stop()
    
    # Read CSV based on configuration
    has_headers = csv_config.get('has_headers', True)
    url_col = csv_config.get('url_column')
    lead_cols = csv_config.get('lead_data_columns', {}) or {}
    if url_col is None or (isinstance(url_col, str) and not url_col.strip()):
        st.error("❌ URL column not configured. Please select a URL column in Step 1.")
        st.stop()
    # Reset file pointer
    uploaded_file.seek(0)
    try:
        if has_headers:
            df_in = pd.read_csv(uploaded_file)
        else:
            df_in = pd.read_csv(uploaded_file, header=None)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()
    if df_in.empty or len(df_in) == 0:
        st.error("❌ CSV file is empty.")
        st.stop()
    # Get URL column index
    try:
        if has_headers:
            if url_col not in df_in.columns:
                st.error(f"❌ URL column '{url_col}' not found in CSV. Columns: {list(df_in.columns)}")
                st.stop()
            url_col_idx = list(df_in.columns).index(url_col)
        else:
            url_col_idx = int(str(url_col).replace("Column ", "").strip()) - 1
            if url_col_idx < 0 or url_col_idx >= len(df_in.columns):
                st.error(f"❌ Invalid column index. Select Column 1 to {len(df_in.columns)}.")
                st.stop()
    except (ValueError, TypeError) as e:
        st.error(f"❌ Invalid URL column selection: {e}")
        st.stop()
    
    # Extract URLs with original row indices (for lead columns)
    url_series = df_in.iloc[:, url_col_idx].fillna("").astype(str)
    url_list_with_idx = [(u.strip(), i) for i, u in enumerate(url_series) if u and str(u).strip()]
    urls = [u for u, _ in url_list_with_idx]
    total = len(urls)
    if total == 0:
        st.error("❌ No valid URLs found in the selected column. Check your CSV and URL column.")
        st.stop()
    # Prepare lead data mapping (filtered index -> lead data dict)
    lead_data_map = {}
    for idx, (url, orig_row) in enumerate(url_list_with_idx):
        lead_data = {'url': url}
        default_company = url.replace('https://', '').replace('http://', '').split('/')[0] if url else ''
        for key, col_name in (lead_cols or {}).items():
            try:
                ci = list(df_in.columns).index(col_name) if has_headers else int(str(col_name).replace("Column ", "")) - 1
                val = df_in.iloc[orig_row, ci] if orig_row < len(df_in) else None
                v = "" if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val).strip()
                lead_data[key] = v
            except (ValueError, TypeError, IndexError):
                lead_data[key] = ""
        if not lead_data.get('company_name'):
            lead_data['company_name'] = default_company
        lead_data_map[idx] = lead_data
    
    # Store lead data map in session state
    st.session_state['lead_data_map'] = lead_data_map
    
    # Show warning and info for large files
    if total > 10000:
        st.warning(f"⚠️ **Large dataset detected:** {total:,} URLs. This may take a while and generate large files.")
        with st.expander("💡 Tips for Large Datasets", expanded=True):
            st.markdown(f"""
            **Your dataset:** {total:,} URLs
            
            **Recommendations:**
            - ✅ **Max chars per site:** Use 20,000-50,000 to keep files manageable
            - ✅ **Rows per file:** Use 2,000-5,000 for easier handling
            - ✅ **Concurrency:** Start with 20-30, increase if stable
            - ✅ **Be patient:** Large datasets can take hours
            
            **Estimated output size:**
            - With 50k chars/site: ~{total * 50000 / (1024*1024):.0f} MB of text data
            - Will be split into multiple files for better performance
            - Excel files may be large - CSV recommended for very large datasets

            **Crash recovery:** Progress is saved automatically every ~10 seconds. If the app crashes,
            scroll to the top → "Resume or Download Partial Results" to download what you have, or
            re-upload your CSV and click Start to resume automatically (no run name needed).
            """)
    elif total > 5000:
        st.info(f"ℹ️ Processing {total:,} URLs. This may take some time. Files will be saved in chunks for better performance.")
    
    # Show warning for large files
    if total > 10000:
        st.warning(f"⚠️ **Large dataset detected:** {total:,} URLs. This may take a while and generate large files. Consider:")
        st.info("""
        - **Reduce max_chars per site** to keep file sizes manageable
        - **Increase rows_per_file** to reduce number of output files
        - **Monitor progress** - the app will process in chunks
        - **Be patient** - large datasets can take hours to complete
        """)
    elif total > 5000:
        st.info(f"ℹ️ Processing {total:,} URLs. This may take some time. Files will be saved in chunks for better performance.")

    # AUTO-RESUME: Find existing run with same URLs before creating new folder
    output_dir = None
    run_folder = None
    url_set = set(urls)
    outputs_dir = "outputs"
    if os.path.isdir(outputs_dir):
        best_match = None  # (output_dir, completed_count)
        for fn in os.listdir(outputs_dir):
            run_path = os.path.join(outputs_dir, fn)
            if not os.path.isdir(run_path):
                continue
            ck = load_checkpoint(get_checkpoint_path(run_path))
            if not ck:
                continue
            stored = set(ck.get("urls", []))
            if stored != url_set:
                continue
            completed = len(ck.get("completed_urls", []))
            total_stored = len(ck.get("urls", []))
            if 0 < completed < total_stored:
                if best_match is None or completed > best_match[1]:
                    best_match = (run_path, completed)
        if best_match:
            output_dir = best_match[0]
            run_folder = os.path.basename(output_dir)
            # Reconcile checkpoint with actual file contents (checkpoint can overstate after crash)
            reconcile_checkpoint_with_files(output_dir)
            actual_rows, _ = get_actual_completed_from_files(output_dir)
            display_count = actual_rows if actual_rows > 0 else best_match[1]
            st.success(f"🔄 **Resuming:** Found existing run with {display_count:,}/{total:,} rows in files. Continuing from where you left off.")

    if output_dir is None:
        run_folder = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", run_folder)
        os.makedirs(output_dir, exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    eta_text = st.empty()
    
    # Real-time activity dashboard placeholders
    dashboard_placeholder = st.empty()
    
    # AI status callback (thread-safe)
    ai_status_messages = {}

    fun_messages = [
        "🔍 Scanning the web...",
        "🛠️ Sharpening the scrapers...",
        "🚀 Launching data rockets...",
        "📡 Tuning into websites...",
        "🧩 Piecing text together..."
    ]
    start_time = time.time()
    progress_state_ui = {"done": 0, "total": max(total, 1), "running": True}
    progress_samples = deque(maxlen=200)
    runtime_logs = deque(maxlen=2000)  # Keep enough for full tracebacks + run history
    
    import threading
    runtime_lock = threading.Lock()
    progress_lock = threading.Lock()
    scrape_status_lock = threading.Lock()
    ai_status_lock = threading.Lock()
    
    scrape_in_progress = {}  # url -> {status, message}
    scrape_recent = deque(maxlen=30)  # [{url, status, message}, ...]
    scrape_errors = deque(maxlen=20)  # [{url, message}, ...]

    def progress_cb(done_count, total_count):
        with progress_lock:
            progress_state_ui["done"] = done_count
            progress_state_ui["total"] = max(total_count, 1)
            progress_samples.append((time.time(), done_count))

    def scrape_status_callback(url: str, status: str, message: str):
        with scrape_status_lock:
            if status in ("scraping", "ai_summarizing", "email_copy"):
                scrape_in_progress[url] = {"status": status, "message": message}
            else:
                scrape_in_progress.pop(url, None)
                entry = {"url": url, "status": status, "message": message}
                scrape_recent.append(entry)
                if status == "error":
                    scrape_errors.append({"url": url, "message": message[:150]})

    def ai_status_callback(url, message):
        with ai_status_lock:
            ai_status_messages[url] = message

    scrape_error = [None]  # Mutable to capture exception from thread
    
    def log_cb(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with runtime_lock:
            runtime_logs.append(line)
    
    ai_enabled_for_run = st.session_state.get("ai_enabled_checkbox", False)
    if not ai_enabled_for_run:
        ai_api_key_run, ai_provider_run, ai_model_run, ai_prompt_run = None, None, None, None
    else:
        ai_api_key_run, ai_provider_run, ai_model_run, ai_prompt_run = ai_api_key, ai_provider, ai_model, ai_prompt

    email_copy_enabled_for_run = st.session_state.get("email_copy_enabled_checkbox", False)
    if not email_copy_enabled_for_run:
        email_copy_api_key_run = email_copy_provider_run = email_copy_model_run = email_copy_prompt_run = None
    else:
        email_copy_api_key_run = email_copy_api_key
        email_copy_provider_run = email_copy_provider_ui
        email_copy_model_run = email_copy_model_ui
        email_copy_prompt_run = email_copy_prompt_ui

    # Auto-resilience: extra retries for unreliable sites (no checkbox)
    effective_timeout = timeout
    effective_retries = min(retries + 2, 8)
    fast_mode = _should_use_fast_mode(total)

    def run_async_scraper():
        try:
            if os.name == "nt":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            progress_state_ui["running"] = True
            log_cb(f"🚀 Run started: urls={total:,}, workers={concurrency}, depth={depth}, retries={effective_retries}, timeout={effective_timeout}s")
            asyncio.run(
                run_scraper(urls, concurrency, effective_retries, effective_timeout, depth, keywords, max_chars,
                            user_agent, rows_per_file, output_dir, progress_cb,
                            ai_enabled_for_run, ai_api_key_run, ai_provider_run, ai_model_run, ai_prompt_run,
                            email_copy_enabled_for_run, email_copy_api_key_run, email_copy_provider_run,
                            email_copy_model_run, email_copy_prompt_run,
                            lead_data_map,
                            ai_status_callback if ai_enabled_for_run else None, scrape_status_callback,
                            run_folder=run_folder, fast_mode=fast_mode,
                            low_resource=low_resource, log_callback=log_cb))
            log_cb("✅ Run completed")
        except Exception as e:
            scrape_error[0] = e
            import traceback
            tb = traceback.format_exc()
            log_cb(f"❌ Run crashed: {type(e).__name__}: {e}")
            log_cb("--- Full traceback (send this for bug reports) ---")
            for line in tb.strip().split("\n"):
                log_cb(line)
            log_cb("--- End traceback ---")
            traceback.print_exc()
            # Persist crash logs to output dir so they survive page refresh
            try:
                with runtime_lock:
                    crash_log_content = "\n".join(runtime_logs)
                crash_log_path = os.path.join(output_dir, f"crash_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                with open(crash_log_path, "w", encoding="utf-8") as f:
                    f.write(crash_log_content)
                log_cb(f"📄 Crash log saved to: {crash_log_path}")
            except Exception:
                pass
        finally:
            progress_state_ui["running"] = False

    with st.spinner("Scraping — runs in-page. Progress is saved; if you refresh or lose connection, re-upload the CSV and click Start to resume."):
        lead_data_map = st.session_state.get('lead_data_map', None)
        with st.expander("📋 Detailed logs", expanded=False):
            logs_placeholder = st.empty()
        
        thread = threading.Thread(target=run_async_scraper, daemon=False)
        thread.start()
        while thread.is_alive():
            with progress_lock:
                d, t = progress_state_ui["done"], progress_state_ui["total"]
            pct = d / max(t, 1)
            now_ts = time.time()
            elapsed = now_ts - start_time
            with progress_lock:
                if not progress_samples or progress_samples[-1][1] != d:
                    progress_samples.append((now_ts, d))
                window = [p for p in progress_samples if now_ts - p[0] <= 120]
            # Throughput from trailing window (more stable ETA than global average)
            if len(window) >= 2:
                dt = max(window[-1][0] - window[0][0], 1e-6)
                dd = max(window[-1][1] - window[0][1], 0)
                rate = dd / dt
            else:
                rate = 0.0
            remaining = ((t - d) / rate) if rate > 1e-6 else float("inf")
            idle_for = int(now_ts - (progress_samples[-1][0] if progress_samples else start_time))
            progress_bar.progress(min(pct, 1.0))
            idx = (d - 1) % len(fun_messages) if d > 0 else 0
            if remaining == float("inf"):
                status_text.text(f"{fun_messages[idx]} ({d}/{t}) — ETA: calculating...")
            else:
                status_text.text(f"{fun_messages[idx]} ({d}/{t}) — ETA: {int(remaining // 60)}m {int(remaining % 60)}s")
            if idle_for >= 30:
                eta_text.warning(f"No progress for {idle_for}s. App will pause and wait for recovery (slow network/system); do not refresh.")
            else:
                eta_text.text(f"⏱️ Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s")
            with runtime_lock:
                logs_text = "\n".join(runtime_logs)
            logs_placeholder.code(logs_text or "(no logs yet)", language=None)
            
            # Real-time activity dashboard (enhanced data & visuals)
            with dashboard_placeholder.container():
                with scrape_status_lock:
                    in_progress = list(scrape_in_progress.items())
                    recent = list(scrape_recent)
                    errors = list(scrape_errors)
                n_scraping = sum(1 for _, d in in_progress if d["status"] == "scraping")
                n_ai = sum(1 for _, d in in_progress if d["status"] == "ai_summarizing")
                n_email = sum(1 for _, d in in_progress if d["status"] == "email_copy")
                rate_per_min = rate * 60 if rate > 0 else 0
                pct_done = (d / max(t, 1)) * 100
                st.markdown("### 📊 Live activity")
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Progress", f"{d:,} / {t:,}", f"{pct_done:.1f}%")
                with m2:
                    st.metric("Rate", f"{rate_per_min:.1f}/min", "URLs" if rate_per_min else "—")
                with m3:
                    st.metric("Elapsed", f"{int(elapsed // 60)}m {int(elapsed % 60)}s", "")
                with m4:
                    st.metric("Errors", len(errors), f"of {d}" if d else "—")
                with m5:
                    phase_str = f"🔍 {n_scraping}  🤖 {n_ai}  ✉️ {n_email}" if in_progress else "Idle"
                    st.metric("Active", phase_str, "scrape | AI | email")
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("#### ⏳ In progress")
                    if in_progress:
                        for url, data in in_progress[-10:]:
                            short_url = escape((url[:42] + "…") if len(url) > 42 else url)
                            status = data.get("status", "scraping")
                            msg = escape(str(data.get("message", ""))[:80])
                            icon = "🔍" if status == "scraping" else ("🤖" if status == "ai_summarizing" else "✉️")
                            badge = "scraping" if status == "scraping" else ("AI" if status == "ai_summarizing" else "email")
                            color = "#3b82f6" if status == "scraping" else ("#8b5cf6" if status == "ai_summarizing" else "#22c55e")
                            st.markdown(f"""<div style="font-size:0.85rem; margin:0.3rem 0; padding:0.3rem; background:#f8fafc; border-radius:6px; border-left:3px solid {color};"><span style="font-weight:600;">{icon} {short_url}</span><br><span style="color:#64748b; font-size:0.8rem;">{msg}</span> <code style="background:#e2e8f0; padding:0.1rem 0.4rem; border-radius:4px; font-size:0.75rem;">{badge}</code></div>""", unsafe_allow_html=True)
                    else:
                        st.caption("_Waiting for workers..._")
                with col2:
                    st.markdown("#### ✅ Recently completed")
                    if recent:
                        for e in reversed(recent[-8:]):
                            u = e.get("url", "")
                            short_url = escape((u[:38] + "…") if len(u) > 38 else u)
                            if e.get("status") == "scraped":
                                msg = escape(str(e.get("message", ""))[:50])
                                st.markdown(f"<div style='font-size:0.85rem; margin:0.2rem 0; color:#16a34a;'>✓ {short_url}</div><div style='font-size:0.75rem; color:#64748b; margin-bottom:0.4rem;'>{msg}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='font-size:0.85rem; margin:0.2rem 0; color:#dc2626;'>✗ {short_url}</div>", unsafe_allow_html=True)
                    else:
                        st.caption("_None yet_")
                with col3:
                    st.markdown("#### ⚠️ Issues")
                    if errors:
                        for e in errors[-6:]:
                            u = e.get("url", "")
                            m = e.get("message", "")
                            short_url = escape((u[:35] + "…") if len(u) > 35 else u)
                            msg = escape((m[:70] + "…") if len(m) > 70 else m)
                            st.markdown(f"""<div style="font-size:0.85rem; margin:0.3rem 0; padding:0.3rem; background:#fef2f2; border-radius:6px; border-left:3px solid #dc2626;"><strong>{short_url}</strong><br><span style="font-size:0.75rem; color:#64748b;">{msg}</span></div>""", unsafe_allow_html=True)
                    else:
                        st.caption("_No issues_")
            
            time.sleep(0.8)
        thread.join()
        with runtime_lock:
            logs_text = "\n".join(runtime_logs) or "(no logs yet)"
        logs_placeholder.code(logs_text, language=None)
        
        # Always show download logs button so users can share for bug reports
        st.download_button(
            "📥 Download run logs",
            logs_text,
            file_name=f"scraper_logs_{run_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Download logs to share when reporting bugs or crashes",
            key="download_run_logs"
        )
        
        if scrape_error[0]:
            st.error(f"❌ Scraper error: {scrape_error[0]}")
            st.warning("💾 **Partial results may have been saved.** Scroll up to **'Resume or Download Partial Results'** at the top to download what you have or re-upload your CSV and click Start to resume automatically.")
            st.info("📋 **Found a bug?** Click **Download run logs** above and send the file when reporting the issue.")

    # Zip all parts at the end with custom name (CSV and Excel files)
    zip_name = f"{run_folder}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    csv_files = []
    excel_files = []
    
    # CRITICAL: Wait a moment for files to be fully written to disk
    import time
    time.sleep(1)
    
    # List files in output directory
    try:
        output_files = os.listdir(output_dir)
        st.info(f"📁 Found {len(output_files)} files in output directory: {', '.join(output_files[:5])}{'...' if len(output_files) > 5 else ''}")
    except Exception as e:
        st.error(f"❌ Error listing output directory: {e}")
        output_files = []
    
    # Create ZIP file
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in output_files:
                file_path = os.path.join(output_dir, f)
                if os.path.isfile(file_path):  # Only process files, not directories
                    if f.endswith(".csv") and f != zip_name and "combined" not in f.lower():
                        try:
                            zf.write(file_path, arcname=f)
                            csv_files.append(f)
                        except Exception as e:
                            st.warning(f"⚠️ Could not add {f} to ZIP: {e}")
                    elif f.endswith(".xlsx") and f != zip_name and "combined" not in f.lower():
                        try:
                            zf.write(file_path, arcname=f)
                            excel_files.append(f)
                        except Exception as e:
                            st.warning(f"⚠️ Could not add {f} to ZIP: {e}")
    except Exception as e:
        st.error(f"❌ Error creating ZIP file: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
    
    # CRITICAL: Verify files were actually created before proceeding
    if not csv_files and not excel_files:
        st.error("⚠️ **WARNING: No CSV or Excel files were generated!**")
        st.error("This might indicate that the writer coroutine failed or no data was processed.")
        st.info("Check the output directory for any error messages or partial files.")
        # Try to list what's actually in the directory
        try:
            actual_files = os.listdir(output_dir)
            if actual_files:
                st.info(f"Files found in output directory: {', '.join(actual_files)}")
            else:
                st.warning("Output directory is empty!")
        except Exception as e:
            st.error(f"Could not list output directory: {e}")
    else:
        st.success(f"✅ Scraping finished! Processed {total:,} website(s). Generated {len(csv_files)} CSV file(s) and {len(excel_files)} Excel file(s).")
    
    # Store file paths and data in session_state for persistence across reruns
    st.session_state['scraping_complete'] = True
    st.session_state['output_dir'] = output_dir
    st.session_state['run_folder'] = run_folder
    st.session_state['zip_path'] = zip_path
    st.session_state['zip_name'] = zip_name
    st.session_state['csv_files'] = csv_files
    st.session_state['excel_files'] = excel_files
    st.session_state['total_processed'] = total
    
    # Read ZIP file data and store in session_state
    try:
        if os.path.exists(zip_path):
            with open(zip_path, 'rb') as f:
                st.session_state['zip_data'] = f.read()
        else:
            st.warning(f"⚠️ ZIP file not found at {zip_path}")
            st.session_state['zip_data'] = None
    except Exception as e:
        st.error(f"❌ Error reading ZIP file: {e}")
        st.session_state['zip_data'] = None
    
    # Show accuracy info for max_chars
    if total > 0:
        st.info(f"💡 **Accuracy:** Max characters limit ({max_chars:,} chars) was accurately enforced per website. Each website's content was limited to this exact amount.")
    
    # Download section
    st.header("📥 Download Results")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        st.subheader("📦 All Files (ZIP)")
        zip_data = st.session_state.get('zip_data')
        if zip_data:
            st.download_button(
                label="⬇️ Download ZIP Archive",
                data=zip_data,
                file_name=zip_name,
                mime="application/zip",
                help="Download all CSV and Excel files in a ZIP archive",
                key="download_zip"
            )
        else:
            # Fallback: read from file if not in session_state
            try:
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                    st.session_state['zip_data'] = zip_data
                    st.download_button(
                        label="⬇️ Download ZIP Archive",
                        data=zip_data,
                        file_name=zip_name,
                        mime="application/zip",
                        help="Download all CSV and Excel files in a ZIP archive",
                        key="download_zip_fallback"
                    )
            except Exception as e:
                st.error(f"Could not read ZIP file: {e}")
        st.caption(f"Contains {len(csv_files)} CSV + {len(excel_files)} Excel files")
    
    with col_dl2:
        st.subheader("📊 Excel Files")
        if excel_files:
            # Create a combined Excel file
            try:
                # CRITICAL: ALWAYS read from CSV files first (source of truth), NOT Excel files
                # Excel files may be corrupted or have wrong data, CSV files are the source of truth
                csv_all_data = []
                if csv_files:
                    for csv_file in sorted(csv_files):
                        # CRITICAL: Skip combined CSV files - they should not be re-read
                        if "combined" in csv_file.lower():
                            continue
                        
                        csv_path = os.path.join(output_dir, csv_file)
                        try:
                            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                                rows_data = []
                                with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                                    csv_reader = csv.reader(f, quoting=csv.QUOTE_ALL, doublequote=True)
                                    header = next(csv_reader, None)
                                    if not header:
                                        continue
                                    
                                    # Normalize header
                                    header_normalized = [str(h).strip().strip('"').strip("'") for h in header]
                                    
                                    # Map header to column indices - support 3 or 4 cols (EmailCopy)
                                    col_indices = {}
                                    for idx, h in enumerate(header_normalized):
                                        h_lower = h.lower().strip()
                                        if h_lower == 'website' or h_lower == 'url':
                                            col_indices['Website'] = idx
                                        elif h_lower == 'scrapedtext' or (h_lower.startswith('scraped') and 'text' in h_lower):
                                            col_indices['ScrapedText'] = idx
                                        elif h_lower == 'companysummary' or (h_lower.startswith('company') and 'summary' in h_lower):
                                            col_indices['CompanySummary'] = idx
                                        elif 'email' in h_lower and 'copy' in h_lower:
                                            col_indices['EmailCopy'] = idx
                                    
                                    if 'Website' not in col_indices and len(header_normalized) >= 1:
                                        col_indices['Website'] = 0
                                    if 'ScrapedText' not in col_indices and len(header_normalized) >= 2:
                                        col_indices['ScrapedText'] = 1
                                    if 'CompanySummary' not in col_indices and len(header_normalized) >= 3:
                                        col_indices['CompanySummary'] = 2
                                    if 'EmailCopy' not in col_indices and len(header_normalized) >= 4:
                                        col_indices['EmailCopy'] = 3
                                    
                                    if len(col_indices) < 3:
                                        st.warning(f"⚠️ CSV file {csv_file} missing columns. Found: {header_normalized}")
                                        continue
                                    
                                    # Read rows - CRITICAL: csv.reader already handles unquoting
                                    for row in csv_reader:
                                        # Skip empty rows
                                        if len(row) == 0:
                                            continue
                                        
                                        # CRITICAL: Ensure row has exactly 3 columns, pad if needed
                                        while len(row) < 3:
                                            row.append("")
                                        # Truncate to 3 columns if somehow more
                                        if len(row) > 3:
                                            row = row[:3]
                                        
                                        website_idx = col_indices.get('Website', 0)
                                        scraped_idx = col_indices.get('ScrapedText', 1)
                                        summary_idx = col_indices.get('CompanySummary', 2)
                                        email_idx = col_indices.get('EmailCopy')
                                        
                                        if website_idx >= len(row) or scraped_idx >= len(row) or summary_idx >= len(row):
                                            continue
                                        
                                        website = str(row[website_idx]).strip()
                                        scraped_text = str(row[scraped_idx]).strip()
                                        company_summary = str(row[summary_idx]).strip()
                                        email_copy = str(row[email_idx]).strip() if email_idx is not None and email_idx < len(row) else ""
                                        
                                        row_dict = {'Website': website, 'ScrapedText': scraped_text, 'CompanySummary': company_summary}
                                        if email_idx is not None:
                                            row_dict['EmailCopy'] = email_copy
                                        rows_data.append(row_dict)
                                
                                if rows_data:
                                    use_cols = EXCEL_COLS_4 if 'EmailCopy' in rows_data[0] else EXCEL_COLS_3
                                    df_csv = pd.DataFrame(rows_data, columns=use_cols)
                                    csv_all_data.append(df_csv)
                        except Exception as e:
                            st.warning(f"⚠️ Error reading CSV {csv_file} for Excel: {e}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
                            continue
                
                # Fallback: Read from Excel files ONLY if CSV reading completely failed
                all_data = []
                if not csv_all_data:
                    st.warning("⚠️ No CSV data found, falling back to Excel files (may have wrong data)")
                    for excel_file in sorted(excel_files):
                        excel_path = os.path.join(output_dir, excel_file)
                        try:
                            # Check if file exists and has content
                            if not os.path.exists(excel_path):
                                st.warning(f"⚠️ Excel file not found: {excel_file}")
                                continue
                            
                            if os.path.getsize(excel_path) == 0:
                                st.warning(f"⚠️ Excel file is empty: {excel_file}")
                                continue
                            
                            # Read Excel file with explicit column handling
                            df_part = pd.read_excel(excel_path, engine='openpyxl', header=0)
                            
                            # Check if DataFrame is empty
                            if df_part.empty:
                                st.warning(f"⚠️ Excel file {excel_file} contains no data")
                                continue
                            
                            # CRITICAL: Normalize column names - handle corrupted/merged headers
                            # Clean column names: strip whitespace, handle case-insensitive matching
                            df_part.columns = [str(col).strip() for col in df_part.columns]
                            
                            # Map columns to standard names (case-insensitive, handle variations)
                            column_mapping = {}
                            for col in df_part.columns:
                                col_lower = col.lower().strip()
                                if 'website' in col_lower or col_lower == 'url':
                                    column_mapping[col] = 'Website'
                                elif 'scraped' in col_lower and 'text' in col_lower:
                                    column_mapping[col] = 'ScrapedText'
                                elif 'company' in col_lower and 'summary' in col_lower:
                                    column_mapping[col] = 'CompanySummary'
                                elif 'summary' in col_lower and 'company' not in col_lower:
                                    column_mapping[col] = 'CompanySummary'
                                elif 'email' in col_lower and 'copy' in col_lower:
                                    column_mapping[col] = 'EmailCopy'
                            
                            # Rename columns
                            df_part = df_part.rename(columns=column_mapping)
                            
                            # If we still don't have the right columns, try to infer from position
                            # This handles cases where headers might be completely corrupted
                            if 'Website' not in df_part.columns and len(df_part.columns) >= 1:
                                df_part.columns.values[0] = 'Website'
                            if 'ScrapedText' not in df_part.columns and len(df_part.columns) >= 2:
                                df_part.columns.values[1] = 'ScrapedText'
                            if 'CompanySummary' not in df_part.columns and len(df_part.columns) >= 3:
                                df_part.columns.values[2] = 'CompanySummary'
                            
                            # Ensure all required columns exist
                            if "Website" not in df_part.columns:
                                st.warning(f"⚠️ Excel file {excel_file} missing Website column. Found columns: {list(df_part.columns)}")
                                continue
                            
                            if "ScrapedText" not in df_part.columns:
                                # If ScrapedText is missing, create it from the second column or empty
                                if len(df_part.columns) >= 2:
                                    second_col = df_part.columns[1]
                                    df_part = df_part.rename(columns={second_col: 'ScrapedText'})
                                else:
                                    df_part['ScrapedText'] = ""
                            
                            if "CompanySummary" not in df_part.columns:
                                df_part["CompanySummary"] = ""
                            if "EmailCopy" not in df_part.columns:
                                df_part["EmailCopy"] = ""
                            standard_cols = [c for c in EXCEL_COLS_4 if c in df_part.columns] or EXCEL_COLS_3
                            df_part = df_part[standard_cols]
                            
                            # Convert all columns to string and clean
                            for col in df_part.columns:
                                df_part[col] = df_part[col].astype(str).replace('nan', '').replace('None', '')
                            
                            # Only add if DataFrame has rows
                            if len(df_part) > 0:
                                all_data.append(df_part)
                            else:
                                st.warning(f"⚠️ Excel file {excel_file} has no valid rows after processing")
                        except Exception as e:
                            st.warning(f"⚠️ Error reading Excel file {excel_file}: {e}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
                            continue
                
                if csv_all_data:
                    combined_df = pd.concat(csv_all_data, ignore_index=True)
                    for c in EXCEL_COLS_3:
                        if c not in combined_df.columns:
                            combined_df[c] = ""
                    if "EmailCopy" in combined_df.columns:
                        combined_df["EmailCopy"] = combined_df["EmailCopy"].fillna("")
                    else:
                        combined_df["EmailCopy"] = ""
                    combined_cols = EXCEL_COLS_4 if "EmailCopy" in combined_df.columns else EXCEL_COLS_3
                    combined_df = combined_df[combined_cols]
                elif all_data:
                    st.warning("⚠️ CSV reading failed, using Excel files (may have incomplete data)")
                    combined_df = pd.concat(all_data, ignore_index=True)
                    for c in EXCEL_COLS_3:
                        if c not in combined_df.columns:
                            combined_df[c] = ""
                    if "EmailCopy" not in combined_df.columns:
                        combined_df["EmailCopy"] = ""
                    combined_cols = EXCEL_COLS_4 if "EmailCopy" in combined_df.columns else EXCEL_COLS_3
                    combined_df = combined_df[combined_cols]
                else:
                    st.error("❌ No data available to combine")
                    raise ValueError("No data available to combine for Excel file")
                
                # Clean the combined DataFrame
                def clean_dataframe_for_excel(df):
                    """Clean all string columns in DataFrame for Excel"""
                    import re
                    for col in df.columns:
                        df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
                        df[col] = df[col].str.replace('\x00', '', regex=False)
                        df[col] = df[col].str.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', regex=True)
                        df[col] = df[col].str.strip()
                    return df
                combined_df = clean_dataframe_for_excel(combined_df.copy())
                
                from openpyxl import Workbook
                excel_buffer = BytesIO()
                wb = Workbook()
                ws = wb.active
                combined_cols = [c for c in EXCEL_COLS_4 if c in combined_df.columns] or EXCEL_COLS_3
                _write_excel_sheet(ws, combined_cols, combined_df)
                
                # CRITICAL: Save with explicit error handling
                try:
                    wb.save(excel_buffer)
                except Exception as save_error:
                    st.error(f"❌ Excel save failed: {save_error}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    raise save_error
                
                excel_buffer.seek(0)
                excel_data = excel_buffer.read()
                
                # Store in session_state for persistence
                st.session_state['combined_excel_data'] = excel_data
                st.session_state['combined_excel_filename'] = f"{run_folder}_combined.xlsx"
                st.session_state['combined_df'] = combined_df  # Store for CSV too
                
                st.download_button(
                    label="⬇️ Download Combined Excel",
                    data=excel_data,
                    file_name=f"{run_folder}_combined.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Download all data in a single Excel file",
                    key="download_excel"
                )
                st.caption(f"{len(combined_df)} total rows")
            except Exception as e:
                st.warning(f"Could not create combined Excel: {e}")
                import traceback
                st.error(f"Error details: {traceback.format_exc()}")
        else:
            st.info("No Excel files generated")
    
    with col_dl3:
        st.subheader("📄 CSV Files")
        if csv_files:
            # Create a combined CSV file
            try:
                all_data = []
                for csv_file in sorted(csv_files):
                    # CRITICAL: Skip combined CSV files - they should not be re-read
                    if "combined" in csv_file.lower():
                        continue
                    
                    csv_path = os.path.join(output_dir, csv_file)
                    try:
                        # Check if file exists and has content
                        if not os.path.exists(csv_path):
                            st.warning(f"⚠️ CSV file not found: {csv_file}")
                            continue
                        
                        if os.path.getsize(csv_path) == 0:
                            st.warning(f"⚠️ CSV file is empty: {csv_file}")
                            continue
                        
                        # CRITICAL: Read CSV using Python's csv.reader (not pandas) for perfect round-trip compatibility
                        # pandas read_csv can misinterpret CSV files even with QUOTE_ALL
                        rows_data = []
                        with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                            csv_reader = csv.reader(f, quoting=csv.QUOTE_ALL, doublequote=True)
                            header = next(csv_reader, None)
                            if not header:
                                st.warning(f"⚠️ CSV file {csv_file} has no header")
                                continue
                            
                            # Normalize header - handle any variations
                            header_normalized = []
                            for h in header:
                                h_clean = str(h).strip().strip('"').strip("'")
                                header_normalized.append(h_clean)
                            
                            # CRITICAL: Expect exactly 3 columns: Website, ScrapedText, CompanySummary
                            # If header doesn't match, log warning but try to map by position
                            if len(header_normalized) != 3:
                                st.warning(f"⚠️ CSV file {csv_file} has {len(header_normalized)} columns instead of 3: {header_normalized}")
                            
                            # Map header to standard columns - be more strict
                            col_indices = {}
                            for idx, h in enumerate(header_normalized):
                                h_lower = h.lower().strip()
                                if h_lower == 'website' or h_lower == 'url':
                                    col_indices['Website'] = idx
                                elif h_lower == 'scrapedtext' or (h_lower.startswith('scraped') and 'text' in h_lower):
                                    col_indices['ScrapedText'] = idx
                                elif h_lower == 'companysummary' or (h_lower.startswith('company') and 'summary' in h_lower):
                                    col_indices['CompanySummary'] = idx
                            
                            # If we couldn't map by name, use position (first 3 columns) as fallback
                            if 'Website' not in col_indices and len(header_normalized) >= 1:
                                col_indices['Website'] = 0
                            if 'ScrapedText' not in col_indices and len(header_normalized) >= 2:
                                col_indices['ScrapedText'] = 1
                            if 'CompanySummary' not in col_indices and len(header_normalized) >= 3:
                                col_indices['CompanySummary'] = 2
                            
                            # Verify we have all 3 columns mapped
                            if len(col_indices) != 3:
                                st.warning(f"⚠️ CSV file {csv_file} column mapping incomplete. Expected 3 columns, mapped {len(col_indices)}: {col_indices}")
                                continue
                            
                            # Read all rows
                            for row_num, row in enumerate(csv_reader, start=2):
                                # Skip empty rows
                                if len(row) == 0:
                                    continue
                                
                                # CRITICAL: Ensure row has at least 3 columns, pad if needed
                                while len(row) < 3:
                                    row.append("")
                                
                                # Extract values by column index - csv.reader already unquotes values
                                # CRITICAL: csv.reader with QUOTE_ALL automatically handles unquoting
                                website_idx = col_indices.get('Website', 0)
                                scraped_idx = col_indices.get('ScrapedText', 1)
                                summary_idx = col_indices.get('CompanySummary', 2)
                                
                                # Extract values - csv.reader has already unquoted them
                                website = str(row[website_idx]).strip() if website_idx < len(row) else ""
                                scraped_text = str(row[scraped_idx]).strip() if scraped_idx < len(row) else ""
                                company_summary = str(row[summary_idx]).strip() if summary_idx < len(row) else ""
                                
                                # CRITICAL: Don't strip quotes here - csv.reader already did that
                                # The values are already clean strings without quotes
                                # Any remaining quotes are part of the actual data content
                                
                                # Only add rows with valid URLs
                                if website:
                                    rows_data.append({
                                        'Website': website,
                                        'ScrapedText': scraped_text,
                                        'CompanySummary': company_summary
                                    })
                        
                        if not rows_data:
                            st.warning(f"⚠️ CSV file {csv_file} contains no valid data rows")
                            continue
                        
                        # Create DataFrame from clean data
                        df_part = pd.DataFrame(rows_data, columns=["Website", "ScrapedText", "CompanySummary"])
                        
                        # Only add if DataFrame has rows
                        if len(df_part) > 0:
                            all_data.append(df_part)
                        else:
                            st.warning(f"⚠️ CSV file {csv_file} has no valid rows after processing")
                    except Exception as e:
                        st.warning(f"⚠️ Error reading CSV file {csv_file}: {e}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
                        continue
                
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    
                    # Ensure final DataFrame has correct structure (always 3 columns)
                    if "CompanySummary" not in combined_df.columns:
                        combined_df["CompanySummary"] = ""
                    combined_df = combined_df[["Website", "ScrapedText", "CompanySummary"]]
                    
                    # PERFECT CSV CLEANING for combined CSV
                    def clean_dataframe_for_csv(df):
                        """Clean all string columns in DataFrame for perfect CSV formatting"""
                        import re
                        for col in df.columns:
                            df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
                            df[col] = df[col].str.replace('\x00', '', regex=False)
                            df[col] = df[col].str.replace(r'[\n\r\f\v\t]', ' ', regex=True)
                            df[col] = df[col].str.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', regex=True)
                            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                            df[col] = df[col].str.strip()
                            df[col] = df[col].replace('', '')
                        return df
                    combined_df = clean_dataframe_for_csv(combined_df.copy())
                    
                    # PERFECT CSV WRITING with manual writer for absolute control
                    # CRITICAL: Ensure DataFrame has exactly 3 columns in correct order
                    if "CompanySummary" not in combined_df.columns:
                        combined_df["CompanySummary"] = ""
                    combined_df = combined_df[["Website", "ScrapedText", "CompanySummary"]]
                    
                    # CRITICAL: Write CSV directly to ensure Excel compatibility
                    # Use StringIO for text-based CSV writing, then encode to bytes
                    csv_buffer = StringIO()
                    # Use QUOTE_ALL and explicit settings for Excel compatibility
                    csv_writer = csv.writer(
                        csv_buffer, 
                        quoting=csv.QUOTE_ALL,  # Quote all fields to handle commas/quotes
                        doublequote=True,  # Escape quotes by doubling them
                        lineterminator='\n',  # Unix line endings
                        quotechar='"'  # Use double quotes
                    )
                    
                    csv_cols = [c for c in EXCEL_COLS_4 if c in combined_df.columns] or EXCEL_COLS_3
                    csv_writer.writerow(csv_cols)
                    
                    # Clean function for CSV values - ensures Excel compatibility
                    def clean_csv_value_for_excel(val_str):
                        """Clean a single CSV value ensuring Excel compatibility"""
                        if not val_str:
                            return ""
                        import re
                        # Convert to string if not already
                        val_str = str(val_str)
                        # Remove null bytes (can break CSV parsing)
                        val_str = val_str.replace('\x00', '')
                        # CRITICAL: Replace newlines/carriage returns with space (Excel can't handle newlines in CSV fields)
                        val_str = val_str.replace('\n', ' ').replace('\r', ' ')
                        # Replace tabs with space
                        val_str = val_str.replace('\t', ' ')
                        # Remove other control characters (except space)
                        val_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', val_str)
                        # Collapse multiple spaces to single space
                        val_str = re.sub(r'\s+', ' ', val_str)
                        # Strip leading/trailing whitespace
                        val_str = val_str.strip()
                        # Note: csv.writer with QUOTE_ALL will automatically quote the entire field
                        # and escape any internal quotes by doubling them (Excel standard)
                        return val_str
                    
                    # Write rows - ensure exactly 3 values per row in correct order
                    # CRITICAL: Use iloc instead of iterrows() to prevent misalignment
                    for idx in range(len(combined_df)):
                        row = combined_df.iloc[idx]
                        
                        row_vals = [clean_csv_value_for_excel("" if pd.isna(row.get(c, "")) else str(row[c])) for c in csv_cols]
                        csv_writer.writerow(row_vals)
                    
                    csv_buffer.seek(0)
                    
                    # Get string content and encode to bytes with UTF-8 BOM for Excel compatibility
                    csv_data = csv_buffer.getvalue().encode('utf-8-sig')
                    
                    # CRITICAL: Validate the CSV by reading it back
                    try:
                        csv_buffer.seek(0)
                        test_reader = csv.reader(csv_buffer, quoting=csv.QUOTE_ALL)
                        test_rows = list(test_reader)
                        if len(test_rows) > 0:
                            expected_cols = len(csv_cols)
                            if list(test_rows[0]) != csv_cols:
                                st.warning(f"⚠️ CSV header validation failed. Expected {csv_cols}, got: {test_rows[0]}")
                            for i, test_row in enumerate(test_rows[1:], start=2):
                                if len(test_row) != expected_cols:
                                    st.warning(f"⚠️ CSV row {i} has {len(test_row)} columns instead of {expected_cols}")
                    except Exception as e:
                        st.warning(f"⚠️ CSV validation error: {e}")
                    
                    # Store for persistence
                    st.session_state['combined_csv_data'] = csv_data
                    st.session_state['combined_df'] = combined_df
                    st.session_state['combined_csv_filename'] = f"{run_folder}_combined.csv"
                    
                    st.download_button(
                        label="⬇️ Download Combined CSV",
                        data=csv_data,
                        file_name=f"{run_folder}_combined.csv",
                        mime="text/csv",
                        help="Download all data in a single CSV file (Excel/Google Sheets compatible)",
                        key="download_csv"
                    )
                    st.caption(f"{len(combined_df)} total rows")
            except Exception as e:
                st.warning(f"Could not create combined CSV: {e}")
                import traceback
                st.error(f"Error details: {traceback.format_exc()}")
        else:
            st.info("No CSV files generated")
    
    
    # File list
    with st.expander("📋 View Generated Files", expanded=False):
        st.write("**CSV Files:**")
        for f in sorted(csv_files):
            st.code(f, language=None)
        st.write("**Excel Files:**")
        for f in sorted(excel_files):
            st.code(f, language=None)
        st.write(f"**Location:** `{output_dir}`")
    
    st.info("💡 **Tip:** CSV files use UTF-8 encoding with BOM for Excel compatibility. Excel files (.xlsx) are ready to open directly!")
    
    # Store max_chars info for display
    st.session_state['max_chars_info'] = max_chars

# Show download section even after rerun (if scraping was completed)
if st.session_state.get('scraping_complete', False):
    # Retrieve data from session_state
    output_dir = st.session_state.get('output_dir')
    run_folder = st.session_state.get('run_folder')
    zip_path = st.session_state.get('zip_path')
    zip_name = st.session_state.get('zip_name')
    csv_files = st.session_state.get('csv_files', [])
    excel_files = st.session_state.get('excel_files', [])
    total = st.session_state.get('total_processed', 0)
    max_chars = st.session_state.get('max_chars_info', 10000)
    
    st.success(f"✅ Scraping completed! Processed {total:,} website(s).")
    
    if total > 0:
        st.info(f"💡 **Accuracy:** Max characters limit ({max_chars:,} chars) was accurately enforced per website.")
    
    # Download section (persistent)
    st.header("📥 Download Results")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        st.subheader("📦 All Files (ZIP)")
        zip_data = st.session_state.get('zip_data')
        if zip_data:
            st.download_button(
                label="⬇️ Download ZIP Archive",
                data=zip_data,
                file_name=zip_name,
                mime="application/zip",
                help="Download all CSV and Excel files in a ZIP archive",
                key="download_zip_persistent"
            )
        else:
            # Fallback: read from file
            try:
                if zip_path and os.path.exists(zip_path):
                    with open(zip_path, 'rb') as f:
                        zip_data = f.read()
                        st.session_state['zip_data'] = zip_data
                        st.download_button(
                            label="⬇️ Download ZIP Archive",
                            data=zip_data,
                            file_name=zip_name,
                            mime="application/zip",
                            help="Download all CSV and Excel files in a ZIP archive",
                            key="download_zip_persistent2"
                        )
                else:
                    st.info("ZIP file not available")
            except Exception as e:
                st.error(f"Could not read ZIP file: {e}")
        st.caption(f"Contains {len(csv_files)} CSV + {len(excel_files)} Excel files")
    
    with col_dl2:
        st.subheader("📊 Excel Files")
        excel_data = st.session_state.get('combined_excel_data')
        excel_filename = st.session_state.get('combined_excel_filename')
        combined_df = st.session_state.get('combined_df')
        
        if excel_data and excel_filename:
            st.download_button(
                label="⬇️ Download Combined Excel",
                data=excel_data,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download all data in a single Excel file",
                key="download_excel_persistent"
            )
            if combined_df is not None:
                st.caption(f"{len(combined_df)} total rows")
        elif excel_files:
            # Regenerate combined Excel file from existing Excel files
            if not output_dir:
                # Try to get output_dir from session_state
                output_dir = st.session_state.get('output_dir')
                if not output_dir:
                    st.warning("⚠️ Output directory not found. Please restart scraping to regenerate Excel file.")
                    output_dir = None
            
            if output_dir:
                st.info("🔄 Regenerating combined Excel file from existing files...")
                try:
                    # CRITICAL: Read from CSV files first (source of truth), NOT Excel files
                    csv_all_data = []
                    if csv_files:
                        for csv_file in sorted(csv_files):
                            if "combined" in csv_file.lower():
                                continue
                            csv_path = os.path.join(output_dir, csv_file)
                            try:
                                if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                                    rows_data = []
                                    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                                        csv_reader = csv.reader(f, quoting=csv.QUOTE_ALL, doublequote=True)
                                        header = next(csv_reader, None)
                                        if not header:
                                            continue
                                        header_normalized = [str(h).strip().strip('"').strip("'") for h in header]
                                        col_indices = {}
                                        for idx, h in enumerate(header_normalized):
                                            h_lower = h.lower().strip()
                                            if h_lower == 'website' or h_lower == 'url':
                                                col_indices['Website'] = idx
                                            elif h_lower == 'scrapedtext' or (h_lower.startswith('scraped') and 'text' in h_lower):
                                                col_indices['ScrapedText'] = idx
                                            elif h_lower == 'companysummary' or (h_lower.startswith('company') and 'summary' in h_lower):
                                                col_indices['CompanySummary'] = idx
                                        if 'Website' not in col_indices and len(header_normalized) >= 1:
                                            col_indices['Website'] = 0
                                        if 'ScrapedText' not in col_indices and len(header_normalized) >= 2:
                                            col_indices['ScrapedText'] = 1
                                        if 'CompanySummary' not in col_indices and len(header_normalized) >= 3:
                                            col_indices['CompanySummary'] = 2
                                        if len(col_indices) != 3:
                                            continue
                                        for row in csv_reader:
                                            if len(row) == 0:
                                                continue
                                            while len(row) < 3:
                                                row.append("")
                                            if len(row) > 3:
                                                row = row[:3]
                                            website_idx = col_indices.get('Website', 0)
                                            scraped_idx = col_indices.get('ScrapedText', 1)
                                            summary_idx = col_indices.get('CompanySummary', 2)
                                            if website_idx >= len(row) or scraped_idx >= len(row) or summary_idx >= len(row):
                                                continue
                                            website = str(row[website_idx]).strip()
                                            scraped_text = str(row[scraped_idx]).strip()
                                            company_summary = str(row[summary_idx]).strip()
                                            if website:
                                                rows_data.append({
                                                    'Website': website,
                                                    'ScrapedText': scraped_text,
                                                    'CompanySummary': company_summary
                                                })
                                    if rows_data:
                                        df_csv = pd.DataFrame(rows_data, columns=["Website", "ScrapedText", "CompanySummary"])
                                        csv_all_data.append(df_csv)
                            except Exception as e:
                                st.warning(f"⚠️ Error reading CSV {csv_file}: {e}")
                                continue
                    
                    # Use CSV data if available, otherwise fall back to Excel files
                    if csv_all_data:
                        combined_df = pd.concat(csv_all_data, ignore_index=True)
                        if "CompanySummary" not in combined_df.columns:
                            combined_df["CompanySummary"] = ""
                        combined_df = combined_df[["Website", "ScrapedText", "CompanySummary"]]
                    elif excel_files:
                        # Fallback to Excel files
                        all_data = []
                        for excel_file in sorted(excel_files):
                            excel_path = os.path.join(output_dir, excel_file)
                            try:
                                if os.path.exists(excel_path) and os.path.getsize(excel_path) > 0:
                                    df_part = pd.read_excel(excel_path, engine='openpyxl', header=0)
                                    if df_part.empty:
                                        continue
                                    df_part.columns = [str(col).strip() for col in df_part.columns]
                                    column_mapping = {}
                                    for col in df_part.columns:
                                        col_lower = col.lower().strip()
                                        if 'website' in col_lower or col_lower == 'url':
                                            column_mapping[col] = 'Website'
                                        elif 'scraped' in col_lower and 'text' in col_lower:
                                            column_mapping[col] = 'ScrapedText'
                                        elif 'company' in col_lower and 'summary' in col_lower:
                                            column_mapping[col] = 'CompanySummary'
                                    df_part = df_part.rename(columns=column_mapping)
                                    if 'Website' not in df_part.columns and len(df_part.columns) >= 1:
                                        df_part.columns.values[0] = 'Website'
                                    if 'ScrapedText' not in df_part.columns and len(df_part.columns) >= 2:
                                        df_part.columns.values[1] = 'ScrapedText'
                                    if 'CompanySummary' not in df_part.columns and len(df_part.columns) >= 3:
                                        df_part.columns.values[2] = 'CompanySummary'
                                    if "Website" not in df_part.columns:
                                        continue
                                    if "ScrapedText" not in df_part.columns:
                                        if len(df_part.columns) >= 2:
                                            second_col = df_part.columns[1]
                                            df_part = df_part.rename(columns={second_col: 'ScrapedText'})
                                        else:
                                            df_part['ScrapedText'] = ""
                                    if "CompanySummary" not in df_part.columns:
                                        df_part["CompanySummary"] = ""
                                    if "EmailCopy" not in df_part.columns:
                                        df_part["EmailCopy"] = ""
                                    std_cols = [c for c in EXCEL_COLS_4 if c in df_part.columns] or EXCEL_COLS_3
                                    df_part = df_part[std_cols]
                                    for col in df_part.columns:
                                        df_part[col] = df_part[col].astype(str).replace('nan', '').replace('None', '')
                                    if len(df_part) > 0:
                                        all_data.append(df_part)
                            except Exception as e:
                                st.warning(f"⚠️ Error reading Excel {excel_file}: {e}")
                                continue
                        if all_data:
                            combined_df = pd.concat(all_data, ignore_index=True)
                            for c in EXCEL_COLS_3:
                                if c not in combined_df.columns:
                                    combined_df[c] = ""
                            if "EmailCopy" not in combined_df.columns:
                                combined_df["EmailCopy"] = ""
                            combined_cols = EXCEL_COLS_4 if "EmailCopy" in combined_df.columns else EXCEL_COLS_3
                            combined_df = combined_df[combined_cols]
                        else:
                            st.error("❌ No data available to combine")
                            combined_df = None
                    else:
                        st.error("❌ No data available to combine")
                        combined_df = None
                    
                    if combined_df is not None and len(combined_df) > 0:
                        # Clean DataFrame
                        def clean_dataframe_for_excel(df):
                            import re
                            for col in df.columns:
                                df[col] = df[col].astype(str).replace('nan', '').replace('None', '')
                                df[col] = df[col].str.replace('\x00', '', regex=False)
                                df[col] = df[col].str.replace(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', regex=True)
                                df[col] = df[col].str.strip()
                            return df
                        combined_df = clean_dataframe_for_excel(combined_df.copy())
                        
                        # Generate combined Excel file
                        from openpyxl import Workbook
                        from io import BytesIO
                        excel_buffer = BytesIO()
                        wb = Workbook()
                        ws = wb.active
                        cols_out = [c for c in EXCEL_COLS_4 if c in combined_df.columns] or EXCEL_COLS_3
                        _write_excel_sheet(ws, cols_out, combined_df)
                        
                        # CRITICAL: Save with explicit error handling
                        try:
                            wb.save(excel_buffer)
                        except Exception as save_error:
                            st.error(f"❌ Excel save failed: {save_error}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
                            raise save_error
                        
                        excel_buffer.seek(0)
                        excel_data = excel_buffer.read()
                        
                        # Store in session_state
                        st.session_state['combined_excel_data'] = excel_data
                        st.session_state['combined_excel_filename'] = f"{run_folder}_combined.xlsx"
                        st.session_state['combined_df'] = combined_df
                        
                        st.download_button(
                            label="⬇️ Download Combined Excel",
                            data=excel_data,
                            file_name=f"{run_folder}_combined.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Download all data in a single Excel file",
                            key="download_excel_persistent"
                        )
                        st.caption(f"{len(combined_df)} total rows")
                    else:
                        st.warning("⚠️ No data found in Excel files to combine")
                except Exception as e:
                    st.error(f"❌ Error generating combined Excel file: {e}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.info("Individual Excel files are available in the ZIP archive")
                    # Still show that Excel files exist
                    if excel_files:
                        st.info(f"Found {len(excel_files)} Excel file(s): {', '.join(excel_files[:3])}{'...' if len(excel_files) > 3 else ''}")
        elif excel_files:
            # Excel files exist but output_dir is missing - this shouldn't happen, but handle it
            st.warning(f"⚠️ Found {len(excel_files)} Excel file(s) but output directory is not set. Please restart scraping.")
        else:
            st.info("No Excel files generated")
    
    with col_dl3:
        st.subheader("📄 CSV Files")
        csv_data = st.session_state.get('combined_csv_data')
        csv_filename = st.session_state.get('combined_csv_filename')
        combined_df = st.session_state.get('combined_df')
        
        if csv_data and csv_filename:
            st.download_button(
                label="⬇️ Download Combined CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                help="Download all data in a single CSV file (Excel/Google Sheets compatible)",
                key="download_csv_persistent"
            )
            if combined_df is not None:
                st.caption(f"{len(combined_df)} total rows")
        elif csv_files:
            st.info("Click 'Start Scraping' again to generate CSV file")
        else:
            st.info("No CSV files generated")
    
    
    # File list
    with st.expander("📋 View Generated Files", expanded=False):
        st.write("**CSV Files:**")
        for f in sorted(csv_files):
            st.code(f, language=None)
        st.write("**Excel Files:**")
        for f in sorted(excel_files):
            st.code(f, language=None)
        if output_dir:
            st.write(f"**Location:** `{output_dir}`")
    
    st.info("💡 **Tip:** CSV files use UTF-8 encoding with BOM for Excel compatibility. Excel files (.xlsx) are ready to open directly!")
    
    # Option to clear results
    if st.button("🗑️ Clear Results", help="Clear download buttons and start fresh", key="clear_results"):
        st.session_state['scraping_complete'] = False
        st.session_state.pop('output_dir', None)
        st.session_state.pop('zip_data', None)
        st.session_state.pop('combined_csv_data', None)
        st.session_state.pop('combined_excel_data', None)
        st.session_state.pop('combined_df', None)
        st.rerun()
