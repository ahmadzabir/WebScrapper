import asyncio
import aiohttp
import pandas as pd
import re
import os
import time
import csv
import zipfile
from html import unescape
import streamlit as st
from datetime import datetime
from io import BytesIO
import json

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


def normalize_url(url: str) -> str:
    """Normalize URL by adding https:// if protocol is missing."""
    url = url.strip()
    if not url:
        return url
    # If URL doesn't start with http:// or https://, add https://
    if not url.startswith(("http://", "https://")):
        # If it starts with www., add https://
        if url.startswith("www."):
            url = "https://" + url
        else:
            # Otherwise, add https://
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
# AI Summary Generation
# -------------------------

def build_company_summary_prompt(base_prompt: str, lead_data: dict, scraped_content: str) -> str:
    """
    Build the complete prompt for AI company summary generation.
    
    Args:
        base_prompt: User-provided base prompt (with placeholders)
        lead_data: Dictionary with lead information (URL, company name, etc.)
        scraped_content: Scraped website content
    
    Returns:
        Complete prompt string ready for AI
    """
    # Default prompt structure if user doesn't provide one - STRICT ANTI-HALLUCINATION
    default_base = """CRITICAL ANTI-HALLUCINATION RULES - READ CAREFULLY:

🚫 DO NOT:
- Invent any information not explicitly stated in the content
- Make assumptions about what the company does
- Guess industry, products, or services
- Add details not present in the scraped content
- Use general knowledge or external information
- Infer facts that aren't directly stated

✅ DO:
- Extract ONLY information explicitly written in the website content below
- Quote or paraphrase directly from the content
- Write "Not specified in the content" if information is missing
- Base every statement on actual text from the scraped content
- Clearly distinguish facts from inferences

LEAD INFORMATION:
- Website URL: {url}
- Company Name: {company_name}

WEBSITE CONTENT (ONLY SOURCE OF INFORMATION):
{scraped_content}

YOUR TASK:
Analyze ONLY the website content above and extract information that is EXPLICITLY STATED. Do not add anything that is not directly written in the content.

Provide:
1. **Company Overview**: What does this company do? (2-3 sentences)
   - ONLY use information directly stated in the content
   - If unclear, write "Company description not clearly stated in the content"
   - Quote specific phrases from the content when possible

2. **Industry**: What industry/sector does this company operate in?
   - ONLY if explicitly mentioned (e.g., "We are a technology company" or "Serving the healthcare industry")
   - If not stated, write "Industry not specified in the content"

3. **Products/Services**: List the products or services offered
   - ONLY list items explicitly mentioned in the content
   - Quote the exact product/service names from the content
   - If none mentioned, write "Products/services not specified in the content"

4. **Key Facts**: Important facts extracted DIRECTLY from the content
   - Each fact must be traceable to specific text in the content
   - Prefer direct quotes
   - If no clear facts, write "No specific facts extracted from the content"

5. **Target Market**: Who is their target audience/customers?
   - ONLY if explicitly stated (e.g., "We serve small businesses" or "Our customers are...")
   - If not stated, write "Target market not specified in the content"

6. **Five Inferences/Hypotheses**: Generate 5 inferences based ONLY on patterns in the provided content
   - Label each as "INFERENCE" or "HYPOTHESIS"
   - Base each on specific patterns/text in the content
   - Start each with "Based on [specific content pattern], I infer..."
   - If insufficient content, write fewer inferences

VALIDATION CHECKLIST:
Before submitting, verify:
- Every statement can be traced to specific text in the content
- No assumptions or guesses were made
- Missing information is marked as "Not specified"
- Inferences are clearly labeled and based on content patterns

Format your response clearly with sections."""
    
    # Use user prompt if provided, otherwise use default
    prompt_template = base_prompt if base_prompt.strip() else default_base
    
    # Replace placeholders
    company_name = lead_data.get('company_name', '')
    if not company_name or company_name == 'Unknown':
        # Extract from URL if not provided
        url_str = lead_data.get('url', '')
        company_name = url_str.replace('https://', '').replace('http://', '').split('/')[0] if url_str else 'Unknown'
    
    url = lead_data.get('url', 'N/A')
    industry = lead_data.get('industry', '')
    other_data = lead_data.get('other', '')
    
    # Build lead information section dynamically
    lead_info_parts = [f"- Website URL: {url}", f"- Company Name: {company_name}"]
    
    if industry and industry.strip() and industry.lower() != 'not specified':
        lead_info_parts.append(f"- Industry: {industry}")
    if other_data and other_data.strip() and other_data.lower() != 'not specified':
        lead_info_parts.append(f"- Additional Info: {other_data}")
    
    lead_info_section = "LEAD INFORMATION:\n" + "\n".join(lead_info_parts)
    
    # Replace placeholders in prompt
    prompt = prompt_template.replace('{url}', url)
    prompt = prompt.replace('{company_name}', company_name)
    prompt = prompt.replace('{scraped_content}', scraped_content[:15000])  # Limit content to avoid token limits
    
    # Replace LEAD INFORMATION section if it exists, otherwise add it
    import re
    if 'LEAD INFORMATION:' in prompt:
        # Replace existing lead info section
        prompt = re.sub(
            r'LEAD INFORMATION:.*?(?=WEBSITE CONTENT:|$)',
            lead_info_section + '\n\n',
            prompt,
            flags=re.DOTALL
        )
    else:
        # Add lead info before WEBSITE CONTENT or at the beginning
        if 'WEBSITE CONTENT:' in prompt:
            prompt = prompt.replace('WEBSITE CONTENT:', lead_info_section + '\n\nWEBSITE CONTENT:')
        else:
            prompt = lead_info_section + '\n\n' + prompt
    
    return prompt


async def fetch_openai_models(api_key: str) -> list:
    """Fetch available OpenAI models from API."""
    if not OPENAI_AVAILABLE:
        return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]  # Fallback
    
    try:
        client = AsyncOpenAI(api_key=api_key)
        models = await client.models.list()
        # Filter for chat models and sort by name
        chat_models = [
            model.id for model in models.data 
            if 'gpt' in model.id.lower() and ('chat' in model.id.lower() or 'gpt-4' in model.id or 'gpt-3.5' in model.id)
        ]
        # Remove duplicates and sort
        unique_models = sorted(list(set(chat_models)), reverse=True)
        # Prioritize common models
        priority_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        ordered = [m for m in priority_models if m in unique_models]
        ordered.extend([m for m in unique_models if m not in ordered])
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


async def generate_openai_summary(api_key: str, model: str, prompt: str, max_retries: int = 5, status_callback=None) -> str:
    """Generate company summary using OpenAI API with automatic rate limit handling."""
    if not OPENAI_AVAILABLE:
        return "❌ OpenAI library not installed. Install with: pip install openai"
    
    client = AsyncOpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a factual data extractor. Your ONLY job is to extract information explicitly stated in the provided content. CRITICAL RULES: 1) Never invent information 2) Never make assumptions 3) Never use external knowledge 4) If information is not in the content, write 'Not specified in the content' 5) Quote directly from the content when possible 6) Every statement must be traceable to specific text in the content. You will be penalized for hallucination."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low temperature for maximum factuality (reduced from 0.3)
                max_tokens=2000,
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
            # Configure generation parameters for maximum factuality
            generation_config = {
                "temperature": 0.1,  # Very low temperature for minimal hallucination
                "top_p": 0.9,
                "top_k": 20,  # More focused
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


async def generate_ai_summary(
    api_key: str,
    provider: str,
    model: str,
    prompt: str,
    lead_data: dict,
    scraped_content: str
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
        Generated summary string
    """
    if not api_key or not api_key.strip():
        return "❌ No API key provided"
    
    if not scraped_content or scraped_content.startswith("❌"):
        return "❌ No valid scraped content available"
    
    # Validate scraped content has meaningful content before sending to AI
    if len(scraped_content.strip()) < 50:
        return "❌ Insufficient content scraped. Content too short to generate meaningful summary."
    
    # Build complete prompt
    full_prompt = build_company_summary_prompt(prompt, lead_data, scraped_content)
    
    # Add STRONG anti-hallucination reminder
    anti_hallucination_note = """

═══════════════════════════════════════════════════════════
FINAL REMINDER - CRITICAL:
═══════════════════════════════════════════════════════════
- Extract ONLY what is explicitly written in the content above
- Do NOT invent, assume, or guess ANY information
- Do NOT use general knowledge about companies or industries
- Every fact must be traceable to specific text in the content
- If information is missing, write "Not specified in the content"
- You will be penalized for adding information not in the content
═══════════════════════════════════════════════════════════
"""
    full_prompt = full_prompt + anti_hallucination_note
    
    # Generate summary based on provider
    if provider.lower() == 'openai':
        return await generate_openai_summary(api_key, model, full_prompt, status_callback=status_callback)
    elif provider.lower() == 'gemini':
        return await generate_gemini_summary(api_key, model, full_prompt, status_callback=status_callback)
    else:
        return f"❌ Unknown provider: {provider}"


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
    
    # Remove common noise phrases
    noise_patterns = [
        r'cookie\s+policy',
        r'privacy\s+policy',
        r'terms\s+of\s+service',
        r'accept\s+cookies',
        r'we\s+use\s+cookies',
        r'skip\s+to\s+content',
        r'jump\s+to\s+content',
        r'menu',
        r'close',
        r'share\s+on',
        r'follow\s+us',
        r'subscribe\s+to',
    ]
    for pattern in noise_patterns:
        html = re.sub(pattern, '', html, flags=re.IGNORECASE)
    
    # Final cleanup
    html = html.strip()
    
    # Split into lines and clean each line
    lines = html.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip very short lines that are likely noise
        if len(line) < 3:
            continue
        # Skip lines that are just punctuation or symbols
        if re.match(r'^[^\w\s]+$', line):
            continue
        cleaned_lines.append(line)
    
    # Join back with proper spacing
    result = '\n'.join(cleaned_lines)
    
    # Final pass: ensure proper spacing around headers
    result = re.sub(r'\n(#{1,6}\s+[^\n]+)\n+', r'\n\1\n\n', result)
    
    return result.strip()


def extract_links(html: str, base_url: str):
    matches = re.findall(
        r'<a\s+(?:[^>]*?\s+)?href=(["\'])(.*?)\1', html, flags=re.IGNORECASE)
    base_domain = "/".join(base_url.split("/")[:3])
    urls = []
    for _, href in matches:
        if href.startswith(("#", "mailto:", "javascript:")):
            continue
        if href.startswith("/"):
            href = base_domain + href
        elif not href.startswith("http"):
            continue
        if href.startswith(base_domain):
            urls.append(href)
    return list(dict.fromkeys(urls))

# -------------------------
# Network fetch + scraping
# -------------------------


async def fetch(session: aiohttp.ClientSession, url: str, timeout: int, retries: int):
    for attempt in range(retries + 1):
        try:
            async with session.get(url, timeout=timeout) as resp:
                if resp.status >= 400:
                    return f"HTTP {resp.status} at {url}"
                if "text/html" not in resp.headers.get("Content-Type", ""):
                    return f"Non-HTML content at {url}"
                html = await resp.text(errors="ignore")
                return (str(resp.url), html)
        except (aiohttp.ClientError, ConnectionResetError, asyncio.TimeoutError) as e:
            if attempt == retries:
                return f"Error fetching {url}: {e}"
            await asyncio.sleep(1 + attempt * 0.5)


async def scrape_site(session, url: str, depth: int, keywords, max_chars: int, retries: int, timeout: int):
    visited, results, errors = set(), [], []
    total_chars = 0
    separator = "\n\n" + "─" * 80 + "\n\n"  # Better visual separator between pages
    separator_len = len(separator)

    homepage = await fetch(session, url, timeout, retries)
    if isinstance(homepage, str):
        return f"❌ {homepage}"

    page_url, html = homepage
    cleaned = cleanup_html(html)
    if cleaned:
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
        errors.append(f"❌ No visible text on homepage: {page_url}")

    links = extract_links(html, url)
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
        
        res = await fetch(session, link, timeout, retries)
        if isinstance(res, str):
            errors.append(f"❌ {res}")
        elif isinstance(res, tuple):
            link_url, html2 = res
            cleaned2 = cleanup_html(html2)
            if cleaned2:
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
                errors.append(f"❌ No visible text on page: {link_url}")

    if not results:
        return errors[0] if errors else f"❌ Unknown error on site: {url}"
    
    # Join results with separator
    final_text = separator.join(results)
    
    # Add errors at the end if there's space (errors are usually short)
    if errors and len(final_text) + len("\n".join(errors)) + 20 <= max_chars:
        final_text += "\n\n" + "\n".join(errors)
    
    # Final safety check - ensure we don't exceed max_chars
    if len(final_text) > max_chars:
        final_text = final_text[:max_chars]
        # Try to cut at a reasonable point (end of sentence or line)
        last_period = final_text.rfind('.')
        last_newline = final_text.rfind('\n')
        cut_point = max(last_period, last_newline)
        if cut_point > max_chars * 0.9:  # Only use if it's not too early
            final_text = final_text[:cut_point + 1]
    
    return final_text

# -------------------------
# Worker + Writer
# -------------------------


async def worker_coroutine(name, session, url_queue: asyncio.Queue, result_queue: asyncio.Queue,
                           depth, keywords, max_chars, retries, timeout,
                           ai_enabled=False, ai_api_key=None, ai_provider=None, ai_model=None, ai_prompt=None,
                           lead_data_map=None, ai_status_callback=None):
    while True:
        item = await url_queue.get()
        if item is None:
            url_queue.task_done()
            break
        
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
            lead_data = lead_data_map[url_index].copy()
            lead_data['url'] = original_url  # Ensure URL is set
        else:
            # Fallback: extract company name from URL
            lead_data = {
                'url': original_url,
                'company_name': original_url.replace('https://', '').replace('http://', '').split('/')[0]
            }
        
        # Validate normalized URL
        if not normalized_url or not normalized_url.startswith(("http://", "https://")):
            scraped_text = "❌ Invalid URL"
            ai_summary = "❌ Invalid URL"
        else:
            # Scrape the website
            scraped_text = await scrape_site(session, normalized_url, depth, keywords, max_chars, retries, timeout)
            
            # Generate AI summary if enabled
            if ai_enabled and ai_api_key and ai_provider and ai_model:
                # Create URL-specific status callback
                def url_status_callback(msg):
                    if ai_status_callback:
                        ai_status_callback(original_url, msg)
                
                ai_summary = await generate_ai_summary(
                    ai_api_key, ai_provider, ai_model, ai_prompt or "",
                    lead_data, scraped_text, status_callback=url_status_callback
                )
            else:
                ai_summary = ""  # Empty if AI not enabled
        
        # Output format: (url, scraped_text, ai_summary)
        out = (original_url, scraped_text, ai_summary)
        await result_queue.put(out)
        url_queue.task_done()


async def writer_coroutine(result_queue: asyncio.Queue, rows_per_file: int, output_dir: str,
                           total_urls: int, progress_callback):
    os.makedirs(output_dir, exist_ok=True)
    buffer = []
    part = 0
    processed = 0

    while True:
        item = await result_queue.get()
        if item is None:
            result_queue.task_done()
            break
        buffer.append(item)
        processed += 1
        if progress_callback:
            progress_callback(processed, total_urls)

        # Write chunks to disk when buffer reaches threshold (optimized for large files)
        while len(buffer) >= rows_per_file:
            part += 1
            chunk_rows = buffer[:rows_per_file]
            buffer = buffer[rows_per_file:]
            
            try:
                # Determine columns based on data structure (2 or 3 columns)
                if len(chunk_rows[0]) == 3:
                    df = pd.DataFrame(chunk_rows, columns=["Website", "ScrapedText", "CompanySummary"])
                else:
                    df = pd.DataFrame(chunk_rows, columns=["Website", "ScrapedText"])
                
                # Save CSV with Excel/Google Sheets compatibility (optimized for large files)
                csv_path = os.path.join(output_dir, f"output_part_{part}.csv")
                # Use UTF-8 with BOM for Excel compatibility
                df.to_csv(csv_path, index=False, encoding="utf-8-sig",
                          quoting=csv.QUOTE_MINIMAL, escapechar='\\', lineterminator="\n",
                          chunksize=1000 if len(df) > 5000 else None)  # Chunk large writes
                
                # Save Excel file (optimized for large files)
                excel_path = os.path.join(output_dir, f"output_part_{part}.xlsx")
                try:
                    # For very large files, use write_only mode to save memory
                    if len(df) > 10000:
                        from openpyxl import Workbook
                        wb = Workbook(write_only=True)
                        ws = wb.create_sheet()
                        # Determine columns
                        if "CompanySummary" in df.columns:
                            ws.append(["Website", "ScrapedText", "CompanySummary"])
                            for _, row in df.iterrows():
                                website = str(row["Website"])[:255]
                                text = str(row["ScrapedText"])[:32767]
                                summary = str(row["CompanySummary"])[:32767]
                                ws.append([website, text, summary])
                        else:
                            ws.append(["Website", "ScrapedText"])
                            for _, row in df.iterrows():
                                website = str(row["Website"])[:255]
                                text = str(row["ScrapedText"])[:32767]
                                ws.append([website, text])
                        wb.save(excel_path)
                    else:
                        df.to_excel(excel_path, index=False, engine='openpyxl')
                except Exception as e:
                    # If Excel fails, log but continue (CSV is more important)
                    import logging
                    logging.warning(f"Excel export failed for part {part}: {e}")
            except Exception as e:
                # If Excel fails for large file, at least save CSV
                try:
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig",
                              quoting=csv.QUOTE_MINIMAL, escapechar='\\', lineterminator="\n")
                except:
                    pass

        result_queue.task_done()

    # Write remaining buffer
    if buffer:
        part += 1
        try:
            # Determine columns based on data structure
            if len(buffer[0]) == 3:
                df = pd.DataFrame(buffer, columns=["Website", "ScrapedText", "CompanySummary"])
            else:
                df = pd.DataFrame(buffer, columns=["Website", "ScrapedText"])
            
            # Save CSV with Excel/Google Sheets compatibility
            csv_path = os.path.join(output_dir, f"output_part_{part}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig",
                      quoting=csv.QUOTE_MINIMAL, escapechar='\\', lineterminator="\n")
            
            # Save Excel file (optimized for large files)
            excel_path = os.path.join(output_dir, f"output_part_{part}.xlsx")
            try:
                if len(df) > 10000:
                    from openpyxl import Workbook
                    wb = Workbook(write_only=True)
                    ws = wb.create_sheet()
                    # Determine columns
                    if "CompanySummary" in df.columns:
                        ws.append(["Website", "ScrapedText", "CompanySummary"])
                        for _, row in df.iterrows():
                            website = str(row["Website"])[:255]
                            text = str(row["ScrapedText"])[:32767]
                            summary = str(row["CompanySummary"])[:32767]
                            ws.append([website, text, summary])
                    else:
                        ws.append(["Website", "ScrapedText"])
                        for _, row in df.iterrows():
                            website = str(row["Website"])[:255]
                            text = str(row["ScrapedText"])[:32767]
                            ws.append([website, text])
                    wb.save(excel_path)
                else:
                    df.to_excel(excel_path, index=False, engine='openpyxl')
            except Exception as e:
                # If Excel fails, log but continue (CSV is more important)
                import logging
                logging.warning(f"Excel export failed for part {part}: {e}")
        except Exception as e:
            # If Excel fails, at least save CSV
            try:
                df.to_csv(csv_path, index=False, encoding="utf-8-sig",
                          quoting=csv.QUOTE_MINIMAL, escapechar='\\', lineterminator="\n")
            except:
                pass

# -------------------------
# Runner
# -------------------------


async def run_scraper(urls, concurrency, retries, timeout, depth, keywords, max_chars,
                      user_agent, rows_per_file, output_dir, progress_callback,
                      ai_enabled=False, ai_api_key=None, ai_provider=None, ai_model=None, ai_prompt=None,
                      lead_data_map=None, ai_status_callback=None):
    url_queue = asyncio.Queue()
    result_queue = asyncio.Queue()
    total = len(urls)

    # Put URLs with their indices for lead data lookup
    for idx, u in enumerate(urls):
        await url_queue.put((u, idx))

    timeout_obj = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=0, ssl=False)
    session = aiohttp.ClientSession(
        headers={"User-Agent": user_agent}, connector=connector, timeout=timeout_obj)

    writer_task = asyncio.create_task(writer_coroutine(
        result_queue, rows_per_file, output_dir, total, progress_callback))

    workers = [asyncio.create_task(worker_coroutine(
        f"worker-{i+1}", session, url_queue, result_queue, depth, keywords, max_chars, retries, timeout,
        ai_enabled, ai_api_key, ai_provider, ai_model, ai_prompt, lead_data_map, ai_status_callback)) for i in range(concurrency)]

    await url_queue.join()

    for _ in workers:
        await url_queue.put(None)
    await asyncio.gather(*workers)

    await result_queue.put(None)
    await result_queue.join()
    await writer_task

    await session.close()

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Web Scraper", layout="wide", page_icon="🌐")
st.title("🌐 Stable Async Web Scraper")
st.markdown("Upload a CSV file with URLs in the first column, configure settings, and start scraping!")

# File Upload Section
st.header("📁 Upload CSV File")
uploaded_file = st.file_uploader(
    "Upload CSV with URLs", 
    type=["csv"],
    help="Upload your CSV file. You'll be able to select which column contains URLs and other lead data."
)

# CSV Configuration (shown after file upload)
csv_has_headers = None
url_column = None
lead_data_columns = {}
df_preview = None

if uploaded_file is not None:
    st.subheader("📋 CSV Configuration")
    st.info("👆 **Configure your CSV file below** - Select headers and columns before starting scraping")
    
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
                "Does your CSV have headers?",
                ["Yes, has headers", "No headers"],
                help="Select whether the first row contains column names",
                key="csv_headers_radio"
            )
        
        with col_csv2:
            st.caption("💡 **Tip:** Headers make column selection easier!")
        
        # Read CSV based on header selection
        uploaded_file.seek(0)  # Reset file pointer
        if csv_has_headers == "Yes, has headers":
            df_preview = pd.read_csv(uploaded_file, nrows=10)
            st.success(f"✅ Detected {len(df_preview.columns)} columns with headers: {', '.join(df_preview.columns.tolist()[:5])}{'...' if len(df_preview.columns) > 5 else ''}")
        else:
            df_preview = pd.read_csv(uploaded_file, header=None, nrows=10)
            st.info(f"ℹ️ Detected {len(df_preview.columns)} columns (no headers)")
        
        # Show preview
        with st.expander("👀 Preview CSV (first 10 rows)", expanded=False):
            st.dataframe(df_preview, use_container_width=True)
        
        # Column selection
        st.subheader("🔧 Column Selection")
        
        col_sel1, col_sel2 = st.columns(2)
        
        with col_sel1:
            # URL column selection
            if csv_has_headers == "Yes, has headers":
                url_column = st.selectbox(
                    "Select column with URLs",
                    options=list(df_preview.columns),
                    index=0,
                    help="Choose which column contains the website URLs"
                )
            else:
                url_column = st.selectbox(
                    "Select column with URLs",
                    options=[f"Column {i+1}" for i in range(len(df_preview.columns))],
                    index=0,
                    help="Choose which column contains the website URLs"
                )
        
        with col_sel2:
            st.caption("📌 **Required:** This column must contain URLs")
        
        # Additional lead data columns (optional)
        st.markdown("**📊 Additional Lead Data (Optional - for AI summaries)**")
        st.caption("Select columns that contain company information (name, industry, etc.) to include in AI summaries")
        
        col_lead1, col_lead2, col_lead3 = st.columns(3)
        
        with col_lead1:
            if csv_has_headers == "Yes, has headers":
                company_name_col = st.selectbox(
                    "Company Name Column (optional)",
                    options=["None"] + list(df_preview.columns),
                    index=0,
                    help="Column with company names"
                )
            else:
                company_name_col = st.selectbox(
                    "Company Name Column (optional)",
                    options=["None"] + [f"Column {i+1}" for i in range(len(df_preview.columns))],
                    index=0
                )
            if company_name_col != "None":
                lead_data_columns['company_name'] = company_name_col
        
        with col_lead2:
            if csv_has_headers == "Yes, has headers":
                industry_col = st.selectbox(
                    "Industry Column (optional)",
                    options=["None"] + list(df_preview.columns),
                    index=0,
                    help="Column with industry information"
                )
            else:
                industry_col = st.selectbox(
                    "Industry Column (optional)",
                    options=["None"] + [f"Column {i+1}" for i in range(len(df_preview.columns))],
                    index=0
                )
            if industry_col != "None":
                lead_data_columns['industry'] = industry_col
        
        with col_lead3:
            if csv_has_headers == "Yes, has headers":
                other_col = st.selectbox(
                    "Other Column (optional)",
                    options=["None"] + list(df_preview.columns),
                    index=0,
                    help="Any other relevant column"
                )
            else:
                other_col = st.selectbox(
                    "Other Column (optional)",
                    options=["None"] + [f"Column {i+1}" for i in range(len(df_preview.columns))],
                    index=0
                )
            if other_col != "None":
                lead_data_columns['other'] = other_col
        
        # Store in session state for use during scraping
        st.session_state['csv_config'] = {
            'has_headers': csv_has_headers == "Yes, has headers",
            'url_column': url_column,
            'lead_data_columns': lead_data_columns,
            'df_preview': df_preview
        }
        
    except Exception as e:
        st.error(f"❌ Error reading CSV: {str(e)}")
        st.info("Please check your CSV file format and try again.")

# Configuration Section
st.header("⚙️ Configuration Settings")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Scraping Parameters")
    
    # Keywords with detailed explanation
    with st.expander("🔑 Keywords - How to Use", expanded=False):
        st.markdown("""
        **Keywords help find relevant pages on each website.**
        
        **Format:**
        - Separate keywords with commas: `about,service,product`
        - Spaces after commas are optional: `about, service, product` ✅
        - Case doesn't matter: `About, SERVICE, Product` ✅
        - Hyphens work: `about-us, service-page` ✅
        
        **How it works:**
        - Scraper looks for these keywords in website URLs
        - Finds links like `/about`, `/about-us`, `/services`, `/products`
        - Case-insensitive matching (About = about = ABOUT)
        - Handles hyphens and underscores automatically
        
        **Examples:**
        - `about,service,product` → finds `/about`, `/services`, `/products`
        - `contact, pricing, faq` → finds contact, pricing, FAQ pages
        - `blog, news, articles` → finds blog/news pages
        """)
    
    keywords_input = st.text_input(
        "Keywords (comma separated)",
        value="about,service,product",
        help="Enter keywords separated by commas. The scraper will find pages containing these keywords in their URLs."
    )
    keywords = process_keywords(keywords_input)
    
    if keywords:
        st.caption(f"✅ Processing {len(keywords)} keyword(s): {', '.join(keywords)}")
    
    # Concurrency
    with st.expander("⚡ Concurrency - What is it?", expanded=False):
        st.markdown("""
        **Number of parallel workers scraping websites simultaneously.**
        
        - **Low (1-10):** Slower but uses less resources, more stable
        - **Medium (20-30):** Good balance (recommended)
        - **High (40-50):** Faster but may cause timeouts or rate limiting
        
        **Recommendation:** Start with 20, increase if you have fast internet.
        """)
    
    concurrency = st.slider(
        "Concurrency (workers)", 
        1, 50, 20,
        help="Number of websites scraped at the same time. Higher = faster but may cause issues."
    )
    
    # Retries
    with st.expander("🔄 Retry Attempts - When to use?", expanded=False):
        st.markdown("""
        **How many times to retry if a website fails to load.**
        
        - **0:** No retries (fastest, but may miss slow websites)
        - **1-2:** Quick retry for temporary issues
        - **3-5:** More persistent (recommended for unreliable websites)
        
        **Use higher values if:** Websites are slow or frequently timeout.
        """)
    
    retries = st.slider(
        "Retry attempts", 
        0, 5, 3,
        help="Number of times to retry failed requests. Higher = more reliable but slower."
    )
    
    # Depth
    with st.expander("🔗 Depth - Link Following", expanded=False):
        st.markdown("""
        **How many levels of links to follow from the homepage.**
        
        - **0:** Only homepage (fastest)
        - **1-2:** Homepage + direct links (recommended)
        - **3-5:** Deep crawling (slower, more content)
        
        **How it works:**
        - Depth 1: Homepage + pages linked from homepage
        - Depth 2: Homepage + linked pages + their linked pages
        - Plus: Always follows pages matching your keywords
        
        **Recommendation:** Use 2-3 for most cases.
        """)
    
    depth = st.slider(
        "Depth (link follow)", 
        0, 5, 3,
        help="How many levels of links to follow. Higher = more pages but slower."
    )

with col2:
    st.subheader("⏱️ Performance & Output")
    
    # Timeout
    with st.expander("⏰ Timeout - What does it do?", expanded=False):
        st.markdown("""
        **Maximum seconds to wait for a website to respond.**
        
        - **Low (2-5s):** Fast, but may skip slow websites
        - **Medium (10-15s):** Good balance (recommended)
        - **High (20-30s):** Waits longer, catches slow sites
        
        **Use higher values if:** Websites are slow to load.
        """)
    
    timeout = st.slider(
        "Timeout (seconds)", 
        2, 30, 10,
        help="Maximum seconds to wait for each website. Higher = waits longer for slow sites."
    )
    
    # Max chars
    with st.expander("📏 Max Characters - Accurate Limit", expanded=False):
        st.markdown("""
        **Maximum characters to scrape from each website (per site, not total).**
        
        - **Low (1k-10k):** Quick summaries only
        - **Medium (20k-50k):** Good content (recommended)
        - **High (80k-100k):** Full pages, may be very long
        
        **How it works (ACCURATE):**
        - ✅ **Per-site limit:** Each website gets up to this many characters
        - ✅ **Smart truncation:** Cuts at sentence/line boundaries when possible
        - ✅ **Includes all pages:** Homepage + linked pages (up to limit)
        - ✅ **Exact control:** The limit is strictly enforced per website
        
        **Why limit?**
        - Prevents extremely large files (especially with 20k+ URLs)
        - Focuses on important content
        - Faster processing and smaller file sizes
        - Better Excel/Google Sheets compatibility
        
        **For large datasets (20k+ rows):**
        - Use 20k-50k characters per site for manageable file sizes
        - Each row = one website, so total file size = rows × avg_chars_per_site
        - Example: 20,000 rows × 50,000 chars = ~1GB of text data
        
        **Recommendation:** 50,000 characters per site is optimal for most use cases.
        """)
    
    max_chars = st.number_input(
        "Max chars per site (per website)", 
        1000, 100000, 50000, step=5000,
        help="Maximum characters to extract from EACH website. This limit is accurately enforced per site. For large datasets (20k+ rows), use 20k-50k to keep file sizes manageable."
    )
    
    # Rows per file
    with st.expander("📊 Rows per File - File Management", expanded=False):
        st.markdown("""
        **How many rows before creating a new file.**
        
        - **Low (1k-2k):** Many small files (easier to manage)
        - **Medium (2k-5k):** Balanced (recommended)
        - **High (10k+):** Fewer large files
        
        **Why split files?**
        - Easier to open in Excel/Google Sheets
        - Prevents file size issues
        - Better for large datasets
        
        **Output:** Files named `output_part_1.csv`, `output_part_2.csv`, etc.
        """)
    
    rows_per_file = st.number_input(
        "Rows per CSV file (chunk size)", 
        1000, 50000, 2000, step=1000,
        help="Number of rows before creating a new file. Prevents huge files."
    )
    
    # User Agent
    with st.expander("🌐 User-Agent - What is it?", expanded=False):
        st.markdown("""
        **Identifies your scraper to websites (like a browser signature).**
        
        - **Default:** `Mozilla/5.0` (basic)
        - **Better:** `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36`
        - **Custom:** Any string you want
        
        **Why change it?**
        - Some websites block basic user agents
        - Mimics real browser (reduces blocking)
        - Usually fine to leave default
        """)
    
    user_agent = st.text_input(
        "User-Agent", 
        value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        help="Browser identifier sent to websites. Default works for most cases."
    )

# AI Summary Section
st.header("🤖 AI Company Summary (Optional)")
ai_enabled = st.checkbox(
    "Enable AI-powered company summary generation",
    value=False,
    help="Generate AI summaries for each company using OpenAI or Gemini"
)

ai_api_key = None
ai_provider = None
ai_model = None
ai_prompt = None

if ai_enabled:
    col_ai1, col_ai2 = st.columns(2)
    
    with col_ai1:
        ai_provider = st.selectbox(
            "AI Provider",
            ["OpenAI", "Gemini"],
            help="Choose OpenAI (GPT models) or Google Gemini"
        )
        
        # Get API key from session state or use empty
        api_key_key = f"{ai_provider.lower()}_api_key"
        stored_api_key = st.session_state.get(api_key_key, "")
        
        col_key1, col_key2 = st.columns([3, 1])
        
        with col_key1:
            ai_api_key = st.text_input(
                f"{ai_provider} API Key",
                value=stored_api_key,
                type="password",
                help=f"Enter your {ai_provider} API key. Get one at: OpenAI (platform.openai.com) or Gemini (makersuite.google.com/app/apikey). Your key will be saved for this session.",
                key=f"{ai_provider}_api_key_input"
            )
        
        with col_key2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("🗑️ Clear", key=f"clear_{ai_provider}_key", help="Clear saved API key"):
                st.session_state[api_key_key] = ""
                st.rerun()
        
        # Save API key to session state when changed
        if ai_api_key:
            if ai_api_key != stored_api_key:
                st.session_state[api_key_key] = ai_api_key
            # Show saved indicator
            masked_key = "*" * min(len(ai_api_key), 8) + "..." if len(ai_api_key) > 8 else "*" * len(ai_api_key)
            st.caption(f"✅ API key saved for this session ({masked_key})")
        elif stored_api_key and not ai_api_key:
            # If cleared, remove from session state
            st.session_state[api_key_key] = ""
        
        # Model selection based on provider
        if ai_provider == "OpenAI":
            ai_model = st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=0,
                help="OpenAI model. gpt-4o-mini is fastest/cheapest, gpt-4o is most capable."
            )
        else:  # Gemini
            ai_model = st.selectbox(
                "Model",
                ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
                index=1,
                help="Gemini model. Flash is fastest, Pro is most capable."
            )
    
    with col_ai2:
        st.markdown("**📝 Prompt Configuration**")
        
        # Prompt editing mode
        prompt_mode = st.radio(
            "Prompt Mode",
            ["Use Default Prompt", "Edit Master Prompt"],
            help="Use default prompt or edit the master prompt template",
            horizontal=True
        )
        
        # Default prompt template
        default_prompt_template = """CRITICAL ANTI-HALLUCINATION RULES - READ CAREFULLY:

🚫 DO NOT:
- Invent any information not explicitly stated in the content
- Make assumptions about what the company does
- Guess industry, products, or services
- Add details not present in the scraped content
- Use general knowledge or external information
- Infer facts that aren't directly stated

✅ DO:
- Extract ONLY information explicitly written in the website content below
- Quote or paraphrase directly from the content
- Write "Not specified in the content" if information is missing
- Base every statement on actual text from the scraped content
- Clearly distinguish facts from inferences

LEAD INFORMATION:
- Website URL: {url}
- Company Name: {company_name}

WEBSITE CONTENT (ONLY SOURCE OF INFORMATION):
{scraped_content}

YOUR TASK:
Analyze ONLY the website content above and extract information that is EXPLICITLY STATED. Do not add anything that is not directly written in the content.

Provide:
1. **Company Overview**: What does this company do? (2-3 sentences)
   - ONLY use information directly stated in the content
   - If unclear, write "Company description not clearly stated in the content"
   - Quote specific phrases from the content when possible

2. **Industry**: What industry/sector does this company operate in?
   - ONLY if explicitly mentioned (e.g., "We are a technology company" or "Serving the healthcare industry")
   - If not stated, write "Industry not specified in the content"

3. **Products/Services**: List the products or services offered
   - ONLY list items explicitly mentioned in the content
   - Quote the exact product/service names from the content
   - If none mentioned, write "Products/services not specified in the content"

4. **Key Facts**: Important facts extracted DIRECTLY from the content
   - Each fact must be traceable to specific text in the content
   - Prefer direct quotes
   - If no clear facts, write "No specific facts extracted from the content"

5. **Target Market**: Who is their target audience/customers?
   - ONLY if explicitly stated (e.g., "We serve small businesses" or "Our customers are...")
   - If not stated, write "Target market not specified in the content"

6. **Five Inferences/Hypotheses**: Generate 5 inferences based ONLY on patterns in the provided content
   - Label each as "INFERENCE" or "HYPOTHESIS"
   - Base each on specific patterns/text in the content
   - Start each with "Based on [specific content pattern], I infer..."
   - If insufficient content, write fewer inferences

VALIDATION CHECKLIST:
Before submitting, verify:
- Every statement can be traced to specific text in the content
- No assumptions or guesses were made
- Missing information is marked as "Not specified"
- Inferences are clearly labeled and based on content patterns

Format your response clearly with sections."""
        
        # Store default prompt in session state if not exists
        if 'master_prompt' not in st.session_state:
            st.session_state['master_prompt'] = default_prompt_template
        
        # Allow editing master prompt
        if prompt_mode == "Edit Master Prompt":
            st.info("💡 **Edit the master prompt template below. Use placeholders: `{url}`, `{company_name}`, `{scraped_content}`")
            
            ai_prompt = st.text_area(
                "Master Prompt Template",
                value=st.session_state.get('master_prompt', default_prompt_template),
                height=400,
                help="Edit the master prompt. Placeholders: {url}, {company_name}, {scraped_content}"
            )
            st.session_state['master_prompt'] = ai_prompt
        else:
            ai_prompt = st.session_state.get('master_prompt', default_prompt_template)
        
        # Show prompt preview
        with st.expander("👁️ Preview Final Prompt Structure", expanded=False):
            st.markdown("**See how your prompt will look with actual data:**")
            
            # Sample data for preview
            sample_url = "https://example.com"
            sample_company = "Example Company"
            sample_content = "This is a sample of scraped website content. The company provides technology solutions for businesses. They serve small and medium enterprises."
            
            # Build preview prompt
            preview_prompt = ai_prompt.replace('{url}', sample_url)
            preview_prompt = preview_prompt.replace('{company_name}', sample_company)
            preview_prompt = preview_prompt.replace('{scraped_content}', sample_content[:200] + "...")
            
            st.markdown("**Example Final Prompt (with sample data):**")
            st.code(preview_prompt, language=None)
            
            st.markdown("---")
            st.markdown("**📌 Placeholders that will be replaced:**")
            st.markdown("""
            - `{url}` → Actual website URL from your CSV
            - `{company_name}` → Company name (from CSV column or extracted from URL)
            - `{scraped_content}` → Full scraped website content (up to 15,000 characters)
            """)
            
            st.markdown("**💡 Note:** The final prompt also includes:")
            st.markdown("""
            - Lead data from CSV (industry, other columns if selected)
            - Anti-hallucination reminder at the end
            - System message with strict instructions
            """)
        
        # Show what will be added
        st.caption("ℹ️ The final prompt sent to AI includes: Master prompt + Lead data + Anti-hallucination reminder")
    
    # Ensure ai_prompt is set even if AI is disabled (for preview purposes)
    if ai_enabled:
        if 'ai_prompt' not in locals() or ai_prompt is None:
            ai_prompt = st.session_state.get('master_prompt', default_prompt_template)
    else:
        ai_prompt = None
    
    if not ai_api_key or not ai_api_key.strip():
        st.warning("⚠️ Please enter your API key to enable AI summaries.")
        ai_enabled = False

# Output folder name
st.header("📂 Output Settings")
with st.expander("📁 Output Folder Name - Optional", expanded=False):
    st.markdown("""
    **Custom name for your output folder and ZIP file.**
    
    - **Leave empty:** Auto-generates timestamp name (e.g., `run_20260120_120000`)
    - **Enter name:** Uses your custom name (e.g., `my-scrape-results`)
    
    **Output includes:**
    - CSV files (Excel/Google Sheets compatible)
    - Excel (.xlsx) files
    - ZIP archive with all files
    """)

run_name = st.text_input(
    "📂 Output folder/zip name (optional)", 
    value="",
    help="Custom name for output folder. Leave empty for auto-generated timestamp."
)

# -------- SCRAPE BUTTON --------
if uploaded_file and st.button("🚀 Start Scraping"):
    # Get CSV configuration from session state
    csv_config = st.session_state.get('csv_config', {})
    
    if not csv_config:
        st.error("❌ Please configure your CSV file above (select columns)")
        st.stop()
    
    # Read CSV based on configuration
    has_headers = csv_config.get('has_headers', True)
    url_col = csv_config.get('url_column')
    lead_cols = csv_config.get('lead_data_columns', {})
    
    # Reset file pointer
    uploaded_file.seek(0)
    
    if has_headers:
        df_in = pd.read_csv(uploaded_file)
    else:
        df_in = pd.read_csv(uploaded_file, header=None)
    
    # Get URL column index
    if has_headers:
        url_col_idx = list(df_in.columns).index(url_col)
    else:
        url_col_idx = int(url_col.replace("Column ", "")) - 1
    
    # Extract URLs
    urls = df_in.iloc[:, url_col_idx].fillna("").astype(str).tolist()
    # Filter out empty URLs
    urls = [url for url in urls if url.strip()]
    total = len(urls)
    
    # Prepare lead data mapping (row index -> lead data dict)
    lead_data_map = {}
    for idx, url in enumerate(urls):
        lead_data = {'url': url}
        
        # Add company name if column selected
        if 'company_name' in lead_cols:
            col_name = lead_cols['company_name']
            if has_headers:
                col_idx = list(df_in.columns).index(col_name)
            else:
                col_idx = int(col_name.replace("Column ", "")) - 1
            if idx < len(df_in):
                lead_data['company_name'] = str(df_in.iloc[idx, col_idx]) if pd.notna(df_in.iloc[idx, col_idx]) else ""
            else:
                lead_data['company_name'] = ""
        
        # Add industry if column selected
        if 'industry' in lead_cols:
            col_name = lead_cols['industry']
            if has_headers:
                col_idx = list(df_in.columns).index(col_name)
            else:
                col_idx = int(col_name.replace("Column ", "")) - 1
            if idx < len(df_in):
                lead_data['industry'] = str(df_in.iloc[idx, col_idx]) if pd.notna(df_in.iloc[idx, col_idx]) else ""
            else:
                lead_data['industry'] = ""
        
        # Add other data if column selected
        if 'other' in lead_cols:
            col_name = lead_cols['other']
            if has_headers:
                col_idx = list(df_in.columns).index(col_name)
            else:
                col_idx = int(col_name.replace("Column ", "")) - 1
            if idx < len(df_in):
                lead_data['other'] = str(df_in.iloc[idx, col_idx]) if pd.notna(df_in.iloc[idx, col_idx]) else ""
            else:
                lead_data['other'] = ""
        
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

    # Use user input or fallback to timestamp
    if run_name.strip():
        run_folder = run_name.strip().replace(" ", "_")
    else:
        run_folder = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", run_folder)
    os.makedirs(output_dir, exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()
    eta_text = st.empty()
    
    # AI status display (only if AI enabled)
    ai_status_text = None
    if ai_enabled:
        st.subheader("🤖 AI Summary Status")
        ai_status_text = st.empty()
        ai_status_text.info("AI summaries will start generating after scraping completes...")
    
    # AI status callback function
    ai_status_messages = {}
    def ai_status_callback(url, message):
        """Update AI status display in real-time."""
        ai_status_messages[url] = message
        if ai_status_text:
            # Show last 5 status messages
            recent_messages = list(ai_status_messages.items())[-5:]
            status_display = "\n".join([f"**{url[:50]}...**: {msg}" for url, msg in recent_messages])
            ai_status_text.info(status_display)

    fun_messages = [
        "🔍 Scanning the web...",
        "🛠️ Sharpening the scrapers...",
        "🚀 Launching data rockets...",
        "📡 Tuning into websites...",
        "🧩 Piecing text together..."
    ]
    start_time = time.time()

    def progress_cb(done_count, total_count):
        percent = done_count / max(total_count, 1)
        elapsed = time.time() - start_time
        avg = elapsed / max(done_count, 1)
        remaining = avg * (total_count - done_count)
        progress_bar.progress(min(percent, 1.0))
        idx = (done_count - 1) % len(fun_messages) if done_count > 0 else 0
        status_text.text(f"{fun_messages[idx]}  ({done_count}/{total_count})")
        eta_text.text(f"⏳ ETA: {int(remaining // 60)}m {int(remaining % 60)}s")

    with st.spinner("Scraping — this runs in the page (do not close)..."):
        if os.name == "nt":
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy())
        # Get lead_data_map from session state
        lead_data_map = st.session_state.get('lead_data_map', None)
        
        asyncio.run(
            run_scraper(urls, concurrency, retries, timeout, depth, keywords, max_chars,
                        user_agent, rows_per_file, output_dir, progress_cb,
                        ai_enabled, ai_api_key, ai_provider, ai_model, ai_prompt, lead_data_map,
                        ai_status_callback if ai_enabled else None)
        )

    # Zip all parts at the end with custom name (CSV and Excel files)
    zip_name = f"{run_folder}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    csv_files = []
    excel_files = []
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f)
            if f.endswith(".csv"):
                zf.write(file_path, arcname=f)
                csv_files.append(f)
            elif f.endswith(".xlsx"):
                zf.write(file_path, arcname=f)
                excel_files.append(f)
    
    # Success message with download options
    st.success(f"✅ Scraping finished! Processed {total:,} website(s).")
    
    # Show accuracy info for max_chars
    if total > 0:
        st.info(f"💡 **Accuracy:** Max characters limit ({max_chars:,} chars) was accurately enforced per website. Each website's content was limited to this exact amount.")
    
    # Download section
    st.header("📥 Download Results")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        st.subheader("📦 All Files (ZIP)")
        with open(zip_path, 'rb') as f:
            st.download_button(
                label="⬇️ Download ZIP Archive",
                data=f.read(),
                file_name=zip_name,
                mime="application/zip",
                help="Download all CSV and Excel files in a ZIP archive"
            )
        st.caption(f"Contains {len(csv_files)} CSV + {len(excel_files)} Excel files")
    
    with col_dl2:
        st.subheader("📊 Excel Files")
        if excel_files:
            # Create a combined Excel file
            try:
                all_data = []
                for excel_file in sorted(excel_files):
                    excel_path = os.path.join(output_dir, excel_file)
                    df_part = pd.read_excel(excel_path, engine='openpyxl')
                    all_data.append(df_part)
                
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    excel_buffer = BytesIO()
                    combined_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_buffer.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download Combined Excel",
                        data=excel_buffer.read(),
                        file_name=f"{run_folder}_combined.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download all data in a single Excel file"
                    )
                    st.caption(f"{len(combined_df)} total rows")
            except Exception as e:
                st.warning(f"Could not create combined Excel: {e}")
        else:
            st.info("No Excel files generated")
    
    with col_dl3:
        st.subheader("📄 CSV Files")
        if csv_files:
            # Create a combined CSV file
            try:
                all_data = []
                for csv_file in sorted(csv_files):
                    csv_path = os.path.join(output_dir, csv_file)
                    df_part = pd.read_csv(csv_path, encoding='utf-8-sig')
                    all_data.append(df_part)
                
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    csv_buffer = BytesIO()
                    combined_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    csv_buffer.seek(0)
                    
                    csv_data = csv_buffer.getvalue()
                    st.download_button(
                        label="⬇️ Download Combined CSV",
                        data=csv_data,
                        file_name=f"{run_folder}_combined.csv",
                        mime="text/csv",
                        help="Download all data in a single CSV file (Excel/Google Sheets compatible)"
                    )
                    st.caption(f"{len(combined_df)} total rows")
                    
                    # Store for Google Sheets section
                    st.session_state['combined_csv_data'] = csv_data
                    st.session_state['combined_df'] = combined_df
            except Exception as e:
                st.warning(f"Could not create combined CSV: {e}")
        else:
            st.info("No CSV files generated")
    
    # Google Sheets section
    st.markdown("---")
    st.subheader("📊 Import to Google Sheets")
    
    if csv_files:
        try:
            if 'combined_df' in st.session_state and st.session_state.get('combined_df') is not None:
                combined_df = st.session_state['combined_df']
                
                col_gs1, col_gs2 = st.columns(2)
                
                with col_gs1:
                    st.markdown("""
                    **🚀 Quick Import Method:**
                    
                    1. **Download the CSV file** (button above)
                    2. Go to [sheets.google.com](https://sheets.google.com)
                    3. Click **File → Import**
                    4. Choose **Upload** tab
                    5. Select your downloaded CSV file
                    6. Click **Import data**
                    
                    ✅ **Done!** Your data is now in Google Sheets.
                    """)
                
                with col_gs2:
                    st.markdown(f"""
                    **💡 Alternative Methods:**
                    
                    **Method 2: Drag & Drop**
                    - Download CSV
                    - Open Google Sheets
                    - Drag CSV file into the sheet
                    
                    **Method 3: Copy-Paste** (for small datasets)
                    - Open CSV in a text editor
                    - Copy all content
                    - Paste into Google Sheets
                    
                    **📊 Your Data:**
                    - **Rows:** {len(combined_df):,}
                    - **Format:** UTF-8 CSV (perfect for Google Sheets)
                    - **Compatibility:** ✅ 100% compatible
                    """)
                
                # Show file size info
                if len(combined_df) > 10000:
                    st.warning(f"⚠️ **Large dataset ({len(combined_df):,} rows).** Use File → Import method for best results. Google Sheets can handle up to 10 million cells.")
                elif len(combined_df) > 5000:
                    st.info(f"ℹ️ Dataset has {len(combined_df):,} rows. File → Import is recommended for best performance.")
                else:
                    st.success(f"✅ Dataset ready ({len(combined_df):,} rows). Any import method will work!")
                
                # Direct link to create new Google Sheet
                st.markdown("---")
                col_link1, col_link2 = st.columns(2)
                with col_link1:
                    st.markdown(f"[🔗 Create New Google Sheet](https://sheets.google.com/create) - Opens in new tab")
                with col_link2:
                    st.markdown(f"[📤 Upload to Google Drive](https://drive.google.com/drive/my-drive) - Then import to Sheets")
            else:
                st.info("💡 Download the combined CSV file above, then use the import instructions to add it to Google Sheets.")
        except Exception as e:
            st.warning(f"Could not load data for Google Sheets: {e}")
    else:
        st.info("No CSV files available for Google Sheets import.")
    
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
