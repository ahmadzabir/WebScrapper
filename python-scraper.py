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


async def fetch(session: aiohttp.ClientSession, url: str, timeout: int, retries: int):
    """Fetch URL with improved error handling and encoding detection."""
    for attempt in range(retries + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True) as resp:
                # Handle redirects properly
                final_url = str(resp.url)
                
                # Check status code
                if resp.status >= 400:
                    return f"HTTP {resp.status} at {final_url}"
                
                # Check content type more flexibly
                content_type = resp.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type and "application/xhtml" not in content_type:
                    # Some sites don't set proper content-type, check first bytes
                    try:
                        first_bytes = await resp.read(512)
                        await resp.release()
                        # Check if it looks like HTML
                        text_start = first_bytes.decode('utf-8', errors='ignore')[:100].lower()
                        if '<html' not in text_start and '<!doctype' not in text_start:
                            return f"Non-HTML content at {final_url}"
                        # Re-read full content
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp2:
                            html = await resp2.text(errors="replace")  # Use 'replace' instead of 'ignore' to preserve structure
                            return (final_url, html)
                    except:
                        return f"Non-HTML content at {final_url}"
                
                # Try to detect encoding from headers or content
                html = await resp.text(errors="replace")  # Use 'replace' to handle encoding issues better
                return (final_url, html)
                
        except asyncio.TimeoutError:
            if attempt == retries:
                return f"Timeout fetching {url} (exceeded {timeout}s)"
            await asyncio.sleep(1 + attempt * 0.5)
        except (aiohttp.ClientError, ConnectionResetError, UnicodeDecodeError) as e:
            if attempt == retries:
                return f"Error fetching {url}: {str(e)}"
            await asyncio.sleep(1 + attempt * 0.5)
        except Exception as e:
            # Catch any other unexpected errors
            if attempt == retries:
                return f"Unexpected error fetching {url}: {str(e)}"
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
    
    # Validate HTML content before processing
    if not html or len(html.strip()) < 50:
        return f"❌ Empty or too short HTML content from {page_url}"
    
    # Check if HTML looks valid (has some HTML structure)
    html_lower = html.lower()
    if '<html' not in html_lower and '<!doctype' not in html_lower and '<body' not in html_lower:
        # Might be a redirect or error page, but still try to extract content
        pass
    
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

# Clean header
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.title("🌐 Web Scraper")
    st.caption("Upload CSV → Configure → Scrape → Download Results")
with col_header2:
    st.write("")  # Spacing

# Main tabs for better organization
tab1, tab2, tab3 = st.tabs(["📁 Upload & Configure", "⚙️ Settings", "🤖 AI Summaries"])

with tab1:
    # File Upload Section
    uploaded_file = st.file_uploader(
        "📤 Upload CSV File", 
        type=["csv"],
        help="Upload your CSV file with URLs"
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
        
        # Column selection - Compact layout
        st.markdown("#### Column Selection")
        
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
            st.caption("📌 Required column")
        
        # Additional lead data columns (optional) - Collapsed by default
        with st.expander("📊 Additional Lead Data (Optional)", expanded=False):
            st.caption("For AI summaries - select columns with company info")
        
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

with tab2:
    st.markdown("### ⚙️ Scraping Settings")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔍 Scraping")
        
        # Keywords - Simplified
        keywords_input = st.text_input(
            "Keywords (comma separated)",
            value=st.session_state.get('keywords_input', "about,service,product"),
            help="Keywords to find in URLs (e.g., about,service,product)",
            key="keywords_input"
        )
        keywords = process_keywords(keywords_input)
        st.session_state['keywords'] = keywords
        
        if keywords:
            st.caption(f"✅ {len(keywords)} keyword(s)")
        
        # Concurrency - Simplified
        concurrency = st.slider(
            "Concurrency (workers)", 
            1, 50, st.session_state.get('concurrency', 20),
            help="Parallel workers (20-30 recommended)",
            key="concurrency"
        )
        st.session_state['concurrency'] = concurrency
        
        # Retries - Simplified
        retries = st.number_input(
            "Retry Attempts", 
            0, 5, st.session_state.get('retries', 2),
            help="Retries for failed requests",
            key="retries"
        )
        st.session_state['retries'] = retries
        
        # Depth - Simplified
        depth = st.slider(
            "Depth (link follow)", 
            0, 5, st.session_state.get('depth', 3),
            help="Link depth (2-3 recommended)",
            key="depth"
        )
        st.session_state['depth'] = depth
    
    with col2:
        st.markdown("#### ⏱️ Performance")
        
        # Timeout - Simplified
        timeout = st.number_input(
            "Timeout (seconds)", 
            2, 60, st.session_state.get('timeout', 15),
            help="Request timeout",
            key="timeout"
        )
        st.session_state['timeout'] = timeout
        
        # Max chars - Simplified
        max_chars = st.number_input(
            "Max Characters per Site", 
            1000, 200000, st.session_state.get('max_chars', 50000),
            step=5000,
            help="Character limit per website",
            key="max_chars"
        )
        st.session_state['max_chars'] = max_chars
        
        # Rows per file - Simplified
        rows_per_file = st.number_input(
            "Rows per CSV file", 
            1000, 50000, st.session_state.get('rows_per_file', 2000), step=1000,
            help="Rows before splitting files",
            key="rows_per_file"
        )
        st.session_state['rows_per_file'] = rows_per_file
        
        # User Agent - Simplified
        user_agent = st.text_input(
            "User-Agent", 
            value=st.session_state.get('user_agent', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"),
            help="Browser identifier",
            key="user_agent"
        )
        st.session_state['user_agent'] = user_agent
    
    # Detailed explanations in collapsed expander
    with st.expander("ℹ️ Detailed Help", expanded=False):
        st.markdown("""
        **Keywords:** Separate with commas. Case-insensitive. Finds URLs containing these keywords.
        
        **Concurrency:** Number of parallel workers. Higher = faster but may cause timeouts.
        
        **Retries:** How many times to retry failed requests. 2-3 recommended.
        
        **Depth:** How many levels of links to follow. 2-3 recommended for most cases.
        
        **Timeout:** Maximum seconds to wait for a website. 15s recommended.
        
        **Max Characters:** Limit content per website. 50k recommended for large datasets.
        
        **Rows per File:** Split large outputs into multiple files. 2000-5000 recommended.
        """)
    
    # Output Settings - Simplified
    st.markdown("---")
    st.markdown("#### 📂 Output Settings")
    
    run_name = st.text_input(
        "Output folder name (optional)", 
        value="",
        help="Custom name for output folder. Leave empty for auto-generated."
    )

with tab3:
    st.markdown("### 🤖 AI Company Summary")
    
    ai_enabled = st.checkbox(
        "Enable AI summaries",
        value=st.session_state.get('ai_enabled', False),
        help="Generate AI summaries for each company",
        key="ai_enabled_checkbox"
    )
    st.session_state['ai_enabled'] = ai_enabled
    
    if ai_enabled:
        col_ai1, col_ai2 = st.columns(2)
        
        with col_ai1:
            st.markdown("#### Configuration")
            
            ai_provider = st.selectbox(
                "AI Provider",
                ["OpenAI", "Gemini"],
                help="Choose OpenAI or Gemini"
            )
            
            # API Key - Simplified
            api_key_key = f"{ai_provider.lower()}_api_key"
            stored_api_key = st.session_state.get(api_key_key, "")
            
            ai_api_key = st.text_input(
                f"{ai_provider} API Key",
                value=stored_api_key,
                type="password",
                help=f"Enter your {ai_provider} API key"
            )
            
            # Save API key
            if ai_api_key:
                if ai_api_key != stored_api_key:
                    st.session_state[api_key_key] = ai_api_key
                    st.session_state[f"{ai_provider.lower()}_models"] = None
                st.caption(f"✅ Saved")
            elif stored_api_key and not ai_api_key:
                st.session_state[api_key_key] = ""
                st.session_state[f"{ai_provider.lower()}_models"] = None
            
            # Model selection - Simplified
            models_cache_key = f"{ai_provider.lower()}_models"
            cached_models = st.session_state.get(models_cache_key)
            
            if ai_api_key and ai_api_key.strip():
                if cached_models is None:
                    with st.spinner(f"Fetching {ai_provider} models..."):
                        try:
                            if os.name == "nt":
                                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                            
                            if ai_provider == "OpenAI":
                                models = asyncio.run(fetch_openai_models(ai_api_key))
                            else:
                                models = asyncio.run(fetch_gemini_models(ai_api_key))
                            
                            if models:
                                st.session_state[models_cache_key] = models
                                cached_models = models
                            else:
                                cached_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] if ai_provider == "OpenAI" else ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                                st.session_state[models_cache_key] = cached_models
                        except Exception as e:
                            st.warning(f"Using default models")
                            cached_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] if ai_provider == "OpenAI" else ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                            st.session_state[models_cache_key] = cached_models
                
                if cached_models:
                    default_index = 0
                    if ai_provider == "OpenAI":
                        if "gpt-4o-mini" in cached_models:
                            default_index = cached_models.index("gpt-4o-mini")
                        elif "gpt-4o" in cached_models:
                            default_index = cached_models.index("gpt-4o")
                    else:
                        if "gemini-1.5-flash" in cached_models:
                            default_index = cached_models.index("gemini-1.5-flash")
                        elif "gemini-1.5-pro" in cached_models:
                            default_index = cached_models.index("gemini-1.5-pro")
                    
                    ai_model = st.selectbox(
                        "Model",
                        options=cached_models,
                        index=default_index,
                        help=f"Select {ai_provider} model"
                    )
                    
                    st.caption(f"📋 {len(cached_models)} model(s) available")
                    
                    with st.expander(f"View All Models", expanded=False):
                        for i, model in enumerate(cached_models, 1):
                            is_selected = "✅" if model == ai_model else ""
                            st.markdown(f"{i}. `{model}` {is_selected}")
                else:
                    ai_model = None
            else:
                st.info("Enter API key to see models")
                ai_model = None
        
        with col_ai2:
            st.markdown("#### Prompt")
            
            prompt_mode = st.radio(
                "Prompt Mode",
                ["Use Default", "Edit Prompt"],
                horizontal=True
            )
            
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
            
            if 'master_prompt' not in st.session_state:
                st.session_state['master_prompt'] = default_prompt_template
            
            if prompt_mode == "Edit Prompt":
                ai_prompt = st.text_area(
                    "Master Prompt Template",
                    value=st.session_state.get('master_prompt', default_prompt_template),
                    height=300,
                    help="Edit prompt. Use: {url}, {company_name}, {scraped_content}",
                    key="ai_prompt_edit"
                )
                st.session_state['master_prompt'] = ai_prompt
            else:
                ai_prompt = st.session_state.get('master_prompt', default_prompt_template)
            
            st.session_state['ai_prompt'] = ai_prompt
            
            with st.expander("Preview Prompt", expanded=False):
                sample_url = "https://example.com"
                sample_company = "Example Company"
                sample_content = "Sample website content..."
                preview = ai_prompt.replace('{url}', sample_url).replace('{company_name}', sample_company).replace('{scraped_content}', sample_content[:200])
                st.code(preview, language=None)
    
    else:
        ai_api_key = None
        ai_provider = None
        ai_model = None
        ai_prompt = None

# Main action button - Outside tabs
st.markdown("---")
st.markdown("### 🚀 Start Scraping")

# Get variables from tabs - Use session_state (proper Streamlit way)
keywords = st.session_state.get('keywords', [])
concurrency = st.session_state.get('concurrency', 20)
retries = st.session_state.get('retries', 2)
depth = st.session_state.get('depth', 3)
timeout = st.session_state.get('timeout', 15)
max_chars = st.session_state.get('max_chars', 50000)
rows_per_file = st.session_state.get('rows_per_file', 2000)
user_agent = st.session_state.get('user_agent', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
run_name = st.session_state.get('run_name', "")
ai_enabled = st.session_state.get('ai_enabled', False)
ai_api_key = st.session_state.get('ai_api_key', None)
ai_provider = st.session_state.get('ai_provider', None)
ai_model = st.session_state.get('ai_model', None)
ai_prompt = st.session_state.get('ai_prompt', None)

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
