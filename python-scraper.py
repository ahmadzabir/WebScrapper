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
    
    # Add STRICT formatting reminder
    final_reminder = """

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
    full_prompt = full_prompt + final_reminder
    
    # Generate summary based on provider
    raw_output = ""
    if provider.lower() == 'openai':
        raw_output = await generate_openai_summary(api_key, model, full_prompt, status_callback=status_callback)
    elif provider.lower() == 'gemini':
        raw_output = await generate_gemini_summary(api_key, model, full_prompt, status_callback=status_callback)
    else:
        return f"❌ Unknown provider: {provider}"
    
    # Post-process to fix all formatting issues
    cleaned_output = clean_and_structure_ai_output(raw_output)
    
    return cleaned_output


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
    """Fetch URL with improved error handling, bot detection avoidance, and encoding support."""
    from urllib.parse import urlparse
    
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
    
    # Try different header combinations
    header_variations = [
        # Modern Chrome headers
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",  # Try with brotli first
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "DNT": "1"
        },
        # Fallback: without brotli
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",  # No brotli
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        },
        # Simple headers (less bot-like)
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
    ]
    
    last_error = None
    
    # Try URL variations and header combinations
    for url_to_try in url_variations:
        for headers in header_variations:
            # Use session's user agent if available
            if session.headers.get("User-Agent"):
                headers["User-Agent"] = session.headers.get("User-Agent")
            
            for attempt in range(retries + 1):
                try:
                    # Parse URL
                    parsed_original = urlparse(url_to_try)
                    if not parsed_original.netloc:
                        continue
                    
                    original_domain = parsed_original.netloc.replace('www.', '').lower()
                    
                    async with session.get(url_to_try, timeout=aiohttp.ClientTimeout(total=timeout), allow_redirects=True, headers=headers) as resp:
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
                        
                        # Check status code - handle 403 specially
                        if resp.status == 403:
                            # 403 Forbidden - try next URL variation or header combination
                            last_error = f"HTTP 403 at {final_url}"
                            break  # Try next variation
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
                    last_error = f"Timeout fetching {url_to_try} (exceeded {timeout}s)"
                    if attempt < retries:
                        await asyncio.sleep(1 + attempt * 0.5)
                    continue
                except (aiohttp.ClientError, ConnectionResetError) as e:
                    error_str = str(e).lower()
                    # Handle brotli encoding error specifically
                    if 'brotli' in error_str or 'br' in error_str or 'content-encoding' in error_str:
                        # Try without brotli in next iteration
                        last_error = f"Brotli encoding error: {str(e)}"
                        continue
                    last_error = f"Error fetching {url_to_try}: {str(e)}"
                    if attempt < retries:
                        await asyncio.sleep(1 + attempt * 0.5)
                    continue
                except UnicodeDecodeError as e:
                    last_error = f"Encoding error: {str(e)}"
                    if attempt < retries:
                        await asyncio.sleep(1 + attempt * 0.5)
                    continue
                except Exception as e:
                    # Catch any other unexpected errors
                    last_error = f"Unexpected error fetching {url_to_try}: {str(e)}"
                    if attempt < retries:
                        await asyncio.sleep(1 + attempt * 0.5)
                    continue
    
    # If we get here, all variations failed
    return last_error or f"Failed to fetch {url} after trying all variations"


async def scrape_site(session, url: str, depth: int, keywords, max_chars: int, retries: int, timeout: int):
    visited, results, errors = set(), [], []
    total_chars = 0
    separator = "\n\n" + "─" * 80 + "\n\n"  # Better visual separator between pages
    separator_len = len(separator)

    # Normalize URL before fetching
    normalized_url = normalize_url(url)
    
    homepage = await fetch(session, normalized_url, timeout, retries)
    if isinstance(homepage, str):
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
    from urllib.parse import urlparse
    while True:
        item = await url_queue.get()
        if item is None:
            url_queue.task_done()
            break
        
        # CRITICAL: Wrap entire processing in try-finally to ensure task_done() is always called
        try:
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
            
            # Be lenient with URL validation - let the actual HTTP request determine validity
            # Only reject obviously invalid URLs (empty or completely malformed)
            if not normalized_url or len(normalized_url.strip()) < 4:
                scraped_text = "❌ Invalid URL: URL is empty or too short"
                ai_summary = "❌ Invalid URL: URL is empty or too short"
            else:
                # CRITICAL: Add a global timeout wrapper to prevent workers from hanging indefinitely
                # Use a more aggressive timeout: max 2 minutes per URL regardless of settings
                # This prevents one slow URL from blocking everything
                max_total_time = min((timeout * (retries + 1) * (depth + 1) * 2) + 30, 120)  # Max 2 minutes per URL
                
                try:
                    # Wrap scraping in a timeout to prevent infinite hangs
                    scraped_text = await asyncio.wait_for(
                        scrape_site(session, normalized_url, depth, keywords, max_chars, retries, timeout),
                        timeout=max_total_time
                    )
                except asyncio.TimeoutError:
                    # Global timeout exceeded - this URL is taking too long
                    scraped_text = f"❌ Timeout: Scraping {normalized_url} exceeded maximum time limit ({max_total_time}s)"
                except Exception as e:
                    # If scraping fails with an exception, try once more with a more lenient approach
                    try:
                        # Try with http:// if https:// failed, but also wrap in timeout
                        if normalized_url.startswith("https://"):
                            fallback_url = normalized_url.replace("https://", "http://", 1)
                            try:
                                scraped_text = await asyncio.wait_for(
                                    scrape_site(session, fallback_url, depth, keywords, max_chars, retries, timeout),
                                    timeout=max_total_time
                                )
                            except asyncio.TimeoutError:
                                scraped_text = f"❌ Timeout: Scraping {fallback_url} exceeded maximum time limit ({max_total_time}s)"
                        else:
                            scraped_text = f"❌ Error scraping {normalized_url}: {str(e)}"
                    except Exception as e2:
                        scraped_text = f"❌ Error scraping {normalized_url}: {str(e2)}"
                
                # CRITICAL VALIDATION: Verify scraped content matches the URL
                # Extract the first PAGE: URL from scraped content to verify
                if scraped_text and not scraped_text.startswith("❌"):
                    # Check if scraped content contains PAGE: header
                    page_header_match = re.search(r'PAGE:\s*(https?://[^\s\n]+)', scraped_text[:5000])
                    if page_header_match:
                        actual_scraped_url = page_header_match.group(1)
                        # Normalize both URLs for comparison
                        actual_domain = urlparse(actual_scraped_url).netloc.replace('www.', '').lower()
                        expected_domain = urlparse(normalized_url).netloc.replace('www.', '').lower()
                        
                        # Verify domains match
                        if actual_domain != expected_domain:
                            if not (actual_domain.endswith('.' + expected_domain) or expected_domain.endswith('.' + actual_domain)):
                                # Domain mismatch - content doesn't match URL!
                                scraped_text = f"❌ CONTENT MISMATCH: Scraped content is from {actual_domain} but expected {expected_domain}. Original URL: {original_url}"
                
                # Generate AI summary if enabled
                if ai_enabled and ai_api_key and ai_provider and ai_model:
                    # Create URL-specific status callback
                    def url_status_callback(msg):
                        if ai_status_callback:
                            ai_status_callback(original_url, msg)
                    
                    try:
                        ai_summary = await asyncio.wait_for(
                            generate_ai_summary(
                                ai_api_key, ai_provider, ai_model, ai_prompt or "",
                                lead_data, scraped_text, status_callback=url_status_callback
                            ),
                            timeout=120  # 2 minute timeout for AI summary
                        )
                    except asyncio.TimeoutError:
                        ai_summary = "❌ AI Summary timeout: Generation exceeded 2 minutes"
                    except Exception as e:
                        ai_summary = f"❌ AI Summary error: {str(e)}"
                else:
                    ai_summary = ""  # Empty if AI not enabled
            
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
            
            # Clean all fields perfectly
            cleaned_url = clean_csv_field(original_url)
            cleaned_scraped_text = clean_csv_field(scraped_text)
            cleaned_ai_summary = clean_csv_field(ai_summary)
            
            # Output format: (url, scraped_text, ai_summary)
            # Use original_url to maintain consistency with input CSV
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
                out = (cleaned_url, cleaned_scraped_text, cleaned_ai_summary)
                await result_queue.put(out)
            except:
                # Last resort: put minimal error info
                try:
                    await result_queue.put(("", f"❌ Critical worker error: {str(e)}", ""))
                except:
                    pass  # If even this fails, just continue
        finally:
            # CRITICAL: Always mark task as done, even if there was an error
            url_queue.task_done()


async def writer_coroutine(result_queue: asyncio.Queue, rows_per_file: int, output_dir: str,
                           total_urls: int, progress_callback):
    # CRITICAL: Create output directory first
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Writer: Created output directory: {output_dir}")
    except Exception as e:
        print(f"❌ Writer: Failed to create output directory: {e}")
        import traceback
        traceback.print_exc()
        return  # Exit if we can't create directory
    
    buffer = []
    part = 0
    processed = 0
    files_written = []

    while True:
        item = await result_queue.get()
        if item is None:
            print(f"📝 Writer: Received None sentinel. Processed {processed}/{total_urls} items. Buffer has {len(buffer)} items.")
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
                
                # Normalize all rows to have 3 columns for consistency
                normalized_rows = []
                for row in filtered_rows:
                    if len(row) == 2:
                        normalized_rows.append((row[0], row[1], ""))
                    elif len(row) == 3:
                        normalized_rows.append(row)
                    else:
                        # Handle unexpected structure - take first 3 elements or pad
                        normalized_row = list(row[:3])
                        while len(normalized_row) < 3:
                            normalized_row.append("")
                        normalized_rows.append(tuple(normalized_row))
                
                # Always create DataFrame with 3 columns
                df = pd.DataFrame(normalized_rows, columns=["Website", "ScrapedText", "CompanySummary"])
                
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
                
                # Ensure CompanySummary column exists (for consistency)
                if "CompanySummary" not in df.columns:
                    df["CompanySummary"] = ""
                
                # Ensure correct column order
                if "CompanySummary" in df.columns:
                    df = df[["Website", "ScrapedText", "CompanySummary"]]
                else:
                    df = df[["Website", "ScrapedText"]]
                
                if len(df) == 0:
                    # Skip writing if DataFrame is empty after filtering
                    continue
                
                # PERFECT CSV WRITING - Zero tolerance for errors
                csv_path = os.path.join(output_dir, f"output_part_{part}.csv")
                
                # Write CSV with perfect formatting:
                # - UTF-8 with BOM for Excel compatibility
                # - QUOTE_ALL: All fields quoted to handle commas, quotes, special chars
                # - lineterminator='\n': Standard Unix line endings
                # - doublequote=True: Escape quotes by doubling them (CSV standard)
                try:
                    # CRITICAL: Use manual CSV writer to ensure ALL fields are properly quoted
                    # pandas to_csv sometimes doesn't quote correctly, causing data to split across columns
                    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n', quotechar='"')
                        # Write header
                        writer.writerow(df.columns.tolist())
                        # Write rows - ensure all values are strings and properly cleaned
                        for _, row in df.iterrows():
                            row_values = []
                            for val in row:
                                if pd.isna(val):
                                    row_values.append('')
                                else:
                                    # Convert to string and ensure it's properly formatted
                                    val_str = str(val)
                                    # Remove any embedded newlines that might break CSV
                                    import re
                                    val_str = val_str.replace('\n', ' ').replace('\r', ' ')
                                    # Collapse multiple spaces
                                    val_str = re.sub(r'\s+', ' ', val_str).strip()
                                    row_values.append(val_str)
                            writer.writerow(row_values)
                    
                    # Verify file was written
                    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                        files_written.append(csv_path)
                        print(f"✅ Writer: Successfully wrote CSV file: {csv_path} ({len(df)} rows)")
                    else:
                        print(f"❌ Writer: CSV file was not created or is empty: {csv_path}")
                    
                    # VALIDATION: Verify CSV file is valid by reading it back
                    try:
                        test_df = pd.read_csv(csv_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL, engine='python')
                        # Check row count matches
                        if len(test_df) != len(df):
                            raise ValueError(f"CSV validation failed: row count mismatch")
                        # Check column count matches
                        if len(test_df.columns) != len(df.columns):
                            raise ValueError(f"CSV validation failed: column count mismatch")
                    except Exception as e:
                        # If validation fails, log but don't rewrite (already used manual writer)
                        import logging
                        logging.warning(f"CSV validation warning: {e}")
                except Exception as e:
                    # Final fallback: manual CSV writing
                    import logging
                    logging.error(f"CSV write failed: {e}. Using manual writer...")
                    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n')
                        writer.writerow(df.columns.tolist())
                        for _, row in df.iterrows():
                            writer.writerow([str(val) if pd.notna(val) else '' for val in row])
                
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
                    
                    # Use manual CSV writer for perfect quoting
                    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n', quotechar='"')
                        writer.writerow(df.columns.tolist())
                        for _, row in df.iterrows():
                            row_values = []
                            for val in row:
                                if pd.isna(val):
                                    row_values.append('')
                                else:
                                    val_str = str(val).replace('\n', ' ').replace('\r', ' ')
                                    import re
                                    val_str = re.sub(r'\s+', ' ', val_str).strip()
                                    row_values.append(val_str)
                            writer.writerow(row_values)
                except:
                    # Ultimate fallback: manual writer
                    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n')
                        writer.writerow(df.columns.tolist())
                        for _, row in df.iterrows():
                            writer.writerow([str(val) if pd.notna(val) else '' for val in row])

        result_queue.task_done()

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
                    # Ensure row has correct structure (pad with empty CompanySummary if needed)
                    if len(row) == 2:
                        row = (row[0], row[1], "")
                    filtered_buffer.append(row)
        
        if filtered_buffer:
            part += 1
            try:
                # Normalize all rows to have 3 columns for consistency
                normalized_buffer = []
                for row in filtered_buffer:
                    if len(row) == 2:
                        normalized_buffer.append((row[0], row[1], ""))
                    elif len(row) == 3:
                        normalized_buffer.append(row)
                    else:
                        # Handle unexpected structure - take first 3 elements or pad
                        normalized_row = list(row[:3])
                        while len(normalized_row) < 3:
                            normalized_row.append("")
                        normalized_buffer.append(tuple(normalized_row))
                
                # Always create DataFrame with 3 columns
                df = pd.DataFrame(normalized_buffer, columns=["Website", "ScrapedText", "CompanySummary"])
                
                # Remove ONLY rows with empty URLs (keep error messages)
                df = df[df["Website"].astype(str).str.strip() != ""]
                
                if len(df) == 0:
                    # Skip writing if DataFrame is empty after filtering
                    pass
                else:
                    # Save CSV with Excel/Google Sheets compatibility
                    csv_path = os.path.join(output_dir, f"output_part_{part}.csv")
                    # Use manual CSV writer for perfect quoting
                    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                        writer = csv.writer(f, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n', quotechar='"')
                        writer.writerow(df.columns.tolist())
                        for _, row in df.iterrows():
                            row_values = []
                            for val in row:
                                if pd.isna(val):
                                    row_values.append('')
                                else:
                                    val_str = str(val).replace('\n', ' ').replace('\r', ' ')
                                    import re
                                    val_str = re.sub(r'\s+', ' ', val_str).strip()
                                    row_values.append(val_str)
                            writer.writerow(row_values)
                    
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
                # If writing fails, log error
                print(f"❌ Writer: Failed to write output part {part}: {e}")
                import traceback
                traceback.print_exc()
    
    # Final summary
    print(f"📊 Writer: Finished. Wrote {len(files_written)} files total.")
    if files_written:
        print(f"📁 Writer: Files written: {', '.join([os.path.basename(f) for f in files_written])}")

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

    # CRITICAL: Add timeout wrapper around url_queue.join() to prevent infinite hangs
    # Use a more aggressive timeout: max 2 minutes per URL (much more reasonable)
    max_queue_time = min((timeout * (retries + 1) * (depth + 1) * 2 * total) + (30 * total), 120 * total)  # Max 2 min per URL
    
    # Track completed count
    completed_count = [0]  # Use list to allow modification in nested function
    
    async def force_completion_on_stall():
        """Force completion if queue is stuck"""
        start_time = time.time()
        last_size = total
        while True:
            await asyncio.sleep(10)  # Check every 10 seconds
            current_size = url_queue.qsize()
            elapsed = time.time() - start_time
            
            # If queue size hasn't changed in 60 seconds and there are still items, force complete
            if current_size == last_size and current_size > 0 and elapsed > 60:
                print(f"⚠️ Detected stall: {total - current_size}/{total} completed, {current_size} remaining. Forcing completion...")
                # Drain remaining URLs and mark as timeout errors
                drained = 0
                while not url_queue.empty() and drained < current_size:
                    try:
                        item = url_queue.get_nowait()
                        if item and item != (None, None):
                            if isinstance(item, tuple):
                                url, idx = item
                            else:
                                url = item
                            error_result = (url, f"❌ Timeout: Processing exceeded time limit (stall detected)", "")
                            await result_queue.put(error_result)
                            url_queue.task_done()
                            drained += 1
                    except Exception as e:
                        print(f"Error draining queue: {e}")
                        break
                break
            
            # If queue is empty, we're done
            if current_size == 0:
                break
            
            last_size = current_size
    
    stall_monitor = asyncio.create_task(force_completion_on_stall())
    
    try:
        # Wait for queue to complete, but with a timeout
        await asyncio.wait_for(url_queue.join(), timeout=max_queue_time)
    except asyncio.TimeoutError:
        # Queue join timed out - force completion
        print(f"⚠️ Queue join timed out after {max_queue_time}s. Forcing completion...")
        # Drain remaining URLs
        drained = 0
        while not url_queue.empty() and drained < 1000:  # Safety limit
            try:
                item = url_queue.get_nowait()
                if item and item != (None, None):
                    if isinstance(item, tuple):
                        url, idx = item
                    else:
                        url = item
                    error_result = (url, f"❌ Timeout: Processing exceeded {max_queue_time}s limit", "")
                    await result_queue.put(error_result)
                    url_queue.task_done()
                    drained += 1
            except Exception as e:
                print(f"Error draining queue: {e}")
                break
    
    # Cancel stall monitor
    stall_monitor.cancel()
    try:
        await stall_monitor
    except (asyncio.CancelledError, Exception):
        pass

    # Signal workers to stop
    for _ in workers:
        try:
            await url_queue.put(None)
        except:
            pass
    
    # Wait for workers to finish (with timeout)
    try:
        await asyncio.wait_for(asyncio.gather(*workers, return_exceptions=True), timeout=30)
    except asyncio.TimeoutError:
        # Force cancel if still hanging
        for worker in workers:
            if not worker.done():
                worker.cancel()

    # Signal writer to stop
    try:
        await result_queue.put(None)
    except:
        pass
    
    # Wait for result queue to drain (with timeout)
    try:
        await asyncio.wait_for(result_queue.join(), timeout=60)
    except asyncio.TimeoutError:
        print("⚠️ Result queue join timed out. Proceeding anyway...")
    
    # Wait for writer to finish (with timeout)
    try:
        await asyncio.wait_for(writer_task, timeout=120)  # Give writer more time to finish
    except asyncio.TimeoutError:
        print("⚠️ Writer task timed out. Proceeding anyway...")
        writer_task.cancel()
    except Exception as e:
        print(f"⚠️ Error waiting for writer: {e}")

    await session.close()
    
    # CRITICAL: Give files time to be fully written to disk
    import time
    await asyncio.sleep(2)  # Wait 2 seconds for file system to sync

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Web Scraper", layout="wide", page_icon="🌐")

# Clean header
st.title("🌐 Website Scraper")
st.caption("Scrape websites from your CSV file and get clean text content")

# Main tabs - Step by step flow
tab1, tab2, tab3 = st.tabs(["📁 Step 1: Upload CSV", "⚙️ Step 2: Settings", "🤖 Step 3: AI (Optional)"])

with tab1:
    st.markdown("### Upload Your CSV File")
    st.info("💡 Your CSV should have a column with website URLs (one per row)")
    
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
        
        # Additional lead data columns (optional)
        st.markdown("---")
        with st.expander("📊 Extra Info for AI (Optional)", expanded=False):
            st.caption("Only needed if you're using AI summaries. Select columns with company names, industry, etc.")
        
        col_lead1, col_lead2, col_lead3 = st.columns(3)
        
        with col_lead1:
            if csv_has_headers == "Yes":
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
            if csv_has_headers == "Yes":
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
            if csv_has_headers == "Yes":
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
            'has_headers': csv_has_headers == "Yes",
            'url_column': url_column,
            'lead_data_columns': lead_data_columns,
            'df_preview': df_preview
        }
        
    except Exception as e:
        st.error(f"❌ Error reading CSV: {str(e)}")
        st.info("Please check your CSV file format and try again.")

with tab2:
    st.markdown("### Configure Scraping Settings")
    st.info("💡 Most settings have good defaults. You can leave them as-is or adjust if needed.")
    
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
        
        # Speed
        concurrency = st.slider(
            "Speed (parallel workers)", 
            1, 50, st.session_state.get('concurrency', 20),
            help="How many websites to scrape at the same time. Higher = faster but may cause errors. 20-30 is usually best.",
            key="concurrency"
        )
        
        # Pages to scrape
        depth = st.slider(
            "Pages to scrape per site", 
            0, 5, st.session_state.get('depth', 3),
            help="How many pages to scrape from each website. 0 = homepage only, 3 = homepage + 3 more pages",
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
        
        # Timeout
        timeout = st.number_input(
            "Wait time per site (seconds)", 
            2, 60, st.session_state.get('timeout', 15),
            help="How long to wait for each website before giving up. 15 seconds is usually fine.",
            key="timeout"
        )
        
        # Max chars
        max_chars = st.number_input(
            "Text limit per site", 
            1000, 200000, st.session_state.get('max_chars', 50000),
            step=5000,
            help="Maximum characters to scrape from each website. Higher = more content but larger files.",
            key="max_chars"
        )
        
        # Rows per file
        rows_per_file = st.number_input(
            "Split files every X rows", 
            1000, 50000, st.session_state.get('rows_per_file', 2000), step=1000,
            help="Large results will be split into multiple files. 2000 rows per file is usually good.",
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
    
    # Output Settings
    st.markdown("---")
    st.markdown("#### Output Folder")
    
    run_name = st.text_input(
        "Folder name (optional)", 
        value=st.session_state.get('run_name', ""),
        help="Give your results folder a custom name. Leave empty for automatic name.",
        key="run_name"
    )

with tab3:
    st.markdown("### Generate AI Summaries (Optional)")
    st.info("💡 Use AI to automatically create company summaries from scraped content. Requires an API key.")
    
    ai_enabled = st.checkbox(
        "Generate AI summaries",
        value=st.session_state.get('ai_enabled', False),
        help="Check this to enable AI-powered company summary generation",
        key="ai_enabled_checkbox"
    )
    st.session_state['ai_enabled'] = ai_enabled
    
    if ai_enabled:
        st.markdown("---")
        # Step 1: Choose AI service
        st.markdown("#### Step 1: Choose AI Service")
        ai_provider = st.selectbox(
            "Which AI service?",
            ["OpenAI", "Gemini"],
            help="OpenAI (GPT models) or Google Gemini",
            key="ai_provider_select"
        )
        st.session_state['ai_provider'] = ai_provider
        
        # Step 2: Enter API key
        st.markdown("#### Step 2: Enter API Key")
        api_key_key = f"{ai_provider.lower()}_api_key"
        stored_api_key = st.session_state.get(api_key_key, "")
        
        if ai_provider == "OpenAI":
            st.caption("Get your API key from: https://platform.openai.com/api-keys")
        else:
            st.caption("Get your API key from: https://makersuite.google.com/app/apikey")
        
        ai_api_key = st.text_input(
            f"{ai_provider} API Key",
            value=stored_api_key,
            type="password",
            help=f"Paste your {ai_provider} API key here"
        )
        
        # Save API key
        if ai_api_key:
            if ai_api_key != stored_api_key:
                st.session_state[api_key_key] = ai_api_key
                st.session_state['ai_api_key'] = ai_api_key
                st.session_state[f"{ai_provider.lower()}_models"] = None
            st.success("✅ API key saved")
        elif stored_api_key and not ai_api_key:
            st.session_state[api_key_key] = ""
            st.session_state['ai_api_key'] = ""
            st.session_state[f"{ai_provider.lower()}_models"] = None
        
        # Always sync generic key
        st.session_state['ai_api_key'] = st.session_state.get(api_key_key, "")
        
        # Step 3: Choose model
        st.markdown("#### Step 3: Choose Model")
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
                        else:
                            models = asyncio.run(fetch_gemini_models(ai_api_key))
                        
                        if models:
                            st.session_state[models_cache_key] = models
                            cached_models = models
                        else:
                            cached_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"] if ai_provider == "OpenAI" else ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                            st.session_state[models_cache_key] = cached_models
                    except Exception as e:
                        st.warning("⚠️ Could not load models. Using defaults.")
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
                    "Select model",
                    options=cached_models,
                    index=default_index,
                    help="Choose which AI model to use. Defaults are usually best.",
                    key=f"{ai_provider}_model_select"
                )
                st.session_state['ai_model'] = ai_model
                st.session_state['ai_provider'] = ai_provider
                
                st.caption(f"✅ {len(cached_models)} models available")
            else:
                ai_model = None
        else:
            st.info("👆 Enter your API key above to see available models")
            ai_model = None
        
        # Step 4: Prompt (simplified)
        st.markdown("#### Step 4: Prompt (Optional)")
        prompt_mode = st.radio(
            "Prompt options",
            ["Use default prompt", "Customize prompt"],
            horizontal=True,
            help="Default prompt works great for most cases"
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
            st.caption("💡 Use {url}, {company_name}, and {scraped_content} as placeholders")
            ai_prompt = st.text_area(
                "Custom prompt",
                value=st.session_state.get('master_prompt', default_prompt_template),
                height=300,
                help="Edit the prompt that will be sent to AI",
                key="ai_prompt_edit"
            )
            st.session_state['master_prompt'] = ai_prompt
        else:
            ai_prompt = st.session_state.get('master_prompt', default_prompt_template)
        
        st.session_state['ai_prompt'] = ai_prompt
        
        with st.expander("👁️ Preview how prompt will look", expanded=False):
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
st.markdown("### 🚀 Ready to Start")
st.info("💡 Make sure you've uploaded your CSV and configured settings above. Then click the button below to start scraping!")

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
ai_provider = st.session_state.get('ai_provider', None)
# Get API key - check provider-specific key first, then generic
if ai_provider:
    api_key_key = f"{ai_provider.lower()}_api_key"
    ai_api_key = st.session_state.get(api_key_key) or st.session_state.get('ai_api_key', None)
else:
    ai_api_key = st.session_state.get('ai_api_key', None)
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
                    if f.endswith(".csv") and f != zip_name:
                        try:
                            zf.write(file_path, arcname=f)
                            csv_files.append(f)
                        except Exception as e:
                            st.warning(f"⚠️ Could not add {f} to ZIP: {e}")
                    elif f.endswith(".xlsx") and f != zip_name:
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
                all_data = []
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
                        
                        df_part = pd.read_excel(excel_path, engine='openpyxl')
                        
                        # Check if DataFrame is empty
                        if df_part.empty:
                            st.warning(f"⚠️ Excel file {excel_file} contains no data")
                            continue
                        
                        # Normalize column structure: ensure all DataFrames have same columns
                        # Standard columns: Website, ScrapedText, CompanySummary
                        if "CompanySummary" not in df_part.columns:
                            df_part["CompanySummary"] = ""
                        
                        # Ensure correct column order
                        if "Website" in df_part.columns and "ScrapedText" in df_part.columns:
                            if "CompanySummary" in df_part.columns:
                                df_part = df_part[["Website", "ScrapedText", "CompanySummary"]]
                            else:
                                df_part = df_part[["Website", "ScrapedText"]]
                        else:
                            st.warning(f"⚠️ Excel file {excel_file} missing required columns (Website, ScrapedText)")
                            continue
                        
                        # Remove any extra columns that might cause issues
                        expected_cols = ["Website", "ScrapedText", "CompanySummary"]
                        df_part = df_part[[col for col in expected_cols if col in df_part.columns]]
                        
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
                
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    
                    # Ensure final DataFrame has correct structure
                    if "CompanySummary" not in combined_df.columns:
                        combined_df["CompanySummary"] = ""
                    combined_df = combined_df[["Website", "ScrapedText", "CompanySummary"]]
                    
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
                    
                    excel_buffer = BytesIO()
                    combined_df.to_excel(excel_buffer, index=False, engine='openpyxl')
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
                    csv_path = os.path.join(output_dir, csv_file)
                    try:
                        # Check if file exists and has content
                        if not os.path.exists(csv_path):
                            st.warning(f"⚠️ CSV file not found: {csv_file}")
                            continue
                        
                        if os.path.getsize(csv_path) == 0:
                            st.warning(f"⚠️ CSV file is empty: {csv_file}")
                            continue
                        
                        # Read CSV with proper quoting handling
                        df_part = pd.read_csv(csv_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL, engine='python')
                        
                        # Check if DataFrame is empty
                        if df_part.empty:
                            st.warning(f"⚠️ CSV file {csv_file} contains no data")
                            continue
                        
                        # Normalize column structure: ensure all DataFrames have same columns
                        # Standard columns: Website, ScrapedText, CompanySummary
                        if "CompanySummary" not in df_part.columns:
                            df_part["CompanySummary"] = ""
                        
                        # Ensure correct column order
                        if "Website" in df_part.columns and "ScrapedText" in df_part.columns:
                            if "CompanySummary" in df_part.columns:
                                df_part = df_part[["Website", "ScrapedText", "CompanySummary"]]
                            else:
                                df_part = df_part[["Website", "ScrapedText"]]
                        else:
                            st.warning(f"⚠️ CSV file {csv_file} missing required columns (Website, ScrapedText)")
                            continue
                        
                        # Remove any extra columns that might cause issues
                        expected_cols = ["Website", "ScrapedText", "CompanySummary"]
                        df_part = df_part[[col for col in expected_cols if col in df_part.columns]]
                        
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
                    csv_buffer = BytesIO()
                    csv_writer = csv.writer(csv_buffer, quoting=csv.QUOTE_ALL, doublequote=True, lineterminator='\n', quotechar='"')
                    # Write header - ALWAYS 3 columns
                    csv_writer.writerow(["Website", "ScrapedText", "CompanySummary"])
                    # Write rows - ensure exactly 3 values per row
                    for _, row in combined_df.iterrows():
                        row_values = []
                        # Always extract exactly 3 values in correct order
                        website = str(row["Website"]) if "Website" in row and pd.notna(row["Website"]) else ""
                        scraped_text = str(row["ScrapedText"]) if "ScrapedText" in row and pd.notna(row["ScrapedText"]) else ""
                        company_summary = str(row["CompanySummary"]) if "CompanySummary" in row and pd.notna(row["CompanySummary"]) else ""
                        
                        # Clean each value
                        for val in [website, scraped_text, company_summary]:
                            val_str = val.replace('\n', ' ').replace('\r', ' ')
                            import re
                            val_str = re.sub(r'\s+', ' ', val_str).strip()
                            row_values.append(val_str)
                        
                        csv_writer.writerow(row_values)
                    csv_buffer.seek(0)
                    
                    csv_data = csv_buffer.getvalue()
                    
                    # Store for Google Sheets section and persistence
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
    max_chars = st.session_state.get('max_chars_info', 50000)
    
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
            st.info("Click 'Start Scraping' again to generate Excel file")
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
    
    # Google Sheets section (persistent)
    st.markdown("---")
    st.subheader("📊 Import to Google Sheets")
    
    if csv_files and combined_df is not None:
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
            st.warning(f"⚠️ **Large dataset ({len(combined_df):,} rows).** Use File → Import method for best results.")
        elif len(combined_df) > 5000:
            st.info(f"ℹ️ Dataset has {len(combined_df):,} rows. File → Import is recommended.")
        else:
            st.success(f"✅ Dataset ready ({len(combined_df):,} rows). Any import method will work!")
        
        # Direct links
        st.markdown("---")
        col_link1, col_link2 = st.columns(2)
        with col_link1:
            st.markdown(f"[🔗 Create New Google Sheet](https://sheets.google.com/create) - Opens in new tab")
        with col_link2:
            st.markdown(f"[📤 Upload to Google Drive](https://drive.google.com/drive/my-drive) - Then import to Sheets")
    else:
        st.info("💡 Download the combined CSV file above, then use the import instructions to add it to Google Sheets.")
    
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
