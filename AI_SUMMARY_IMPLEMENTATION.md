# 🤖 AI Company Summary Feature - Implementation Plan

## Overview
Adding AI-powered company summary generation that runs in parallel with web scraping.

## Architecture

### Data Flow:
1. **Scrape website** → Get scraped content
2. **Generate AI summary** → Use scraped content + lead data → Get company summary
3. **Output**: Website | ScrapedText | CompanySummary

### Components:

1. **AI Functions** ✅ (Added)
   - `generate_openai_summary()` - OpenAI API
   - `generate_gemini_summary()` - Gemini API
   - `generate_ai_summary()` - Unified interface
   - `build_company_summary_prompt()` - Prompt builder

2. **Worker Modification** (In Progress)
   - Modify `worker_coroutine()` to generate summaries after scraping
   - Return tuple: (url, scraped_text, ai_summary)

3. **Writer Modification** (In Progress)
   - Update to handle 3 columns: Website, ScrapedText, CompanySummary

4. **UI Elements** (In Progress)
   - API key input (password field)
   - Provider selection (OpenAI/Gemini)
   - Model selection dropdown
   - Prompt input (with default)

## Current Status

✅ AI functions added
⏳ Worker modification needed
⏳ Writer modification needed
⏳ UI elements needed

## Next Steps

1. Modify worker to generate summaries
2. Update writer to handle new column
3. Add UI for AI settings
4. Test with sample data
