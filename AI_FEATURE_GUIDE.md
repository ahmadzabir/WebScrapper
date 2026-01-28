# 🤖 AI Company Summary Feature - User Guide

## Overview
The app now includes AI-powered company summary generation that runs in parallel with web scraping!

## Features

✅ **Dual AI Support**: OpenAI (GPT models) and Google Gemini  
✅ **Parallel Processing**: Generates summaries while scraping (fast!)  
✅ **Customizable Prompts**: Use default or customize your own  
✅ **Smart Integration**: Automatically includes scraped content + lead data  
✅ **Excel/CSV Compatible**: Summaries included in output files  

---

## How to Use

### Step 1: Enable AI Summaries
1. Check the box: **"Enable AI-powered company summary generation"**
2. Choose your AI provider: **OpenAI** or **Gemini**

### Step 2: Enter API Key
- **OpenAI**: Get key from [platform.openai.com](https://platform.openai.com)
- **Gemini**: Get key from [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- Enter your API key (stored securely, not saved)

### Step 3: Select Model
- **OpenAI Models:**
  - `gpt-4o-mini` - Fastest/cheapest (recommended)
  - `gpt-4o` - Most capable
  - `gpt-4-turbo` - Balanced
  - `gpt-3.5-turbo` - Budget option

- **Gemini Models:**
  - `gemini-1.5-flash` - Fastest (recommended)
  - `gemini-1.5-pro` - Most capable
  - `gemini-pro` - Standard

### Step 4: Customize Prompt (Optional)
- **Default Prompt**: Works great for most cases (recommended)
- **Custom Prompt**: Use placeholders:
  - `{url}` - Website URL
  - `{company_name}` - Company name
  - `{scraped_content}` - Scraped website content

### Step 5: Run Scraping
- Upload CSV with URLs
- Click "Start Scraping"
- AI summaries generate automatically in parallel!

---

## Output Format

Your CSV/Excel files will have **3 columns**:

1. **Website** - The URL
2. **ScrapedText** - Scraped website content
3. **CompanySummary** - AI-generated company summary

### Summary Includes:
- Company Overview
- Industry
- Products/Services
- Key Facts
- Target Market
- 5 Inferences/Hypotheses

---

## Default Prompt Structure

The default prompt asks the AI to provide:

1. **Company Overview** - What the company does (2-3 sentences)
2. **Industry** - Industry/sector
3. **Products/Services** - Main offerings
4. **Key Facts** - Important facts from website
5. **Target Market** - Target audience/customers
6. **Five Inferences/Hypotheses** - Strategic insights

---

## Custom Prompt Example

You can customize the prompt. Here's an example:

```
Analyze this company and provide:

COMPANY: {company_name}
URL: {url}

CONTENT:
{scraped_content}

Provide:
1. What they do
2. Their industry
3. Main products
4. Target customers
5. 5 strategic insights
```

---

## Cost Considerations

### OpenAI Pricing (approximate):
- `gpt-4o-mini`: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- `gpt-4o`: ~$2.50 per 1M input tokens, ~$10 per 1M output tokens

### Gemini Pricing (approximate):
- `gemini-1.5-flash`: Free tier available, then ~$0.075 per 1M tokens
- `gemini-1.5-pro`: ~$1.25 per 1M tokens

**For 1000 URLs:**
- Using gpt-4o-mini: ~$1-3 (depending on content length)
- Using gemini-1.5-flash: ~$0.50-1.50

---

## Tips

✅ **Start with default prompt** - It's optimized for company summaries  
✅ **Use gpt-4o-mini or gemini-1.5-flash** for cost efficiency  
✅ **Test with small batch first** (10-20 URLs)  
✅ **Monitor API usage** in your provider dashboard  
✅ **AI summaries add time** - Allow extra processing time  

---

## Troubleshooting

### "API Error" messages
- Check API key is correct
- Verify API key has credits/quota
- Check internet connection

### "Library not installed"
- Run: `pip install openai` or `pip install google-generativeai`
- Or: `pip install -r requirements.txt`

### Summaries are empty
- Check scraped content is valid
- Verify API key permissions
- Try different model

### Slow processing
- Reduce concurrency if using AI
- Use faster models (gpt-4o-mini, gemini-flash)
- AI adds ~2-5 seconds per URL

---

## What You Need from Me

To finalize this feature, please provide:

1. **Your current prompt** - So I can integrate it as default
2. **Any specific requirements** - Format, length, sections
3. **Test feedback** - After you try it!

---

**The feature is ready to test! 🚀**
