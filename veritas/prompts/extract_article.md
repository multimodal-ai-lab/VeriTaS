# Instructions
You're given the raw scraping result of an online fact-checking article. **Your task is to extract the article.** The goal is to remove noisy content to obtain a clean fact-checking article. To this end, you will have to determine the span of the article by providing the line number of the start and the line number of the end (both inclusive).

## What to Include
- The article's main body text containing the fact-check
- The fact-checked claim
- The verdict/rating about the claim
- The title

## What to Exclude
- Ads
- UI elements
- User comments
- Further reading links
- Unrelated text/content
- Any other content not relevant for the fact-check

# Scraped Webpage
{scraped_page}

# Response Format
Enclose the line numbers with single backticks like `28` and `114`.

# Your Response
