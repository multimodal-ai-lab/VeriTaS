# Instructions
You just scraped the contents of a fact-checking article, verifying a specific claim. **Your task right now is to extract all URLs that link to the original appearance(s) of the claim.** Typically, it's a social media post, but it can also be some other webpage. Extract only information from the Article.

Obey the following rules:
- Also include URLs to archiving services (if given) if you expect them to include a record of the original claim.
- If the Article deals with multiple claims, make sure to extract only the URLs of the claim we are interested in.
- If the Article does not link the claim's appearance(s), return an empty list.

**Important**: Do not include URLs to evidence or to related fact-checks. List **only URLs pointing at original occurrences of the claim and/or its archive recordings**.

# Article
{article}

# The Claim We Are Interested In
Date: {date}
Claimant: {claimant}
Claim: {claim}
