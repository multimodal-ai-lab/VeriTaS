import pytest

from veritas.pipeline.stage_3 import extract_article_content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url", [
        "https://www.politifact.com/article/2017/sep/19/fact-checking-donald-trumps-speech-un/",
        "https://factcheck.afp.com/doc.afp.com.349J6TQ",
        "https://leadstories.com/hoax-alert/2025/12/fact-check-rob-reiner-did-not-post-anti-trump-tweet-2023.html",
        "https://www.thequint.com/news/webqoof/delhi-cm-atishi-stopped-by-police-from-campaigning-for-aap-delhi-elections-viral-video-fact-check",
        "https://www.factcrescendo.com/false-claim-of-muslims-attacking-railway-gateman-in-hardoi-video-of-mutual-quarrel-viral-with-false-claim/",
        "https://www.vishvasnews.com/viral/fact-check-air-india-concession-for-senior-citizens-is-25-of-basic-fare/",
        "https://newschecker.in/fact-check/old-unrelated-visuals-of-showing-cache-of-arms-falsely-linked-to-recent-raid-in-lucknow",
        "https://english.factcrescendo.com/2024/03/18/this-new-york-times-paper-clip-mocking-pm-modi-prior-to-lok-sabha-election-is-fake/",
        "https://www.thequint.com/news/webqoof/ai-generate-a-clip-show-elephant-falling-on-a-truck-fact-check",
    ]
)
async def test_extract_article_content(db, url: str):
    from ezmm import MultimodalSequence
    article = await db.get_article_by_url(url)
    page = article.scraped_page
    extracted = await extract_article_content(page)

    assert extracted
    print(extracted)

    page_len = len(page)
    extracted_len = len(extracted)
    print(f"\n{url}\nShortened {page_len} -> {extracted_len} | Reduction rate: {1 - extracted_len / page_len:.1%}.")
    assert extracted_len < page_len
    assert extracted_len > 0

    # Count removed media
    extracted_media = MultimodalSequence(extracted).unique_items()
    for medium in MultimodalSequence(page).unique_items():
        if medium not in extracted_media:
            print(f"Excluded medium {medium.reference} ({medium.size:.0f} bytes)")
