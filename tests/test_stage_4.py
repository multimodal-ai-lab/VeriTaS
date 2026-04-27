import pytest
from ezmm import MultimodalSequence

from veritas.common.appearance import appearance_from_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "review_url, urls_to_extract",
    [
        ("https://leadstories.com/hoax-alert/2025/02/fact-check-javier-milei-sister-photo.html", [
            "https://x.com/HugoMartingale/status/1891972262791684588",
            "https://archive.ph/kYi4w"
        ]),
        ("https://factcheck.afp.com/doc.afp.com.33C97FY", [
            "https://archive.ph/d75Po",
            "https://archive.is/4glhe",
            "https://archive.ph/eGXoa",
        ]),
        ("https://newschecker.in/fact-check/tulsi-gabbards-video-shared-as-new-zealands-pm-jacinda", [
            "https://www.facebook.com/geetasociety/posts/1231591460520469",
            "https://www.facebook.com/RajasekaranRJ/posts/3558132897530528",
        ]),
        ("https://www.snopes.com/fact-check/sheila-jackson-lee-homicide/", [
            "https://www.facebook.com/permalink.php?story_fbid=1694445400792130&id=1690570001179670",
        ]),
        ("https://www.thequint.com/news/webqoof/old-video-of-fans-singing-to-support-palestine-in-morocco-linked-to-2022-fifa-world-cup-fact-check", [
            "https://perma.cc/S6Z3-6NT5",  # Included in image caption
            "https://perma.cc/ABM2-XZNP",
            "https://perma.cc/BK9B-8B2Y",
        ]),
    ]
)
async def test_appearance_extraction(db, review_url: str, urls_to_extract: list[str]):
    from veritas.pipeline.stage_4 import extract_appearance_urls

    review = await db.get_review_by_url(review_url)
    urls = await extract_appearance_urls(review)

    print(f"Extracted URLs:\n{'\n'.join(urls)}")

    for url in urls_to_extract:
        assert url in urls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://archive.fo/CA6cb"
    ]
)
async def test_url_to_appearance(url):
    app = await appearance_from_url(url)
    assert app
    assert app.url
    assert app.archive_url
    assert isinstance(app.scraped_content, MultimodalSequence)
