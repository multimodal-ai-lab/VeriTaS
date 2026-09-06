import aiohttp
import pytest
from bs4 import BeautifulSoup

from scrapemm.download.common import HEADERS
from scrapemm.util import (
    _extract_media_elements,
    get_markdown_hyperlinks,
    unshorten,
)


@pytest.mark.parametrize("input,target",
                         [
                             (
                                     '![](https://factly.in/wp-content/uploads//2023/12/Bombay-high-court-building-featured-image-103x65.jpeg "Review: Bombay High Court Rules That Human Need for an Organ Transplant is Directly a Facet of Right to Life as Guaranteed Under Article 21 of the Constitution")',
                                     [
                                         "https://factly.in/wp-content/uploads//2023/12/Bombay-high-court-building-featured-image-103x65.jpeg"
                                     ]
                             )
                         ]
                         )
def test_media_link_extraction(input, target):
    match_hypertext_url_triples = get_markdown_hyperlinks(input)
    urls = [triple[2] for triple in match_hypertext_url_triples]
    assert urls == target


def test_extract_background_image_from_photo_wrap():
    html = (
        '<a class="tgme_widget_message_photo_wrap" '
        'style="width:641px;background-image:url(\'https://cdn4.telegram-cdn.org/file/abc123.jpg\')"></a>'
    )
    soup = BeautifulSoup(html, "html.parser")
    elements = _extract_media_elements(soup)
    assert len(elements) == 1
    assert elements[0].get("src") == "https://cdn4.telegram-cdn.org/file/abc123.jpg"


def test_extract_skips_emoji_background_image():
    html = (
        '<i class="emoji" '
        'style="background-image:url(\'//telegram.org/img/emoji/40/F09FA681.png\')"></i>'
    )
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_media_elements(soup) == []


async def do_unshorten(short_url):
    async with aiohttp.ClientSession(headers=HEADERS) as session:
        return await unshorten(short_url, session)


@pytest.mark.asyncio
@pytest.mark.parametrize("short_url,long_url", [
    ("https://t.co/GcXDN4zbRx", "https://twitter.com/mossos/status/1415935363567263744/photo/1"),
])
async def test_unshorten(short_url: str, long_url: str):
    extended = await do_unshorten(short_url)
    assert extended == long_url
