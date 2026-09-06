import pytest

from scrapemm import retrieve
from test_social_media import assert_expectations


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://archive.is/uTVE4", dict(image=1)),  # X post
    ("https://archive.is/0VrgI", dict(image=1)),  # Facebook post
    ("https://archive.md/SI9Yy", dict(image=8)),  # Old Twitter hashtag thread
    ("https://archive.li/Ubqsd", dict(image=1)),  # Webpage
    ("http://archive.today/6OttS", dict(image=2)),  # Facebook post
    ("http://archive.today/2022.05.05-091515/https:/twitter.com/SamvanRooy1/status/1521438261130014721", dict(image=2)), # Twitter post
    ("https://archive.ph/movd4", dict(image=1)),  # Online article
    ("https://archive.vn/Edqcv", dict(image=1)),  # Online article
    ("https://archive.fo/jnN0O", dict(image=1)),  # Older Facebook post
    ("https://archive.md/yACAz", dict(image=2)),  # News article with images

])
async def test_archive_today(url: str, expected: dict[str, int]):
    result = await retrieve(url)
    assert_expectations(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://perma.cc/N8MR-NS96", dict(video=1)),  # TikTok video
    ("https://perma.cc/HU7Z-B24D", dict(image=1)),  # Screenshot of Twitter post
    ("https://perma.cc/HW47-P32Z", dict(image=1)),  # Screenshot of Instagram post
    ("https://perma.cc/8J8J-2CEB", dict(image=4)),  # 4 images in old Twitter UI
    ("https://perma.cc/38CZ-MMFV", dict(image=1)),  # Web article with one image
    ("https://perma.cc/ZD7Z-B3U7?type=image", dict(image=1)),  # Screenshot of Facebook video
    ("https://perma.cc/D5GB-S4E9", dict(image=1)),  # Recorded screenshot of Facebook post
    ("https://perma.cc/Z76P-4TE3", dict(image=1)),  # Image in a Telegram post
    ("https://perma.cc/T5T9-QQ77", dict(video=1)),  # Video in a Telegram post
    ("https://perma.cc/K5L8-V3LZ", dict(video=1)),  # Video from Facebook post
])
async def test_perma_cc(url: str, expected: dict[str, int]):
    result = await retrieve(url)
    assert_expectations(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://web.archive.org/web/20260629131321/https://www.facebook.com/peter.hamilton.54/posts/pfbid0dbCRo43miXqEyeLW9pbyPXrNLAuuNrGJw8nBWGQaNaGs4gcDq4GayR1hkUdkC4Lnl", dict()),
    ("https://web.archive.org/web/20260408100822/https://www.facebook.com/plugins/post.php?href=https%3A%2F%2Fwww.facebook.com%2Fluftari.spartak%2Fposts%2Fpfbid037cXTv7r3rVpd8A5UoGSN1FWaff65bm4C5Yi6FxD2X9qfV9bWcgG7fwM1LMBu2nXhl", dict(image=1)),
    ("https://web.archive.org/web/20260602042101/https://www.threads.com/@yisangworks/post/DZCPCucmsFy?xmt=AQG0cP3v6OLRdeQGg40ajMQntJ7IM1r3dIK8lPjQDOaK2A", dict(image=1)),
    ("https://web.archive.org/web/20210604181412/https://www.tiktok.com/@realstewpeters/video/6969789589590379781?is_copy_url=1", dict(video=1)),
    ("https://web.archive.org/web/20231020082452/https://www.tiktok.com/@greenbrigade2006/video/7291677940834766113?q=madrid%20filistin%20&t=1697788782212", dict(video=1)),
    ("https://web.archive.org/web/20260618201112/https://www.kwai.com/@SHREKPAPORETOO/video/5199555880897549098?photoId=5199555880897549098&share_item_info=5199555880897549098&fid=150001695016757&timestamp=1781550538452&share_uid=150001695016757&kpn=KWAI&userId=150001380085634&cc=COPY_LINK&language=pt-BR&share_item_type=photo&share_device_id=8090ECD7-ED11-41FE-A603-B0B76BBF3706&share_id=8090ECD7-ED11-41FE-A603-B0B76BBF3706_1781550538452&authorKwaiId=SHREKPAPORETOO&translateKey=news_share_text_081001&shareBucket=br&pwa_source=share&shareCountry=BRA&shareBiz=photo&short_key=xp0cCBGW&PWA_share_N_string=20&request_source=1001&share_redirect_switch_choice=pwa", dict(video=1)),
])
async def test_internet_archive(url: str, expected: dict[str, int]):
    result = await retrieve(url)
    assert_expectations(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://ghostarchive.org/archive/d63Zd", dict(image=1)),
    ("https://ghostarchive.org/archive/gz80n", dict(video=1)),
    ("https://www.awesomescreenshot.com/image/40260716?key=c50159e80af3be1d003aa5235d3edaa9", dict(image=1)),
    ("https://www.awesomescreenshot.com/video/11862930?key=b021c372bb96716fdbb2316d0eb37c69", dict(video=1)),
    ("https://mvau.lt/media/b286c959-00da-4765-8f49-88d4ca87a555", dict(video=1)),
    ("https://mvau.lt/media/ccfa5e89-a89d-4a12-aee1-cf68dcb205ce", dict(image=1)),
])
async def test_other_archiving_services(url: str, expected: dict[str, int]):
    result = await retrieve(url)
    assert_expectations(result, expected)
