import pytest
from ezmm import MultimodalSequence
from scrapemm.common import ScrapingResponse

from scrapemm import retrieve
from test_social_media import assert_expectations


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://factcheckarabic.afp.com/doc.afp.com.9DD6J8", dict(image=5)),
    ("https://www.vishvasnews.com/viral/fact-check-upsc-has-not-reduced-the-maximum-age-limit-for-ias-and-ips-exams/", dict(image=1)),
    ("https://health.medicaldialogues.in/fact-check/brain-health-fact-check/fact-check-is-sprite-the-best-remedy-for-headaches-in-the-world-140368", dict(image=1)),
    ("https://assamese.factcrescendo.com/viral-claim-that-the-video-shows-the-incident-from-uttar-pradesh-and-the-youth-on-the-bike-and-the-youth-being-beaten-and-taken-away-by-the-police-are-the-same-youth-named-abdul-is-false/", dict(image=5, video=1)),
    ("https://factuel.afp.com/doc.afp.com.43ZN7NP", dict(image=5)),
    ("https://factcheck.afp.com/doc.afp.com.B44R6WE", dict(image=4)),
    ("https://leadstories.com/365cb414b83e29d26fecae374d55c743a3eac4c7.png", dict(image=1)),
    ("https://leadstories.com/assets_c/2025/08/193f14f06dd6f15b89bf8050e553ad7fb1be6530-thumb-900xauto-3165872.png", dict(image=1)),
])
async def test_generic_retrieval(url: str, expected: dict[str, int]):
    result = await retrieve(url)
    assert_expectations(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("method, url", [
    ("firecrawl", "https://www.vishvasnews.com/viral/fact-check-upsc-has-not-reduced-the-maximum-age-limit-for-ias-and-ips-exams/"),
    ("decodo", "https://www.zeit.de/politik/deutschland/2025-07/spionage-iran-festnahme-anschlag-juden-berlin-daenemark"),
    ("firecrawl", "https://factnameh.com/fa/fact-checks/2025-04-16-araghchi-witkoff-fake-photo"),
    ("decodo", "https://www.thip.media/health-news-fact-check/fact-check-can-a-kalava-on-the-wrist-prevent-paralysis/74724/"),
    ("Perma.cc", "https://perma.cc/U3JC-79UN"),
])
async def test_html_retrieval(url, method):
    result = await retrieve(url, output_format="html", methods=[method])
    assert isinstance(result, ScrapingResponse)
    assert result.success, result.errors
    html = result.get()
    print(html)
    assert isinstance(html, str)
    assert html
    # No format beyond the requested one is produced
    assert result.content.markdown is None
    assert result.content.multimodal is None


@pytest.mark.asyncio
@pytest.mark.parametrize("url, max_video_size, download_expected", [
    ("https://www.facebook.com/reel/2038221060315031", None, True),  # ~12 MB
    ("https://www.facebook.com/reel/2038221060315031", 128_000_000, True),
    ("https://www.facebook.com/reel/2038221060315031", 1_000_000, False),
    ("https://www.youtube.com/shorts/cE0zgN6pYOc", None, True),  # ~1.2 MB
    ("https://www.youtube.com/shorts/cE0zgN6pYOc", 6_000_000, True),
    ("https://www.youtube.com/shorts/cE0zgN6pYOc", 500_000, False),
])
async def test_max_video_size(url, max_video_size, download_expected):
    result = await retrieve(url, max_video_size=max_video_size)
    assert isinstance(result, ScrapingResponse)
    content = result.get()
    assert isinstance(content, MultimodalSequence)
    assert content.has_videos() == download_expected
    if max_video_size and content.has_videos():
        video = content.videos[0]
        assert video.size <= max_video_size


@pytest.mark.asyncio
@pytest.mark.parametrize("urls, methods", [
    ([
         "https://www.youtube.com/shorts/cE0zgN6pYOc"
     ], [
         "YouTube"
     ]),
    ([
         "https://www.facebook.com/reel/2038221060315031",
         "https://factnameh.com/fa/fact-checks/2025-04-16-araghchi-witkoff-fake-photo",
     ], [
         ["Facebook"],
         ["firecrawl"]
     ]),
    ([
         "https://theconversation.com/dandelions-are-a-lifeline-for-bees-on-the-brink-we-should-learn-to-love-them-204504",
         "https://yussus.wixsite.com/newsspoilers/post/leaked-durex-to-launch-reversible-condom-to-circumvent-single-use-plastic-law",
         "https://www.bbc.co.uk/iplayer/episode/b09zg78h/question-time-2018-28062018#t=19m38s",
     ], [
         "decodo",
     ]),
    ([
         "https://factuel.afp.com/doc.afp.com.43ZN7NP",
         "https://x.com/realDonaldTrump"
     ],
     None
    ),
    ([
         "https://factuel.afp.com/doc.afp.com.43ZN7NP",
         "https://x.com/realDonaldTrump",
         "https://www.facebook.com/reel/2038221060315031",
     ],
     "auto"
    ),
])
async def test_methods(urls: list[str], methods: list[str] | list[list[str]] | None):
    results = await retrieve(urls, methods=methods)
    assert results
    print(results)
    if methods and methods != "auto":
        if isinstance(methods[0], str):
            methods = [methods] * len(urls)
        for result, method_list in zip(results, methods):
            assert result.method in method_list
