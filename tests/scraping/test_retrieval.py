import pytest
from ezmm import MultimodalSequence
from scrapemm.common import ScrapingResponse

from scrapemm import retrieve


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "https://factcheckarabic.afp.com/doc.afp.com.9DD6J8",
    "https://www.vishvasnews.com/viral/fact-check-upsc-has-not-reduced-the-maximum-age-limit-for-ias-and-ips-exams/",
    "https://health.medicaldialogues.in/fact-check/brain-health-fact-check/fact-check-is-sprite-the-best-remedy-for-headaches-in-the-world-140368",
    "https://assamese.factcrescendo.com/viral-claim-that-the-video-shows-the-incident-from-uttar-pradesh-and-the-youth-on-the-bike-and-the-youth-being-beaten-and-taken-away-by-the-police-are-the-same-youth-named-abdul-is-false/",
    "https://factuel.afp.com/doc.afp.com.43ZN7NP",
    "https://leadstories.com/365cb414b83e29d26fecae374d55c743a3eac4c7.png",
    "https://leadstories.com/assets_c/2025/08/193f14f06dd6f15b89bf8050e553ad7fb1be6530-thumb-900xauto-3165872.png"
])
async def test_generic_retrieval(url):
    result = await retrieve(url)
    assert isinstance(result, ScrapingResponse)
    print(result)
    assert result
    content = result.content
    assert isinstance(content, MultimodalSequence)
    assert content.has_images()


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "https://www.vishvasnews.com/viral/fact-check-upsc-has-not-reduced-the-maximum-age-limit-for-ias-and-ips-exams/",
    "https://www.zeit.de/politik/deutschland/2025-07/spionage-iran-festnahme-anschlag-juden-berlin-daenemark",
    "https://factnameh.com/fa/fact-checks/2025-04-16-araghchi-witkoff-fake-photo",
    "https://www.thip.media/health-news-fact-check/fact-check-can-a-kalava-on-the-wrist-prevent-paralysis/74724/",
])
@pytest.mark.parametrize("method", ["firecrawl", "decodo"])
async def test_html_retrieval(url, method):
    result = await retrieve(url, format="html", methods=[method])
    assert isinstance(result, ScrapingResponse)
    print(result)
    content = result.content
    print(content)
    assert content
    assert isinstance(content, str)


@pytest.mark.asyncio
@pytest.mark.parametrize("url, max_video_size, download_expected", [
    ("https://www.facebook.com/reel/1089214926521000", None, True),
    ("https://www.facebook.com/reel/1089214926521000", 128_000_000, True),
    ("https://www.facebook.com/reel/1089214926521000", 1_000_000, False),
    ("https://www.youtube.com/shorts/cE0zgN6pYOc", None, True),
    ("https://www.youtube.com/shorts/cE0zgN6pYOc", 6_000_000, True),
    ("https://www.youtube.com/shorts/cE0zgN6pYOc", 3_000_000, False),
])
async def test_max_video_size(url, max_video_size, download_expected):
    result = await retrieve(url, max_video_size=max_video_size)
    assert isinstance(result, ScrapingResponse)
    content = result.content
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
         "https://www.facebook.com/reel/1089214926521000",
         "https://www.zeit.de/politik/deutschland/2025-07/spionage-iran-festnahme-anschlag-juden-berlin-daenemark",
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
         "https://www.facebook.com/reel/1089214926521000",
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
