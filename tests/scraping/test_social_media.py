import pytest
from ezmm import MultimodalSequence

from scrapemm import retrieve
from scrapemm.common import ScrapingResponse


def assert_expectations(response: ScrapingResponse, expected: dict[str, int]):
    """Assert that the content has the expected number of images and videos."""
    assert isinstance(response, ScrapingResponse)
    content = response.content
    print(content or response.errors)
    assert isinstance(content, MultimodalSequence)
    for medium, count in expected.items():
        match medium:
            case "image":
                n_images = len(content.images)
                assert n_images >= count, f"Expected at least {count} images, got {n_images}"
            case "video":
                n_videos = len(content.videos)
                assert n_videos >= count, f"Expected at least {count} videos, got {n_videos}"


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=RxJtYsCQ0jo",
    "https://www.youtube.com/watch?v=mUBdMLfkI54",
    "https://www.youtube.com/watch?v=A4dVOznX6Kk",
    "https://www.youtube.com/shorts/1Cgvb17edsQ",
    "https://www.youtube.com/shorts/mM0i832urK0",
    "https://www.youtube.com/shorts/cE0zgN6pYOc",
])
async def test_youtube(url):
    """Test YouTube video and shorts retrieval"""
    result = await retrieve(url)
    assert_expectations(result, dict(video=1))


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://www.instagram.com/p/DRJ94KKDhpx", dict(image=1)),
    ("https://www.instagram.com/p/DRNcdbPiPCj", dict(image=1)),
    ("https://www.instagram.com/p/CqJDbyOP839", dict(image=1)),
    ("https://www.instagram.com/p/DMuOe6th94D", dict(video=1)),  # Yes, this is a video, marked as misinfo
    ("https://www.instagram.com/reel/DRKtWnhAI0j", dict(video=1)),
    ("https://www.instagram.com/reel/DRE38jKDIYb", dict(video=1)),
    ("https://www.instagram.com/reel/DKqPQqpTDW4", dict(video=1)),  # Age-restricted content
])
async def test_instagram(url: str, expected: dict[str, int]):
    result = await retrieve(url)
    assert_expectations(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://www.facebook.com/esta.spencer/posts/pfbid0zTQ51pwQ1ZGvoTnh6dWKaSqMTcjwpis7tcX1XE5YZ7jhnoc9ZCY8TiUw1Ewr8Wq3l", dict()),
    ("https://www.facebook.com/photo/?fbid=1721085455188778&set=a.107961589834514&_rdc=1&_rdr", dict(image=1)),
    ("https://www.facebook.com/photo/?fbid=1287391456760943&set=a.644291054404323", dict(image=1)),
    ("https://www.facebook.com/photo/?fbid=1148555903931627&set=a.859126372874583", dict(image=1)),
    # Login redirect URL:
    ("https://www.facebook.com/login/?next=https%3A%2F%2Fwww.facebook.com%2Fphoto%3Ffbid%3D860758296160977%26set%3Da.513585680878242",
     dict(image=1)),
    # Post embedding URL:
    ("https://www.facebook.com/plugins/post.php?href=https%3A%2F%2Fwww.facebook.com%2FTheWolverLing%2Fposts%2Fpfbid02FRKrQUdWmULer6VnVZsoDshBBdqYbsz4dERZiN52499HzGFSRJQtsrQFdZf8dXXyl&show_text=true&width=500",
     dict(image=4)),
    ("https://www.facebook.com/reel/2038221060315031", dict(video=1)),
    ("https://www.facebook.com/reel/1954696035077530", dict(video=1)),
    ("https://www.facebook.com/reel/1089214926521000", dict(video=1)),
    ("https://www.facebook.com/reel/3466446073497470", dict(video=1)),  # restricted for misinformation
    ("https://www.facebook.com/watch/?v=502482344935053", dict(video=1)),
    # restricted for misinformation, accessible only with cookie:
    ("https://www.facebook.com/groups/1973976962823632/posts/3992825270938781/", dict(image=1)),
])
async def test_facebook(url: str, expected: dict[str, int]):
    """Test Facebook reel and photo retrieval"""
    result = await retrieve(url)
    assert_expectations(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://t.me/sheyhtamir1974/93691%C2%A0", dict(image=1)),  # One image
    ("https://t.me/durov/404", dict(image=1)),  # One image
    ("https://t.me/tglobaleye/16172", dict(image=2)),  # Multiple images
    ("https://t.me/tglobaleye/16178", dict(video=1)),  # Video and quote
    ("https://t.me/tglobaleye/6289", dict(video=1)),  # GIF (treated as video)
    ("https://t.me/tglobaleye/16192", dict(image=2, video=1)),  # Images and video
])
async def test_telegram(url: str, expected: dict[str, int]):
    result = await retrieve(url)
    assert_expectations(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://www.tiktok.com/@realdonaldtrump/video/7433870905635409198", dict(video=1)),  # Post
    ("https://www.tiktok.com/@xxxx.xxxx5743/video/7521704371109793046", dict(video=1)),  # Post
    ("https://www.tiktok.com/@tomekfoodemprior", dict(image=1)),  # User profile (with profile image)
    ("https://www.tiktok.com/@policebodycam6741", dict(image=1)),  # User profile (with profile image)
])
async def test_tiktok(url: str, expected: dict[str, int]):
    result = await retrieve(url)
    assert_expectations(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://x.com/PopBase/status/1938496291908030484", dict(image=1)),
    ("https://twitter.com/Cow__Momma/status/1729708561934991851/photo/2", dict(image=1)),
    ("https://x.com/realDonaldTrump", dict()),
    ("https://t.co/GcXDN4zbRx", dict(image=1)),
    ("https://x.com/HPhobiaWatch/status/1875842972216066547", dict(image=2)),
    ("https://x.com/TigerRajaSingh/status/1125050909338169345", dict(video=1)),
    ("https://x.com/visegrad24/status/2072633510725071224", dict(image=1, video=3)),
    ("https://x.com/visegrad24/status/2072633510725071224/video/2", dict(video=1)),
    ("https://publish.twitter.com/?query=https%3A%2F%2Ftwitter.com%2FRain2097952%2Fstatus%2F1719525724279976200&widget=Tweet",
     dict(image=1)),
])
async def test_x(url: str, expected: dict[str, int]):
    result = await retrieve(url)
    assert_expectations(result, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected", [
    ("https://bsky.app/profile/eliothiggins.bsky.social/post/3mqrktuon4s2l", dict()),
    ("https://bsky.app/profile/mabellogan.bsky.social/post/3mqtbpexyks2i", dict(image=1)),
    ("https://bsky.app/profile/dennishorn.de/post/3mqt2k6c7oc2s", dict(image=4)),
    ("https://bsky.app/profile/acyn.bsky.social/post/3mqspt2uqxz22", dict(video=1)),
])
async def test_bluesky(url: str, expected: dict[str, int]):
    result = await retrieve(url)
    assert_expectations(result, expected)
