from unittest import TestCase

import pytest

from veritas.util.parsing import remove_wrapping_quotes, extract_last_code_block
from veritas.util.url import get_domain, is_domain_root, resolve_archiving_url


class TestRemoveWrappingQuotes(TestCase):
    def test(self):
        test_cases = [
            '"X said "abc" while being in ..."',  # Should remove (wrapping)
            '"abc def ghi"',  # Should remove (wrapping)
            '""Y", better known as X, said "abc""',  # Should remove (wrapping)
            '"Y", better known as X, said "abc"',  # Should NOT remove (structural)
            '"abc" has been said by X',  # Should NOT remove (structural)
            'X said "abc"',  # Should NOT remove (structural)
            '"Simple quote"',  # Should remove (wrapping)
            '"Name", also known as Y',  # Should NOT remove (structural)
            '"Balanced "inner" quotes here"',  # Should remove (wrapping)
            '"Unbalanced "quote here',  # Should NOT remove (unbalanced)
        ]

        expected_results = [
            'X said "abc" while being in ...',
            "abc def ghi",
            '"Y", better known as X, said "abc"',
            '"Y", better known as X, said "abc"',
            '"abc" has been said by X',
            'X said "abc"',
            "Simple quote",
            '"Name", also known as Y',
            'Balanced "inner" quotes here',
            '"Unbalanced "quote here',
        ]
        for case, expected in zip(test_cases, expected_results):
            with self.subTest(case=case):
                result = remove_wrapping_quotes(case)
                self.assertEqual(result, expected, f"Failed for case: {case}")


@pytest.mark.parametrize(
    "url,target",
    [
        ("https://www.facebook.com/AmericasLastLineofDefense/posts", "facebook.com"),
        ("http://t.me", "t.me"),
        ("example.com", "example.com"),
        ("www.example.com", "example.com"),
        ("subdomain.example.com", "example.com"),
        ("example.co.uk", "example.co.uk"),
        ("https://www.example.co.uk/some/path", "example.co.uk"),
        (
                "https://web.archive.org/web/20250604085431/https://www.facebook.com/AmericasLastLineofDefense/posts",
                "archive.org",
        ),
    ],
)
def test_get_domain(url, target):
    assert get_domain(url) == target


@pytest.mark.parametrize(
    "input,target",
    [
        (
                "https://web.archive.org/web/20250604085431/https://www.facebook.com/AmericasLastLineofDefense/posts",
                "https://www.facebook.com/AmericasLastLineofDefense/posts",
        ),
        (
                "https://web.archive.org/web/20250604000000/https://twitter.com/nair_nandu08/status/1234567890",
                "https://twitter.com/nair_nandu08/status/1234567890",
        ),
        (
                "https://www.reddit.com/r/india/submit?url=https%3A%2F%2Ffactly.in%2Fan-old-video-from-maharashtra-is-falsely-shared-as-rcb-fans-celebrating-the-2025-ipl-victory-in-bengaluru%2F",
                None,
        ),
        (
                "https://rejouer.perma.cc/replay-web-page/w/id-505328c00239/:37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f/20250415134357mp_/https://x.com/afd_wallduern/status/1908101935770562722",
                "https://x.com/afd_wallduern/status/1908101935770562722",
        ),
        (
                "http://archive.today/2025.05.19-155334/https://x.com/DonaldJTrumpJr/status/1924250965965709808",
                "https://x.com/DonaldJTrumpJr/status/1924250965965709808",
        ),
        (
                "https://rejouer.perma.cc/replay-web-page/w/id-8964e5b7ae9d/:37a8eec1ce19687d132fe29051dca629d164e2c4958ba141d5f4133a33f0688f/20250206151010mp_/file:///9LEL-AKMR/upload.png",
                None,
        ),
        (
                "https://archive.is/20230807093054/https:/twitter.com/The_Iram__/status/1688201784588894208#selection-1885.0-1885.47",
                "https:/twitter.com/The_Iram__/status/1688201784588894208"
        )
    ],
)
@pytest.mark.asyncio
async def test_resolve_archive_url_long_format(input, target):
    result = await resolve_archiving_url(input)
    if result:
        result = result.get("original_url")
    assert result == target


# Test for Archive Today short format URLs (requires network access and Decodo Advanced plan)
# @pytest.mark.parametrize(
#     "input,target",
#     [
#         ("https://archive.ph/nLdE3", "https://x.com/DerPhysiker21/status/1916960065073873035"),
#         (
#                 "https://archive.ph/zuhsW",
#                 "https://www.facebook.com/themalaengtad/posts/pfbid0261K9XKx9sUbebr9BHkV8FSpXWRmkA5iLpK7ZirJomwsLz49jzDJKFMcUJASXrAarl"
#         ),
#         (
#                 "https://archive.md/Pl8lU",
#                 "https://twitter.com/mjavinod/status/1468050808281309188"
#         ),
#         (
#                 "https://archive.ph/F4GGE",
#                 "https://twitter.com/markiank/status/1704958766230159720"
#         )
#     ],
# )
# @pytest.mark.asyncio
# async def test_resolve_archive_today_short_format(input, target):
#     """Test resolving Archive Today short code format URLs.
#     This test requires network access and Decodo Advanced plan subscription.
#     Example: https://archive.ph/nLdE3 should resolve to the original URL.
#     """
#     result = await resolve_archiving_url(input)
#     if result:
#         result = result.get("original_url")
#     assert result == target


@pytest.mark.parametrize(
    "input,target",
    [
        ("https://perma.cc/G7R6-9XD4", "https://www.emich.edu/"),
        ("https://perma.cc/9265-T4NB", "http://www.whitehouse.gov"),
        ("https://perma.cc/T8U2-994F", "http://www.fbi.gov/wanted/topten"),
        ("https://perma.cc/48VC-ZS62", "http://www.esquire.com/blogs/politics/occupy-wall-street-violence-6575448"),
        ("https://perma.cc/8H4G-2P7K", "http://www.drake.edu/law/future/academics/jd/trial-practicum/"),
        ("https://perma.cc/B7Z7D9DJ", "http://www.sec.gov/comments/s7-03-13/s70313-178.pdf"),
        ("https://perma.cc/U2AN-2TXE", "http://blog.ericgoldman.org/archives/2013/09/when_should_sea.htm"),
        ("https://perma.cc/C6UP-96HN",
         "http://www.huffingtonpost.com/2013/07/25/puerto-rico-status-debate_n_3651755.html"),
    ],
)
@pytest.mark.asyncio
async def test_resolve_perma_cc(input, target):
    result = await resolve_archiving_url(input)
    if result:
        result = result.get("original_url")
    assert result == target


@pytest.mark.parametrize(
    "input, target",
    [
        ("example.com", True),
        ("www.example.com", True),
        ("www.example.co.uk", True),
        ("example.com/hello", False),
        ("http://example.com", True),
        ("https://example.com", True),
        ("https://example.com/", True),
        ("https://www.example.com/", True),
        ("https://www.an.example.com/", True),
        ("https://www.an.example.com/hello", False),
    ],
)
def test_is_domain_root(input, target):
    assert is_domain_root(input) == target


@pytest.mark.parametrize(
    "input, target",
    [
        ("```Hello World```", "Hello World"),
        ("```python\nprint('Hello World')\n```", "print('Hello World')"),
        ("```\nThis is a text.\n```", "This is a text."),
        ("```text\nWith language tag```", "With language tag"),
        ("```c++\n\n```", ""),
    ],
)
def test_last_code_block_extraction(input, target):
    result = extract_last_code_block(input)
    assert result == target
