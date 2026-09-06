"""Media-reference handling - the part the multimodal rendering depends on."""

import pytest

from webui.parsing import (
    describe_multimodal,
    domain_of,
    find_references,
    has_media,
    parse_reference,
    segment,
    unique_references,
)


@pytest.mark.parametrize("reference,expected", [
    ("<image:42>", ("image", 42)),
    ("<video:7>", ("video", 7)),
    ("<audio:105202>", ("audio", 105202)),
    ("  <image:1>  ", ("image", 1)),
])
def test_parse_reference(reference, expected):
    assert parse_reference(reference) == expected


@pytest.mark.parametrize("reference", [
    "<image:>", "<pdf:12>", "image:12", "<image:12", "<image:1.5>", "",
])
def test_parse_reference_rejects_non_references(reference):
    assert parse_reference(reference) is None


def test_find_references_keeps_order_and_duplicates():
    text = "See <image:2> and <video:9>, again <image:2>."
    assert find_references(text) == ["<image:2>", "<video:9>", "<image:2>"]
    assert unique_references(text) == ["<image:2>", "<video:9>"]


def test_find_references_on_empty_input():
    assert find_references(None) == []
    assert find_references("") == []
    assert has_media(None) is False


def test_segment_keeps_the_reference_as_its_own_segment():
    segments = segment("Before <image:5> after")
    assert segments == [
        {"type": "text", "text": "Before "},
        {"type": "media", "reference": "<image:5>", "kind": "image", "id": 5},
        {"type": "text", "text": " after"},
    ]


def test_segment_handles_adjacent_and_edge_references():
    segments = segment("<image:1><video:2>tail")
    assert [part["type"] for part in segments] == ["media", "media", "text"]
    assert segments[-1]["text"] == "tail"


def test_segment_reassembles_to_the_original_text():
    text = "A <image:1> B <video:2><audio:3> C"
    rebuilt = "".join(
        part["text"] if part["type"] == "text" else part["reference"]
        for part in segment(text)
    )
    assert rebuilt == text


def test_segment_of_plain_text_is_a_single_segment():
    assert segment("no media here") == [{"type": "text", "text": "no media here"}]


def test_segment_of_empty_text_is_empty():
    assert segment("") == []
    assert segment(None) == []


def test_describe_multimodal_counts_per_kind():
    described = describe_multimodal("<image:1> x <image:2> y <video:3> z <image:1>")
    assert described["n_media"] == 3
    assert described["counts"] == {"image": 2, "video": 1, "audio": 0}
    assert described["is_multimodal"] is True


def test_describe_multimodal_on_text_only():
    described = describe_multimodal("just text")
    assert described["is_multimodal"] is False
    assert described["n_media"] == 0


@pytest.mark.parametrize("url,expected", [
    ("https://www.example.com/a/b?c=1", "example.com"),
    ("http://sub.example.co.uk/", "sub.example.co.uk"),
    ("https://Example.COM", "example.com"),
    ("https://user@example.com:8443/x", "example.com"),
    ("example.com/path", "example.com"),
    ("not a url", None),
    (None, None),
    ("", None),
])
def test_domain_of(url, expected):
    assert domain_of(url) == expected
