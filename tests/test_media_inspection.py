import pytest
from ezmm import Item

from veritas.pipeline.stage_5 import _inspect_medium, is_similar


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "medium_ref, review_id, expected",
    [
        # Completely unrelated:
        ("<image:12345>", 129305, dict(related=False, inherent=False, original=False)),
        # Image is screenshot from Facebook page:
        ("<image:490224>", 139998, dict(original=False)),
        # Image is actually relevant:
        ("<image:490230>", 139998, dict(related=True, inherent=True, original=True, unmodified=True)),
        # Image is related but not original:
        ("<image:129043>", 234344, dict(related=True, original=False)),
        # Image is related but not original:
        ("<image:491588>", 91037, dict(related=True, original=False)),
        # Image is actually relevant:
        ("<image:452578>", 66081, dict(related=True, inherent=True, original=True, unmodified=True)),
        # Image is not even related to claim:
        ("<image:452187>", 236591, dict(related=False)),
        # Video is fully relevant:
        ("<video:5055>", 217885, dict(related=True, inherent=True, original=True, unmodified=True)),
        # Video is fully relevant:
        ("<video:5060>", 269110, dict(related=True, inherent=True, original=True, unmodified=True)),
    ]
)
async def test_media_inspection(db, medium_ref: str, review_id: int, expected: dict):
    medium = Item.from_reference(medium_ref)
    review = await db.get_review_by_id(review_id)
    result = await _inspect_medium(
        medium=medium,
        raw_claim=review.raw_claim,
        date=review.raw_claim_date,
        article=await review.article
    )
    assert result
    assert result.model_dump().items() >= expected.items()  # Subset


@pytest.mark.parametrize(
    "ref_1, ref_2, expected",
    [
        ("<image:494029>", "<image:494013>", True),  # Same except for color
        ("<image:446636>", "<image:446637>", True),  # Same except for color
        ("<image:679536>", "<image:679554>", True),  # Same except for black border and red circle
        ("<image:605918>", "<image:605792>", True),  # Same except for cropping and text
        ("<image:901292>", "<image:901297>", False),  # Somewhat similar, yet different here
        ("<image:829772>", "<image:829738>", False),  # Same overlay but different photos
        ("<image:827952>", "<image:827954>", False),  # Same overlay but different photos
        ("<image:452079>", "<image:452205>", False),  # Same topic but very different photos
        ("<image:862275>", "<image:862274>", False),  # Very different
        ("<image:743562>", "<image:743585>", False),  # Very different
    ]
)
def test_media_similarity(ref_1: str, ref_2: str, expected: bool):
    medium_1 = Item.from_reference(ref_1)
    medium_2 = Item.from_reference(ref_2)
    print(f"Cosine similarity: {medium_1.cos_sim(medium_2):.3f}")
    assert is_similar(medium_1, medium_2) == expected
