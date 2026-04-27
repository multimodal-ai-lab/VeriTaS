import pytest
from ezmm import Item

from veritas.pipeline.stage_5 import check_medium_fact_check_exposure


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "medium_ref, expected",
    [
        ("<image:105202>", True),
        ("<image:727305>", True),
        ("<image:129027>", True),
        ("<image:428733>", True),
        ("<image:743453>", False),
        ("<image:432188>", False),
        ("<image:420779>", False),
    ]
)
async def test_media_fact_check_exposure_check(db, medium_ref: str, expected: bool):
    medium = Item.from_reference(medium_ref)
    if medium:
        result = await check_medium_fact_check_exposure(medium)
        assert result == expected
