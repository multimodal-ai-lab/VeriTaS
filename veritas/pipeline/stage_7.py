import logging
from typing import Optional

from ezmm import MultimodalSequence

from veritas.common import Claim, Prompt
from veritas.models import gpt_cheap, gemini_strong, gpt_strong, gemini_cheap
from veritas.pipeline.util.stage import Stage
from veritas.util import run_with_semaphore

logger = logging.getLogger("VeriTaS")


class Stage7(Stage):
    """Rectifies claims to ensure dataset balance."""

    id = 7
    name = "Claim Rectification"
    default_interval_seconds = 60

    def __init__(self) -> None:
        super().__init__()
        self.quarters: dict[tuple[int, int], bool] = {}

    async def step(self, *,
                   start: tuple[int, int] = (2016, 1),
                   end: tuple[int, int] | None = None,
                   max_per_quarter: int = 500) -> None:
        items = await self.get_queued_items(start=start, end=end,
                                            max_per_quarter=max_per_quarter)
        if items:
            await rectify_claims(items)

    async def get_queued_items(self, *,
                               start: tuple[int, int],
                               end: tuple[int, int] | None,
                               max_per_quarter: int) -> list[Claim]:
        """Fetches a batch of claims to rectify, respecting per-quarter limits."""
        from veritas.db import db
        from veritas.util.util import get_quarter_date_range, get_quarters

        if not self.quarters:
            self.quarters.update({q: False for q in get_quarters(start, end)})

        all_claims: list[Claim] = []
        for quarter, is_complete in self.quarters.items():
            if not is_complete:
                quarter_start, quarter_end = get_quarter_date_range(*quarter)
                n_intact_claims = await db.count_intact_claims(quarter_start, quarter_end)
                if n_intact_claims < max_per_quarter:
                    missing = max_per_quarter - n_intact_claims
                    claims = await db.get_claims_to_rectify(start_date=quarter_start,
                                                            end_date=quarter_end,
                                                            limit=min(1000, missing))
                    if claims:
                        all_claims.extend(claims)
                        logger.debug(f"Added {len(claims)} original claims from Q{quarter[1]} {quarter[0]} to the queue.")
                else:
                    self.quarters[quarter] = True
                    logger.info(f"✅ Rectification of quarter {quarter} completed successfully!")

        if self.quarters and all(self.quarters.values()):
            self.done = True
        return all_claims


async def rectify_claims(claims: list[Claim]):
    """Takes a list of false claims and corrects them."""
    # Deduplicate and omit claims explicitly marked as non-rectifiable or being itself a rectification
    original_claims = {c.id: c for c in claims if c.rectifiable != False and c.is_original}
    logger.info(f"Rectifying {len(original_claims)} claims...")
    tasks = [rectify_claim(claim) for claim in original_claims.values()]
    await run_with_semaphore(tasks, limit=20)


async def rectify_claim(claim: Claim):
    assert claim.rectifiable != False, "Cannot rectify an unrectifiable claim."
    assert not claim.is_rectified, "Cannot rectify a rectification."

    if rectified_claim := await claim.variant:
        # Claim already has a rectified version
        if not rectified_claim.check_completed:
            await validate_rectified_claim(rectified_claim)
        return

    logger.debug(f"Rectifying claim {claim.id}: '{claim.data}'")
    verdict = await claim.current_verdict
    assert verdict, "Cannot rectify a claim without a verdict."
    reviews = await claim.reviews
    articles = [await review.article for review in reviews]
    publishers = [await review.publisher for review in reviews]

    prompt = Prompt("veritas/prompts/rectify_claim.md.j2",
                    reviews=zip(articles, publishers),
                    claim=claim,
                    reason=verdict.integrity.explanation,
                    max_words=claim.n_words)
    llm = gemini_strong if prompt.has_videos() else gpt_strong
    try:
        response = await llm.generate(prompt, extract="last_code_span")
        if not response:
            logger.info(f"Rectification result is empty of claim {claim.id}.")
            await set_unrectifiable(claim)
            return
        rectification = MultimodalSequence(str(response))
    except Exception as e:
        logger.warning(f"Could not rectify claim {claim.id}. {type(e).__name__}: {e}")
        await set_unrectifiable(claim)
        return

    # Replace all media from the rectified version with the original media
    rectification = " ".join(item for item in rectification if isinstance(item, str))
    media = MultimodalSequence(claim.data).unique_items()
    rectification = str(MultimodalSequence(*media, rectification))

    # Compose new claim
    claim_rectified = Claim(
        data=rectification,
        date=claim.date,
        appearance_ids=claim.appearance_ids,
        review_ids=claim.review_ids,
        is_rectified=True,
        variant_id=claim.id,
        dismissed=False,
        language=claim.language
    )
    await claim_rectified.save_to_db()
    logger.debug(f"Rectified claim {claim.id}, resulting in claim {claim_rectified.id}")

    # Assign this variant to the original claim
    claim.variant_id = claim_rectified.id
    await claim.save_to_db()

    await validate_rectified_claim(claim_rectified)


async def set_unrectifiable(claim: Claim):
    claim.rectifiable = False
    await claim.save_to_db()
    logger.debug(f"Claim {claim.id} cannot be rectified.")


async def validate_rectified_claim(claim: Claim):
    """Runs several validation tests to check whether the given rectified claim is valid."""
    if not claim.is_rectified:
        raise ValueError("Claim is not rectified.")

    n_words = len(claim.data.split())
    if n_words > 70:
        await claim.dismiss(f"Claim is too long ({n_words} words).")
        return

    # Run validation tests
    try:
        claim.is_inconsistent = not await check_claim_consistency(claim)
        claim.is_unshareable = not await check_claim_shareability(claim)
        from veritas.pipeline.stage_5 import check_missing_referenced_media
        claim.missing_referenced_media = await check_missing_referenced_media(claim)
    except Exception as e:
        await claim.dismiss(f"Could not validate claim: {e}")
        return

    claim.check_completed = True

    # Dismiss if any validation problem is present
    if claim.is_inconsistent or claim.is_unshareable or claim.missing_referenced_media:
        await claim.dismiss("Claim failed validation.")
    else:
        await claim.save_to_db()


async def check_claim_consistency(claim: Claim) -> Optional[bool]:
    """Returns True if the claim is consistent (contains contradictions)."""
    prompt = Prompt(
        "veritas/prompts/claim_validation/check_consistency.md",
        date=claim.date_str,
        language=claim.language_name,
        claim=claim,
    )
    llm = gemini_cheap if prompt.has_videos() else gpt_cheap
    response = await llm.generate(prompt, extract="last_code_span")
    if response:
        decision = str(response).strip().lower()
        if decision == "consistent":
            return True
        if decision == "inconsistent":
            return False
        # Unknown output
        return None
    return None


async def check_claim_shareability(claim: Claim) -> Optional[bool]:
    """Returns True if the claim is shareable."""
    prompt = Prompt(
        "veritas/prompts/claim_validation/check_shareability.md",
        claim=claim,
    )
    llm = gemini_cheap if prompt.has_videos() else gpt_cheap
    response = await llm.generate(prompt, extract="last_code_span")
    if response:
        decision = str(response).strip().lower()
        if decision == "shareable":
            return True
        if decision == "unshareable":
            return False
        # Unknown output
        return None
    return None
