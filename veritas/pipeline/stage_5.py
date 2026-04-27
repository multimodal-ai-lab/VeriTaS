import asyncio
import logging
import traceback
from datetime import datetime, date
from typing import cast, Optional, Collection

from ezmm import MultimodalSequence, Item, Image, Video
from pydantic import BaseModel, Field, ValidationError

from veritas.common import Appearance, Claim, Prompt, Review, Article
from veritas.common.platforms import PLATFORMS
from veritas.db import db
from veritas.models import gemini_strong, gemini_cheap, QuotaExceededError, gpt_cheap, gpt_strong
from veritas.pipeline import max_video_duration, max_media_per_claim, max_appearances_per_claim
from veritas.pipeline.stage_7 import validate_rectified_claim
from veritas.pipeline.util.stage import Stage
from veritas.util import run_with_semaphore, get_domain
from veritas.util.parsing import remove_wrapping_quotes, detect_hallucinated_media_refs
from veritas.util.url import ARCHIVING_SITES

logger = logging.getLogger("VeriTaS")


class Stage5(Stage):
    """Normalizes, matches, and validates claims for reviews at stage 4."""

    id = 5
    name = "Claim Normalization"

    def __init__(self) -> None:
        super().__init__()
        self.quarters: dict[tuple[int, int], bool] = {}

    async def step(self, *,
                   start: tuple[int, int] = (2016, 1),
                   end: tuple[int, int] | None = None,
                   max_per_quarter: int = 1000) -> None:
        items = await self.get_queued_items(start=start, end=end,
                                            max_per_quarter=max_per_quarter)
        if items:
            await process_claims(items)

    async def get_queued_items(self, *,
                               start: tuple[int, int],
                               end: tuple[int, int] | None,
                               max_per_quarter: int) -> list[Review]:
        """Fetches a batch of reviews at stage 4, respecting per-quarter claim limits."""
        from veritas.util.util import get_quarter_date_range, get_quarters

        if not self.quarters:
            self.quarters.update({q: False for q in get_quarters(start, end)})

        all_reviews: list[Review] = []
        for quarter, is_complete in self.quarters.items():
            if not is_complete:
                quarter_start, quarter_end = get_quarter_date_range(*quarter)
                n_claims = await db.count_claims(start_date=quarter_start, end_date=quarter_end)

                if n_claims < max_per_quarter:
                    missing = max_per_quarter - n_claims
                    reviews = await db.get_reviews(stage=4, limit=missing,
                                                   start_date=quarter_start, end_date=quarter_end)
                    if reviews:
                        all_reviews.extend(reviews)
                else:
                    self.quarters[quarter] = True
                    logger.info(f"✅ Stage 5 processing of quarter {quarter} completed successfully!")

        if self.quarters and all(self.quarters.values()):
            self.done = True
        return all_reviews


async def process_claims(reviews: list[Review]):
    logger.info(f"Normalizing, matching, and validating {len(reviews)} claims...")
    tasks = [process_claim(r) for r in reviews]
    await run_with_semaphore(tasks, limit=20)
    logger.info("Claim processing completed successfully.")


async def process_claim(review: Review):
    claim = await review.claim or await normalize_claim(review)
    if claim:
        await validate_original_claim(claim)
        await review.set_stage(5)


async def register_claim(claim_data: MultimodalSequence, review: Review) -> Optional[Claim]:
    """Checks for matching claims in the DB. If a matching claim is found,
    assigns the proper IDs to reviews, etc. Otherwise, saves the claim to the
    DB and returns it."""
    try:
        if match_id := await match(claim_data):
            claim = await assign_review_to_claim(match_id, review)
        else:
            claim = await register_new_claim(claim_data, review)
        review.claim_id = claim.id
        await review.save_to_db()
        if not match_id:
            return claim
    except Exception as e:
        await review.dismiss(f"Could not register claim: {e}\n{traceback.format_exc()}")


async def assign_review_to_claim(match_id: int, review: Review) -> Claim:
    """Adds the new review and appearances to the known claim."""
    claim = cast(Claim, await Claim.get(match_id))
    review_other = (await claim.reviews)[0]

    # Update claim details
    if review.appearance_ids:
        claim.appearance_ids.union(review.appearance_ids)
    if isinstance(review.id, int):
        claim.review_ids.add(review.id)
    # Set language on the claim if not set yet
    if not claim.language and review.language:
        claim.language = review.language
    await claim.save_to_db()

    # Update review details
    review.stage = review_other.stage
    review.dismissed = review_other.dismissed
    review.dismissed_reason = review_other.dismissed_reason
    await review.save_to_db()

    return claim


async def register_new_claim(claim: MultimodalSequence, review: Review) -> Claim:
    assert review.id, "Review ID must be set before processing claims"
    result_claim = Claim(
        data=str(claim),
        date=review.raw_claim_date or review.published or review.modified,
        language=review.language,
        appearance_ids=review.appearance_ids or set(),
        review_ids={review.id},
    )
    await result_claim.save_to_db()
    return result_claim


async def match(claim: MultimodalSequence) -> int | None:
    """Takes a normalized claim and determines the ID of any matching
    claim in the DB. If there is no matching claim, returns None."""
    # Exact matches
    return await db.get_claim_id_by_data(claim)
    # TODO: Reverse image search
    # TODO: Matches by cos similarity
    # TODO: Matches by LLM as a judge


class MediumInspection(BaseModel):
    related: bool
    inherent: bool
    original: bool
    unmodified: bool


async def normalize_claim(review: Review) -> Claim | None:
    """Normalizes a single claim, if necessary. Expects the appearances and the review
    to be scraped already."""
    appearances = [app for app in await review.appearances if not app.dismissed][:max_appearances_per_claim]
    article = await review.article
    assert article, "Article must be scraped before processing claims"

    # Prepare raw claim by removing empty spaces and wrapping quotes
    raw_claim = remove_wrapping_quotes(review.raw_claim.strip())

    # 1. Identify the relevant claim media
    media, media_origin = await _extract_media(appearances, article, raw_claim, review.raw_claim_date)

    # 2. Reformulate the claim
    reformulation = await _reformulate_claim(raw_claim, review, article, appearances, media)

    # Save the claim into the DB
    if reformulation:
        claim = await register_claim(reformulation, review)
        if claim:
            # Persist where the used media originated from; only set if media exist
            claim.media_origin = media_origin if media else None
            await claim.save_to_db()
        return claim


async def _extract_media(
        appearances: list[Appearance],
        article: Article,
        raw_claim: str,
        raw_claim_date: date | datetime | None
) -> tuple[list[Item], str]:
    """Uses heuristics and cheap LLMs to extract relevant claim media from the appearances/article.
    Includes archived media only if no original media were obtained. Includes article media only
    if no archived media were obtained. The returned string
    indicates the origin of the media ('original', 'archived', or 'article')."""

    original_media, archived_media, article_media = _get_candidate_media(appearances, article)

    async def filter_relevant(media: list[Item], origin: str) -> list[Item]:
        return await _filter_relevant_media(media, raw_claim=raw_claim, date=raw_claim_date,
                                            article=article, origin=origin)

    if relevant_media := await filter_relevant(original_media, 'original'):
        return relevant_media, 'original'
    elif relevant_media := await filter_relevant(archived_media, 'archived'):
        return relevant_media, 'archived'
    # Never use article media anymore - too noisy (rule introduced in Q2 2026)
    # elif relevant_media := await filter_relevant(article_media, 'article'):
    #     return relevant_media, 'article'

    return [], ""


async def _filter_relevant_media(media: list[Item], **kwargs) -> list[Item]:
    """Filters out irrelevant media through LLM prompting. Caps at max_media_per_claim
    and prioritizes videos over images."""
    if not media:
        return []
    tasks = [_medium_is_relevant(item, **kwargs) for item in media]
    results = await asyncio.gather(*tasks)
    relevant_media = [item for item, is_relevant in zip(media, results) if is_relevant]

    # Prioritize videos over images and cap at max_media_per_claim
    videos = [medium for medium in relevant_media if isinstance(medium, Video)]
    images = [medium for medium in relevant_media if isinstance(medium, Image)]
    media = (videos + images)[:max_media_per_claim]
    return media


async def _medium_is_relevant(medium: Item, origin: str, **kwargs) -> bool | None:
    """Returns True if the given medium is relevant to the claim, False otherwise.
    The lower the medium's origin quality, the stricter the criteria for being relevant."""
    inspection = await _inspect_medium(medium, **kwargs)
    if inspection:
        match origin:
            case "original":
                return inspection.related and inspection.inherent
            case "archived":
                return (inspection.related and inspection.inherent
                        and inspection.original)
            case _:  # article
                return (inspection.related and inspection.inherent
                        and inspection.original and inspection.unmodified)


async def _inspect_medium(
        medium: Item, raw_claim: str, date: datetime | None, article: Article
) -> MediumInspection | None:
    """Applies LLM prompting to determine if the given medium is a relevant part of the claim."""
    # text_elements = [t for t in article.scraped_content if isinstance(t, str)]
    # article_str = str(" ".join(text_elements))[:50_000]
    prompt = Prompt(
        "veritas/prompts/inspect_medium.md.j2",
        medium=medium,
        claim=raw_claim,
        date=date.strftime("%B %d, %Y") if date else None,
        article=str(article.content)[:50_000],
    )
    llm = gemini_cheap if isinstance(medium, Video) else gpt_cheap
    try:
        response = await llm.generate(prompt, extract="last_code_span")
    except QuotaExceededError:
        raise
    except Exception as e:
        logger.debug(f"Could not inspect medium {medium.reference}. Reason: {e}")
        return None

    if response:
        try:
            return MediumInspection.model_validate_json(str(response))
        except ValidationError:
            logger.debug(f"Could not parse medium inspection response: {response}")
            return None


def _get_candidate_media(
        appearances: list[Appearance], article: Article
) -> tuple[list[Item], list[Item], list[Item]]:
    """Gathers the media from the appearances, their archived records, and the article.
    Returns a tuple of unique media items, deduplicated by cos-similarity, pre-filtered
    by removing blank images and videos longer than max_video_duration. The returned tuple
    contains the original media, the archived media, and the article media."""

    # Gather media
    original_media: list[Item] = []
    archived_media: list[Item] = []
    article_media: list[Item] = []

    for app in appearances:
        if app.original_scrape_ok:
            original_media.extend(app.original_scraped_content.unique_items())
        elif app.archived_scrape_ok:
            archived_media.extend(app.archived_scraped_content.unique_items())
    article_media.extend(article.content.unique_items())

    # Always keep order original > archived > article media
    media = original_media + archived_media + article_media

    # Validate the media
    media = [item for item in media if _is_valid_item(item)]

    # Deduplicate
    deduplicated_media = []
    for item in media:
        if not _contained_in(item, deduplicated_media):
            deduplicated_media.append(item)

    # Compose result
    original_out = []
    archived_out = []
    article_out = []
    for item in deduplicated_media:
        if item in original_media:
            original_out.append(item)
        elif item in archived_media:
            archived_out.append(item)
        else:
            article_out.append(item)

    return original_out, archived_out, article_out


def _is_valid_item(item: Item) -> bool:
    # Filter overly long videos
    if max_video_duration is not None and isinstance(item, Video):
        try:
            if item.duration > max_video_duration:
                return False
        except OSError:
            return False
    # Filter solid color images
    if isinstance(item, Image):
        # getextrema() returns (min, max) per channel; if all channels have min==max, it's a solid color
        extrema = item.image.getextrema()
        if all(abs(lo - hi) < 10 for (lo, hi) in extrema):
            return False
    return True


def _contained_in(medium: Item, media: Collection[Item]) -> bool:
    """Checks if a medium is contained in a collection of media, either identical or similar."""
    for existing_medium in media:
        if medium == existing_medium:  # Check identity
            return True
        if type(medium) == type(existing_medium):  # Check similarity
            try:
                if is_similar(medium, existing_medium):
                    return True
            except ValueError:  # Some videos cannot be read, needs fix in ezMM
                pass
    return False


def is_similar(medium1: Item, medium2: Item) -> bool:
    return medium1.cos_sim(medium2) > 0.85


async def _reformulate_claim(
        raw_claim: str, review: Review, article: Article, appearances: list[Appearance], media: list[Item],
) -> MultimodalSequence | None:
    """Uses strong LLMs to reformulate the claim based on the extracted media,
    the article, and the appearances."""

    try:
        prompt = Prompt(
            "veritas/prompts/normalize_claim.md",
            article=str(article.content)[:50_000],
            appearances=appearances_to_string(appearances)[:50_000],
            date=review.raw_claim_date.strftime("%B %d, %Y") if review.raw_claim_date else None,
            claimant=review.raw_claimant_name,
            language=review.language_name,
            claim=str(MultimodalSequence(*media, raw_claim))
        )
        llm = gemini_strong if prompt.has_videos() else gpt_strong
        reformulation = await llm.generate(prompt, extract="last_code_span")

        # Validate output
        if not reformulation:
            raise AssertionError("Reformulation is empty.")

        detect_hallucinated_media_refs(str(reformulation))

        # Add media to the reformulated version
        reformulation = " ".join(item for item in reformulation if isinstance(item, str))
        return MultimodalSequence(*media, reformulation)

    except AssertionError as e:
        await review.dismiss(f"Could not reformulate claim. Reason: {e}")

    except Exception as e:  # Runtime/code errors
        await review.dismiss(f"Could not normalize claim.\n"
                             f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def appearances_to_string(appearances: list[Appearance]) -> str:
    """Turns the appearances into an LLM-friendly string. Truncates lengthy appearances."""
    strings = []
    for appearance in appearances:
        if appearance.scraped_content:
            string = f"## From {appearance.url}\n{str(appearance.scraped_content)[:10_000]}"
            strings.append(string)
    return "\n\n".join(strings)


async def validate_claim(claim: Claim):
    if claim.is_rectified:
        await validate_rectified_claim(claim)
    else:
        await validate_original_claim(claim)


async def validate_original_claim(claim: Claim):
    """Validates a single claim with three separate quality checks."""
    if claim.is_rectified:
        # Rectified claims are validated at a different place
        raise ValueError("Claim is rectified.")

    n_words = len(claim.data.split())
    if n_words > 70:
        await claim.dismiss(f"Claim is too long ({n_words} words).")
        return

    try:
        # Check 1: Ambiguity
        # claim.is_ambiguous = await check_ambiguity(claim)

        # Check 2: Does any media expose a fact-check?
        claim.media_expose_verdict = await check_media_fact_check_exposure(claim)

        # Check 3: Does the claim expose a verdict through text or images?
        claim.text_exposes_verdict = await check_text_exposes_verdict(claim)

        # Check 4: Does the claim reference media that's not attached?
        claim.missing_referenced_media = await check_missing_referenced_media(claim)

        claim.check_completed = True

        # Evaluate results
        if claim.missing_referenced_media is None or claim.text_exposes_verdict is None:
            await claim.dismiss("Could not perform all validation tests.")
        elif claim.is_valid:
            logger.debug(f"Claim {claim.id} passed validation.")
            await claim.save_to_db()
        else:
            await claim.dismiss("Claim failed validation.")

    except QuotaExceededError:
        raise

    except Exception as e:
        error_msg = f"Error validating claim {claim.id}: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        await claim.dismiss(error_msg)


async def check_ambiguity(claim: Claim) -> Optional[bool]:
    # FIXME: This check is too strict. Almost all the failed claims are actually clear enough.
    prompt = Prompt(
        "veritas/prompts/claim_validation/check_ambiguity.md",
        claim=claim.data,
        claim_date=claim.date.strftime("%B %d, %Y")
    )
    response = await gpt_cheap.generate(prompt, extract="last_code_span")
    if response:
        return str(response).strip().lower() == "ambiguous"
    return None


async def check_media_fact_check_exposure(claim: Claim) -> bool:
    media = claim.as_multimodal_sequence().unique_items()
    for medium in media:
        if await check_medium_fact_check_exposure(medium):
            return True
    return False


async def check_medium_fact_check_exposure(medium: Item) -> bool | None:
    """Returns True if the medium contains clues that hint at a
     fact-check."""
    origin_domain = get_domain(medium.source_url)
    if origin_domain in PLATFORMS or origin_domain in ARCHIVING_SITES:
        return False  # We may assume that the medium is original, thus cannot contain fact-checking cues

    prompt = Prompt(
        "veritas/prompts/claim_validation/check_fact_check_exposure.md",
        medium=medium.reference,
    )
    llm = gemini_cheap if isinstance(medium, Video) else gpt_cheap
    response = await llm.generate(prompt, extract="last_code_span")
    if response:
        return str(response).strip().lower() == "exposing"


async def check_text_exposes_verdict(claim: Claim) -> bool | None:
    """Returns True if the claim text exposes a verdict through labels or fact-checking language."""

    # TODO: Rework to avoid response_format

    class VerdictExposureResult(BaseModel):
        exposes_verdict: bool = Field(
            description="Whether the claim text exposes a verdict.", examples=[True, False]
        )

    prompt = Prompt(
        "veritas/prompts/claim_validation/check_text_verdict_exposure.md",
        claim=claim.data,
    )
    response: VerdictExposureResult = await gpt_cheap.generate(prompt, response_format=VerdictExposureResult)
    if response:
        return response.exposes_verdict


async def check_missing_referenced_media(claim: Claim) -> bool | None:
    """Returns True if the claim text references media that is not attached."""

    # TODO: Rework to avoid response_format

    class MediaReferenceResult(BaseModel):
        has_image_reference: bool = Field(
            description="Whether the claim refers to one or more images.", examples=[True, False]
        )
        has_video_reference: bool = Field(
            description="Whether the claim refers to one or more videos.", examples=[True, False]
        )

    prompt = Prompt(
        "veritas/prompts/claim_validation/check_missing_media.md",
        claim=claim.data,
    )

    response: MediaReferenceResult = await gpt_cheap.generate(prompt, response_format=MediaReferenceResult)
    if response:
        mm_seq = claim.as_multimodal_sequence()

        # Return True if there's a mismatch (missing media)
        missing_image = response.has_image_reference and not mm_seq.has_images()
        missing_video = response.has_video_reference and not mm_seq.has_videos()

        return missing_image or missing_video
