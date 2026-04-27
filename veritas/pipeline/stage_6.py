import traceback
from typing import Collection

from ezmm import Item, MultimodalSequence

from veritas import logger
from veritas.common import Claim, Prompt, Verdict
from veritas.common.annotation import Rating, PROPERTIES, Property
from veritas.common.annotation.rating import RatingAggregated, Category3Bin
from veritas.common.verdict import MediumVerdict
from veritas.ensemble import ensemble, ModelResponse
from veritas.models import QuotaExceededError, RateLimitError, gemini_cheap
from veritas.pipeline.util.stage import Stage
from veritas.util import run_with_semaphore
from veritas.util.parsing import extract_last, extract_all, extract_last_code_block

CERTAINTY_OPTIONS = {
    "certain": 1,
    "rather certain": 2 / 3,
    "rather uncertain": 1 / 3,
}
LABEL_REGEX = r"[a-zA-Z\- ]+"


class Stage6(Stage):
    """Predicts verdicts for claims without verdicts."""

    id = 6
    name = "Verdict Standardization"

    def __init__(self) -> None:
        super().__init__()
        self.quarters: dict[tuple[int, int], bool] = {}

    async def step(self, *,
                   start: tuple[int, int] = (2016, 1),
                   end: tuple[int, int] | None = None,
                   max_verdicts_per_quarter: int = 1200,
                   max_intact_claims_per_quarter: int = 500) -> None:
        items = await self.get_queued_items(
            start=start, end=end,
            max_verdicts_per_quarter=max_verdicts_per_quarter,
            max_intact_claims_per_quarter=max_intact_claims_per_quarter,
        )
        if items:
            await predict_verdicts(items)

    async def get_queued_items(self, *,
                               start: tuple[int, int],
                               end: tuple[int, int] | None,
                               max_verdicts_per_quarter: int,
                               max_intact_claims_per_quarter: int) -> list[Claim]:
        """Fetches a batch of claims without verdicts, respecting per-quarter limits."""
        import random
        from veritas.db import db
        from veritas.util.util import get_quarter_date_range, get_quarters

        if not self.quarters:
            self.quarters.update({q: False for q in get_quarters(start, end)})

        all_claims: list[Claim] = []
        for quarter, is_complete in self.quarters.items():
            if not is_complete:
                quarter_start, quarter_end = get_quarter_date_range(*quarter)
                n_claims_with_verdicts = await db.count_claims_with_verdicts(quarter_start, quarter_end)
                n_intact_claims = await db.count_intact_claims(quarter_start, quarter_end)

                if n_claims_with_verdicts < max_verdicts_per_quarter:
                    missing = max_verdicts_per_quarter - n_claims_with_verdicts
                    claims = await db.get_claims_without_verdicts(limit=missing,
                                                                  start_date=quarter_start,
                                                                  end_date=quarter_end)
                    if claims:
                        all_claims.extend(claims)
                        logger.debug(f"Added {len(claims)} original claims from Q{quarter[1]} {quarter[0]} to the queue.")

                if n_intact_claims < max_intact_claims_per_quarter:
                    missing = max_intact_claims_per_quarter - n_intact_claims
                    logger.debug(f"Q{quarter[1]} {quarter[0]} is missing {missing} claims.")
                    claims = await db.get_claims_without_verdicts(limit=missing,
                                                                  start_date=quarter_start,
                                                                  end_date=quarter_end,
                                                                  is_rectified=True)
                    if claims:
                        all_claims.extend(claims)
                        logger.debug(f"Added {len(claims)} rectified claims from Q{quarter[1]} {quarter[0]} to the queue.")
                else:
                    self.quarters[quarter] = True
                    logger.info(f"✅ Prediction of quarter {quarter} completed successfully!")

        random.shuffle(all_claims)
        if self.quarters and all(self.quarters.values()):
            self.done = True
        return all_claims


async def predict_verdicts(claims: Collection[Claim]) -> None:
    """Maps the raw verdicts to the standardized verdict schema."""
    # Each claim can be handled separately and concurrently
    claims = list(set(claims))
    logger.info(f"Predicting verdicts for {len(claims)} claims...")
    tasks = [predict_verdict_single(claim) for claim in claims]
    await run_with_semaphore(tasks, limit=20)
    logger.info("Verdict prediction complete.")


async def predict_verdict_single(claim: Claim) -> Verdict | None:
    current_verdict = await claim.current_verdict
    if current_verdict:
        logger.debug(f"Completing verdict for claim {claim.id}...")
    else:
        logger.debug(f"Predicting verdict for claim {claim.id}...")
    veracity = context_coverage = incorrect_contextualization = None
    media_verdicts = []

    original_claim = await claim.variant if claim.is_rectified else None
    original_verdict = await original_claim.current_verdict if original_claim else None

    try:
        # Assess the media
        media = MultimodalSequence(claim.data).unique_items()
        for medium in media:
            current_medium_verdict = current_verdict.get_medium_verdict(medium.reference) if current_verdict else None

            # Asses the authenticity (only if this is not a rectified claim)
            if claim.is_original:
                authenticity = await assess_property("authenticity", claim, medium)
            else:
                # Recycle the original verdict for the rectified claim
                assert original_verdict, "Rectified claim without original verdict!"
                original_medium_verdict = original_verdict.get_medium_verdict(medium.reference)
                assert original_medium_verdict, "Rectified claim without original medium verdict!"
                authenticity = original_medium_verdict.authenticity

            # Assess the contextualization
            current_contextualization = current_medium_verdict.contextualization if current_medium_verdict else None
            contextualization = await assess_property("contextualization", claim, medium,
                                                      incomplete_rating=current_contextualization)

            medium_verdict = MediumVerdict(reference=medium.reference,
                                           authenticity=authenticity,
                                           contextualization=contextualization)
            media_verdicts.append(medium_verdict)
            if contextualization.as_3_bin() == Category3Bin.NEGATIVE:
                incorrect_contextualization = True

        # Assess the veracity only if everything is correct so far
        if not incorrect_contextualization:
            # Assess the veracity
            current_veracity = current_verdict.veracity if current_verdict else None
            veracity = await assess_property("veracity", claim,
                                             incomplete_rating=current_veracity)

            if veracity.score > 0:
                # Assess the context coverage
                current_context_coverage = current_verdict.context_coverage if current_verdict else None
                context_coverage = await assess_property("context_coverage", claim,
                                                         incomplete_rating=current_context_coverage)

        # Construct verdict object
        if current_verdict:
            await current_verdict.save_to_db()
            verdict = current_verdict
        else:
            verdict = Verdict(
                claim_id=claim.id,
                review_ids=claim.review_ids,
                media_verdicts=media_verdicts,
                veracity=veracity,
                context_coverage=context_coverage,
            )
            await verdict.save_to_db()  # Automatically invalidates old verdicts of this claim

            claim.verdict_ids.add(verdict.id)
            await claim.save_to_db()

    except (QuotaExceededError, RateLimitError):
        raise

    except Exception as e:
        logger.error(f"Could not predict verdict of claim {claim.id}: {e}")
        logger.debug(traceback.format_exc())
        await claim.dismiss(f"Could not predict verdict: {e}")
        return None

    await validate_verdict(verdict, claim)

    if not claim.dismissed:
        # Update stage tracker for all associated reviews
        reviews = await claim.reviews
        for review in reviews:
            await review.set_stage(7 if claim.is_rectified else 6)

    logger.debug(f"Successfully predicted verdict for claim {claim.id}.")
    return verdict


async def assess_property(
        property_name: str,
        claim: Claim,
        medium: Item | None = None,
        incomplete_rating: RatingAggregated | None = None,
) -> RatingAggregated:
    """Prompts the LLMs to assess the specified property, subject to the
    claim or, if specified, the given Medium. If an incomplete rating is provided,
    fills in the ratings for the missing LLMs."""
    logger.debug(f"Assessing property {property_name} for claim {claim.id}...")
    prop = PROPERTIES[property_name]

    # Compose prompt
    subject = "Claim" if medium is None else medium.kind.capitalize()
    reviews = await claim.reviews
    articles = [await review.article for review in reviews]
    publishers = [await review.publisher for review in reviews]
    article_contents = [article.content for article in articles]
    prompt = Prompt("veritas/prompts/assess_property.md.j2",
                    reviews=zip(article_contents, publishers),
                    claim=None if prop.name == "Authenticity" else claim,
                    medium=medium,
                    subject=subject,
                    property=prop)

    def extract_label(response: ModelResponse) -> Rating:
        return extract_label_from_response(response, prop)

    def aggregate_labels(ratings: list[Rating]) -> list[Rating]:
        return ratings

    # Select models to call if incomplete rating is provided
    missing_model_names = None  # Call all models by default
    if incomplete_rating:
        missing_model_names = [model_name for model_name in ensemble.model_names
                               if model_name not in incomplete_rating.raters]
        if missing_model_names is not None and len(missing_model_names) == 0:
            return incomplete_rating  # No models needed to call, the rating is complete

    # Execute prompt
    ratings = await ensemble.generate(prompt,
                                      response_extraction_fn=extract_label,
                                      aggregation_fn=aggregate_labels,
                                      models=missing_model_names)
    if not ratings:
        raise RuntimeError(f"All models failed to assess {property_name} for claim {claim.id}. "
                           f"Possible content policy violations or model failures.")

    if incomplete_rating:
        incomplete_rating.add_ratings(ratings)
        rating_aggregated = incomplete_rating
    else:
        rating_aggregated = RatingAggregated(individual_ratings=ratings, rater="ensemble")

    await merge_justifications(rating_aggregated, property_name, claim, medium)
    return rating_aggregated


def extract_label_from_response(response: ModelResponse, prop: Property) -> Rating:
    """Extracts the label from the LLM response for the given property."""
    output = str(response.output) if response.output is not None else ""

    # Handle None or empty output
    if not output or output == "None":
        raise ValueError(f"Model {response.model} returned empty/None response")

    # Handle content policy refusals
    refusal_patterns = [
        "i'm sorry, i can't assist",
        "i cannot assist",
        "i'm unable to",
        "i can't help with that",
        "i cannot provide",
        "i must decline",
    ]
    output_lower = output.lower()
    for pattern in refusal_patterns:
        if pattern in output_lower and len(output) < 200:  # Short refusal messages
            raise ValueError(f"Model {response.model} refused to respond (content policy)")

    candidate_categories = [cat.lower() for cat in prop.categories] + ["unknown"]

    category = extract_last(output, "`", allowed_symbols=LABEL_REGEX)
    certainty = extract_last(output, "_", allowed_symbols=LABEL_REGEX)
    tags = extract_all(output, ":", allowed_symbols=LABEL_REGEX) if prop.tags else []
    explanation = extract_last_code_block(output)

    # Validate the response
    # Category
    assert category, f"No category provided by model {response.model}!\nResponse was:\n{output}"
    category = category.lower().strip()
    assert category in candidate_categories, (f"Invalid category '{category}' for property {prop.name}. "
                                              f"Must be one of {candidate_categories}.")
    is_positive = category == prop.positive_category.lower()

    # Certainty
    if not category == "unknown":
        assert certainty, f"No certainty provided.\nResponse was:\n{output}"
        assert certainty.lower() in CERTAINTY_OPTIONS, f"Invalid certainty '{certainty}' for property {prop.name}."

    # Tags
    candidate_tags = prop.positive_tags if is_positive else prop.negative_tags
    candidate_tags = [tag.name.lower() for tag in candidate_tags]
    for tag in tags:
        assert tag.lower() in candidate_tags, (f"Invalid tag '{tag}' for property {prop.name}. Must be one "
                                               f"of {candidate_tags}.")

    # Explanation
    assert explanation, f"No explanation provided."

    # Turn into Label object
    tendency = 0 if category == "unknown" else CERTAINTY_OPTIONS[certainty.lower()] * (1 if is_positive else -1)
    return Rating(
        score=tendency,
        explanation=explanation,
        tags=tags,
        rater=response.model.specifier
    )


async def validate_verdict(verdict: Verdict, claim: Claim) -> None:
    """Checks if...
     1. The verdict has at least three LLM predictions.
     2. The agreement level within the verdict is sufficient.
     3. The claim is intact if it's a rectified claim."""
    if verdict.n_ratings < 3:
        await claim.dismiss("Not enough predictions.")
    elif not verdict.sufficient_agreement:
        await claim.dismiss("Verdict agreement too low.")
    elif claim.is_rectified:
        if verdict.integrity.as_3_bin() != Category3Bin.POSITIVE:
            await claim.dismiss("Rectified claim is still compromised.", also_dismiss_reviews=False)
        # else:
        #     # Discard original claim in favor of rectified version
        #     original_claim = await claim.variant
        #     await original_claim.dismiss("In favor of rectified version.", also_dismiss_reviews=False)


async def merge_justifications(
        rating: RatingAggregated,
        property_name: str,
        claim: Claim,
        medium: Item | None = None,
) -> None:
    """Uses a cheap LLM to generate a single, concise justification for the
    given aggregated rating using the individual ratings."""
    justifications = "\n\n".join(r.explanation for r in rating.individual_ratings if r.explanation)
    prop = PROPERTIES[property_name]
    claim_text = " ".join(i for i in MultimodalSequence(claim.data) if isinstance(i, str))
    prompt = Prompt("veritas/prompts/merge_justifications.md.j2",
                    justifications=justifications,
                    property=prop,
                    subject="Claim" if medium is None else medium.kind.capitalize(),
                    claim=claim_text,
                    rating=rating.category_str_7_bin(prop))
    rating.explanation = str(await gemini_cheap.generate(prompt, extract="last_code_span"))
