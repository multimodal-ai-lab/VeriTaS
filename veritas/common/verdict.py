from typing import Optional, TYPE_CHECKING

from ezmm import Item
from pydantic import BaseModel, Field

from veritas.common.base_model import VeritasBaseModel
from veritas.common.annotation import PROPERTIES, Rating, RatingAggregated

if TYPE_CHECKING:
    from veritas.common.claim import Claim


class MediumVerdict(BaseModel):
    """Classification of a medium (image or video) based on the standardized labeling scheme."""

    reference: str  # Reference to the medium
    authenticity: RatingAggregated
    contextualization: RatingAggregated

    def model_post_init__(self):
        assert self.medium

    @property
    def medium(self) -> Optional[Item]:
        return Item.from_reference(self.reference)


class Verdict(VeritasBaseModel):
    """Classifications of a claim based on the standardized annotation scheme.
    Can comprise one or more ratings from different predictors."""
    claim_id: int  # The claim that the verdict is scoring
    review_ids: set[int]  # The reviews this verdict is based on

    # Properties from the annotation scheme
    media_verdicts: list[MediumVerdict] = Field(default_factory=list)
    veracity: RatingAggregated | None = None
    context_coverage: RatingAggregated | None = None

    @property
    def integrity(self) -> Rating:
        compromising_rating = self.compromising_rating
        explanation = compromising_rating.to_text(PROPERTIES[self.compromising_property_name])
        return Rating(
            score=compromising_rating.score,
            explanation=f"{explanation}\n\n{compromising_rating.explanation}",
            rater=compromising_rating.rater
        )

    @property
    def compromising_property_name(self) -> str:
        """Determines the property that is decisive for the integrity and returns
        its name. It's the property of [context_coverage, veracity, contextualization]
        with the lowest score. On score ties, the precedence is context_coverage >
        veracity > contextualization."""
        candidates = []
        if self.context_coverage:
            candidates.append("context_coverage")
        if self.veracity:
            candidates.append("veracity")
        if self.contextualization:
            candidates.append("contextualization")
        return min(candidates, key=lambda c: getattr(self, c).score)

    @property
    def contextualization(self) -> Optional[Rating]:
        """Returns the contextualization rating that scores the worst (lowest)
        among all media verdicts."""
        ratings = [medium_verdict.contextualization for medium_verdict in self.media_verdicts]
        if ratings:
            return min(ratings, key=lambda r: r.score)
        return None

    @property
    def authenticity(self) -> Optional[Rating]:
        """Returns the authenticity rating that scores the worst (lowest)
        among all media verdicts."""
        ratings = [medium_verdict.authenticity for medium_verdict in self.media_verdicts]
        if ratings:
            return min(ratings, key=lambda r: r.score)
        return None

    @property
    def compromising_rating(self) -> RatingAggregated:
        """The rating that scores the worst (lowest) among all integrity-influencing
        properties."""
        return getattr(self, self.compromising_property_name)

    @property
    def sufficient_agreement(self) -> bool:
        """Returns True if the highest internal score difference in the compromising property is below
        a certain threshold."""
        return self.compromising_rating.sufficient_agreement

    @property
    def n_ratings(self) -> int:
        """Returns the number of ratings included in the compromising property."""
        compromising_rating = self.compromising_rating
        if isinstance(compromising_rating, RatingAggregated):
            return compromising_rating.n_ratings
        return 1

    @property
    async def claim(self) -> "Claim":
        from veritas.db import db
        return await db.get_claim_by_id(self.claim_id)

    def get_medium_verdict(self, medium_ref: str) -> MediumVerdict | None:
        """Returns the MediumVerdict for the given medium reference."""
        for medium_verdict in self.media_verdicts:
            medium = medium_verdict.medium
            assert medium
            if medium.reference == medium_ref:
                return medium_verdict
        return None

    def is_complete(self, target: int = 4) -> bool:
        """Returns True if all properties are aggregated from at least `target`
        individual ratings."""
        if target == 1:
            return True

        return (all(mv.authenticity.n_ratings >= target for mv in self.media_verdicts)
                and all(mv.contextualization.n_ratings >= target for mv in self.media_verdicts)
                and (self.veracity is None or self.veracity.n_ratings >= target)
                and (self.context_coverage is None or self.context_coverage.n_ratings >= target))
