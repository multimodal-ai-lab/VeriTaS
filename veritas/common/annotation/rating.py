from collections import Counter
from dataclasses import field
from enum import Enum

from pydantic.dataclasses import dataclass

from veritas.common.annotation.property import Property


class Category3Bin(int, Enum):
    """Labels for the 3-bin discretization of ratings."""
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1


class Category7Bin(float, Enum):
    """Labels for the 7-bin discretization of ratings."""
    NEG_CERTAIN = -1
    NEG_RATHER_CERTAIN = -2 / 3
    NEG_RATHER_UNCERTAIN = -1 / 3
    NEUTRAL = 0
    POS_RATHER_UNCERTAIN = 1 / 3
    POS_RATHER_CERTAIN = 2 / 3
    POS_CERTAIN = 1


def map_3_bins(score: float) -> Category3Bin:
    """Discretizes the rating score by mapping it to one of 3 discrete Categories."""
    if score < -1 / 3:
        return Category3Bin.NEGATIVE
    elif score > 1 / 3:
        return Category3Bin.POSITIVE
    else:
        return Category3Bin.NEUTRAL


def map_7_bins(score: float) -> Category7Bin:
    """Discretizes the rating score by mapping it to one of 7 discrete Categories."""
    if score < -5 / 6:
        return Category7Bin.NEG_CERTAIN
    elif score < -3 / 6:
        return Category7Bin.NEG_RATHER_CERTAIN
    elif score < -1 / 6:
        return Category7Bin.NEG_RATHER_UNCERTAIN
    elif score > 5 / 6:
        return Category7Bin.POS_CERTAIN
    elif score > 3 / 6:
        return Category7Bin.POS_RATHER_CERTAIN
    elif score > 1 / 6:
        return Category7Bin.POS_RATHER_UNCERTAIN
    else:
        return Category7Bin.NEUTRAL


@dataclass
class Rating:
    """An assessment of a property, encoding uncertainty."""
    score: float  # The scalar rating, in [-1, 1]
    rater: str  # Name of the original model/human who produced the rating
    explanation: str | None = None  # Textual justification for the rating
    tags: list[str] = field(default_factory=list)  # Any applicable tags to categorize the rating further

    def as_3_bin(self) -> Category3Bin:
        return map_3_bins(self.score)

    def as_7_bin(self) -> Category7Bin:
        return map_7_bins(self.score)

    def category_str_3_bin(self, prop: Property) -> str:
        match self.as_3_bin():
            case Category3Bin.NEGATIVE:
                return prop.negative_category
            case Category3Bin.POSITIVE:
                return prop.positive_category
            case Category3Bin.NEUTRAL:
                return "Not Enough Information"

    def category_str_7_bin(self, prop: Property) -> str:
        """Converts the rating to a human-readable string using the
        semantics of the given Property."""
        match self.as_7_bin():
            case Category7Bin.NEG_CERTAIN:
                return f"{prop.negative_category} (certain)"
            case Category7Bin.NEG_RATHER_CERTAIN:
                return f"{prop.negative_category} (rather certain)"
            case Category7Bin.NEG_RATHER_UNCERTAIN:
                return f"{prop.negative_category} (rather uncertain)"
            case Category7Bin.NEUTRAL:
                return "Not Enough Information"
            case Category7Bin.POS_RATHER_UNCERTAIN:
                return f"{prop.positive_category} (rather uncertain)"
            case Category7Bin.POS_RATHER_CERTAIN:
                return f"{prop.positive_category} (rather certain)"
            case Category7Bin.POS_CERTAIN:
                return f"{prop.positive_category} (certain)"

    def to_text(self, prop: Property) -> str:
        """Converts the rating to a human-readable string using the
        semantics of the given Property."""
        cat_name = self.category_str_3_bin(prop)
        return f"The {prop.name} is {cat_name}."


@dataclass
class RatingAggregated(Rating):
    """Merges a list of ratings into a single rating by averaging the scores and
    applying majority vote on the tags."""

    # Hide attributes from constructor (compute them later dynamically)
    score: float = field(init=False)
    tags: list[str] = field(init=False)

    individual_ratings: list[Rating] = None

    explanation: str | None = None  # Will be generated separately from individual explanations

    def __post_init__(self):
        assert self.individual_ratings, "Aggregated rating must have at least one rating."

        ratings = self.individual_ratings

        # Average tendency
        self.score = sum(rating.score for rating in ratings) / len(ratings)

        # Count tag occurrences
        tag_counts = Counter(tag for label in ratings for tag in label.tags)
        # Keep tags with a count of at least half of the "votes"
        self.tags = [tag for tag, count in tag_counts.items() if count >= len(ratings) / 2]

    @property
    def scores(self) -> list[float]:
        """Returns the individual scores of the ratings."""
        return [rating.score for rating in self.individual_ratings]

    @property
    def raters(self) -> list[str]:
        """Returns the individual rater names of the ratings."""
        return [rating.rater for rating in self.individual_ratings]

    @property
    def max_score_diff(self) -> float:
        """Returns the difference between the highest and lowest scores. Useful to
        analyze internal disagreements between predictors."""
        scores = self.scores
        return max(scores) - min(scores)

    @property
    def sufficient_agreement(self) -> bool:
        """Returns True if the highest internal score difference is at most 1."""
        from veritas.pipeline import sufficient_agreements_threshold
        return self.max_score_diff <= sufficient_agreements_threshold

    @property
    def n_ratings(self) -> int:
        """Returns the number of individual ratings aggregated into this rating."""
        return len(self.individual_ratings)

    def add_ratings(self, ratings: list[Rating]) -> None:
        """Adds a list of ratings to the aggregated rating."""
        self.individual_ratings.extend(ratings)
        self.__post_init__()
