import asyncio

from scripts.stats.wordcloud_utils import plot_wordcloud_base
from veritas.db.veritas_db import db


async def plot_review_wordcloud(language: str, save: str | None = None):
    print(f"Fetching reviews for language: {language}...")
    # Fetching reviews from stage 1 to 6
    reviews = []
    for stage in range(1, 7):
        stage_reviews = await db.get_reviews(stage=stage, language=language, dismissed=False)
        reviews.extend(stage_reviews)

    if not reviews:
        print(f"No non-dismissed reviews found for language: {language}")
        return

    print(f"Found {len(reviews)} reviews. Generating word cloud...")

    # Extract raw_rating and combine into one large string
    ratings = [r.raw_rating for r in reviews if r.raw_rating]
    text = "\nand\n".join(ratings)

    plot_wordcloud_base(
        text=text,
        language=language,
        title=f"Word Cloud of Raw Ratings ({language})",
        save=save
    )


async def main(language: str):
    await db.connect_maybe_initialize()

    await plot_review_wordcloud(language=language, save=f"plots/wordclouds/raw_ratings_{language}.pdf")


if __name__ == "__main__":
    asyncio.run(main(
        language="en"
    ))
