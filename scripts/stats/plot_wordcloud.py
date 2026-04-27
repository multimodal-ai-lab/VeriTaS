import asyncio

from ezmm import MultimodalSequence

from scripts.stats.wordcloud_utils import plot_wordcloud_base
from veritas.db.veritas_db import db



async def plot_wordcloud(language: str, save: str | None = None, released: bool = None):
    print(f"Fetching claims for language: {language}...")
    claims = await db.get_claims_by_language(language, released=released)

    if not claims:
        print(f"No non-dismissed claims found for language: {language}")
        return

    print(f"Found {len(claims)} claims. Generating word cloud...")

    # Combine all claim data into one large string
    text = " ".join(_preprocess_claim(claim.data) for claim in claims)

    plot_wordcloud_base(
        text=text,
        language=language,
        title=f"Word Cloud of Claims ({language})",
        save=save
    )


async def plot_wordclouds(released: bool = None):
    await db.connect_maybe_initialize()
    languages = ["en", "es", "hi", "de", "pt", "fr", "tr", "ru", "pl", "ar"]
    for lang in languages:
        await plot_wordcloud(language=lang, save=f"plots/wordclouds/{lang}.pdf", released=released)


def _preprocess_claim(claim: str) -> str:
    mm_seq = MultimodalSequence(claim)
    text = [item for item in mm_seq if isinstance(item, str)]
    return " ".join(text)


if __name__ == "__main__":
    asyncio.run(plot_wordclouds(released=True))
