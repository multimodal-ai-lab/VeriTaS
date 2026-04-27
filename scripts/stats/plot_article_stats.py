import argparse
import asyncio
from typing import Sequence

import matplotlib.pyplot as plt

from veritas.db import db


async def fetch_article_lengths() -> list[dict[str, int]]:
    """Fetch lengths of scraped_page and extracted_article for all articles.

    Uses SQL char_length to avoid transferring large text blobs.
    """
    await db.connect_maybe_initialize()
    query = """
        SELECT char_length(scraped_page) AS scraped_len,
               char_length(extracted_article) AS extracted_len
        FROM articles
        WHERE scraped_page IS NOT NULL
    """
    records: Sequence = await db.fetch(query)  # type: ignore[attr-defined]
    return [
        {"scraped": int(r["scraped_len"]), "extracted": int(r["extracted_len"]) if r["extracted_len"] is not None else 0}
        for r in records
    ]


async def async_main(save: str | None = None, show: bool = True, bins: int = 100):
    data = await fetch_article_lengths()

    if not data:
        print("No article content found in the database.")
        return

    scraped_lengths = [d["scraped"] for d in data]
    extracted_lengths = [d["extracted"] for d in data if d["extracted"] > 0]

    # Print summary
    if extracted_lengths:
        reductions = [d["scraped"] - d["extracted"] for d in data if d["extracted"] > 0]
        avg_scraped = sum(scraped_lengths) / len(scraped_lengths)
        avg_extracted = sum(extracted_lengths) / len(extracted_lengths)
        avg_reduction = sum(reductions) / len(reductions)
        print(f"Summary of {len(data)} articles:")
        print(f"  Average scraped_page length:    {avg_scraped:,.0f} chars")
        print(f"  Average extracted_article length: {avg_extracted:,.0f} chars")
        print(f"  Average length reduction:        {avg_reduction:,.0f} chars")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot 1: Scraped Page Lengths
    ax1.hist(scraped_lengths, bins=bins, color="#4C78A8", edgecolor="white", range=(0, 50_000))
    ax1.set_title("Distribution of Article scraped_page Lengths")
    ax1.set_xlabel("scraped_page length (characters)")
    ax1.set_ylabel("Number of articles")
    ax1.grid(axis="y", alpha=0.2)

    # Plot 2: Extracted Article Lengths
    if extracted_lengths:
        ax2.hist(extracted_lengths, bins=bins, color="#F58518", edgecolor="white", range=(0, 50_000))
        ax2.set_title("Distribution of Article extracted_article Lengths")
        ax2.set_xlabel("extracted_article length (characters)")
        ax2.set_ylabel("Number of articles")
        ax2.grid(axis="y", alpha=0.2)
    else:
        ax2.text(0.5, 0.5, "No extracted content available", ha='center', va='center')
        ax2.set_title("Distribution of Article extracted_article Lengths")

    plt.tight_layout()

    if save:
        plt.savefig(save)
        print(f"Saved histograms to {save}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot distribution of string length of articles (scraped_page and extracted_article)"
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save the plot image")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins")
    args = parser.parse_args()

    asyncio.run(async_main(save=args.save, show=not args.no_show, bins=args.bins))


if __name__ == "__main__":
    main()
