import logging
from datetime import date, datetime

from bs4 import BeautifulSoup
from scrapemm import retrieve
from scrapemm.common import ScrapingResponse

from veritas.common import Article, Review, Prompt
from veritas.db import db
from veritas.models import gpt_nano
from veritas.pipeline import max_video_size, min_scraped_article_length, min_extracted_article_length, \
    max_extracted_article_length
from veritas.pipeline.util.stage import Stage
from veritas.util import get_domain, run_with_semaphore
from veritas.util.parsing import extract_all

logger = logging.getLogger("VeriTaS")

PAYWALL_DOMAINS = {
    "washingtonpost.com",
}

PUBLISHED_META_TAGS = [
    ("property", "article:published_time"),
    ("name", "pubdate"),
    ("name", "publish_date"),
    ("name", "date"),
    ("property", "og:pubdate"),
    ("name", "dc.date"),
]

MODIFIED_META_TAGS = [
    ("property", "article:modified_time"),
    ("name", "lastmod"),
    ("name", "modified"),
    ("name", "revised"),
    ("name", "updated"),
    ("name", "date.modified"),
    ("name", "dc.modified"),
]


class Stage3(Stage):
    """Continuously processes articles for reviews at stage 2."""

    id = 3
    name = "Article Scraping"

    async def step(self, *,
                   start: tuple[int, int] = (2016, 1),
                   end: tuple[int, int] | None = None,
                   batch_size: int = 100) -> None:
        reviews = await self.get_queued_items(start=start, end=end, batch_size=batch_size)
        if reviews:
            await process_articles(reviews)

    async def get_queued_items(self, *,
                               start: tuple[int, int],
                               end: tuple[int, int] | None,
                               batch_size: int) -> list[Review]:
        """Fetches a batch of reviews at stage 2 within the given date range."""
        from veritas.util.util import get_quarter_date_range
        start_date, _ignored = get_quarter_date_range(*start)
        end_date = None
        if end:
            _ignored, end_date = get_quarter_date_range(*end)
        return await db.get_reviews(stage=2, limit=batch_size,
                                    start_date=start_date, end_date=end_date)


async def process_articles(reviews: list[Review]):
    """Retrieves the articles of all given reviews, if known, otherwise
    creates new article instances."""
    logger.info(f"Checking articles of {len(reviews)} reviews...")

    # Exclude scraped articles and paywalled articles
    urls = set()
    for review in reviews:
        if not await review.has_article():
            url = str(review.url)
            if get_domain(url) in PAYWALL_DOMAINS:
                await review.dismiss("Article is paywalled/blocked.")
            else:
                urls.add(url)

    # Scrape all unscraped articles concurrently and dynamically
    logger.info(f"Downloading articles of {len(urls)} reviews...")
    responses = await retrieve(list(urls), prioritize="speed", show_progress=False, max_video_size=max_video_size)
    assert isinstance(responses, list)
    scraped_pages = [response.content for response in responses]
    url_to_response: dict[str, ScrapingResponse] = dict(zip(urls, responses))
    url_to_scraped = dict(zip(urls, scraped_pages))

    # Log retrieval statistics
    if urls:
        n_successes = sum(response.successful for response in responses)
        logger.info(f"Successfully retrieved {n_successes} articles from {len(urls)} URLs.")
        avg_time = sum(r.retrieval_time for r in responses) / len(responses)
        logger.info(f"Avg time per URL: {avg_time:.2f}s")
        max_time = max(r.retrieval_time for r in responses)
        logger.info(f"Max time per URL: {max_time:.2f}s")
        sorted_responses = sorted(responses, key=lambda r: r.retrieval_time, reverse=True)
        logger.info(
            f"Slowest 10 URLs:\n{'\n'.join(f"{r.retrieval_time:.1f}s | {r.url}" for r in sorted_responses[:10])}")

    # Extract article content from scraped pages concurrently
    logger.info("Extracting article content...")
    tasks, success_urls = zip(*[(extract_article_content(str(page)), url)
                                for url, page in url_to_scraped.items()
                                if page])
    extracted_contents = await run_with_semaphore(tasks, limit=200)
    url_to_extracted = dict(zip(success_urls, extracted_contents))

    # Validate scraped content and save to DB. Must be sequential to avoid duplicates
    for review in reviews:
        if not await review.has_article():
            # Register new article
            url = str(review.url)
            scraped = url_to_scraped.get(url)
            if scraped:
                article = Article(url=review.url,
                                  publisher_id=review.publisher_id,
                                  review_ids={review.id},
                                  scraped_page=str(scraped),
                                  extracted_article=url_to_extracted.get(url))
                await article.save_to_db()
                if await validate_article(article):
                    await review.set_stage(3)
            else:
                error_msgs = ""
                response = url_to_response[url]
                if response.errors:
                    error_msgs = " Errors by scraping method: "
                    error_msgs += ", ".join(f"{method}: {error}" for method, error in response.errors.items())
                await review.dismiss(reason=f"Unable to scrape article.{error_msgs}")
        else:
            await review.set_stage(3)


async def validate_article(article: Article) -> bool:
    """Checks whether the article content has sufficient length."""
    if not article.dismissed:
        if article.extracted_article:  # Extracted version takes precedence
            if len(article.extracted_article) < min_extracted_article_length:
                await article.dismiss(reason="Extracted article content too short.")
                return False
            elif len(article.extracted_article) > max_extracted_article_length:
                await article.dismiss(reason="Extracted article content too long.")
                return False
        elif article.scraped_page:
            if len(article.scraped_page) < min_scraped_article_length:
                await article.dismiss(reason="Scraped article content too short.")
                return False
            elif len(article.scraped_page) > max_extracted_article_length:  # Serves also as max for raw scrape
                await article.dismiss(reason="Scraped article content too long.")
                return False
    return True


async def register_new_article(review: Review) -> Article:
    """Scrapes the actual review webpage and returns a new review instance
    from the information of the ClaimReview instance."""
    publisher = await review.publisher
    article = Article(url=review.url, publisher_id=publisher.id, review_ids={review.id})
    await article.save_to_db()
    return article


def extract_date_meta(html: str) -> date | datetime | None:
    date_str = extract_meta(html, PUBLISHED_META_TAGS)
    if date_str:
        return datetime.fromisoformat(date_str).replace(tzinfo=None)


def extract_modified_meta(html: str) -> date | datetime | None:
    date_str = extract_meta(html, MODIFIED_META_TAGS)
    if date_str:
        return datetime.fromisoformat(date_str).replace(tzinfo=None)


def extract_meta(html: str, meta_tags: list[tuple[str, str]]) -> str | None:
    """Returns the first occurrence of a meta tag matching any of the
    given `meta_tags` in the given HTML string."""
    soup = BeautifulSoup(html, "html.parser")

    for attr, key in meta_tags:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            return tag["content"].strip()


async def extract_article_content(page: str) -> str | None:
    """Takes the full scrape of the article webpage and returns the relevant
    article content (particularly the title and the main text). Returns None
    if extraction fails."""
    if not page:
        return None

    # Truncate to respect context window
    page = page[:100_000]

    # Enumerate lines to allow for line number references in the prompt
    page_lines = page.splitlines()
    page_lines_enumerated = "\n".join(
        [f"{i}: {line}" for i, line in enumerate(page_lines)]
    )

    # Format and execute prompt
    prompt = Prompt(file_path="veritas/prompts/extract_article.md", scraped_page=page_lines_enumerated)
    response = await gpt_nano.generate(prompt, resolve_media=False)

    if response:
        try:
            start_lineno, end_lineno = extract_all(str(response), delimiter="`")
            start_lineno = int(start_lineno)
            end_lineno = int(end_lineno)
            article_lines = page_lines[start_lineno:end_lineno + 1]
            return "\n".join(article_lines)
        except Exception:
            pass
