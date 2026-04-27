import json
import re
from collections.abc import Iterable
from datetime import datetime

import aiohttp
from asyncpg import DataError, UniqueViolationError
from bs4 import BeautifulSoup
from langdetect import detect
from pydantic import HttpUrl, ValidationError
from scrapemm.util import URL_REGEX

from veritas import logger
from veritas.common import Appearance, Review
from veritas.common.appearance import appearance_from_url
from veritas.db import db
from veritas.pipeline.util.cr_parsing import jsonld_parser, microdata_parser, custom_parser
from veritas.pipeline.util.datacommons_feed import DataCommonsFeedRetriever
from veritas.pipeline.util.google_factcheck_explorer import GoogleFactCheckExplorerRetriever
from veritas.pipeline.util.stage import Stage
from veritas.util import run_with_semaphore
from veritas.util.parsing import get_lang_from_code, determine_date
from veritas.util.scraping import get_static_htmls
from veritas.util.url import get_domain, is_domain_root
from veritas.util.util import validate, get_quarter_start_date

datacommons = DataCommonsFeedRetriever()
gfce = GoogleFactCheckExplorerRetriever()


class Stage1(Stage):
    """Continuously retrieves new reviews."""

    id = 1
    name = "Review Discovery"
    db_max_connections = 10
    default_interval_seconds = 60 * 60 * 24

    async def step(self, *, start: tuple[int, int] = (2016, 1)) -> None:
        await retrieve_reviews(after=get_quarter_start_date(start))


async def retrieve_reviews(after: datetime = None):
    """Scans Google Fact Check Tools API and DataCommons for new ClaimReviews.
    Processes them and saves them into the DB."""
    logger.info("Retrieving reviews...")

    # 1. Retrieve all the latest raw ClaimReview dictionaries and save new ones in the DB
    datacommons_crs, google_crs = await _retrieve_latest_claim_reviews(after)
    await _save_new_claim_reviews(datacommons_crs, google_crs)

    # 2. Try to get the original raw ClaimReview from the review article page directly,
    # parse the CLaimReview data and save structured info in the DB
    while reviews := await db.get_reviews(stage=0, limit=1000, dismissed=False):
        await _retrieve_claim_reviews_directly(reviews)
        await _process_raw_claim_reviews_from_reviews(reviews)

    logger.info(f"Retrieved and processed {len(reviews)} reviews.")


async def _retrieve_latest_claim_reviews(after: datetime = None) -> tuple[list[dict], list[dict]]:
    datacommons_crs = datacommons.retrieve(after=after, latest=False)
    google_crs = gfce.retrieve(after=after, latest=False)
    logger.debug(f"Retrieved {len(datacommons_crs) + len(google_crs)} ClaimReviews.")
    return datacommons_crs, google_crs


async def _save_new_claim_reviews(datacommons_crs: list[dict], google_crs: list[dict]):
    """Saves all ClaimReview dictionaries that are new to the DB.
    Returns the corresponding Review objects for all new/updated reviews."""
    logger.debug("Saving new ClaimReviews...")
    if datacommons_crs:
        datacommons_crs = _claim_reviews_to_organized_dict(datacommons_crs)
        await db.save_claim_review(datacommons_crs, source="datacommons")
    if google_crs:
        google_crs = _claim_reviews_to_organized_dict(google_crs)
        await db.save_claim_review(google_crs, source="google")


def _claim_reviews_to_organized_dict(crs: list[dict]) -> dict:
    """Organizes ClaimReviews in a dictionary, identified by URL-claim-pairs.
    Implicitly drop sCRs that are not clearly identifiable, i.e., are lacking
    URL or claim fields."""
    dict_crs = dict()
    for cr in crs:
        claim = cr.get("claimReviewed") or cr.get("claim")
        url = cr.get("url")
        if claim and url and not is_domain_root(url):  # Skip reviews that only point at org main site
            # Sanitize URLs
            if isinstance(url, str) and not url.startswith("http"):
                url = "https://" + url
            dict_crs[(url, claim)] = cr
    return dict_crs


async def keep_new(crs: list[dict]) -> list[dict]:
    """Removes known (i.e., DB-contained) ClaimReviews from the list of ClaimReviews."""
    logger.debug(f"Filtering {len(crs)} ClaimReviews for new ones...")
    url_claim_pairs = [(cr.get("url"), cr.get("claimReviewed")) for cr in crs]
    is_known = await db.has_claim_reviews(url_claim_pairs)
    return [cr for cr, known in zip(crs, is_known) if not known]


async def _retrieve_claim_reviews_directly(reviews: list[Review]):
    """Adds additional ClaimReview information retrieved from the
    fact-checking article websites directly. Also obtains the review's language."""
    logger.debug(f"Retrieving ClaimReviews from {len(reviews)} websites directly...")

    # Filter for reviews that have no direct ClaimReview yet
    reviews = [review for review in reviews if review.direct_claim_review is None]

    # Get their URLs, download and extract the direct ClaimReviews
    urls = [review.url for review in reviews]
    direct_crs_all, htmls = await download_and_extract_claim_reviews(urls)
    direct_crs_all = dict(zip(urls, direct_crs_all))
    htmls = dict(zip(urls, htmls))

    n_direct_crs = sum(bool(cr) for cr in direct_crs_all.values())
    logger.debug(
        f"Successfully retrieved ClaimReviews from {n_direct_crs} URLs directly.\n"
        f"Matching them to reviews and detecting their languages..."
    )

    n_matches = 0
    for review in reviews:
        # Obtain direct ClaimReview
        direct_crs = direct_crs_all.get(review.url)
        if direct_crs:
            for direct_cr in direct_crs:
                direct_claim = direct_cr.get("claimReviewed") or direct_cr.get("claim")
                if direct_claim == review.raw_claim:
                    review.direct_claim_review = direct_cr
                    # await review.save_to_db()  Review will be saved later
                    n_matches += 1
                    break

        # Detect and save language
        lang_code = detect_language(review.raw_claim)  # The claim has the highest precedence

        if not lang_code:
            if html := htmls.get(review.url):
                soup = BeautifulSoup(html, "html.parser")
                lang_code = (
                        get_lang_from_html(soup)
                        or extract_meta_language(soup)
                        or detect_language(soup.get_text())
                )

        language = get_lang_from_code(lang_code)
        if language:
            review.language = language.language

    logger.debug(f"Assigned {n_matches} direct ClaimReviews.")


async def _process_raw_claim_reviews_from_reviews(reviews: list[Review]):
    """Reads the raw ClaimReview dictionaries from the reviews and saves the relevant
    data into the corresponding Review object."""
    logger.debug(f"Processing ClaimReview dicts of {len(reviews)} reviews...")
    await run_with_semaphore(
        (_process_raw_claim_reviews_from_review(review) for review in reviews),
        limit=1000,
    )


async def _process_raw_claim_reviews_from_review(review: Review):
    """Reads the raw ClaimReview dictionaries from the review and saves the relevant
    data into the Review object. Precedence:
    direct_claim_review > datacommons_claim_review > google_claim_review."""

    direct_cr = review.direct_claim_review
    datacommons_cr = review.datacommons_claim_review
    google_cr = review.google_claim_review

    # Extract the data according to precedence
    published = _get_published(direct_cr) or _get_published(datacommons_cr) or _get_published(google_cr)
    modified = _get_modified(direct_cr) or _get_modified(datacommons_cr) or _get_modified(google_cr)
    rating = _get_rating(direct_cr) or _get_rating(datacommons_cr) or _get_rating(google_cr)
    appearances = set(
        await _get_appearances_for_claim_review(direct_cr)
        + await _get_appearances_for_claim_review(datacommons_cr)
        + await _get_appearances_for_claim_review(google_cr)
    )
    claimant = _get_claimant(direct_cr) or _get_claimant(datacommons_cr) or _get_claimant(google_cr)
    claim_date = _get_claim_date(direct_cr) or _get_claim_date(datacommons_cr) or _get_claim_date(google_cr)
    publisher_details = (
            _get_publisher_name_url(direct_cr)
            or _get_publisher_name_url(datacommons_cr)
            or _get_publisher_name_url(google_cr)
    )
    publisher_name, publisher_url = publisher_details if publisher_details else (None, None)
    publisher = await db.get_publisher_by_url(get_domain(publisher_url or review.url))
    author = (
            _get_author_name_url(direct_cr)
            or _get_author_name_url(datacommons_cr)
            or _get_author_name_url(google_cr)
    )

    # Save the extracted data to the review object and to the DB
    try:
        review.published = published
        review.modified = modified
        review.raw_rating = rating
        review.raw_claimant_name = claimant[0] if claimant else None
        review.raw_claimant_url = claimant[1] if claimant else None
        review.raw_claim_date = claim_date
        review.raw_publisher_name = publisher_details[0] if publisher_details else None
        review.raw_publisher_url = publisher_details[1] if publisher_details else None
        review.publisher_id = publisher.id if publisher else None

        review.author_name = author[0] if author else None
        review.author_url = author[1] if author else None

        review.appearance_ids = {appearance.id for appearance in appearances}

        review.stage = 1

    except ValidationError as e:
        await review.dismiss(f"Could not parse ClaimReview data: {e}")
        return

    try:
        await review.save_to_db()
    except DataError as e:
        await review.dismiss(f"Unable to save review instance into DB. Reason: {e}")
        return
    except UniqueViolationError:
        logger.error(f"Found a review duplicate: {review}")
        return

    # Validate the review, dismiss if claim date is after review publication
    if review.raw_claim_date and review.published and review.raw_claim_date > review.published:
        await review.dismiss("Future claim: The review publication date predates the claim date.")


def _get_published(cr: dict | None) -> datetime | None:
    if cr:
        date = cr.get("datePublished") or cr.get("dateModified")
        if date:
            return determine_date(date)
    return None


def _get_modified(cr: dict | None) -> datetime | None:
    if cr is None:
        return None
    try:
        if date_modified := cr.get("dateModified").strip():
            return datetime.fromisoformat(date_modified).replace(tzinfo=None)
    except Exception:
        pass


def _get_rating(cr: dict | None) -> str | None:
    if cr is None:
        return None
    try:
        if review_rating := cr.get("reviewRating"):
            if isinstance(review_rating, dict):
                rating = review_rating.get("alternateName")
                if isinstance(rating, str):
                    return rating
                elif isinstance(rating, list):
                    return rating[0]
            elif isinstance(review_rating, str):
                return review_rating
        else:
            # Sometimes 'reviewRating' is inside 'properties'
            return cr.get("properties", {}).get("reviewRating")
    except Exception:
        pass


async def _get_appearances_for_claim_review(cr: dict | None) -> list[Appearance]:
    if cr is None:
        return []
    if item_reviewed := cr.get("itemReviewed"):
        if item_reviewed.get("@type") == "Claim":
            appearance_urls = _get_appearance_urls_for_item(item_reviewed)
            appearances = []
            for url in appearance_urls:
                if not url.startswith("http"):
                    url = "https://" + url
                if validate(url, HttpUrl):  # Sometimes the url field is ill-populated
                    appearance = await appearance_from_url(url)
                    if appearance:
                        appearances.append(appearance)
            return appearances
    return []


def _get_appearance_urls_for_item(item_reviewed: dict) -> list[str]:
    try:
        if appearances_raw := item_reviewed.get("appearance"):
            if isinstance(appearances_raw, list):
                urls = [app.get("url") for app in appearances_raw]
                urls = [url for url in urls if isinstance(url, str) and len(url) < 2084]
                return urls
            elif isinstance(appearances_raw, str):
                return [appearances_raw]
        elif first_appearance := item_reviewed.get("firstAppearance"):
            if url := first_appearance.get("url"):
                if isinstance(url, str):
                    return [url]
        elif url := item_reviewed.get("url"):
            if isinstance(url, str):
                return [url]
    except Exception:
        pass
    return []


def _get_claimant(cr: dict | None) -> tuple[str, str] | None:
    if cr is None:
        return None
    if item_reviewed := cr.get("itemReviewed"):
        if item_reviewed.get("@type") == "Claim":
            if author := item_reviewed.get("author"):
                claimant_name = author.get("name")
                claimant_url = author.get("url")
                if not claimant_url:
                    if same_as := author.get("sameAs"):
                        if isinstance(same_as, list):
                            claimant_url = same_as[0]
                return claimant_name, claimant_url


def _get_claim_date(cr: dict | None) -> datetime | None:
    if cr is None:
        return None
    if item_reviewed := cr.get("itemReviewed"):
        if date_published := item_reviewed.get("datePublished"):
            return determine_date(date_published)
    return None


def _get_publisher_name_url(cr: dict | None) -> tuple[str, str] | None:
    if cr is None:
        return None
    if author := cr.get("author"):
        authors = author if isinstance(author, list) else [author]
        for author in authors:
            if isinstance(author, dict) and author.get("@type") == "Organization":
                publisher_name = author.get("name")
                publisher_url = author.get("url")
                if publisher_url and not publisher_url.startswith("http"):
                    publisher_url = "https://" + publisher_url
                return publisher_name, publisher_url
            elif isinstance(author, str):  # Africa Check likes to do this
                # Use regex to extrac URL from author field
                pattern = re.compile(URL_REGEX)
                match = re.search(pattern, author)
                if match:
                    publisher_url = match.group(0)
                    return author, publisher_url


def _get_author_name_url(cr: dict | None) -> tuple[str, str | None] | None:
    if cr is None:
        return None
    if author := cr.get("author"):
        authors = author if isinstance(author, list) else [author]
        for author in authors:
            if isinstance(author, dict) and author.get("@type") == "Person":
                author_name = author.get("name")
                author_url = author.get("url")
                if author_url and not author_url.startswith("http"):
                    author_url = "https://" + author_url
                return author_name, author_url
            elif isinstance(author, str):
                return author, None


def get_lang_from_html(soup: BeautifulSoup) -> str | None:
    """Gets the lang code from the HTML tag (first line of HTML DOM)"""
    html_tag = soup.find("html")
    if html_tag and html_tag.has_attr("lang"):
        lang_code = html_tag["lang"].strip()
        if lang_code == "en":
            # If language is 'en', there's a high chance that it's wrong,
            # perhaps due to inappropriate template usage or wrong CMS config.
            # Therefore, invalidate it.
            lang_code = None
        return lang_code


def detect_language(text: str) -> str | None:
    """Returns the language code from the given HTML."""
    # Use langdetect to determine the true language
    if not text:
        logger.warning("Empty text passed to langdetect.")
    try:
        return detect(text)
    except Exception:
        return None


def extract_meta_language(soup: BeautifulSoup):
    tag = soup.find("meta", attrs={"http-equiv": "content-language"})
    if tag and tag.get("content"):
        return tag["content"].strip()
    tag = soup.find("meta", attrs={"name": "language"})
    if tag and tag.get("content"):
        return tag["content"].strip()


async def download_and_extract_claim_reviews(urls: Iterable[str]) -> (list[list[dict]], list[str | None]):
    """Retrieve the ClaimReview metadata from a URL. Can have multiple ClaimReviews.
    Also returns the raw page HTMLs"""
    async with aiohttp.ClientSession() as session:
        htmls = await get_static_htmls(urls, session)
    logger.debug(f"Extracting ClaimReviews from {sum(bool(page) for page in htmls)} pages...")
    claim_reviews = []
    for html in htmls:
        if html and "ClaimReview" in html:
            claim_reviews.append(extract_claim_reviews(html))
        else:
            claim_reviews.append([])
    return claim_reviews, htmls


def extract_claim_reviews(page_text: str) -> list[dict]:
    # TODO: Improve this
    try:
        claim_reviews = jsonld_parser(page_text)
    except (json.decoder.JSONDecodeError, ValueError):
        claim_reviews = []

    if not claim_reviews:
        try:
            claim_reviews = microdata_parser(page_text)
        except Exception:
            pass

    if not claim_reviews:
        try:
            claim_reviews = custom_parser(page_text)
        except Exception:
            pass

    if not claim_reviews and "ClaimReview" in page_text:
        if "https://newschecker.in" not in page_text:
            # "https://newschecker.in" encapsulates the ClaimReview in wildly escaped JS code
            # TODO: Improve _custom_parser to minimize these cases
            logger.debug(
                "ClaimReview extraction failed despite 'ClaimReview' being contained in page_text."
            )

    return claim_reviews
