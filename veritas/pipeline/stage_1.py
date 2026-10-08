import json
import re
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse

import aiohttp
from asyncpg import DataError, UniqueViolationError
from bs4 import BeautifulSoup
from langdetect import detect
from pydantic import HttpUrl, ValidationError

from veritas import logger
from veritas.common import Appearance, Review
from veritas.common.appearance import appearance_from_url
from veritas.db import db
from veritas.models import QuotaExceededError
from veritas.pipeline.util.cr_parsing import jsonld_parser, microdata_parser, custom_parser
from veritas.pipeline.util.datacommons_feed import DataCommonsFeedRetriever
from veritas.pipeline.util.google_factcheck_explorer import GoogleFactCheckExplorerRetriever
from veritas.pipeline.util.stage import Stage
from veritas.util import run_with_semaphore
from veritas.util.parsing import get_lang_from_code, determine_date
from veritas.util.scraping import get_static_htmls
from veritas.util.url import URL_REGEX, get_domain, is_domain_root
from veritas.util.util import validate, get_quarter_start_date

datacommons = DataCommonsFeedRetriever()
gfce = GoogleFactCheckExplorerRetriever()


class Stage1(Stage):
    """Continuously retrieves new reviews."""

    id = 1
    name = "Review Discovery"
    db_max_connections = 10
    default_interval_seconds = 60 * 60 * 24

    reextracted: bool = False

    async def step(self, *,
                   start: tuple[int, int] = (2016, 1),
                   reextract_from: date | str | None = None,
                   reextract_until: date | str | None = None) -> None:
        """Retrieves new reviews. If `reextract_from` is set, afterward re-extracts
        the data of the reviews published since then (until `reextract_until`, or
        today), from their just refreshed ClaimReviews. That happens only once per
        pipeline start: re-extraction is a deliberate, one-off operation."""
        await retrieve_reviews(after=get_quarter_start_date(start))

        if reextract_from and not self.reextracted:
            until = to_date(reextract_until) if reextract_until else date.today()
            await reextract_reviews(to_date(reextract_from), until)
            self.reextracted = True


def to_date(value: date | str) -> date:
    """Turns a date (as given in the config) into a `date`. Accepts `date` and
    `datetime` objects as well as ISO strings like '2026-07-01'."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


async def retrieve_reviews(after: datetime = None):
    """Scans Google Fact Check Tools API and DataCommons for new ClaimReviews.
    Processes them and saves them into the DB."""
    logger.info("Retrieving reviews...")

    # 1. Retrieve all the latest raw ClaimReview dictionaries and save new ones in the DB
    datacommons_crs, google_crs = await _retrieve_latest_claim_reviews(after)
    await _save_new_claim_reviews(datacommons_crs, google_crs)

    # 2. Try to get the original raw ClaimReview from the review article page directly,
    # parse the CLaimReview data and save structured info in the DB
    n_processed = 0
    while reviews := await db.get_reviews(stage=0, limit=1000, dismissed=False):
        await _retrieve_claim_reviews_directly(reviews)
        await _process_raw_claim_reviews_from_reviews(reviews)
        n_processed += len(reviews)

    logger.info(f"Retrieved and processed {n_processed} reviews.")


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
        n_new, n_updated = await db.save_claim_review(datacommons_crs, source="datacommons")
        logger.info(f"DataCommons: {n_new} new and {n_updated} updated reviews.")
    if google_crs:
        google_crs = _claim_reviews_to_organized_dict(google_crs)
        n_new, n_updated = await db.save_claim_review(google_crs, source="google")
        logger.info(f"Google: {n_new} new and {n_updated} updated reviews.")


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
    data = await _extract_review_data(review)

    # Save the extracted data to the review object and to the DB
    try:
        for column, value in data.items():
            setattr(review, column, value)
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


async def _extract_review_data(review: Review) -> dict[str, Any]:
    """Extracts the review data (one value per column of `REVIEW_EXTRACTED_COLUMNS`)
    from the review's raw ClaimReview dictionaries. Precedence:
    direct_claim_review > datacommons_claim_review > google_claim_review.
    Appearances are united over all three."""
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

    # Validate claim date for publisher provereno.media because this publisher
    # frequently submits claims with wrong claim dates (in the future, or more
    # than 30 days before the review). In such cases, remove the raw claim date.
    if publisher_url and "provereno.media" in publisher_url and claim_date and published:
        if (claim_date > published) or ((published - claim_date).days > 30):
            claim_date = None

    return {
        "published": published,
        "modified": modified,
        "raw_rating": rating,
        "raw_claimant_name": claimant[0] if claimant else None,
        "raw_claimant_url": claimant[1] if claimant else None,
        "raw_claim_date": claim_date,
        "raw_publisher_name": publisher_name,
        "raw_publisher_url": publisher_url,
        "publisher_id": publisher.id if publisher else None,
        "author_name": author[0] if author else None,
        "author_url": author[1] if author else None,
        "appearance_ids": {appearance.id for appearance in appearances},
    }


async def reextract_reviews(start: date, end: date, batch_size: int = 500) -> int:
    """Extracts the review data anew from the current ClaimReviews of all reviews
    that went through Stage 1 already, are not dismissed, and were published within
    the given date range (both inclusive). Updates only the extracted columns (see
    `merge_extracted_review_data` for how). Returns the number of updated reviews."""
    logger.info(f"Re-extracting the data of reviews published between {start} and {end}...")
    n_seen = n_updated = 0
    last_id = 0
    while reviews := await db.get_reviews_for_reextraction(start, end, after_id=last_id, limit=batch_size):
        last_id = reviews[-1].id
        updated = await run_with_semaphore((_reextract_review(review) for review in reviews), limit=100)
        n_seen += len(reviews)
        n_updated += sum(updated)
        logger.debug(f"Re-extracted {n_seen} reviews so far, {n_updated} of them changed.")
    logger.info(f"Re-extracted {n_seen} reviews, updated {n_updated} of them.")
    return n_updated


async def _reextract_review(review: Review) -> bool:
    """Re-extracts and saves the data of a single review. Returns whether it changed."""
    try:
        data = await _extract_review_data(review)
        if merge_extracted_review_data(review, data):
            await db.update_review_extracted_data(review)
            return True
    except QuotaExceededError:
        raise  # Run-level condition, e.g. scrapeMM unreachable
    except Exception as e:
        logger.warning(f"Could not re-extract the data of review {review.id}: {e}")
    return False


def merge_extracted_review_data(review: Review, data: dict[str, Any]) -> list[str]:
    """Merges freshly extracted review data into the review without losing any:
    a value replaces the existing one only if it is not None; appearances are
    united with the existing ones (later stages may have added more); and the
    publisher is only set if there is none yet (Stage 2 may have identified it).
    Returns the names of the changed columns."""
    changed = []
    for column, value in data.items():
        current = getattr(review, column)
        if column == "appearance_ids":
            value = (current or set()) | (value or set())
            if value == (current or set()):
                continue
        elif column == "publisher_id" and current is not None:
            continue
        elif value is None or _same_value(value, current):
            continue
        setattr(review, column, value)
        changed.append(column)
    return changed


def _same_value(a: Any, b: Any) -> bool:
    """Compares two extracted values, treating URLs as equal if they are the same
    after URL parsing (which, e.g., adds the trailing slash of a bare domain)."""
    if a == b:
        return True
    if isinstance(a, (str, HttpUrl)) and isinstance(b, (str, HttpUrl)):
        return _as_url_string(a) == _as_url_string(b)
    return False


def _as_url_string(value: str | HttpUrl) -> str:
    try:
        return str(HttpUrl(str(value)))
    except ValidationError:
        return str(value)


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
    appearances = []
    for url in _get_appearance_urls_for_item(_get_item_reviewed(cr)):
        if not url.startswith("http"):
            url = "https://" + url
        if validate(url, HttpUrl):  # Sometimes the url field is ill-populated
            appearance = await appearance_from_url(url)
            if appearance:
                appearances.append(appearance)
    return appearances


def _get_item_reviewed(cr: dict | None) -> dict | None:
    """Returns the item (usually the claim) reviewed by the ClaimReview. Publishers
    type it differently (`Claim`, `CreativeWork`, `Thing`, a list of types, or not
    at all), so any dict is accepted. If `itemReviewed` is a list, the first item
    typed `Claim` is preferred, falling back to the first dict."""
    if not isinstance(cr, dict):
        return None
    item_reviewed = cr.get("itemReviewed")
    if isinstance(item_reviewed, dict):
        return item_reviewed
    if isinstance(item_reviewed, list):
        items = [item for item in item_reviewed if isinstance(item, dict)]
        for item in items:
            if _has_type(item, "Claim"):
                return item
        if items:
            return items[0]
    return None


def _has_type(item: dict, type_name: str) -> bool:
    item_type = item.get("@type")
    if isinstance(item_type, list):
        return type_name in item_type
    return item_type == type_name


def _get_appearance_urls_for_item(item_reviewed: dict | None) -> list[str]:
    """Collects the URLs of all appearances (`appearance` and `firstAppearance`)
    of the reviewed item, deduplicated. Each may be a list, a dict with a `url`,
    or a plain URL string. Falls back to the item's own `url` if there is none."""
    if not isinstance(item_reviewed, dict):
        return []

    candidates = []
    for key in ("appearance", "firstAppearance"):
        value = item_reviewed.get(key)
        for entry in value if isinstance(value, list) else [value]:
            if isinstance(entry, dict):
                candidates.append(entry.get("url"))
            elif isinstance(entry, str):
                candidates.append(entry)

    if not any(isinstance(url, str) and url for url in candidates):
        candidates = [item_reviewed.get("url")]

    urls = []
    for url in candidates:
        if isinstance(url, str) and url and len(url) < 2084 and url not in urls:
            urls.append(url)
    return urls


def _get_claimant(cr: dict | None) -> tuple[str | None, str | None] | None:
    """Returns the name and URL of the claim's author, or None if neither is known
    (so that the next ClaimReview in precedence order is considered)."""
    item_reviewed = _get_item_reviewed(cr)
    if not item_reviewed:
        return None

    author = item_reviewed.get("author")
    if isinstance(author, list):
        author = next((a for a in author if isinstance(a, (dict, str)) and a), None)
    if isinstance(author, str):
        return (author.strip(), None) if author.strip() else None
    if not isinstance(author, dict):
        return None

    claimant_name = author.get("name")
    if not isinstance(claimant_name, str) or not claimant_name.strip():
        claimant_name = None

    claimant_url = author.get("url")
    if not claimant_url:
        same_as = author.get("sameAs")
        if isinstance(same_as, list):
            same_as = next((url for url in same_as if isinstance(url, str) and url), None)
        claimant_url = same_as
    claimant_url = _absolute_url(claimant_url)

    if claimant_name is None and claimant_url is None:
        return None
    return claimant_name, claimant_url


def _absolute_url(url: Any) -> str | None:
    """Returns the URL if it is absolute, adding a missing scheme to a bare domain
    like 'example.com/page'. Returns None for relative or invalid URLs like '/',
    which some ClaimReviews state and which would fail the Review validation."""
    if not isinstance(url, str) or not (url := url.strip()):
        return None
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    host = urlparse(url).hostname
    return url if host and "." in host else None


def _get_claim_date(cr: dict | None) -> datetime | None:
    item_reviewed = _get_item_reviewed(cr)
    if item_reviewed:
        if date_published := item_reviewed.get("datePublished"):
            if isinstance(date_published, str):
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
                return publisher_name, _absolute_url(author.get("url"))
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
                return author_name, _absolute_url(author.get("url"))
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
