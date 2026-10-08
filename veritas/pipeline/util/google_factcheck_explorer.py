"""
Enhanced retriever for Google Factcheck Explorer with logging
"""

import copy
import json
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import requests

from veritas import logger
from veritas.pipeline.util.review_retriever import ReviewRetriever

PAGE_SIZE = 1000  # Number of raw results requested per GFCE page
PAGE_OVERLAP = 100  # Number of raw results shared by consecutive pages (must be < PAGE_SIZE)


class GoogleFactCheckExplorerRetriever(ReviewRetriever):
    """Review Retriever integrating Google's Fact Check Explorer (GFCE)."""

    def __init__(self, configuration: dict[str, Any] | None = None):
        """Initialize the GFCE retriever.

        Args:
            configuration: Optional configuration for the retriever
        """
        self.id = "google_fact_check_explorer"
        self.name = "Google Fact Check Explorer"
        self.homepage = "https://toolbox.google.com/factcheck/explorer"
        self.description = "Retrieves fact-checks from Google's Fact Check Explorer"
        super().__init__(configuration)

        cookie = os.environ.get("GOOGLE_FACTCHECK_EXPLORER_COOKIE", "")
        if not cookie:
            logger.info("GOOGLE_FACTCHECK_EXPLORER_COOKIE environment variable not set")

        logger.debug(f"Initialized {self.name} retriever")

    def retrieve(
        self, latest: bool = True, after: datetime | None = None, before: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve fact-checks from Google Factcheck Explorer.

        Args:
            latest: Whether to fetch new data (True) or use cached data (False)
            after: Optional start date for filtering results
            before: Optional end date for filtering results

        Returns:
            List of ClaimReview dictionaries.
        """
        logger.debug(f"Starting retrieval of {'latest' if latest else 'all'} ClaimReviews from Google...")
        start_time = time.time()
        if latest and not after:
            after = datetime.now() - timedelta(days=7)
        if after and after.tzinfo is None:
            after = after.replace(tzinfo=UTC)
        results = self._retrieve(after=after, before=before)
        logger.debug(f"Retrieval operation completed. Retrieved {len(results)} ClaimReviews from Google.")
        end_time = time.time()
        # get the duration of the operation in format HH:MM:SS
        duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        logger.debug(f"Retrieval took {duration} h")
        return results

    def _get_recent(
        self, lang: str = "", offset: int = 0, num_results: int = 1000, query: str = "list:recent"
    ) -> list[dict]:
        """Get recent fact-checks from Google Factcheck Explorer.

        Args:
            lang: Language to filter by
            offset: Pagination offset
            num_results: Number of results to fetch
            query: Search query

        Returns:
            List of raw fact-check data

        Raises:
            ValueError: If the request fails
        """
        # logger.debug(f"Fetching recent fact-checks (offset={offset}, num_results={num_results})")

        params = {
            "hl": lang,
            "num_results": num_results,
            "query": query,
            "force": "false",
            "offset": offset,
        }

        headers = {
            "dnt": "1",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-GB,en;q=0.9,it-IT;q=0.8,it;q=0.7,en-US;q=0.6",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
            # noqa: E501
            "accept": "application/json, text/plain, */*",
            "referer": "https://toolbox.google.com/factcheck/explorer/search/list:recent;hl=en;gl=",
            "authority": "toolbox.google.com",
        }

        cookie = os.environ.get("GOOGLE_FACTCHECK_EXPLORER_COOKIE", "")
        if cookie:
            headers["cookie"] = cookie

        try:
            response = requests.get(
                "https://toolbox.google.com/factcheck/api/search",
                params=params,
                headers=headers,
            )

            if response.status_code != 200:
                logger.error(f"Request failed with status code {response.status_code}")
                raise ValueError(f"Request failed with status code {response.status_code}")

            # Google's API returns a prefix before the JSON content
            content = json.loads(response.text[5:])

            try:
                reviews = content[0][1]
            except IndexError:
                logger.warning(f"No reviews found for offset {offset} and num_results {num_results}")
                return []

            return cast(list[dict], reviews)

        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise

    def _retrieve(
        self,
        after: datetime | None = None,
        before: datetime | None = None,
        page_size: int = PAGE_SIZE,
        overlap: int = PAGE_OVERLAP,
    ) -> list[dict[str, Any]]:
        """Fetches ClaimReviews from Google Factcheck Explorer at the given time range.

        Pages through the `list:recent` listing (newest first). Because that listing
        is live, new items get prepended while paging, pushing items across page
        boundaries. To not skip such items, consecutive pages overlap: each page
        has `page_size` items, but the offset only advances by `page_size - overlap`.
        Raw groups seen on several pages are deduplicated. If a page shares no group
        with the previous (full) page, the list shifted by more than `overlap`
        items and some items may have been skipped, which gets logged.

        Paging stops on an empty page or, if `after` is given, as soon as a page
        lies entirely outside the cutoff, i.e., all its dated reviews are older than
        `after` (the listing is only roughly sorted by date, so a single old item
        does not suffice). A page without any dated review never stops the paging.

        Args:
            after: Optional start date for filtering results (also the paging cutoff)
            before: Optional end date for filtering results
            page_size: Number of raw results requested per page
            overlap: Number of raw results shared by consecutive pages, < page_size

        Returns:
            List of ClaimReviews
        """
        if page_size < 1:
            raise ValueError(f"page_size must be positive, got {page_size}.")
        if not 0 <= overlap < page_size:
            raise ValueError(f"overlap must be in [0, page_size), got {overlap} (page_size={page_size}).")

        raws = []
        seen: set[str] = set()
        prev_identities: set[str] | None = None
        prev_was_full = False
        offset = 0

        while True:
            page = self._get_recent(offset=offset, num_results=page_size)

            if not page:
                logger.debug("No more results, stopping fetch.")
                break

            identities = [_group_identity(r) for r in page]
            identity_set = set(identities)

            if overlap and prev_identities is not None and prev_was_full and not identity_set & prev_identities:
                logger.warning(
                    f"GFCE page at offset {offset} shares no item with the previous page. "
                    f"The listing shifted by more than the overlap of {overlap} items "
                    f"while paging, so some items may have been skipped."
                )

            for r, identity in zip(page, identities):
                if identity not in seen:
                    seen.add(identity)
                    raws.append(r)

            if after and _page_is_outside_cutoff(page, after):
                logger.debug(f"All items of the page at offset {offset} are older than {after}, stopping fetch.")
                break

            is_full = len(page) >= page_size
            offset += _offset_step(len(page), page_size, overlap)
            prev_identities = identity_set
            prev_was_full = is_full

        return self._process_raw(raws, after=after, before=before)

    def _process_raw(
        self, raws: list[dict[str | int, Any]], after: datetime | None = None, before: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Process raw fact-check data into structured format.

        Args:
            raws: List of raw fact-check data
            after: Optional start date for filtering results
            before: Optional end date for filtering results

        Returns:
            List of processed ClaimReviews
        """
        logger.debug(f"Processing {len(raws)} raw ClaimReviews...")
        results = []
        appearance_count = 0
        errors = 0

        for i, r in enumerate(raws):
            try:
                claim_reviews, n_failed = claim_reviews_from_raw(r)
            except (IndexError, TypeError) as e:
                errors += 1
                logger.error(f"Error processing claim {i}: {e}")
                logger.debug(f"Problematic data: {json.dumps(r)}")
                continue
            if n_failed:
                errors += n_failed
                logger.debug(f"Skipped {n_failed} unparsable review(s) of claim {i}: {json.dumps(r)}")
            appearance_count += sum("appearance" in cr["itemReviewed"] for cr in claim_reviews)
            results.extend(claim_reviews)

        logger.debug(
            f"Processing completed. {len(results)} successes, {errors} errors. Out of {len(results)}, {appearance_count} have appearances."
            # noqa: E501
        )

        if results:
            results = self._filter_by_date(results, after, before)

        return results


def claim_reviews_from_raw(r: list) -> tuple[list[dict[str, Any]], int]:
    """Converts one raw GFCE result into ClaimReview dictionaries. A result is a
    claim (`r[0][0]`) together with the list of all its reviews (`r[0][3]`), which
    can stem from several publishers or from one article checking several claims.
    Each review yields its own ClaimReview, all sharing the claim's data.
    Returns the ClaimReviews and the number of reviews that could not be parsed.
    Raises IndexError/TypeError if the claim itself is malformed."""
    claim = r[0]
    claim_text = claim[0]
    reviews = claim[3]

    item_reviewed = _item_reviewed_from_raw_claim(claim)

    claim_reviews = []
    n_failed = 0
    for review in reviews or []:
        try:
            claim_reviews.append(_claim_review_from_raw_review(review, claim_text, item_reviewed))
        except (IndexError, TypeError, ValueError, OSError, OverflowError):
            n_failed += 1
    return claim_reviews, n_failed


def _item_reviewed_from_raw_claim(claim: list) -> dict[str, Any]:
    """Builds the schema.org `Claim` reviewed by the ClaimReviews from the raw GFCE
    claim. The observed layout of the raw claim is:
        [0]  claim text
        [1]  [claimant name, claimant KG ID, appearance URLs, ...] (may be empty)
        [2]  claim date (Unix timestamp)
        [3]  reviews
        [4]  appearances, each as [[None, domain], url, ...]
        [12] appearance URLs
    Other positions hold no claim data (e.g., [10] is free text like
    "In a Facebook post", not a URL). Missing or malformed fields are left out."""
    item_reviewed: dict[str, Any] = {"@type": "Claim"}

    claimant = _get(claim, 1, 0)
    if isinstance(claimant, str) and claimant.strip():
        item_reviewed["author"] = {"@type": "Person", "name": claimant}

    claim_date = _timestamp_to_iso(_get(claim, 2))
    if claim_date:
        item_reviewed["datePublished"] = claim_date

    appearance_urls = _appearance_urls_from_raw_claim(claim)
    if appearance_urls:
        item_reviewed["appearance"] = [{"url": url} for url in appearance_urls]

    return item_reviewed


def _appearance_urls_from_raw_claim(claim: list) -> list[str]:
    """Collects the appearance URLs from all positions of the raw claim that list
    them ([4], [12], and [1][2]), deduplicated in order of first occurrence."""
    candidates = []
    for appearance in _get(claim, 4) or []:
        candidates.append(_get(appearance, 1))
    for key in ((12,), (1, 2)):
        urls = _get(claim, *key)
        if isinstance(urls, list):
            candidates.extend(urls)
        elif isinstance(urls, str):
            candidates.append(urls)
    urls = []
    for url in candidates:
        if isinstance(url, str) and url.strip() and url not in urls:
            urls.append(url)
    return urls


def _get(data: Any, *indices: int) -> Any:
    """Returns `data[i][j]...` for the given indices or None if any is missing."""
    for i in indices:
        if not isinstance(data, list) or not -len(data) <= i < len(data):
            return None
        data = data[i]
    return data


def _timestamp_to_iso(timestamp: Any) -> str | None:
    """Converts a Unix timestamp into an ISO date string (UTC); None if invalid."""
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def _claim_review_from_raw_review(review: list, claim_text: str, item_reviewed: dict) -> dict[str, Any]:
    """Builds the ClaimReview dictionary for a single raw review of a claim."""
    date_published = review[2]
    if date_published:
        date_published = datetime.fromtimestamp(date_published, tz=UTC)

    has_rating_values = len(review) > 9 and review[9] and len(review[9])

    claim_review = {
        "@context": "http://schema.org",
        "@type": "ClaimReview",
        "datePublished": date_published.isoformat() if date_published else None,
        "url": review[1],
        "claimReviewed": claim_text,
        "author": {
            "@type": "Organization",
            "name": review[0][0],
            "url": review[0][1],
        },
        "reviewRating": {
            "@type": "Rating",
            "ratingValue": review[9][0] if has_rating_values else -1,
            "worstRating": review[9][1] if has_rating_values else -1,
            "bestRating": review[9][2] if has_rating_values else -1,
            "alternateName": review[3],
        },
        "itemReviewed": copy.deepcopy(item_reviewed),
    }

    # Further review details, if available
    title = _get(review, 8)
    if isinstance(title, str) and title.strip():
        claim_review["name"] = title
    language = _get(review, 6)
    if isinstance(language, str) and language.strip():
        claim_review["inLanguage"] = language

    return claim_review


def _offset_step(page_len: int, page_size: int, overlap: int) -> int:
    """Returns by how much to advance the pagination offset after a page of
    `page_len` raw results. A full page advances by `page_len - overlap` (at least
    1, so paging always progresses), making the next page overlap with this one.
    A short page marks the end of the listing; overlapping it would re-request
    (almost) the same items with little progress, so it advances by `page_len`."""
    if page_len < page_size:
        return max(page_len, 1)
    return max(page_len - overlap, 1)


def _group_identity(r: Any) -> str:
    """Returns a stable identity of a raw GFCE result (a claim group), consisting
    of the claim text and the URLs of all its reviews. Falls back to the full
    JSON serialization if the group is malformed."""
    try:
        claim = r[0]
        review_urls = [review[1] for review in claim[3] or []]
        return json.dumps([claim[0], review_urls], ensure_ascii=False, default=str)
    except (IndexError, TypeError, KeyError):
        return json.dumps(r, sort_keys=True, ensure_ascii=False, default=str)


def _page_dates(page: list) -> list[datetime]:
    """Returns the publication dates of all reviews of all raw groups in the page.
    Reviews without (valid) date are left out."""
    dates = []
    for r in page:
        try:
            reviews = r[0][3] or []
        except (IndexError, TypeError, KeyError):
            continue
        for review in reviews:
            try:
                timestamp = review[2]
                if timestamp is None or isinstance(timestamp, bool):
                    continue
                dates.append(datetime.fromtimestamp(timestamp, tz=UTC))
            except (IndexError, TypeError, KeyError, ValueError, OSError, OverflowError):
                continue
    return dates


def _page_is_outside_cutoff(page: list, after: datetime) -> bool:
    """Tells whether the page lies entirely outside the cutoff, i.e., whether all
    its dated reviews are older than `after`. Undated reviews are ignored; a page
    without any dated review is never considered outside the cutoff."""
    dates = _page_dates(page)
    return bool(dates) and all(date < after for date in dates)
