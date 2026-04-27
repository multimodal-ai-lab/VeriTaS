"""
Enhanced retriever for Google Factcheck Explorer with logging
"""

import json
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import requests

from veritas import logger
from veritas.pipeline.util.review_retriever import ReviewRetriever


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
        self, after: datetime | None = None, before: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Fetches ClaimReviews from Google Factcheck Explorer at the given time range."""
        raws = []

        # Fetch new data
        offset = 0

        while True:
            raw_piece = self._get_recent(offset=offset)

            if not raw_piece:
                logger.debug("No more results, stopping fetch.")
                break

            offset += len(raw_piece)
            raws.extend(raw_piece)

            # Stop collection if oldest example is older than 1 week
            if after:
                oldest_date = None
                for oldest in reversed(raws):
                    try:
                        oldest_date_str = oldest[0][3][0][2]
                        if oldest_date_str:
                            oldest_date = datetime.fromtimestamp(oldest_date_str, tz=UTC)
                            break
                    except (ValueError, TypeError):
                        pass
                if oldest_date and oldest_date < after:
                    break

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
                date_published = r[0][3][0][2]
                if date_published:
                    date_published = datetime.fromtimestamp(date_published, tz=UTC)

                claim_review = {
                    "@context": "http://schema.org",
                    "@type": "ClaimReview",
                    "datePublished": date_published.isoformat() if date_published else None,
                    "url": r[0][3][0][1],
                    "claimReviewed": r[0][0],
                    "author": {
                        "@type": "Organization",
                        "name": r[0][3][0][0][0],
                        "url": r[0][3][0][0][1],
                    },
                    "reviewRating": {
                        "@type": "Rating",
                        "ratingValue": (
                            r[0][3][0][9][0]
                            if (len(r[0][3][0]) > 9 and r[0][3][0][9] and len(r[0][3][0][9]))
                            else -1
                        ),
                        "worstRating": (
                            r[0][3][0][9][1]
                            if (len(r[0][3][0]) > 9 and r[0][3][0][9] and len(r[0][3][0][9]))
                            else -1
                        ),
                        "bestRating": (
                            r[0][3][0][9][2]
                            if (len(r[0][3][0]) > 9 and r[0][3][0][9] and len(r[0][3][0][9]))
                            else -1
                        ),
                        "alternateName": r[0][3][0][3],
                    },
                    "itemReviewed": (
                        {
                            "@type": "CreativeWork",
                            "author": (
                                {
                                    "@type": "Person",
                                    "name": r[0][1][0],
                                    "sameAs": r[0][4][0][1] if r[0][4] and len(r[0][4]) else None,
                                }
                                if len(r[0][1])
                                else {}
                            ),
                            "url": r[0][10],
                        }
                        if len(r[0]) > 10 and r[0][10]
                        else {}
                    ),
                }

                # Add appearance data if available
                try:
                    appearance = r[0][1][2]
                    if appearance:
                        claim_review["itemReviewed"]["appearance"] = [{"url": u} for u in appearance]
                        appearance_count += 1
                except IndexError:
                    pass
                    # logger.debug(f"No appearance for {r[0][3][0][1]}")

                # Add firstAppearance data if available
                try:
                    first_appearance = len(r[0]) > 13 and r[0][13]
                    if first_appearance:
                        claim_review["itemReviewed"]["firstAppearance"] = {
                            "type": "CreativeWork",
                            "url": first_appearance,
                        }
                except IndexError:
                    logger.debug(f"No firstAppearance for {r[0][3][0][1]}")

                results.append(claim_review)
            except IndexError as e:
                errors += 1
                logger.error(f"Error processing review {i}: {e}")
                logger.debug(f"Problematic data: {json.dumps(r)}")

        logger.debug(
            f"Processing completed. {len(results)} successes, {errors} errors. Out of {len(results)}, {appearance_count} have appearances."
            # noqa: E501
        )

        if results:
            results = self._filter_by_date(results, after, before)

        return results
