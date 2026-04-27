"""
Scraper for DataCommons Fact Check Markup Tool Data Feed
"""

import re
from datetime import datetime, timedelta
from typing import Any
from xml.etree import ElementTree

import ijson
import requests

from veritas import logger
from veritas.pipeline.util.review_retriever import ReviewRetriever

# Constants for the data feed
FEED_DIRECTORY = "https://storage.googleapis.com/datacommons-feeds/"
LATEST_FEED = "claimreview/latest/data.json"


class DataCommonsFeedRetriever(ReviewRetriever):
    """Scraper for DataCommons Fact Check Markup Tool Data Feed."""

    def __init__(self, configuration: dict[str, Any] | None = None):
        """Initialize the DataCommons Feeds scraper.

        Args:
            configuration: Optional configuration for the scraper
        """
        self.id = "datacommons_feeds"
        self.homepage = "https://www.datacommons.org/factcheck/download#fcmt-data"
        self.name = "DataCommons - Fact Check Markup Tool Data Feed"
        self.description = """This is a data feed of ClaimReview markups created via the Google Fact Check
            Markup Tool and the new ClaimReview Read/Write API. The data in the feed also follows the
            schema.org ClaimReview standard, namely the same schema as the data in the historical research
            dataset. The feed itself is in DataFeed format."""
        super().__init__(configuration)

    def retrieve(
        self, latest: bool = True, after: datetime | None = None, before: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Gathers ClaimReviews from DataCommons Feeds.

        Args:
            latest: Whether to fetch only the latest data (True) or retrieve all available data (False)
            after: Optional start date for filtering results
            before: Optional end date for filtering results

        Returns:
            List of ClaimReview dictionaries.
        """
        logger.debug(
            f"Starting retrieval of {'latest' if latest else 'all'} ClaimReviews from DataCommons..."
        )

        # Download the latest feed
        if latest and not after:
            after = datetime.now() - timedelta(days=7)

        data_list = self._download_latest_feed(byte_limit=1_000_000 if latest else None)

        # Extract and process claim reviews
        claim_reviews = self._extract_claimreviews(data_list)
        claim_reviews = self._filter_by_date(claim_reviews, after, before)
        logger.debug(
            f"DataCommons Feed retrieval operation completed. Retrieved {len(claim_reviews)} ClaimReviews"
        )
        return claim_reviews

    def _download_feed(self, feed_url: str, byte_limit: int = None) -> list[dict[str, Any]]:
        """Download a specific feed from DataCommons.

        Args:
            feed_url: The URL of the feed to download

        Returns:
            The feed data as a dictionary

        Raises:
            ValueError: If the request fails
        """
        pieces = feed_url.split("/")
        if len(pieces) != 3:
            raise ValueError(f"Invalid feed URL format: {feed_url}")

        url = f"{FEED_DIRECTORY}{feed_url}"
        logger.debug(f"Downloading feed from {url}")

        headers = {"Range": f"bytes=0-{byte_limit}"} if byte_limit else None

        response = requests.get(url, headers=headers)
        if response.status_code not in (200, 206):
            raise ValueError(f"Failed to download feed: HTTP {response.status_code}")

        if byte_limit:
            # The JSON code is truncated and needs to be read object by object
            return _read_truncated_data_feed(response.text)
        else:
            return response.json()["dataFeedElement"]

    def _download_latest_feed(self, **kwargs) -> list[dict[str, Any]]:
        """Download the latest feed from DataCommons.

        Returns:
            The latest feed data as a dictionary
        """
        return self._download_feed(LATEST_FEED, **kwargs)

    def _extract_claimreviews(self, data_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract ClaimReviews from the data feed.

        Args:
            data_list: The data feed elements

        Returns:
            A list of ClaimReviews
        """
        claim_reviews = []
        skipped_count = 0

        for item in data_list:
            if not item.get("item"):
                continue

            for cr in item["item"]:
                if cr:  # some have "item": null
                    # convert datePublished to ISO format
                    if "datePublished" in cr:
                        try:
                            date_published = datetime.fromisoformat(cr["datePublished"].strip())
                            cr["datePublished"] = date_published.isoformat()
                        except ValueError:
                            skipped_count += 1
                            continue

                    if "itemReviewed" in cr and "datePublished" in cr["itemReviewed"]:
                        try:
                            date_published = datetime.fromisoformat(
                                cr["itemReviewed"]["datePublished"].strip()
                            )
                            cr["itemReviewed"]["datePublished"] = date_published.isoformat()
                        except ValueError:
                            skipped_count += 1
                            continue

                    claim_reviews.append(cr)

        logger.debug(
            f"Extracted {len(claim_reviews)} ClaimReviews from the feed where {skipped_count} were skipped due to invalid date formats."
            # noqa: E501
        )

        return claim_reviews

    def download_all_feeds(self) -> list[dict[str, Any]]:
        """Download all available feeds from DataCommons.

        Returns:
            A list of all feed data

        Raises:
            ValueError: If the request fails
        """
        logger.info("Downloading all DataCommons feeds")

        response = requests.get(FEED_DIRECTORY)
        if response.status_code != 200:
            raise ValueError(f"Failed to download feed directory: HTTP {response.status_code}")

        response_content = response.text
        # Remove the namespace to simplify XML parsing
        response_content = re.sub(r"""\s(xmlns="[^"]+"|xmlns='[^']+')""", "", response_content, count=1)

        root = ElementTree.fromstring(response_content)
        all_feeds = []

        for el in root.findall("Contents"):
            try:
                key = el.find("Key").text  # type: ignore
                logger.debug(f"Found feed: {key}")
                feed_data = self._download_feed(key)  # type: ignore
                all_feeds.append(feed_data)
            except Exception as e:
                logger.error(f"Error downloading feed {key}: {e}")

        return all_feeds


def _read_truncated_data_feed(data: str) -> list[dict]:
    """Takes a JSON sting that is cut off and reads all complete elements
    within "dataFeedElement"."""
    prefix = """{
  "@context" : "http://schema.org",
  "@type" : "DataFeed",
  "dataFeedElement" :"""
    data_feed_elements = data.removeprefix(prefix)
    items = []
    parser = ijson.items(data_feed_elements, "item")
    try:
        for i, item in enumerate(parser):
            items.append(item)
    except ijson.JSONError:
        pass
    return items
