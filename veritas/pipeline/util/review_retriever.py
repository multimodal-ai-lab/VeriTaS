from datetime import UTC, datetime
from typing import Any

from veritas import logger


class ReviewRetriever:
    id: str
    homepage: str
    name: str
    description: str

    def __init__(self, configuration=None):
        self.configuration = configuration

    def retrieve(self):
        raise NotImplementedError("override this method")

    def _filter_by_date(
        self, data: list[dict[str, Any]], start_date: datetime | None, end_date: datetime | None
    ) -> list[dict[str, Any]]:
        """Filter data by date range.

        Args:
            data: List of claim reviews
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            Filtered list of claim reviews
        """
        if not start_date and not end_date:
            return data  # No filtering needed

        filtered_data = []
        skipped_count = 0
        missing_date_count = 0

        # Ensure start_date and end_date are timezone-aware if they're not None
        if start_date and start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=UTC)
        if end_date and end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=UTC)

        for item in data:
            date_published = item.get("datePublished")
            if not date_published:
                missing_date_count += 1
                continue

            try:
                # Handle both string dates and datetime objects
                if isinstance(date_published, str):
                    date_obj = datetime.fromisoformat(date_published.replace("Z", "+00:00"))
                else:
                    date_obj = date_published

                # Ensure the date is timezone-aware for comparison
                if date_obj.tzinfo is None:
                    date_obj = date_obj.replace(tzinfo=UTC)

                if start_date and date_obj < start_date:
                    skipped_count += 1
                    continue

                if end_date and date_obj > end_date:
                    skipped_count += 1
                    continue

                filtered_data.append(item)

            except (ValueError, TypeError) as e:
                # Handle malformed date strings
                logger.warning(f"Could not parse date: {date_published}, {type(date_published)} - {e}")
                continue

        logger.debug(
            f"Date filtering: {len(filtered_data)} items matched, {skipped_count} outside date range, {missing_date_count} missing dates"  # noqa: E501
        )
        return filtered_data
