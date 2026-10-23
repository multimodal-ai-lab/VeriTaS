"""The VeriTaS database: the claims' fact-checks (reviews), when they were published and under
which hosts and paths their publishers publish fact-checks (the publishers' fact_check_scopes),
and the fact-check scopes of all IFCN signatories and EFCSN members.

The database is only read: the connection is read-only, so it refuses any change.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import asyncpg
import yaml

# The project's config.yaml, whose "database" section says how to connect
CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"


@dataclass
class Review:
    """A review as the database has it."""
    published: datetime | None
    fact_check_scopes: list[str] = field(default_factory=list)


def lookup_reviews(urls: list[str], config_path: Path = CONFIG_PATH) -> dict[str, Review]:
    """The reviews at `urls`, each with its publishing date and its publisher's fact-check scopes.
    A URL in several reviews (of several claims) gets the earliest date and all scopes. URLs that
    are not in the database are left out."""
    rows = asyncio.run(_fetch(
        config_path,
        """SELECT r.url, r.published, p.fact_check_scopes
           FROM reviews r LEFT JOIN publishers p ON p.id = r.publisher_id
           WHERE r.url = ANY($1)""",
        list(dict.fromkeys(urls)),
    ))
    reviews = {}
    for row in rows:
        review = reviews.setdefault(row["url"], Review(None))
        if row["published"] and (review.published is None or row["published"] < review.published):
            review.published = row["published"]
        review.fact_check_scopes += [s for s in row["fact_check_scopes"] or [] if s not in review.fact_check_scopes]
    return reviews


def signatory_scopes(config_path: Path = CONFIG_PATH) -> list[str]:
    """The distinct fact-check scopes of all publishers that are IFCN signatories (status active
    or in renewal) or EFCSN members (status active), sorted. A publisher without fact-check scopes
    gives its domains instead, which block their hosts as a whole, but not their subdomains."""
    rows = asyncio.run(_fetch(
        config_path,
        """SELECT DISTINCT unnest(CASE WHEN cardinality(fact_check_scopes) > 0 THEN fact_check_scopes ELSE domains END)
                  AS scope
           FROM publishers
           WHERE ifcn_status IN ('active', 'in_renewal') OR efcsn_status = 'active'
           ORDER BY scope""",
    ))
    return [row["scope"] for row in rows]


async def _fetch(config_path: Path, query: str, *args) -> list:
    """The rows of `query`, with the parameters `args`, over a read-only connection to the
    database. Raises RuntimeError on any failure, reading the config included."""
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)["database"]
        connection = await asyncpg.connect(
            host=config["host"], port=config["port"], user=config["user"], password=config["password"],
            database=config["database"], server_settings={"default_transaction_read_only": "on"}, timeout=30,
        )
        try:
            return await connection.fetch(query, *args)
        finally:
            await connection.close()
    except Exception as e:
        raise RuntimeError(f"Could not query the VeriTaS database (see the database section of {config_path}): "
                           f"{type(e).__name__}: {e}") from e
