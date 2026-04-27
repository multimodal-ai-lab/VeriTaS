import asyncio
from veritas.db import db
from veritas.util import get_domain


async def print_publisher_domains():
    await db.connect_maybe_initialize()

    # Fetch URLs of reviews belonging to a released claim
    query = """
        SELECT r.url
        FROM reviews r
        JOIN claims c ON r.claim_id = c.id
        WHERE c.released_quarter IS TRUE OR c.released_longitudinal IS TRUE
    """
    records = await db._fetch(query)
    urls = [record['url'] for record in records]

    # Extract domains and sort domains alphabetically
    domains = set([get_domain(url) for url in urls if url])
    domains = sorted(domains)

    # Print domains separated by newlines
    for domain in domains:
        print(domain)

    await db.close()


if __name__ == "__main__":
    asyncio.run(print_publisher_domains())
