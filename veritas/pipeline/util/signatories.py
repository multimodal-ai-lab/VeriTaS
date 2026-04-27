from datetime import datetime
from typing import Collection, Optional

import aiohttp
from bs4 import BeautifulSoup, Tag
from dateutil import parser

from veritas import logger
from veritas.common import Publisher, SignatoryStatus
from veritas.db import db
from veritas.util import get_domain
from veritas.util.scraping import get_all_links, get_dynamic_htmls, get_static_htmls
from veritas.util.util import get_lang_iso_code

IFCN_CODE_OF_PRINCIPLES_DOMAIN = "https://ifcncodeofprinciples.poynter.org"
IFCN_SIGNATORIES_URL = "https://ifcncodeofprinciples.poynter.org/signatories"

EFCSN_BASE_URL = "https://members.efcsn.com"
EFCSN_SIGNATORIES_URL = "https://members.efcsn.com/signatories"


async def update_signatories():
    """Downloads the current profile information from all IFCN and EFCSN signatories/members
    and saves them to the DB."""
    await db.connect_maybe_initialize()
    logger.info("Updating signatories...")
    await update_ifcn_signatories()
    await update_efcsn_members()
    logger.info("Update finished!")
    await db.close()


async def update_ifcn_signatories():
    logger.debug("Fetching IFCN signatories URLs...")
    urls = await get_ifcn_signatories_profile_urls()
    if not urls:
        logger.error("No signatories found!")
        return None
    logger.info(f"Found {len(urls)} signatory profiles. Reading them...")
    await read_and_save_ifcn_profiles(urls)


async def update_efcsn_members():
    logger.debug("Fetching EFCSN member URLs...")
    urls = await get_efcsn_member_profile_urls()
    if not urls:
        logger.error("No EFCSN members found!")
        return None
    logger.info(f"Found {len(urls)} member profiles. Reading them...")
    await read_and_save_efcsn_profiles(urls)


async def get_ifcn_signatories_profile_urls() -> list[str]:
    """Applies BeautifulSoup to retrieve the URLs of all IFCN signatories' profiles,
    including in renewal and expired signatories."""
    links = [
        *await get_all_links(IFCN_SIGNATORIES_URL),
        *await get_all_links(IFCN_SIGNATORIES_URL, buttons_to_click=['[data-rr-ui-event-key="In Renewal"]']),
        # Officially removed:
        # *await get_all_links(IFCN_SIGNATORIES_URL, buttons_to_click=['[data-rr-ui-event-key="Expired"]']),
    ]
    # Normalize and filter links by the desired scheme
    matching_links = [IFCN_CODE_OF_PRINCIPLES_DOMAIN + link for link in links if link.startswith("/profile/")]

    return list(set(matching_links))


async def read_and_save_ifcn_profiles(urls: list[str]):
    """Scrapes the signatory's profile and extracts their name, URL, status, etc."""
    htmls = await get_dynamic_htmls(urls)
    for html in htmls:
        if html:
            publisher = await _read_ifcn_profile(BeautifulSoup(html, "html.parser"))
            if publisher:
                await save_publisher(publisher)


async def save_publisher(publisher: Publisher):
    """Upserts the publisher into the DB. Handles name collisions, compares them and decides
    whether to keep both or merge."""
    domain = list(publisher.domains)[0]

    if existing_publisher := await db.get_publisher_by_url(domain):
        # Update the existing publisher's signatory information
        await update_signatory_status(existing_publisher, publisher)

    elif existing_publisher := await db.get_publisher_by_name(publisher.name):
        # Find out if existing and new publisher are indeed identical organizations

        if publisher.language and publisher.language != existing_publisher.language:
            # Keep both but change name of new publisher
            publisher.name = f"{publisher.name} ({publisher.language})"
            await publisher.save_to_db()
            logger.info(f"Found and saved new publisher {publisher.name} ({', '.join(publisher.domains)})")

        elif publisher.country and publisher.country != existing_publisher.country:
            # Keep both but change name of new publisher
            publisher.name = f"{publisher.name} ({publisher.country})"
            await publisher.save_to_db()
            logger.info(f"Found and saved new publisher {publisher.name} ({', '.join(publisher.domains)})")

        else:
            # Name, country and language are identical, so this is likely
            # the same publisher with new domain => merge.
            existing_publisher.domains.update(publisher.domains)
            await update_signatory_status(existing_publisher, publisher)

    else:
        # Found a completely new publisher
        await publisher.save_to_db()
        logger.info(f"Found and saved new publisher: {publisher.name} ({', '.join(publisher.domains)})")


async def update_signatory_status(existing_publisher: Publisher, new_publisher: Publisher):
    """Updates existing_publisher with the signatory data of new_publisher."""
    if new_publisher.ifcn_status:
        existing_publisher.ifcn_status = new_publisher.ifcn_status
        existing_publisher.ifcn_expires = new_publisher.ifcn_expires
    if new_publisher.efcsn_status:
        existing_publisher.efcsn_status = new_publisher.efcsn_status
        existing_publisher.efcsn_expires = new_publisher.efcsn_expires
    await existing_publisher.save_to_db()
    logger.info(f"Updated existing publisher {existing_publisher.name} ({', '.join(existing_publisher.domains)})")


async def _read_ifcn_profile(soup: BeautifulSoup) -> Optional[Publisher]:
    # Extracting the fields
    h1 = soup.find("h1")
    if not h1:
        return None

    title = h1.text.strip()
    if "(In-Renewal)" in title:
        name = title.replace("(In-Renewal)", "").strip()
        ifcn_status = SignatoryStatus.IN_RENEWAL
    elif "(Expired)" in title:
        name = title.replace("(Expired)", "").strip()
        ifcn_status = SignatoryStatus.EXPIRED
    else:
        name = title
        ifcn_status = SignatoryStatus.ACTIVE

    country = language = expires = website = None

    # Go through all <p> and <div> tags and find the needed info
    for tag in soup.find_all(["p", "div"]):
        text = tag.get_text(strip=True)
        if text.startswith("Country:"):
            country = text.replace("Country:", "").strip()
        if text.startswith("Language:"):
            language = get_lang_iso_code(text.replace("Language:", "").strip())
        elif text.startswith("Expires on:"):
            expires = text.replace("Expires on:", "").strip()
        elif text.startswith("Website:"):
            a_tag = tag.find("a")
            if a_tag:
                website = a_tag["href"]
                break

    if not website or not isinstance(website, str):
        logger.warning(f"IFCN profile of signatory {name} has no website URL. Skipping it...")
        return None

    if expires:
        try:
            expires = datetime.strptime(expires, "%d %b %Y").date()
        except ValueError:
            expires = parser.parse(expires).date()
    else:
        expires = None

    domain = get_domain(website)

    return Publisher(
        name=name, domains={domain}, ifcn_status=ifcn_status, ifcn_expires=expires,
        country=country, language=language
    )


async def get_efcsn_member_profile_urls() -> list[str]:
    """Applies BeautifulSoup to retrieve the URLs of all IFCN signatories' profiles,
    including in renewal and expired signatories."""
    links = await get_all_links(EFCSN_SIGNATORIES_URL)
    # Normalize and filter links by the desired scheme
    matching_links = [EFCSN_BASE_URL + link for link in links if link.startswith("/organization/")]

    return list(set(matching_links))


async def read_and_save_efcsn_profiles(urls: Collection[str]):
    """Scrapes the signatory's profile and extracts their name, URL, status, etc."""
    async with aiohttp.ClientSession() as session:
        htmls = await get_static_htmls(urls, session)
    for html in htmls:
        if html:
            publisher = await _read_efcsn_profile(BeautifulSoup(html, "html.parser"))
            if publisher:
                await save_publisher(publisher)


async def _read_efcsn_profile(soup: BeautifulSoup) -> Optional[Publisher]:
    h1 = soup.find("h1")
    if not h1:
        return None

    name = h1.text.strip()
    efcsn_status = SignatoryStatus.ACTIVE
    country = language = expires = website = None

    # country is in two spans: <span>Country:</span> <span>Country Name</span>
    country_span = soup.find("span", text="Country:")
    if country_span:
        country_span = country_span.find_next_sibling("span")
        if country_span:
            country = country_span.text.strip()

    # language is in two spans: <span>Language:</span> <span>Language Name</span>
    language_span = soup.find("span", text="Language:")
    if language_span:
        language_span = language_span.find_next_sibling("span")
        if language_span:
            language = language_span.text.strip()
            language = get_lang_iso_code(language)

    # expires is in two spans: <span>Valid Until:</span> <span>Valid until date</span>
    valid_until_span = soup.find("span", text="Valid Until: ")
    if valid_until_span:
        valid_until_span = valid_until_span.find_next_sibling("span")
        if valid_until_span:
            expires = valid_until_span.text.strip()
            try:
                expires = datetime.strptime(expires, "%d %b %Y").date()
            except ValueError:
                expires = parser.parse(expires).date()

    # website is a span followed by an <a> tag: <span>Website:</span> <a href="...">Website Name</a>
    website_span = soup.find("span", text="Website:")
    if website_span:
        website_span = website_span.find_next_sibling("a")
        if website_span and isinstance(website_span, Tag) and "href" in website_span.attrs:
            website = website_span["href"]

    if not website or not isinstance(website, str):
        logger.warning(f"EFCSN profile of signatory {name} has no website URL. Skipping it...")
        return None

    domain = get_domain(website)

    return Publisher(
        name=name,
        domains={domain},
        efcsn_status=efcsn_status,
        efcsn_expires=expires,
        country=country,
        language=language,
    )
