import re
from datetime import datetime
from typing import Optional

from dateutil import parser
from ezmm import Item
from ezmm.util import is_item_ref
from langcodes import Language


def get_lang_from_code(lang_code: str) -> Language | None:
    """Takes the ISO code of a language and turns it into a Language object."""
    try:
        return Language.get(lang_code)
    except Exception:
        return None


def remove_wrapping_quotes(s):
    """
    Remove quotes only if they wrap the entire string (not structural).
    """

    def should_remove_wrapping_quotes(s):
        """
        Determine if outer quotes should be removed using regex patterns.
        Returns True if quotes should be removed, False otherwise.
        """
        # Must start and end with same quote type
        if not ((s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'"))):
            return False

        # Must be long enough
        if len(s) < 2:
            return False

        # Patterns that indicate structural quotes (should NOT remove)
        structural_patterns = [
            r'^"[^"]+",',  # "Name", ... (quoted name followed by comma)
            r'\w+\s+"[^"]*"$',  # word(s) "speech" (attribution pattern)
            r'^"[^"]*"\s+\w+',  # "quote" word(s) (quote followed by words)
            r'^"[^"]*"\s+has\s+been',  # "something" has been ...
            r'said\s+"[^"]*"$',  # said "something" at the end
        ]

        # Check if any structural pattern matches
        for pattern in structural_patterns:
            if re.search(pattern, s):
                return False

        # Additional check: if removing outer quotes leaves unbalanced quotes,
        # it's probably wrapping
        inner_content = s[1:-1]
        quote_count = inner_content.count('"') + inner_content.count("'")

        # If odd number of quotes inside, likely structural
        if quote_count % 2 == 1:
            return False

        # If we get here, likely wrapping quotes
        return True

    if should_remove_wrapping_quotes(s):
        return s[1:-1]
    return s


def strip_string(s: str) -> str:
    """Strips a string of newlines and spaces."""
    return s.strip(' \n')


def extract_last(text: str, delimiter: str, **kwargs) -> Optional[str]:
    """Extracts the contents of the last block of text (enclosed with delimiter)
     appearing in the given string. If no block of text is found, returns ''."""
    matches = find(text, delimiter, **kwargs)
    if matches:
        match = strip_string(matches[-1])
        if match:
            return match


def extract_last_code_block(text: str) -> Optional[str]:
    """Extracts the last code block from the given text. Ignores the language tag."""
    code_block = extract_last(text, "```")
    if code_block:
        lines = code_block.split("\n")
        if len(lines[0]) <= 8:
            # First line seems to be a language tag, so remove it
            return "\n".join(lines[1:]).strip()
        return code_block
    return None


def extract_all(text: str, delimiter: str, **kwargs) -> list[str]:
    """Extracts the contents of all blocks of text (enclosed with delimiter)
     appearing in the given string. If no block of text is found, returns an empty list."""
    matches = find(text, delimiter, **kwargs)
    return [strip_string(match) for match in matches if strip_string(match)]


def find(text: str, delimiter: str, allowed_symbols: str = ".*?"):
    # Composition of negative lookbehind, delimiters, allowed content symbols, and a negative lookahead
    pattern = re.compile(f'(?<!{delimiter}){delimiter}({allowed_symbols}){delimiter}(?!{delimiter})', re.DOTALL)
    matches = pattern.findall(text)
    return matches


def detect_hallucinated_media_refs(text: str):
    """Throws a ValueError if the given text contains hallucinated media references."""
    # Extract all substrings of the form '<...>'
    substrings = re.findall(r'<[^>]*>', text)
    # Check if any substring contains a valid item reference
    for substring in substrings:
        if is_item_ref(substring):
            assert Item.from_reference(substring) is not None
        else:
            raise ValueError(f"Hallucinated media reference found: {substring}")


def determine_date(text: str) -> Optional[datetime]:
    """Attempts to determine the date from the given text. Returns None if no date is found."""
    if text:
        if isinstance(text, datetime):
            return text
        elif isinstance(text, str):
            text = text.strip()
            try:
                return datetime.fromisoformat(text.strip()).replace(tzinfo=None).replace(tzinfo=None)
            except ValueError:
                pass
            try:
                return parser.parse(text.strip()).replace(tzinfo=None)
            except Exception:
                return None
    return None


def perform_extraction(html: str, extraction: dict) -> object | None:
    """Perform a simple BeautifulSoup extraction as specified on the given HTML code.

    Extraction spec supports the following keys:
    - select: CSS selector string to locate an element (preferred)
    - tag: HTML tag name to find (ignored if 'select' is provided)
    - attrs: dict of attributes for find/find_all
    - index: optional index for find_all/select results
    - get: attribute name to extract; use 'text' or omit to get element's text
    Example: {"tag": "input", "attrs": {"name": "q"}, "get": "value"}
    """
    if not html or not extraction:
        return None

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    select = extraction.get("select")
    tag = extraction.get("tag")
    attrs = extraction.get("attrs", {})
    index = extraction.get("index")
    getter = extraction.get("get")

    element = None
    try:
        if select:
            matches = soup.select(select)
            if matches:
                element = matches[index] if index is not None and -len(matches) <= index < len(matches) else matches[0]
        elif tag:
            if index is None:
                element = soup.find(tag, attrs=attrs)
            else:
                matches = soup.find_all(tag, attrs=attrs)
                if matches and -len(matches) <= index < len(matches):
                    element = matches[index]
        else:
            return None
    except Exception:
        element = None

    if not element:
        return None

    if getter and getter != "text":
        return element.get(getter)
    else:
        return element.get_text(strip=True)


def lang_iso_to_name(lang_iso: str | None) -> str | None:
    """Returns the human-readable name of the given language ISO 639-1 language code."""
    if not lang_iso:
        return None
    try:
        lang = Language.get(lang_iso)
        return lang.display_name()
    except Exception:
        pass
    return None
