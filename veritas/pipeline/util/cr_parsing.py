import json
import warnings
from json import JSONDecodeError

import extruct
import flatten_json
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
from json_repair import repair_json

# Hide warnings printed when encountering an XML document
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def jsonld_parser(page) -> list[dict]:
    """Reads the ClaimReview metadata from the page."""
    data = extruct.extract(page, syntaxes=["json-ld"])
    return traverse_graph(data["json-ld"], "ClaimReview")


def traverse_graph(graph: dict | list | None, target_type: str) -> list[dict]:
    """Takes a schema.org @graph and returns all (sub)components
    that match the target @type."""
    if not graph:
        return []
    elif isinstance(graph, dict):
        if target_type == graph.get("@type"):
            return [graph]
        else:
            return traverse_graph(graph.get("@graph"), target_type)
    elif isinstance(graph, list):
        matches = []
        for el in graph:
            matches.extend(traverse_graph(el, target_type))
        return matches


def microdata_parser(page) -> list[dict]:
    """Reads the ClaimReview metadata from the page."""
    data = extruct.extract(page, syntaxes=["microdata"])
    # Filter before flattening otherwise merge errors
    microdata = [el for el in data["microdata"] if el["type"] == "http://schema.org/ClaimReview"]
    jsonld = _to_jsonld(microdata)
    # Get only the ClaimReview, not other microdata
    claim_reviews = [el for el in jsonld if ("@type" in el and "ClaimReview" in el["@type"])]
    return claim_reviews


def custom_parser(page) -> list[dict]:
    soup = BeautifulSoup(page, "html.parser")
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})

    results = []
    for script in scripts:
        if "ClaimReview" in script.text:
            decoded = try_json_loads(script.text)
            if decoded:
                crs = traverse_graph(decoded, "ClaimReview")
                results.extend(crs)

    return results


def try_json_loads(json_str: str) -> list | dict | None:
    """Good to robustly load ill-formatted JSON."""
    # Heal any broken syntax (indeed occurring from time to time)
    json_str = repair_json(json_str)

    try:
        return json.loads(json_str)
    except JSONDecodeError:
        pass

    try:
        return json.loads(f"[{json_str}]")
    except JSONDecodeError:
        pass


def _to_jsonld(microdata):
    context = "http://schema.org"
    properties = "properties_"
    typestr = "type"
    jsonld_data = {}
    jsonld_data["@context"] = context
    for data in microdata:
        data = flatten_json.flatten(data)
        for key in data.keys():
            value = data[key]
            if context in value:
                value = value.replace(context + "/", "")
            if properties in key:
                keyn = key.replace(properties, "")
                jsonld_data[keyn] = value
                if typestr in keyn:
                    keyn = keyn.replace(typestr, "@" + typestr)
                    jsonld_data[keyn] = value
            if typestr is key:
                keyn = key.replace(typestr, "@" + typestr)
                jsonld_data[keyn] = value
        del data
    jsonld_data = flatten_json.unflatten(jsonld_data)
    return [jsonld_data]
