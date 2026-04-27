from dataclasses import dataclass
from typing import Optional

import yaml

from veritas.common.annotation.tag import Tag

@dataclass
class Property:
    """An aspect of a claim or a medium."""
    name: str
    definition: str

    negative_category: str
    negative_definition: str
    negative_tags: list[Tag]

    positive_category: str
    positive_definition: str
    positive_tags: list[Tag]

    hints: Optional[str] = None

    @property
    def categories(self) -> list[str]:
        return [self.negative_category, self.positive_category]

    @property
    def tags(self) -> list[Tag]:
        return self.negative_tags + self.positive_tags


def load_properties() -> dict[str, Property]:
    """Reads the labeling scheme YAML file and turns it into a list of Property objects,
    including names, definitions, and tags."""
    properties = dict()

    # Load the labeling scheme YAML
    labeling_scheme = yaml.safe_load(open("veritas/labels.yaml"))
    for property_id, property_dict in labeling_scheme["properties"].items():
        # Read in tags
        negative_tags = [
            Tag(name=tag["name"], definition=tag["definition"])
            for tag in property_dict["negative"]["tags"]
        ] if "tags" in property_dict["negative"] else []
        positive_tags = [
            Tag(name=tag["name"], definition=tag["definition"])
            for tag in property_dict["positive"]["tags"]
        ] if "tags" in property_dict["positive"] else []

        # Construct property
        prop = Property(
            name=property_dict["name"],
            definition=property_dict["definition"],
            negative_category=property_dict["negative"]["name"],
            negative_definition=property_dict["negative"]["definition"],
            negative_tags=negative_tags,
            positive_category=property_dict["positive"]["name"],
            positive_definition=property_dict["positive"]["definition"],
            positive_tags=positive_tags,
            hints=property_dict.get("hints")
        )
        properties[property_id] = prop

    return properties


PROPERTIES: dict[str, Property] = load_properties()
