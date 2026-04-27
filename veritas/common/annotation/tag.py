from dataclasses import dataclass

@dataclass
class Tag:
    """A finer-grained, non-exclusive label for claims and media."""
    name: str
    definition: str
