"""Small text utilities used by the publishing service."""


def normalize_words(text: str) -> str:
    """Collapse whitespace and apply the project's case-insensitive convention."""
    return " ".join(text.split()).casefold()


def make_slug(title: str) -> str:
    """Create a URL-safe slug while keeping Unicode letters and numbers."""
    raise NotImplementedError
