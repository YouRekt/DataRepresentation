"""Text transforms for the preprocessing variants. B1 is the identity.

Masks are written as `_name_` so B1's default token pattern keeps each one as a single
token that cannot collide with a real word ("phone" vs "_phone_").
"""

from collections.abc import Callable

from smsspam import patterns as P

Transform = Callable[[str], str]


def identity(text: str) -> str:
    return text


def mask_numbers(text: str) -> str:
    """P1: money first (it may contain digits), then phone numbers, then short codes."""
    text = P.MONEY.sub(" _money_ ", text)
    text = P.PHONE.sub(" _phone_ ", text)
    return P.SHORTCODE.sub(" _shortcode_ ", text)


def mask_urls(text: str) -> str:
    """P2a: replace each URL or bare domain with one token."""
    return P.URL.sub(" _url_ ", text)


def remove_urls(text: str) -> str:
    """P2b: drop URLs entirely, to measure how much the model relies on them."""
    return P.URL.sub(" ", text)


def squeeze_repeats(text: str) -> str:
    """P3: three or more repeated letters or ! ? . become two ("sooo" -> "soo")."""
    text = P.REPEATED_LETTER.sub(r"\1\1", text)
    return P.REPEATED_PUNCT.sub(r"\1\1", text)


def compose(*transforms: Transform) -> Transform:
    def run(text: str) -> str:
        for t in transforms:
            text = t(text)
        return text

    return run


# URLs go first in combinations so digits inside a link are not masked as numbers.
VARIANTS: dict[str, Transform] = {
    "B1": identity,
    "P1": mask_numbers,
    "P2a": mask_urls,
    "P2b": remove_urls,
    "P3": squeeze_repeats,
}
