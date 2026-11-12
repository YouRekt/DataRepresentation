"""B0r: hand-written rule from the Milestone 1 EDA markers (no learning)."""

from smsspam import patterns as P

# The four markers that separate the classes most on train (spam 18-62% vs ham <= 0.3%).
RULE_MARKERS = {"phone": P.PHONE, "money": P.MONEY, "shortcode": P.SHORTCODE, "url": P.URL}


def marker_count(text: str) -> int:
    """How many of the four markers occur in the message (0-4); used as the rule's score."""
    return sum(p.search(text) is not None for p in RULE_MARKERS.values())
