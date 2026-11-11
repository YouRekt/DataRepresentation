"""Controlled, reproducible typos for the robustness test."""

import re

import numpy as np

WORD = re.compile(r"[A-Za-z]{3,}")
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _edit(word: str, rng: np.random.Generator) -> str:
    i = int(rng.integers(len(word)))
    match rng.integers(4):
        case 0:  # deletion
            return word[:i] + word[i + 1:]
        case 1:  # swap with the next letter
            i = min(i, len(word) - 2)
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        case 2:  # substitution
            return word[:i] + LETTERS[rng.integers(26)] + word[i + 1:]
        case _:  # duplicated letter
            return word[:i] + word[i] + word[i:]


def add_typos(text: str, rate: float, seed: int, record_id: int) -> str:
    """Edit each word of >= 3 letters with probability `rate`; same (seed, id) -> same output."""
    rng = np.random.default_rng([seed, record_id])
    return WORD.sub(lambda m: _edit(m[0], rng) if rng.random() < rate else m[0], text)
