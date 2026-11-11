"""Surface patterns used by EDA now and by the masking variants (P1-P3) later."""

import re

URL = re.compile(r"(?i)\b(?:https?://\S+|www\.\S+|[a-z0-9-]+\.(?:com|co\.uk|net|org|biz|info|tv)\b\S*)")
# >= 8 digits, optionally split by single spaces/dashes: "08452810075", "0808 145 4742".
PHONE = re.compile(r"(?<!\d)\d(?:[ -]?\d){7,}(?!\d)")
# Standalone 5-6 digit numbers: premium-rate short codes such as 87121 or 86888.
SHORTCODE = re.compile(r"(?<![\d£$€.,])\d{5,6}(?![\d.,]?\d)")
MONEY = re.compile(r"(?i)[£$€]\s?\d[\d,]*(?:\.\d+)?|\b\d{1,4}(?:\.\d{1,2})?\s?(?:p|ppm|pence|gbp|pounds?)\b")
REPEATED_LETTER = re.compile(r"(?i)(?!www\b)([a-z])\1{2,}")
REPEATED_PUNCT = re.compile(r"([!?.])\1{2,}")
HTML_ENTITY = re.compile(r"&(?:lt|gt|amp|quot|#\d+);")
# cp1252 punctuation left undecoded in the source file (e.g. \x92 for an apostrophe).
C1_CONTROL = re.compile(r"[\x80-\x9f]")
EMOTICON = re.compile(r"(?<!\w)[:;=]-?[()DPp](?!\w)|<3")
EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿]")
