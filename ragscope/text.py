"""Tiny, dependency-free text utilities.

Everything ragscope does lexically goes through here: tokenizing, light
stemming, sentence splitting and masking of citation / list markers.
The goal is *predictable* behaviour, not linguistic perfection.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set, Tuple

STOPWORDS = frozenset(
    """
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had
has have having he her here hers herself him himself his how i if in into is it its itself just
me more most my myself of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves may might must shall also us
""".split()
)
# NOTE: "no", "not" and "nor" are deliberately *not* stopwords: negations matter.

# Words that commonly appear in LLM answers but are not factual claims.
_FILLER_WORDS = """
according based context provided document documents source sources mentioned mention states
stated says said information answer question questions sure however therefore additionally
overall summary note yes hope help helps let know feel free glad happy assist great certainly
following below above mentioned specifically generally typically example
""".split()


_TOKEN_RE = re.compile(r"\d+(?:[.,:]\d+)*%?|[^\W\d_][^\W_]*(?:['’][^\W\d_]+)*")
_VOWELS = set("aeiou")

# Citation markers like [1], [2, 3], [doc1] and list bullets like "- ", "1. ", "2) ".
_MARKER_RE = re.compile(
    r"\[(?:[A-Za-z_-]*\d+)(?:\s*[,;-]\s*[A-Za-z_-]*\d+)*\]|^\s*(?:[-*•]|\d+[.)])\s+",
    re.MULTILINE,
)

_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "prof", "fig", "no", "approx",
    "inc", "ltd", "st", "jr", "sr", "cf", "al",
}
_BOUNDARY_RE = re.compile(r"([.!?]+[\"'”’)\]]*)\s+")


def _undouble(word: str) -> str:
    if len(word) >= 3 and word[-1] == word[-2] and word[-1] not in "lsz" and word[-1] not in _VOWELS:
        return word[:-1]
    return word


_ED_EXCEPTIONS = {"embed", "hundred", "sacred", "naked", "wicked", "shed", "bed", "red"}


def stem(word: str) -> str:
    """A deliberately light stemmer: maps common inflections onto one key.

    >>> stem("retrieved") == stem("retrieves") == stem("retrieve")
    True
    """
    w = word.lower().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) <= 3 or w[0].isdigit():
        return w
    if w.endswith(("ies", "ied")) and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "xes", "zes", "ches", "shes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    if w.endswith("ing") and len(w) > 5:
        w = _undouble(w[:-3])
    elif w.endswith("ed") and len(w) >= 4 and w not in _ED_EXCEPTIONS and not w.endswith("eed"):
        w = _undouble(w[:-2])
    if w.endswith("ly") and len(w) > 5:
        w = w[:-2]
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


FILLER = frozenset(stem(w) for w in _FILLER_WORDS)

# Words that shape a question but are not things a retriever should find.
_QUERY_FILLER = frozenset(stem(w) for w in """
explain tell describe show give list possible way ways without within anyone someone
""".split())
_HOW_WORDS = {"long", "many", "much", "often", "far", "big", "old", "soon", "fast"}


def query_keys(query: str) -> List[str]:
    """Content keys of a query, minus question scaffolding ("how long", "explain")."""
    out: List[str] = []
    prev = ""
    for _, _, tok in tokens(query):
        low = tok.lower()
        if prev == "how" and low in _HOW_WORDS:
            prev = low
            continue
        prev = low
        key = token_key(tok)
        if key and key not in _QUERY_FILLER:
            out.append(key)
    return list(dict.fromkeys(out))


def normalize_number(token: str) -> str:
    """'1,200' -> '1200', '40%' -> '40', '3.50' -> '3.5'."""
    t = token.rstrip("%")
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", t):
        t = t.replace(",", "")
    if "." in t and re.fullmatch(r"\d+\.\d+", t):
        t = t.rstrip("0").rstrip(".")
    return t


def is_number_key(key: str) -> bool:
    return bool(key) and key[0].isdigit()


def token_key(token: str, *, drop_filler: bool = False) -> Optional[str]:
    """Return the comparison key for a token, or None if it carries no content."""
    if token[0].isdigit():
        return normalize_number(token)
    low = token.lower().replace("’", "'")
    if low in STOPWORDS or len(low) < 2:
        return None
    key = stem(low)
    if key in STOPWORDS:
        return None
    if drop_filler and key in FILLER:
        return None
    return key


def mask_markers(text: str) -> str:
    """Blank out citation and list markers, preserving character offsets."""
    return _MARKER_RE.sub(lambda m: " " * len(m.group()), text)


def tokens(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in _TOKEN_RE.finditer(text)]


def content_keys(text: str, *, drop_filler: bool = False, mask: bool = False) -> List[str]:
    """Ordered content keys (stemmed, stopwords removed) for ``text``."""
    src = mask_markers(text) if mask else text
    out = []
    for _, _, tok in tokens(src):
        key = token_key(tok, drop_filler=drop_filler)
        if key:
            out.append(key)
    return out


def bigrams(keys: List[str]) -> Set[Tuple[str, str]]:
    return {(a, b) for a, b in zip(keys, keys[1:])}


def surface_forms(text: str) -> dict:
    """Map each content key to the first surface word that produced it."""
    forms: dict = {}
    for _, _, tok in tokens(text):
        key = token_key(tok)
        if key and key not in forms:
            forms[key] = tok
    return forms


def segments(text: str, *, drop_filler: bool = False, mask: bool = False) -> List[list]:
    """Split ``text`` into ``[piece, key_or_None]`` pairs covering the whole string.

    Used by the HTML report to highlight matching words without re-implementing
    the stemmer in JavaScript.
    """
    src = mask_markers(text) if mask else text
    out: List[list] = []
    pos = 0
    for start, end, tok in tokens(src):
        if start > pos:
            out.append([text[pos:start], None])
        out.append([text[start:end], token_key(tok, drop_filler=drop_filler)])
        pos = end
    if pos < len(text):
        out.append([text[pos:], None])
    return out


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` character spans of sentences in ``text``.

    Newlines always end a sentence (so bullet lists split per item); ``.``/``!``/``?``
    end one unless they follow a known abbreviation or the next word is lowercase.
    """
    spans: List[Tuple[int, int]] = []
    for line_m in re.finditer(r"[^\n]+", text):
        line, base, start = line_m.group(), line_m.start(), 0
        for m in _BOUNDARY_RE.finditer(line):
            before = line[start:m.start(1)].split()
            last = before[-1].lower().rstrip(".") if before else ""
            nxt = line[m.end():m.end() + 1]
            if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
                continue
            if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in "\"'“‘([*-"):
                continue
            spans.append((base + start, base + m.end(1)))
            start = m.end()
        rest = line[start:].rstrip()
        if rest.strip():
            spans.append((base + start, base + start + len(rest)))
    cleaned = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        if s < e:
            cleaned.append((s, e))
    return cleaned


def jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
