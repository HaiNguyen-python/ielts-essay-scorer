"""Small, dependency-free text helpers shared by features, feedback and the grounding check.

The same splitting and normalisation is used everywhere on purpose: the grounding checker must
see the essay exactly as the feedback provider saw it.
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")
_PARAGRAPH = re.compile(r"\n\s*\n")


def normalise_ws(text: str) -> str:
    """Collapse every run of whitespace (spaces, tabs, newlines) to one space."""
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list[str]:
    """Lower-case word tokens; punctuation and digits are dropped."""
    return [t.lower() for t in _TOKEN.findall(text)]


def paragraphs(text: str) -> list[str]:
    """Paragraphs are separated by at least one blank line, as in a typed IELTS answer."""
    return [p.strip() for p in _PARAGRAPH.split(text.strip()) if p.strip()]


def sentences(text: str) -> list[str]:
    """Sentences in reading order, split after . ! or ? followed by whitespace."""
    out: list[str] = []
    for para in paragraphs(text):
        out.extend(s.strip() for s in _SENTENCE_END.split(normalise_ws(para)) if s.strip())
    return out


def word_count(text: str) -> int:
    return len(tokens(text))
