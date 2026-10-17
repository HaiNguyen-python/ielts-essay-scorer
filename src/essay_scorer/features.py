"""Hand-made features, grouped by the IELTS criterion they are meant to inform.

Every feature is a small pure function on the essay text (and the prompt, for Task Response),
so each one can be tested on a hand example and explained to a teacher. None of them reads
the L1 group or the prompt id.

The grammar error rate exists in two versions on purpose:
- ``error_rate_all`` counts article omissions like any other error (the naive model);
- ``error_rate_no_articles`` leaves them out (the fairness fix, see fairness.py).
"""

from __future__ import annotations

import statistics
import zlib

from .lexicon import (
    CLAUSE_MARKERS,
    COLLOCATIONS,
    CONCLUSION_MARKERS,
    CONNECTIVES,
    COUNTABLE_NOUNS,
    DETERMINERS,
    ERROR_PATTERNS,
    POSITION_MARKERS,
    SIGNPOSTS,
    STOPWORDS,
    is_common,
)
from .text import normalise_ws, paragraphs, sentences, tokens

# Longest first, so "on the other hand" wins over "on".
_CONNECTIVES_BY_LENGTH = sorted(CONNECTIVES, key=len, reverse=True)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "task_response": (
        "word_count",
        "prompt_overlap",
        "on_topic_share",
        "position_markers",
        "has_conclusion",
    ),
    "coherence_cohesion": (
        "paragraph_count",
        "connective_share",
        "distinct_connectives",
        "topic_sentence_share",
    ),
    "lexical_resource": (
        "mattr",
        "rare_word_share",
        "collocations_per_100w",
        "mean_word_length",
    ),
    "grammatical_range_accuracy": (
        "mean_sentence_length",
        "sentence_length_sd",
        "clauses_per_sentence",
        "error_rate_no_articles",
    ),
}
# The naive feature set: the same, except that article omissions count as grammar errors.
NAIVE_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    c: tuple("error_rate_all" if f == "error_rate_no_articles" else f for f in feats)
    for c, feats in FEATURE_GROUPS.items()
}

N_CONTENT_BUCKETS = 32


def stem(word: str) -> str:
    """Very light stemming so that 'student' matches 'students' when comparing with the prompt."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_stems(text: str) -> set[str]:
    """Stems of the words that carry meaning (no stopwords, at least 3 letters)."""
    return {stem(t) for t in tokens(text) if len(t) >= 3 and t not in STOPWORDS}


# ---- Task Response -------------------------------------------------------------------------


def prompt_overlap(essay: str, prompt: str) -> float:
    """Share of the prompt's content words that the essay uses. Low = possibly off task."""
    keys = content_stems(prompt)
    if not keys:
        return 0.0
    return len(keys & content_stems(essay)) / len(keys)


def on_topic_share(essay: str, prompt: str) -> float:
    """Share of sentences that mention at least one prompt content word."""
    keys = content_stems(prompt)
    sents = sentences(essay)
    if not sents or not keys:
        return 0.0
    return sum(1 for s in sents if content_stems(s) & keys) / len(sents)


def position_markers(essay: str) -> int:
    """How often the writer states a position ('In my opinion', 'I believe', ...)."""
    low = " " + normalise_ws(essay).lower() + " "
    return sum(low.count(" " + m + " ") + low.count(" " + m + ",") for m in POSITION_MARKERS)


def has_conclusion(essay: str) -> float:
    """1 if one of the last two sentences opens with a conclusion marker."""
    last = [s.lower() for s in sentences(essay)[-2:]]
    return float(any(s.startswith(m) for s in last for m in CONCLUSION_MARKERS))


# ---- Coherence and Cohesion ----------------------------------------------------------------


def leading_connective(sentence: str) -> str | None:
    """The linking word or phrase a sentence starts with, if any."""
    low = sentence.lower()
    for conn in _CONNECTIVES_BY_LENGTH:
        if low.startswith(conn) and (len(low) == len(conn) or not low[len(conn)].isalpha()):
            return conn
    return None


def connective_stats(essay: str) -> tuple[float, int]:
    """(share of sentences that open with a connective, number of different connectives).

    Variety matters as much as frequency: 'Also, ... Also, ... And ...' is a typical Band 5
    pattern, a mix of 'However', 'As a result', 'In contrast' a Band 7 one. A ratio such as
    distinct / used would give a perfect score to an essay with a single connective, so the
    plain count of different connectives is used.
    """
    sents = sentences(essay)
    used = [c for c in (leading_connective(s) for s in sents) if c]
    if not sents:
        return 0.0, 0
    return len(used) / len(sents), len(set(used))


def topic_sentence_share(essay: str, prompt: str) -> float:
    """Share of body paragraphs whose first sentence signposts or names the topic.

    Body paragraphs are all but the first and last. Fewer than three paragraphs means there is
    no clear body, which scores 0.
    """
    paras = paragraphs(essay)
    if len(paras) < 3:
        return 0.0
    keys = content_stems(prompt)
    body = paras[1:-1]
    hits = 0
    for para in body:
        first = (sentences(para) or [""])[0]
        low = first.lower()
        signposted = any(low.startswith(s) for s in SIGNPOSTS)
        if not low.startswith("for example") and (signposted or content_stems(first) & keys):
            hits += 1
    return hits / len(body)


# ---- Lexical Resource ----------------------------------------------------------------------


def mattr(words: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio.

    Plain type-token ratio falls as texts get longer, which would punish long essays. MATTR
    averages the ratio over every window of `window` words, so length matters much less.
    """
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    ratios = [len(set(words[i : i + window])) / window for i in range(len(words) - window + 1)]
    return sum(ratios) / len(ratios)


def rare_word_share(words: list[str]) -> float:
    """Share of content words that are not in the shipped list of common words."""
    content = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    if not content:
        return 0.0
    return sum(1 for w in content if not is_common(w)) / len(content)


def collocations_per_100w(essay: str) -> float:
    words = tokens(essay)
    low = " " + " ".join(words) + " "
    n = max(1, len(words))
    hits = sum(low.count(" " + c + " ") for c in COLLOCATIONS)
    return 100.0 * hits / n


def mean_word_length(words: list[str]) -> float:
    return sum(len(w) for w in words) / len(words) if words else 0.0


# ---- Grammatical Range and Accuracy --------------------------------------------------------


def sentence_lengths(essay: str) -> list[int]:
    return [len(tokens(s)) for s in sentences(essay)]


def clauses_per_sentence(essay: str) -> float:
    """Subordinators and relative pronouns per sentence: a proxy for complex structures."""
    sents = sentences(essay)
    if not sents:
        return 0.0
    return sum(1 for t in tokens(essay) if t in CLAUSE_MARKERS) / len(sents)


def grammar_errors(essay: str) -> dict[str, int]:
    """Counts per error pattern, plus sentences that start with a lower-case letter."""
    flat = normalise_ws(essay)
    counts = {name: len(p.findall(flat)) for name, p in ERROR_PATTERNS.items()}
    counts["lowercase_start"] = sum(1 for s in sentences(essay) if s[:1].islower())
    return counts


def article_omissions(words: list[str]) -> int:
    """Singular countable nouns with no determiner in the three words before them.

    'the government should' is fine, 'government should' is counted. Three words of look-back
    allow for adjectives ('a very serious problem'). This is a heuristic with false alarms on
    real text (for example noun compounds), which is part of why it is risky to score with.
    """
    count = 0
    for i, w in enumerate(words):
        if w in COUNTABLE_NOUNS and not any(p in DETERMINERS for p in words[max(0, i - 3) : i]):
            count += 1
    return count


# ---- All features --------------------------------------------------------------------------


def extract_features(essay: str, prompt: str) -> dict[str, float]:
    """Every hand-made feature for one essay, as a flat dict of floats."""
    words = tokens(essay)
    n = max(1, len(words))
    lengths = sentence_lengths(essay)
    share, distinct = connective_stats(essay)
    errors = sum(grammar_errors(essay).values())
    articles = article_omissions(words)
    return {
        "word_count": float(len(words)),
        "prompt_overlap": prompt_overlap(essay, prompt),
        "on_topic_share": on_topic_share(essay, prompt),
        "position_markers": float(position_markers(essay)),
        "has_conclusion": has_conclusion(essay),
        "paragraph_count": float(len(paragraphs(essay))),
        "connective_share": share,
        "distinct_connectives": float(distinct),
        "topic_sentence_share": topic_sentence_share(essay, prompt),
        "mattr": mattr(words),
        "rare_word_share": rare_word_share(words),
        "collocations_per_100w": collocations_per_100w(essay),
        "mean_word_length": mean_word_length(words),
        "mean_sentence_length": statistics.fmean(lengths) if lengths else 0.0,
        "sentence_length_sd": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
        "clauses_per_sentence": clauses_per_sentence(essay),
        "error_rate_no_articles": 100.0 * errors / n,
        "error_rate_all": 100.0 * (errors + articles) / n,
        "article_omissions_per_100w": 100.0 * articles / n,
    }


def content_vector(essay: str, n_buckets: int = N_CONTENT_BUCKETS) -> list[float]:
    """Hashed bag of content words (share of words per bucket).

    These features describe *what* the essay is about, so they let a model recognise the
    prompt. They are only used by the 'HGB + content' model, to show how a random split
    rewards that and a split by prompt does not.
    """
    vec = [0.0] * n_buckets
    words = [t for t in tokens(essay) if len(t) >= 3 and t not in STOPWORDS]
    for w in words:
        vec[zlib.crc32(stem(w).encode()) % n_buckets] += 1.0
    total = max(1, len(words))
    return [v / total for v in vec]
