"""Word lists behind the hand-made features.

All lists are short and readable on purpose: a teacher can check every entry. They were written
for this project. On real essays they would miss many items; that limitation is stated in
docs/how-it-works.md.
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources

STOPWORDS: frozenset[str] = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being
    it its this that these those there their they them he she his her we our you your i my me
    do does did not no so than then too very can could should would will may might must have
    has had about into over such what which who whom whose when where why how all any some
    more most other also just only many much one each every""".split()
)

# Phrases that state the writer's position (Task Response).
POSITION_MARKERS: tuple[str, ...] = (
    "in my opinion",
    "in my view",
    "i think",
    "i believe",
    "i feel",
    "i would argue",
    "i am convinced",
    "i strongly agree",
    "i strongly disagree",
    "i agree",
    "i disagree",
    "this essay will argue",
)

CONCLUSION_MARKERS: tuple[str, ...] = ("in conclusion", "to conclude", "to sum up", "overall,")

# Sentence-initial linking words (Coherence and Cohesion).
CONNECTIVES: tuple[str, ...] = (
    "furthermore",
    "moreover",
    "in addition",
    "additionally",
    "as a result",
    "consequently",
    "therefore",
    "similarly",
    "likewise",
    "however",
    "nevertheless",
    "on the other hand",
    "on the one hand",
    "in contrast",
    "firstly",
    "secondly",
    "finally",
    "for example",
    "for instance",
    "in conclusion",
    "to sum up",
    "also",
    "and",
    "so",
    "but",
)

# Openers that announce what a body paragraph is about.
SIGNPOSTS: tuple[str, ...] = (
    "firstly",
    "first of all",
    "secondly",
    "on the one hand",
    "on the other hand",
    "another",
    "the main reason",
    "one reason",
)

# Subordinators and relative pronouns (Grammatical Range).
CLAUSE_MARKERS: frozenset[str] = frozenset(
    "although though because since whereas while which who whom whose if unless when".split()
)

# Common academic collocations (Lexical Resource).
COLLOCATIONS: tuple[str, ...] = (
    "play a vital role",
    "have a profound impact",
    "raise awareness",
    "pose a serious threat",
    "bridge the gap",
    "meet the needs",
    "take into account",
    "a wide range of",
    "make a significant contribution",
    "strike a balance",
    "long-term consequences",
    "address the problem",
)

# Error patterns for Grammatical Accuracy. Each one is a frequent learner error that a regular
# expression can find with few false alarms. A real system would use a grammar checker.
ERROR_PATTERNS: dict[str, re.Pattern[str]] = {
    "agreement_people_is": re.compile(r"\bpeople is\b", re.I),
    "agreement_they_has": re.compile(r"\bthey has\b", re.I),
    "agreement_it_have": re.compile(r"\b(?:it|he|she|this|that) have\b", re.I),
    "agreement_it_do_not": re.compile(r"\b(?:it|he|she|this) do not\b", re.I),
    "double_comparative": re.compile(r"\bmore (?:better|worse|easier|cheaper)\b", re.I),
    "modal_to": re.compile(r"\b(?:can|could|should|must|will|would|may|might) to\b", re.I),
    "a_before_vowel": re.compile(r"\ba (?!uni|use|usu|eu|one|once)[aeiou]\w*", re.I),
    "uncountable_plural": re.compile(r"\b(?:informations|advices|knowledges|equipments)\b", re.I),
}

# Singular countable nouns that need a determiner in front (article omission check).
COUNTABLE_NOUNS: frozenset[str] = frozenset(
    """problem issue government world country solution result way reason city person child
    job chance step role impact threat gap difference economy""".split()
)
DETERMINERS: frozenset[str] = frozenset(
    """a an the this that every each another any no one my our your their its his her
    which what whose either neither""".split()
)


@cache
def common_words() -> frozenset[str]:
    """The shipped list of very common English words (resources/common_words.txt)."""
    raw = resources.files("essay_scorer").joinpath("resources/common_words.txt").read_text()
    return frozenset(
        line.strip() for line in raw.splitlines() if line.strip() and not line.startswith("#")
    )


def is_common(word: str) -> bool:
    """True if the word, or a simple inflection of it, is in the common list.

    Stripping -s, -es, -ed, -ing and -ly is crude, but it keeps "students" and "really" common
    without shipping a lemmatiser.
    """
    common = common_words()
    if word in common:
        return True
    for suffix in ("s", "es", "ed", "ing", "ly"):
        if word.endswith(suffix) and word[: -len(suffix)] in common:
            return True
    return word.endswith("ied") and word[:-3] + "y" in common
