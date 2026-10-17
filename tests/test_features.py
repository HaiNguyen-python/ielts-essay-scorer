import pytest

from essay_scorer.features import (
    NAIVE_FEATURE_GROUPS,
    article_omissions,
    collocations_per_100w,
    connective_stats,
    extract_features,
    grammar_errors,
    has_conclusion,
    leading_connective,
    mattr,
    position_markers,
    prompt_overlap,
    rare_word_share,
    topic_sentence_share,
)
from essay_scorer.text import tokens


def test_mattr_counts_types_per_window():
    assert mattr(["a", "b", "a", "c"]) == 0.75
    assert mattr(["word"] * 120) == pytest.approx(1 / 50)
    assert mattr([]) == 0.0


def test_prompt_overlap_on_a_hand_example():
    # Prompt content stems: university, education, free, student -> essay hits 3 of 4.
    prompt = "Should university education be free for students?"
    assert prompt_overlap("Free education helps every student.", prompt) == 0.75


def test_connectives_share_variety_and_word_boundaries():
    share, distinct = connective_stats("However, it works. Also, it helps. Also, it lasts. Fine.")
    assert share == 0.75 and distinct == 2
    assert leading_connective("Andrew went home.") is None
    assert leading_connective("On the other hand, it fails.") == "on the other hand"


def test_structure_features_on_the_sample_essay(essay, prompt):
    assert position_markers(essay) == 2  # "In my opinion", "I believe"
    assert has_conclusion(essay) == 1.0
    assert topic_sentence_share(essay, prompt) == 1.0  # both body paragraphs signpost
    one_block = essay.replace("\n\n", " ")
    assert topic_sentence_share(one_block, prompt) == 0.0


def test_rare_words_and_collocations():
    assert rare_word_share(["beneficial", "good", "detrimental", "people"]) == 0.5
    text = "This can play a vital role and raise awareness."
    assert collocations_per_100w(text) == pytest.approx(100 * 2 / 9)


def test_each_error_pattern_fires_once_and_clean_text_scores_zero():
    bad = (
        "People is happy. they has a car. It have value. He do not care. This is more better. "
        "We can to go. It is a interesting idea. I need informations."
    )
    counts = grammar_errors(bad)
    assert all(n == 1 for n in counts.values()), counts
    clean = "An important idea is that a university can help people. They have a useful choice."
    assert sum(grammar_errors(clean).values()) == 0


def test_article_omission_heuristic():
    assert article_omissions(tokens("Government should solve serious problem.")) == 2
    assert article_omissions(tokens("The government should solve a very serious problem.")) == 0


def test_naive_error_rate_adds_article_omissions_only(prompt):
    text = "Government should solve serious problem. People is happy."
    f = extract_features(text, prompt)
    assert f["error_rate_all"] - f["error_rate_no_articles"] == pytest.approx(
        f["article_omissions_per_100w"]
    )
    gra = NAIVE_FEATURE_GROUPS["grammatical_range_accuracy"]
    assert "error_rate_all" in gra and "error_rate_no_articles" not in gra
