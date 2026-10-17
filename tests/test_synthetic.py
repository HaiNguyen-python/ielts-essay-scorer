import numpy as np
import pandas as pd

from essay_scorer import CRITERIA
from essay_scorer.synthetic import GeneratorConfig, generate_essays


def test_same_seed_same_essays_and_scores():
    a = generate_essays(GeneratorConfig(n_essays=40, seed=11))
    b = generate_essays(GeneratorConfig(n_essays=40, seed=11))
    pd.testing.assert_frame_equal(a, b)
    c = generate_essays(GeneratorConfig(n_essays=40, seed=12))
    assert not a["text"].equals(c["text"])


def test_rater_scores_are_half_bands_in_range(ds):
    for c in CRITERIA:
        for r in (1, 2):
            s = ds.df[f"r{r}_{c}"]
            assert s.between(4.0, 9.0).all() and ((s * 2) % 1 == 0).all()
    assert ds.df["prompt_id"].value_counts().nunique() == 1  # balanced prompts


def test_planted_rater_bias_and_quality_signals(ds):
    df, f = ds.df, ds.feats
    assert (df["r1_task_response"] - df["r2_task_response"]).mean() > 0.1
    gra = df["true_grammatical_range_accuracy"]
    assert np.corrcoef(gra, f["error_rate_no_articles"])[0, 1] < -0.4
    assert np.corrcoef(df["true_task_response"], f["word_count"])[0, 1] > 0.4


def test_planted_article_pattern_depends_on_l1_not_on_ability(ds):
    rate = ds.feats["article_omissions_per_100w"].groupby(ds.df["l1"]).mean()
    assert min(rate["Vietnamese"], rate["Finnish"]) > 3 * max(rate["Arabic"], rate["Spanish"])
    ability = ds.df["true_grammatical_range_accuracy"].groupby(ds.df["l1"]).mean()
    assert ability.max() - ability.min() < 0.3
