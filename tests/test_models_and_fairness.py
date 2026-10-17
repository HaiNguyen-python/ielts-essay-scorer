import numpy as np
import pandas as pd
import pytest

from essay_scorer import CRITERIA
from essay_scorer.bands import overall_band
from essay_scorer.evaluate import agreement_table, ceiling_table, cross_validated_bands, fit_model
from essay_scorer.fairness import (
    adjusted_gaps,
    apply_group_offsets,
    compare_fixes,
    fit_group_offsets,
)
from essay_scorer.features import FEATURE_GROUPS, NAIVE_FEATURE_GROUPS
from essay_scorer.models import EssayModel
from essay_scorer.report import no_article_gap
from essay_scorer.split import fold_indices


def test_prompt_folds_never_share_a_prompt_between_train_and_test(ds):
    held_out = []
    for train, test in fold_indices(ds.df, by_prompt=True):
        train_prompts = set(ds.df["prompt_id"].iloc[train])
        test_prompts = set(ds.df["prompt_id"].iloc[test])
        assert not train_prompts & test_prompts
        held_out.extend(test.tolist())
    assert sorted(held_out) == list(range(len(ds.df)))  # every essay held out exactly once
    train, test = fold_indices(ds.df, by_prompt=False)[0]
    assert set(ds.df["prompt_id"].iloc[train]) & set(ds.df["prompt_id"].iloc[test])


def test_unseen_prompt_agreement_is_useful_but_below_the_human_ceiling(ds):
    table = agreement_table(cross_validated_bands(ds, "ridge", by_prompt=True), ds.df)
    ceiling = ceiling_table(ds.df)
    assert table.loc["overall", "qwk"] > 0.6
    assert table.loc["overall", "qwk"] < ceiling.loc["overall", "qwk"]


def test_fitting_is_deterministic_and_survives_save_and_load(ds, tmp_path):
    a = fit_model(ds, "hgb").predict_bands(ds.feats, ds.content)
    model = fit_model(ds, "hgb")
    pd.testing.assert_frame_equal(a, model.predict_bands(ds.feats, ds.content))
    model.save(tmp_path / "m.joblib")
    loaded = EssayModel.load(tmp_path / "m.joblib")
    pd.testing.assert_frame_equal(a, loaded.predict_bands(ds.feats, ds.content))
    for c in CRITERIA:
        assert ((a[c] * 2) % 1 == 0).all()
    assert all(
        row["overall"] == overall_band([row[c] for c in CRITERIA]) for _, row in a.iterrows()
    )


def test_linear_contributions_add_up_to_the_prediction(ds):
    """For ridge, occlusion contributions are exact: typical-essay prediction + sum = prediction."""
    model = fit_model(ds, "ridge")
    one = ds.feats.iloc[[5]]
    raw = model.predict_raw(one)
    contribs = model.contributions(one, top_k=10)
    typical = ds.feats.iloc[[0]].copy()
    for c in CRITERIA:
        feats = list(FEATURE_GROUPS[c])
        typical[feats] = model.medians[c]
        base = float(model.predict_raw(typical)[c][0])
        total = sum(float(r["contribution"]) for r in contribs[c])
        assert base + total == pytest.approx(float(raw[c][0]))


def test_content_model_requires_the_content_matrix(ds):
    model = fit_model(ds, "hgb_content")
    with pytest.raises(ValueError):
        model.predict_raw(ds.feats)
    with pytest.raises(ValueError):
        EssayModel(kind="svm").fit(ds.feats, {c: np.ones(len(ds.feats)) for c in CRITERIA})


def test_adjusted_gap_on_a_hand_example():
    human = np.array([5.0, 6.0, 7.0, 5.0, 6.0, 7.0])
    pred = human + np.array([0.2, 0.2, 0.2, -0.2, -0.2, -0.2])
    gaps = adjusted_gaps(pred, human, np.array(list("AAABBB")))
    assert gaps.loc["A", "gap"] == pytest.approx(0.2) and gaps.loc["B", "gap"] == pytest.approx(
        -0.2
    )


def test_group_offsets_remove_each_group_mean_residual():
    raw = {c: np.array([6.3, 6.3, 5.7, 5.7]) for c in CRITERIA}
    human = pd.DataFrame({c: [6.0, 6.0, 6.0, 6.0] for c in CRITERIA})
    groups = np.array(["X", "X", "Y", "Y"])
    fixed = apply_group_offsets(raw, groups, fit_group_offsets(raw, human, groups))
    assert np.allclose(fixed[CRITERIA[0]], 6.0)


def test_planted_l1_bias_is_found_and_the_article_neutral_fix_shrinks_it(ds):
    result = compare_fixes(ds, "ridge", NAIVE_FEATURE_GROUPS, FEATURE_GROUPS)
    target = "grammatical_range_accuracy"
    naive = no_article_gap(result.gaps["naive"][target])
    fixed = no_article_gap(result.gaps["article_neutral"][target])
    recal = no_article_gap(result.gaps["group_recalibrated"][target])
    assert naive < -0.1  # no-article L1s are under-scored at the same human band
    assert abs(fixed) < abs(naive) / 2
    assert abs(recal) < abs(naive) / 2
