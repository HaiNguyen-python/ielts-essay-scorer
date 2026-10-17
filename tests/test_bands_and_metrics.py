import pytest
from sklearn.metrics import cohen_kappa_score

from essay_scorer.bands import band_labels, overall_band, round_half_band, to_band
from essay_scorer.metrics import agreement, agreement_vs_raters, quadratic_weighted_kappa


@pytest.mark.parametrize(
    ("mean", "expected"),
    [
        (6.0, 6.0),
        (6.125, 6.0),  # below .25 rounds down
        (6.25, 6.5),  # .25 rounds up to the half band
        (6.375, 6.5),
        (6.625, 6.5),  # below .75 stays on the half band
        (6.75, 7.0),  # .75 rounds up to the next whole band
        (6.875, 7.0),
        (6.2499999999, 6.5),  # float noise from averaging still counts as .25
    ],
)
def test_ielts_rounding_rule(mean, expected):
    assert round_half_band(mean) == expected


def test_overall_band_is_the_rounded_mean_of_four_criteria():
    assert overall_band([6.5, 6.0, 6.0, 6.5]) == 6.5  # mean 6.25
    assert overall_band([7.0, 7.0, 6.5, 7.0]) == 7.0  # mean 6.875
    assert overall_band([5.0, 5.5, 5.0, 5.0]) == 5.0  # mean 5.125
    with pytest.raises(ValueError):
        overall_band([6.0, 6.0, 6.0])


def test_continuous_predictions_are_clipped_to_the_band_range():
    assert to_band(3.1) == 4.0 and to_band(9.7) == 9.0 and to_band(6.74) == 6.5
    assert band_labels() == [4.0 + i / 2 for i in range(11)]


def test_qwk_on_a_hand_example():
    # Observed weighted disagreement 0.5, expected 1.625 (worked out in docs/how-it-works.md).
    qwk = quadratic_weighted_kappa([1, 2, 3, 1], [1, 3, 3, 2], labels=[1, 2, 3])
    assert qwk == pytest.approx(9 / 13)


def test_qwk_matches_sklearn_on_half_bands():
    a = [4.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 6.0, 5.0, 9.0]
    b = [4.5, 5.5, 6.5, 6.0, 7.0, 8.0, 7.5, 6.0, 4.5, 8.5]
    expected = cohen_kappa_score(
        [int(x * 2) for x in a], [int(x * 2) for x in b], weights="quadratic"
    )
    assert quadratic_weighted_kappa(a, b) == pytest.approx(expected)


def test_exact_adjacent_and_mae():
    m = agreement([6.0, 6.5, 7.0, 5.0], [6.0, 7.0, 6.0, 5.0])
    assert m["exact"] == 0.5 and m["adjacent"] == 0.75 and m["mae"] == pytest.approx(0.375)
    assert agreement([6.0, 7.0], [6.0, 7.0])["qwk"] == 1.0
    both = agreement_vs_raters([6.0, 7.0], [6.0, 7.0], [6.5, 7.0])
    assert both["exact"] == pytest.approx(0.75)
