"""Agreement metrics used in automated essay scoring.

Quadratic weighted kappa (QWK) is the standard: it compares agreement with what chance would
give, and a miss by two half bands costs four times a miss by one. Exact and adjacent
(within half a band) agreement and MAE are easier to explain to teachers, so they sit next to it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .bands import band_labels


def quadratic_weighted_kappa(
    a: ArrayLike,
    b: ArrayLike,
    labels: Sequence[float] | None = None,
) -> float:
    """QWK = 1 - sum(W * O) / sum(W * E).

    O is the observed confusion matrix, E the matrix expected if the two raters were
    independent (outer product of their marginals, scaled to the same total), and
    W[i, j] = (i - j)^2 / (k - 1)^2 the disagreement weight.
    """
    cats = list(labels) if labels is not None else band_labels()
    index = {round(c * 2): i for i, c in enumerate(cats)}
    k = len(cats)
    x = np.array([index[round(float(v) * 2)] for v in np.asarray(a, dtype=float)])
    y = np.array([index[round(float(v) * 2)] for v in np.asarray(b, dtype=float)])
    observed = np.zeros((k, k))
    np.add.at(observed, (x, y), 1.0)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / max(1.0, observed.sum())
    i, j = np.indices((k, k))
    weights = (i - j) ** 2 / (k - 1) ** 2
    denom = float((weights * expected).sum())
    if denom == 0.0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return 1.0 - float((weights * observed).sum()) / denom


def agreement(a: ArrayLike, b: ArrayLike) -> dict[str, float]:
    """QWK, exact agreement, adjacent agreement (|diff| <= 0.5) and MAE in bands."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = np.abs(x - y)
    return {
        "qwk": quadratic_weighted_kappa(x, y),
        "exact": float(np.mean(diff < 1e-9)),
        "adjacent": float(np.mean(diff <= 0.5 + 1e-9)),
        "mae": float(np.mean(diff)),
    }


def agreement_vs_raters(
    pred: ArrayLike,
    rater1: ArrayLike,
    rater2: ArrayLike,
) -> dict[str, float]:
    """Model vs each human rater, averaged.

    This is the like-for-like comparison with the human-human ceiling (rater 1 vs rater 2):
    in both cases one score is compared with one other single rater's score.
    """
    m1, m2 = agreement(pred, rater1), agreement(pred, rater2)
    return {k: (m1[k] + m2[k]) / 2.0 for k in m1}
