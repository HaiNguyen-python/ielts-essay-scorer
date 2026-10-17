"""IELTS band arithmetic.

IELTS publishes one rounding rule for averaged band scores: an average ending in .25 rounds up
to the next half band, and one ending in .75 rounds up to the next whole band. Averages below
those points round down. This repo applies that rule twice: to turn a model's continuous
prediction into a half band, and to turn the mean of the four criteria into the overall band.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

MIN_BAND = 4.0
MAX_BAND = 9.0
# Guards against float noise such as 6.2499999999 from averaging, which must count as 6.25.
_EPS = 1e-9


def round_half_band(x: float) -> float:
    """Round to the nearest half band, with .25 and .75 rounding up (the IELTS rule)."""
    return math.floor(2.0 * x + 0.5 + _EPS) / 2.0


def clip_band(x: float, low: float = MIN_BAND, high: float = MAX_BAND) -> float:
    """Keep a band inside the range the simulated raters use."""
    return min(high, max(low, x))


def to_band(x: float) -> float:
    """A continuous model output -> a reportable half band between 4.0 and 9.0."""
    return clip_band(round_half_band(x))


def overall_band(criteria: Sequence[float]) -> float:
    """Overall Task 2 band: the IELTS-rounded mean of the four criterion bands."""
    if len(criteria) != 4:
        raise ValueError(f"expected 4 criterion bands, got {len(criteria)}")
    return round_half_band(sum(criteria) / 4.0)


def band_labels(low: float = MIN_BAND, high: float = MAX_BAND) -> list[float]:
    """All half bands in the range, used as the categories for QWK."""
    n = int(round((high - low) * 2)) + 1
    return [low + i / 2 for i in range(n)]
