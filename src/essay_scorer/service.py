"""Score one essay end to end: injection check, features, bands, contributions, feedback.

The CLI `score` command and the API both call `Scorer.score`, so they cannot disagree.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from . import CRITERIA
from .feedback import FeedbackProvider, FeedbackRequest, TemplateProvider, generate_feedback
from .models import EssayModel, content_matrix, feature_frame
from .safety import strip_injections
from .text import word_count


@dataclass
class ScoreResult:
    bands: dict[str, float]
    overall: float
    word_count: int
    contributions: dict[str, list[dict[str, Any]]]
    feedback: dict[str, Any]
    feedback_provider: str
    feedback_attempts: int
    feedback_fallback: bool
    flagged_sentences: list[str]
    model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Scorer:
    def __init__(self, model: EssayModel, provider: FeedbackProvider | None = None):
        self.model = model
        self.provider: FeedbackProvider = provider or TemplateProvider()

    def score(self, essay: str, prompt: str) -> ScoreResult:
        clean, flagged = strip_injections(essay)
        feats = feature_frame([clean], [prompt])
        content = content_matrix([clean])
        bands_df = self.model.predict_bands(feats, content)
        bands = {c: float(bands_df[c].iloc[0]) for c in CRITERIA}
        overall = float(bands_df["overall"].iloc[0])
        contributions = {
            c: [
                {
                    "feature": str(r["feature"]),
                    "value": None if math.isnan(float(r["value"])) else round(float(r["value"]), 3),
                    "contribution": round(float(r["contribution"]), 3),
                }
                for r in rows
            ]
            for c, rows in self.model.contributions(feats, content).items()
        }
        request = FeedbackRequest(
            essay=clean,
            prompt=prompt,
            bands={**bands, "overall": overall},
            contributions=contributions,
            features={str(k): float(v) for k, v in feats.iloc[0].items()},
        )
        fb = generate_feedback(request, self.provider)
        return ScoreResult(
            bands=bands,
            overall=overall,
            word_count=word_count(clean),
            contributions=contributions,
            feedback=fb.criteria,
            feedback_provider=fb.provider,
            feedback_attempts=fb.attempts,
            feedback_fallback=fb.fallback,
            flagged_sentences=flagged,
            model=self.model.kind,
        )
