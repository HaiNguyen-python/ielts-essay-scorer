"""FastAPI service: POST /score and GET /health.

Stateless: the model is loaded once at start-up and each request carries the essay and prompt.
Bands come only from the scoring model; the feedback provider can change wording, never scores.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .service import Scorer


class ScoreRequest(BaseModel):
    essay: str = Field(min_length=20, max_length=20_000)
    prompt: str = Field(min_length=10, max_length=2_000)


class Contribution(BaseModel):
    feature: str
    value: float | None
    contribution: float  # bands, compared with a typical training essay


class CriterionFeedback(BaseModel):
    comment: str
    quotes: list[str]


class ScoreResponse(BaseModel):
    bands: dict[str, float]
    overall: float
    word_count: int
    contributions: dict[str, list[Contribution]]
    feedback: dict[str, CriterionFeedback]
    feedback_provider: str
    feedback_attempts: int
    feedback_fallback: bool
    flagged_sentences: list[str]
    model: str


def create_app(scorer: Scorer) -> FastAPI:
    app = FastAPI(title="IELTS Task 2 essay scorer", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model": scorer.model.kind,
            "feedback_provider": scorer.provider.name,
            "version": __version__,
        }

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        return ScoreResponse.model_validate(scorer.score(req.essay, req.prompt).to_dict())

    return app
