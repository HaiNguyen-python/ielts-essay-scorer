"""One regressor per criterion, the overall band, and per-feature contributions.

Model kinds:
- ``ridge``: standardised features + ridge regression. The baseline, and easy to read.
- ``hgb``: sklearn HistGradientBoostingRegressor on the same features. Catches non-linear
  effects such as "more than about 3 errors per 100 words hurts a lot".
- ``hgb_content``: HGB plus a hashed bag of content words. Included to show prompt leakage:
  it can recognise prompts it has seen, which helps on a random split and not on new prompts.

Each criterion's model only sees that criterion's features (features.FEATURE_GROUPS). That
costs a little accuracy, because the four criteria are correlated through the writer's ability,
but it keeps the explanation honest: a Grammar band is explained by grammar features only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import CRITERIA
from .bands import overall_band, to_band
from .features import FEATURE_GROUPS, N_CONTENT_BUCKETS, content_vector, extract_features

MODEL_KINDS: tuple[str, ...] = ("ridge", "hgb", "hgb_content")
MODEL_LABELS: dict[str, str] = {
    "ridge": "Ridge",
    "hgb": "Gradient boosting",
    "hgb_content": "Gradient boosting + content words",
}
CONTENT = "content_words"


def feature_frame(texts: list[str], prompts: list[str]) -> pd.DataFrame:
    """Hand-made features for many essays (one row per essay)."""
    return pd.DataFrame([extract_features(t, p) for t, p in zip(texts, prompts, strict=True)])


def content_matrix(texts: list[str]) -> np.ndarray:
    return np.array([content_vector(t) for t in texts]).reshape(len(texts), N_CONTENT_BUCKETS)


def make_estimator(kind: str, seed: int = 0) -> Any:
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind in ("hgb", "hgb_content"):
        return HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


@dataclass
class EssayModel:
    """Four fitted regressors and everything needed to explain a prediction."""

    kind: str = "hgb"
    feature_groups: dict[str, tuple[str, ...]] = field(default_factory=lambda: FEATURE_GROUPS)
    seed: int = 0
    estimators: dict[str, Any] = field(default_factory=dict)
    medians: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def uses_content(self) -> bool:
        return self.kind == "hgb_content"

    def _matrix(
        self, criterion: str, feats: pd.DataFrame, content: np.ndarray | None
    ) -> np.ndarray:
        x = feats[list(self.feature_groups[criterion])].to_numpy(dtype=float)
        if self.uses_content:
            if content is None:
                raise ValueError("this model needs the content matrix")
            x = np.hstack([x, content])
        return x

    def fit(
        self,
        feats: pd.DataFrame,
        targets: dict[str, np.ndarray],
        content: np.ndarray | None = None,
    ) -> EssayModel:
        # One thread: on a few thousand rows OpenMP start-up costs more than it saves, and on a
        # shared CPU many threads slowed a single fit down more than tenfold.
        with threadpool_limits(limits=1):
            for c in CRITERIA:
                x = self._matrix(c, feats, content)
                self.estimators[c] = make_estimator(self.kind, self.seed).fit(x, targets[c])
                self.medians[c] = np.median(x, axis=0)
        return self

    def predict_raw(
        self, feats: pd.DataFrame, content: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Continuous predictions per criterion (before rounding to half bands)."""
        with threadpool_limits(limits=1):
            return {
                c: np.asarray(
                    self.estimators[c].predict(self._matrix(c, feats, content)), dtype=float
                )
                for c in CRITERIA
            }

    def predict_bands(self, feats: pd.DataFrame, content: np.ndarray | None = None) -> pd.DataFrame:
        return bands_from_raw(self.predict_raw(feats, content))

    def contributions(
        self, feats: pd.DataFrame, content: np.ndarray | None = None, top_k: int = 3
    ) -> dict[str, list[dict[str, float | str]]]:
        """Top features per criterion for ONE essay, by occlusion.

        For each feature: prediction with the essay's value minus prediction with the value
        replaced by the training median. Positive = this feature pushed the band up compared
        with a typical essay. Simple and model-agnostic; it ignores interactions (SHAP would
        not), which is acceptable for short feedback.
        """
        if len(feats) != 1:
            raise ValueError("contributions are computed for one essay at a time")
        out: dict[str, list[dict[str, float | str]]] = {}
        for c in CRITERIA:
            x = self._matrix(c, feats, content)
            names = list(self.feature_groups[c])
            blocks: list[tuple[str, list[int]]] = [(n, [i]) for i, n in enumerate(names)]
            if self.uses_content:
                blocks.append((CONTENT, list(range(len(names), x.shape[1]))))
            base = float(self.estimators[c].predict(x)[0])
            rows: list[dict[str, float | str]] = []
            for name, cols in blocks:
                x_typ = x.copy()
                x_typ[0, cols] = self.medians[c][cols]
                delta = base - float(self.estimators[c].predict(x_typ)[0])
                value = float(x[0, cols[0]]) if name != CONTENT else float("nan")
                rows.append({"feature": name, "value": value, "contribution": delta})
            rows.sort(key=lambda r: -abs(float(r["contribution"])))
            out[c] = rows[:top_k]
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> EssayModel:
        """Load a model saved by `save`. joblib uses pickle: only load files you created."""
        model = joblib.load(path)
        if not isinstance(model, EssayModel):
            raise TypeError(f"{path} does not contain an EssayModel")
        return model


def bands_from_raw(raw: dict[str, np.ndarray]) -> pd.DataFrame:
    """Round each criterion to a half band (4.0-9.0), then add the IELTS overall band."""
    out = pd.DataFrame({c: [to_band(float(v)) for v in raw[c]] for c in CRITERIA})
    out["overall"] = [overall_band([float(v) for v in row]) for row in out[list(CRITERIA)].values]
    return out
