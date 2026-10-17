"""Cross-validation folds: by prompt (the honest one) or random (the flattering one).

A school will use the scorer on new essay questions, so the honest test holds out whole
prompts: no essay on a test prompt is seen in training. A random split lets the model see other
essays on the very same prompt. Both are provided so the report can show the difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


def fold_indices(
    df: pd.DataFrame, by_prompt: bool, n_folds: int = 4, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train rows, held-out rows) per fold.

    by_prompt=True: GroupKFold on prompt_id, so each fold holds out a quarter of the prompts.
    by_prompt=False: shuffled KFold on essays.
    """
    if by_prompt:
        folds = GroupKFold(n_splits=n_folds).split(df, groups=df["prompt_id"])
    else:
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(df)
    return [(np.asarray(a), np.asarray(b)) for a, b in folds]
