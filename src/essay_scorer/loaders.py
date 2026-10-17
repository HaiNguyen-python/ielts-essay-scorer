"""Loaders for public essay-scoring datasets (not used by tests or CI beyond small samples).

Both return a frame with at least `essay_id`, `prompt_id`, `text`. Scores stay on each
dataset's own scale; neither corpus uses IELTS bands, so a model trained on them predicts
their scale, not IELTS bands.

- **ELLIPSE** (English Language Learner Insight, Proficiency and Skills Evaluation corpus;
  also the Kaggle "Feedback Prize - English Language Learning" data). Columns: text_id,
  full_text, cohesion, syntax, vocabulary, phraseology, grammar, conventions (1.0-5.0 in
  half points). The closest IELTS mapping: CC <- cohesion; LR <- mean(vocabulary,
  phraseology); GRA <- mean(syntax, grammar, conventions). There is no Task Response
  equivalent, and only one (adjudicated) score per trait, so no human-human ceiling.
- **ASAP-AES** (Hewlett Foundation, Kaggle 2012), training_set_rel3.tsv: essay_id,
  essay_set, essay, rater1_domain1, rater2_domain1, domain1_score. Eight prompts with
  different score ranges; two raters, so a human-human ceiling and a split by prompt
  (essay_set) both work. The file is Latin-1 encoded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ELLIPSE_TRAITS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")


def load_ellipse(path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    missing = {"text_id", "full_text", *ELLIPSE_TRAITS} - set(raw.columns)
    if missing:
        raise ValueError(f"not an ELLIPSE file, missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "essay_id": raw["text_id"].astype(str),
            "prompt_id": raw["prompt"].astype(str) if "prompt" in raw else "unknown",
            "text": raw["full_text"].astype(str),
        }
    )
    for trait in ELLIPSE_TRAITS:
        out[trait] = raw[trait].astype(float)
    out["coherence_cohesion"] = out["cohesion"]
    out["lexical_resource"] = out[["vocabulary", "phraseology"]].mean(axis=1)
    out["grammatical_range_accuracy"] = out[["syntax", "grammar", "conventions"]].mean(axis=1)
    return out


def load_asap(path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", encoding="latin-1")
    missing = {"essay_id", "essay_set", "essay", "rater1_domain1", "rater2_domain1"} - set(
        raw.columns
    )
    if missing:
        raise ValueError(f"not an ASAP-AES file, missing columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "essay_id": raw["essay_id"].astype(str),
            "prompt_id": "set" + raw["essay_set"].astype(str),
            "text": raw["essay"].astype(str),
            "rater1": raw["rater1_domain1"].astype(float),
            "rater2": raw["rater2_domain1"].astype(float),
        }
    )
