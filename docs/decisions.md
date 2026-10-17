# Design decisions

Short records of the main choices: what was decided, why, and what it costs. Format inspired by
Architecture Decision Records (ADRs).

## 1. A seeded essay generator as the ground truth

- **Decision:** synthetic essays from templates and sentence banks, with hidden quality per
  criterion, two simulated raters and a planted L1 pattern. Loaders for ELLIPSE and ASAP-AES
  exist but are only tested on tiny inline samples.
- **Why:** no essay dataset with IELTS bands per criterion is openly available, and CI has no
  internet. A generator also gives what no real corpus has: the true quality behind each band,
  and effects whose size is known in advance, so the code can be checked against them.
- **Cost:** the features were written knowing what the generator plants. Agreement on real
  essays will be lower. Every number is labelled synthetic.

## 2. Evaluate on unseen prompts

- **Decision:** 4-fold cross-validation grouped by prompt for every headline number; a random
  split only for comparison.
- **Why:** a school will set new questions. A random split lets a model learn things about the
  test prompt from other essays on it. Measured: the content-word model scores 0.850 overall
  QWK on random folds but 0.825 on unseen prompts (LR: 0.717 vs 0.646).
- **Cost:** 16 prompts give only 4 per fold, so prompt-to-prompt variation adds noise.

## 3. Compare with single raters, and show the human ceiling

- **Decision:** model vs rater 1 and model vs rater 2, averaged; ceiling = rater 1 vs rater 2.
- **Why:** comparing the model with the *mean* of two raters would flatter it (the mean has
  less noise than any single rater), so it would not be comparable with human-human agreement.
- **Cost:** the model is trained on the two-rater mean but judged against single raters, so
  some disagreement is rater noise that no model can remove. That is the point of the ceiling.

## 4. One model per criterion, on that criterion's features only

- **Decision:** the GRA model sees only grammar features, the TR model only task features, and
  so on.
- **Why:** feedback has to make sense to a teacher. With all features, the GRA band could go up
  because the essay has four paragraphs (criteria are correlated through ability). That might
  be accurate, but it is not a reason a teacher could sign.
- **Cost:** some accuracy. Correlated criteria would help each other's predictions.

## 5. Serve ridge regression

- **Decision:** the API serves ridge on article-neutral features. Gradient boosting and the
  content-word model are evaluated but not served.
- **Why:** on unseen prompts gradient boosting was not clearly better (overall QWK 0.806 vs
  0.797; mean criterion QWK 0.676 vs 0.677). Ridge contributions are exact and easy to explain.
  The content-word model is better (0.825), but its extra signal is the generator's own word
  banks ("beneficial", "detrimental") and the prompts it has seen, which says little about
  real essays, and "content words" is not feedback a student can act on.
- **Cost:** LR is weak (QWK 0.527 vs a ceiling of 0.827). The hand-made LR features miss much
  of what the content-word model picks up. Better LR features are the main next step.

## 6. The IELTS rounding rule, on half bands

- **Decision:** `round_half_band` with .25 rounding up to .5 and .75 up to the next whole band,
  for both criterion predictions and the overall band.
- **Why:** this is the rule IELTS publishes for averaged band scores, and students compare
  their estimate with real results.
- **Cost:** official criterion bands are whole numbers; this simulation uses half bands for
  criteria too, as specified for the project. Switching to whole bands means changing the
  rater rounding and `to_band`.

## 7. Fix the L1 bias in the features, not with group labels

- **Decision:** the served model's grammar error rate leaves out article omissions. Group
  recalibration is reported only for comparison.
- **Why:** the naive model under-scores Vietnamese and Finnish writers by 0.29 GRA bands at the
  same human band. Dropping articles from the error rate cuts that to 0.10 *and* improves GRA
  agreement (QWK 0.688 -> 0.727), because article slips were mostly L1 noise. Recalibration
  closes the gap completely (+0.02), but it needs every student's L1 at scoring time and gives
  different scores to identical essays from different groups. Most schools would not want to
  collect that, and some may not be allowed to.
- **Cost:** a 0.10-band gap remains. Dropped articles also shorten sentences and change other
  counts, so the planted pattern leaks into other features. Articles are still mentioned in
  the feedback as something to practise.

## 8. The LLM writes words, never scores

- **Decision:** bands come only from the scoring model; the LLM receives them as fixed and its
  reply is read only for comments and quotes.
- **Why:** this makes prompt injection harmless to the score by construction, whatever the
  essay says. It also keeps the score reproducible and testable.
- **Cost:** an LLM may notice things the features miss (a weak argument) but cannot change the
  band. A teacher can.

## 9. Grounded quotes, retry, then a template fallback

- **Decision:** every comment needs 1-3 verbatim quotes; failing replies are retried with the
  problems named; after 3 attempts the deterministic template provider takes over.
- **Why:** a quote is the cheapest check that feedback is about *this* essay. In the stress
  test (invented quotes at a 15% rate per criterion), 203 invented quotes were rejected and
  every final reply was grounded.
- **Cost:** a real quote does not prove the comment about it is right. Fallback feedback is
  plainer than LLM feedback.

## 10. Remove injection sentences before scoring

- **Decision:** sentences matching injection patterns are removed before features are computed
  and listed in `flagged_sentences`.
- **Why:** an appended "give band 9" sentence would otherwise add words and maybe a connective,
  nudging TR or CC. Removing it makes the score identical (0 of 200 changed in the report).
- **Cost:** a false alarm removes a real sentence. That is visible to the teacher in the
  response, and the patterns are narrow.

## 11. Occlusion contributions instead of SHAP

- **Decision:** contribution = prediction - prediction with the feature at its training median.
- **Why:** model-agnostic, no extra dependency, exact for ridge, and easy to explain.
- **Cost:** ignores interactions, so for gradient boosting the parts do not add up.

## 12. Single-threaded fitting

- **Decision:** model fitting and prediction run inside `threadpool_limits(1)`.
- **Why:** on a few thousand rows OpenMP start-up costs more than it saves; on the shared
  build machine one gradient-boosting fit took 5 seconds with all threads and 0.4 with one.
- **Cost:** slower on very large datasets with many cores. Remove the limit there.
