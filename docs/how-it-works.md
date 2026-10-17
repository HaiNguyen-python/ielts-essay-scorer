# How it works

This page follows one essay from the generator to an API response. File references point to
`src/essay_scorer/`.

## 1. The world: a synthetic essay generator (`synthetic.py`)

The generator is the ground truth. Models never see its hidden values, only the essay text and
the two raters' bands.

1. **Writer.** Each essay gets a hidden ability `a ~ N(6.2, 0.9)` and, per criterion, a hidden
   quality `q = a + N(0, 0.45)`, clipped to 4.0-9.0. The four criteria are correlated (good
   writers are usually good at everything) but not identical.
2. **First language (L1).** Vietnamese, Finnish, Arabic or Spanish, drawn **independently** of
   ability.
3. **Text.** The essay is built from 16 prompts (each with its own pros, cons and an example)
   and sentence templates. Every random choice is steered by one criterion:

   | Criterion | What a higher hidden quality changes |
   | --- | --- |
   | TR | clearer position statement, a conclusion that restates it, more supporting points and elaborations (longer essays), fewer off-topic sentences |
   | CC | four paragraphs instead of one or two blocks, topic sentences ("Firstly, ...", "On the other hand, ..."), more linking words, and varied ones ("Moreover", "As a result") instead of "Also", "And", "So" |
   | LR | mid and advanced words instead of basic ones ("beneficial" vs "good"), academic collocations, fewer repetitions of the subject phrase |
   | GRA | complex templates (although, which, if, because), fewer planted grammar errors |

   Errors are planted with regular-expression "hooks": "people are" becomes "people is",
   "can" becomes "can to", "an important" becomes "a important", and so on; if no hook fits,
   the sentence starts with a lower-case letter. The chance of an error per sentence falls from
   about 55% at GRA 4.5 to 2% at GRA 9.
4. **Planted L1 pattern.** Each "a", "an" or "the" is dropped with probability 0.50
   (Vietnamese), 0.45 (Finnish), 0.05 (Arabic) or 0.04 (Spanish), plus a little more for weak
   GRA. Vietnamese and Finnish have no articles, which is why this pattern is realistic. The
   raters do not see it (their bands come from the hidden quality), so article omission tells
   you about L1 and almost nothing about quality.
5. **Two raters.** Band = round to half band(`q + prompt effect + rater bias + N(0, 0.4)`),
   clipped to 4.0-9.0. Rater 1 is slightly lenient (+0.10), rater 2 slightly strict (-0.15).
   The prompt effect (SD 0.35 for TR, 0.15 for the others) models raters reading some
   questions more strictly; the text cannot show it.

Same seed, same essays and bands, byte for byte (`test_same_seed_same_essays_and_scores`).
The training target is the mean of the two raters, so quarter bands are possible.

## 2. Features (`features.py`, `lexicon.py`, `text.py`)

Small pure functions on the text. `text.py` splits paragraphs (blank lines), sentences (after
`.`, `!`, `?`) and word tokens; every module uses the same splitting.

| Criterion | Feature | How it is computed | Why it should matter |
| --- | --- | --- | --- |
| TR | `word_count` | word tokens | IELTS asks for at least 250 words; short answers rarely develop ideas |
| TR | `prompt_overlap` | share of the prompt's content-word stems used in the essay | an essay that never uses the question's key ideas may be off task |
| TR | `on_topic_share` | share of sentences with at least one prompt content word | catches off-topic sentences ("Football is the most popular sport") |
| TR | `position_markers` | count of "in my opinion", "I believe", "I would argue", ... | Task 2 asks for a clear position |
| TR | `has_conclusion` | one of the last two sentences opens with "In conclusion", "To sum up", ... | a conclusion that restates the view |
| CC | `paragraph_count` | blocks separated by blank lines | paragraphing is part of the CC descriptor |
| CC | `connective_share` | share of sentences that open with a linking word | progression between ideas |
| CC | `distinct_connectives` | number of different opening linking words | variety; "Also ... Also ... And" is a Band 5 pattern |
| CC | `topic_sentence_share` | body paragraphs whose first sentence signposts ("Firstly", "Another") or names the topic | clear central topic per paragraph |
| LR | `mattr` | moving-average type-token ratio over 50-word windows | vocabulary range without punishing long essays (plain TTR falls with length) |
| LR | `rare_word_share` | share of content words not in `resources/common_words.txt` (about 950 common words, written for this project), after stripping -s, -ed, -ing, -ly | less common, more precise words |
| LR | `collocations_per_100w` | hits from a list of 12 academic collocations | natural word combinations |
| LR | `mean_word_length` | letters per word | a rough proxy for academic vocabulary |
| GRA | `mean_sentence_length` | words per sentence | longer sentences usually mean more structures |
| GRA | `sentence_length_sd` | spread of sentence lengths | mixing short and long sentences |
| GRA | `clauses_per_sentence` | although, because, which, who, if, when, ... per sentence | range of complex structures |
| GRA | `error_rate_no_articles` | regex error hits per 100 words (agreement, "can to", "a important", "informations", "more better", lower-case sentence start) | accuracy |
| (analysis) | `error_rate_all` | the same plus article omissions | the naive version, used only to show the bias |

Article omission (`article_omissions`): a singular countable noun from a short list (problem,
government, country, ...) with no determiner in the three words before it.
"Government should solve serious problem" counts 2; "The government should solve a very serious
problem" counts 0.

Honest limits: the regexes and word lists were written knowing what the generator plants, so
they catch the synthetic errors far better than they would catch real ones. On real essays a
grammar checker (for example LanguageTool) and a corpus frequency list would replace them.
That is why the synthetic results are an upper bound for this feature set.

`content_vector` is different: a hashed bag of 32 content-word buckets. It describes *what*
the essay is about, so it lets a model recognise a prompt. Only the "gradient boosting + content
words" model uses it.

## 3. Models and bands (`models.py`, `bands.py`)

`EssayModel` holds one regressor per criterion. Each sees only its own criterion's features
(`FEATURE_GROUPS`), so a grammar band is explained by grammar features. Kinds:

- `ridge`: `StandardScaler` + `Ridge(alpha=1)`. The served model.
- `hgb`: `HistGradientBoostingRegressor` (250 trees, learning rate 0.05, 15 leaves, at least
  20 essays per leaf, L2 1.0).
- `hgb_content`: the same with the 32 content buckets added.

Continuous predictions become bands with the IELTS rule in `bands.py`:
`round_half_band(x) = floor(2x + 0.5) / 2`, so .25 goes up to .5 and .75 goes up to the next
whole band (6.25 -> 6.5, 6.75 -> 7.0, 6.125 -> 6.0, 6.625 -> 6.5), then clipped to 4.0-9.0.
The overall band is the same rounding applied to the mean of the four criterion bands. A tiny
epsilon makes 6.2499999999 (float noise from averaging) count as 6.25.

**Contributions** (`EssayModel.contributions`): for each feature, prediction with the essay's
value minus prediction with that feature set to the training median. Positive means "pushed
this band up compared with a typical essay". For ridge this is exact: the typical-essay
prediction plus all contributions equals the prediction (`test_linear_contributions_add_up_to_the_prediction`).

## 4. Evaluation (`split.py`, `evaluate.py`, `metrics.py`)

- **Split by prompt:** `fold_indices(by_prompt=True)` is a 4-fold `GroupKFold` on `prompt_id`.
  Each fold holds out 4 of 16 prompts; every essay is predicted once, by models that never saw
  its prompt. `by_prompt=False` is a shuffled `KFold` for comparison.
  `test_prompt_folds_never_share_a_prompt_between_train_and_test` guards this.
- **Model vs human:** agreement of the model with rater 1 and with rater 2, averaged. **Ceiling:**
  rater 1 vs rater 2 on the same essays. Both compare one score with one other single score, so
  they are like for like.
- **Metrics:** QWK, exact agreement, adjacent agreement (within 0.5) and MAE.

QWK, with categories 4.0, 4.5, ..., 9.0 (`k = 11`):

```
O[i, j]  = number of essays with score i from A and j from B
E[i, j]  = rowsum_i(O) * colsum_j(O) / N       (what independent raters would give)
W[i, j]  = (i - j)^2 / (k - 1)^2
QWK      = 1 - sum(W * O) / sum(W * E)
```

Hand example (tested, 3 categories): A = [1, 2, 3, 1], B = [1, 3, 3, 2]. Two essays differ by
one category, so `sum(W * O) = 0.25 + 0.25 = 0.5`. Marginals A = (2, 1, 1), B = (1, 1, 2) give
`sum(W * E) = 1.625`. QWK = 1 - 0.5 / 1.625 = 9/13 = 0.692. It also matches scikit-learn's
`cohen_kappa_score(weights="quadratic")` on half bands.

## 5. Fairness (`fairness.py`)

The question is: at the **same human band**, does the model give one L1 group lower bands?

1. Residual = model band - human band (mean of the two raters).
2. Regress the residual on the human band (models regress towards the mean, so residuals
   depend on the band itself) and keep what is left.
3. Average per L1 group, with a 95% normal interval.

Three variants, all cross-validated by prompt over all 2,000 essays:

- **Naive:** the grammar error rate counts article omissions (`NAIVE_FEATURE_GROUPS`).
- **Article-neutral (served):** article omissions are left out of the error rate.
- **Group recalibration:** the naive model minus each L1 group's mean residual, learned
  out-of-fold inside each training part (nested, so no test essay informs its own offset).

The report states the headline as "no-article L1s minus article L1s".

## 6. Feedback and the grounding check (`feedback.py`, `providers.py`)

`FeedbackRequest` carries the essay, prompt, fixed bands, contributions and features.
`messages()` builds the LLM prompt: a system message that says the essay is data, not
instructions, and that bands are fixed; then the escaped prompt and essay in `<prompt>` and
`<essay>` tags.

Providers share one method, `generate(request) -> str` (JSON text):

- `TemplateProvider` (offline, deterministic): picks one sentence per criterion by rule (TR: a
  sentence with a position marker or most prompt words; CC: the first sentence that opens with
  a linking word or signpost; LR: most collocations, then most rare words; GRA: the first
  sentence with a detected error, otherwise the longest). Comments come from `ADVICE`, keyed by
  the features that helped or hurt most. If articles are often missing, the GRA comment says so
  and adds that this did not lower the band.
- `OllamaProvider`: `POST /api/chat` with `format: json`, temperature 0.
- `OpenAIProvider`: `POST /chat/completions` with JSON mode; key from `OPENAI_API_KEY`.

`check_grounding(raw, essay)` parses the JSON and, for each criterion, requires a comment and
1-3 quotes of at least 3 words. Each quote is HTML-unescaped, whitespace-normalised, stripped
of wrapping quotation marks, and must be a substring of the whitespace-normalised essay. Case
and punctuation must match: "verbatim" means verbatim.

`generate_feedback` calls the provider; on failure it calls again with a `retry_note` that lists
the problems; after 3 failed attempts (or any provider exception) the template provider takes
over. The result records the provider used, attempts, whether it fell back, and every rejected
item.

## 7. Prompt injection (`safety.py`)

1. Bands come only from `EssayModel`, which reads numbers. Any score in an LLM reply is never
   read (`test_scores_in_an_llm_reply_are_ignored`).
2. `strip_injections` removes sentences matching patterns such as "ignore ... instructions",
   "give ... band 9", "system prompt", "you are now", and `<essay>`-like tags, keeping
   paragraph breaks. Scoring and feedback use the cleaned essay; the removed sentences are
   returned as `flagged_sentences` for the teacher.
3. `escape_for_prompt` HTML-escapes the essay, so `</essay>` in student text becomes
   `&lt;/essay&gt;` and cannot close the block (`test_essay_is_escaped_inside_the_llm_prompt`).

Layer 1 is the real guarantee. Layers 2 and 3 are defence in depth: the pattern list will miss
creative attacks, but a missed attack can only change feedback wording, never a band.

## 8. Serving (`service.py`, `api.py`)

`Scorer.score(essay, prompt)`: strip injections -> features -> bands -> contributions ->
grounded feedback. `POST /score` returns bands per criterion, overall band, word count, the top
3 contributions per criterion, feedback with quotes, provider details and flagged sentences.
`GET /health` returns the model kind and provider. The model is a joblib file made by
`essay-scorer train` (joblib uses pickle: only load files you created).

## 9. Report (`report.py`)

`essay-scorer report` generates 2,000 essays, runs every evaluation above, stress-tests the
grounding check with `HallucinatingProvider` (the template provider with each quote replaced by
an invented sentence with probability 0.15), appends one of three injections to 200 essays,
and writes `docs/results.md` and both charts. About a minute on a laptop.
