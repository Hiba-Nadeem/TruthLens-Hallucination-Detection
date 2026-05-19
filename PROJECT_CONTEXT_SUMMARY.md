# TruthLens Project Context Summary

## Project Goal

TruthLens is an evidence-based hallucination detection system. It checks whether an LLM response is hallucinated by comparing the LLM's True/False answer against Wikipedia evidence and FEVER ground truth.

The project has two main parts:

1. **Rule-based TruthLens system**
   - LLM answers a FEVER claim as True/False.
   - Wikipedia evidence is retrieved using FAISS plus direct entity lookup.
   - NLI classifies evidence as entailment, contradiction, or neutral.
   - A deterministic rule combines the LLM label and NLI label into a final verdict.

2. **Trained ML meta-classifier**
   - Uses features extracted from the TruthLens pipeline.
   - Predicts whether the LLM hallucinated.
   - Trained on combined outputs from multiple LLMs.

## Dataset And Evidence

- Main experiment uses `data/fever_1000.csv`.
- Four LLMs were run on 1000 claims each.
- Wikipedia summaries are used as the evidence source.
- Full Wikipedia articles were not used because they would increase embedding cost and may introduce more noisy chunks.
- Low coverage is a known limitation because many FEVER facts are not present in short Wikipedia summaries.

## LLMs Compared

The saved scored runs are:

- `google/flan-t5-base`
- `google/flan-t5-small`
- `declare-lab/flan-alpaca-base`
- `Qwen/Qwen2.5-0.5B-Instruct`

## TruthLens Rule-Based Metrics

These metrics evaluate the rule-based TruthLens system only on binary verdict rows:

- `Hallucination`
- `Not Hallucination`

Rows such as `True | No Evidence`, `False | No Evidence`, and `Not Enough Evidence` are excluded from this selective evaluation.

| Model | Coverage | LLM Hallucination Rate | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `google/flan-t5-base` | 30.20% | 39.74% | 79.14% | 68.39% | 88.33% | 77.09% |
| `google/flan-t5-small` | 23.60% | 48.31% | 76.27% | 75.89% | 74.56% | 75.22% |
| `declare-lab/flan-alpaca-base` | 44.70% | 38.70% | 78.08% | 68.29% | 80.92% | 74.07% |
| `Qwen/Qwen2.5-0.5B-Instruct` | 28.70% | 56.45% | 75.61% | 92.59% | 61.73% | 74.07% |

Interpretation:

- `google/flan-t5-base` has the strongest named-model F1.
- `declare-lab/flan-alpaca-base` has the best coverage.
- `Qwen/Qwen2.5-0.5B-Instruct` has the highest precision but lower recall.
- The system is selective: it abstains when evidence is insufficient.

## Top-K Retrieval Decision

Different retrieval depths were tested for `google/flan-t5-base`.

| Top-K | Accuracy | Coverage | Outcome |
|---:|---:|---:|---|
| 3 | 79.28% | 30.40% | Slightly better than k=5 |
| 5 | 79.14% | 30.20% | Selected balanced baseline |
| 10 | Lower | Lower | Added noisy evidence chunks |

Final choice: **top_k = 5**.

Reason:

- `top_k=3` improved only marginally.
- `top_k=10` added noisy/irrelevant chunks.
- `top_k=5` is a balanced and defensible retrieval setting.

## Trained ML Meta-Classifier

The final ML script is:

```text
scripts/model.py
```

It trains on the combined scored outputs from all four LLMs.

Initial rows:

```text
1000 rows x 4 LLMs = 4000 rows
```

Usable rows after filtering:

```text
2919 rows
```

Rows are kept if:

- FEVER label is `SUPPORTS` or `REFUTES`
- LLM label is parseable as `TRUE` or `FALSE`

Rows by LLM:

| LLM | Usable Rows |
|---|---:|
| `declare-lab/flan-alpaca-base` | 998 |
| `google/flan-t5-base` | 691 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 647 |
| `google/flan-t5-small` | 583 |

The ML model includes both:

- binary verdict rows
- no-evidence rows

This is intentional because the ML model tries to predict hallucination even when the rule-based system abstains.

## ML Features

The trained classifier uses tabular features such as:

- `llm_label_enc`
- `nli_label_enc`
- `nli_confidence`
- `llm_nli_agree`
- `claim_length`
- `answer_length`
- `max_entailment`
- `max_contradiction`
- `max_neutral`
- `avg_entailment`
- `avg_contradiction`
- `avg_neutral`
- `entail_contra_diff`
- `n_chunks`
- `n_direct_chunks`
- `n_faiss_chunks`
- `avg_faiss_score`
- one-hot encoded LLM model identity

## ML Train/Test Split

The final script uses a grouped train/test split by claim ID.

Reason:

- The same FEVER claim appears across multiple LLM files.
- Grouping prevents the same claim from appearing in both train and test sets.
- This gives a more honest generalization test.

Final split:

| Split | Rows | Unique Claims |
|---|---:|---:|
| Train | 2332 | 800 |
| Test | 587 | 200 |

## ML Model Results

Saved in:

```text
data/model_hallucination_results.csv
```

| Model | Accuracy | Precision Hallucinated | Recall Hallucinated | F1 Hallucinated | F1 Not Hallucinated | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 79.39% | 76.54% | 76.83% | 76.69% | 81.53% | 79.11% |
| Logistic Regression | 79.05% | 73.45% | 82.24% | 77.60% | 80.32% | 78.96% |
| Gradient Boosting | 78.71% | 76.59% | 74.52% | 75.54% | 81.15% | 78.34% |
| Rule-based | 66.61% | 73.68% | 37.84% | 50.00% | 74.94% | 62.47% |

Interpretation:

- Random Forest has the best macro F1.
- Logistic Regression has the best hallucination F1 and recall.
- Rule-based system performs worse on full-coverage data because it cannot handle neutral/no-evidence rows well.

## Per-LLM ML Breakdown

The per-LLM table shows how the best trained model performs separately on each LLM's test rows.

It does **not** train separate models.

It uses one trained Random Forest model and evaluates it separately by LLM group.

Saved in:

```text
data/model_hallucination_per_llm_results.csv
```

## Architecture Selection

Final recommended model family: **classical ML on tabular features**.

Why not CNN?

- CNNs are designed for image/spatial data or local sequence patterns.
- This project uses structured numeric/categorical features from retrieval and NLI.

Why classical ML?

- Dataset is small-to-medium tabular data.
- Logistic Regression and Random Forest are easier to interpret.
- Random Forest gives feature importances.
- Logistic Regression gives coefficients.
- Classical ML is less likely to overfit than a neural network on this feature set.

## Neural Network Failure Analysis

NN script:

```text
scripts/model_nn.py
```

Saved outputs:

```text
data/nn_results.csv
data/nn_training_history.csv
data/nn_training_validation_curve.png
data/model_hallucination_nn.pt
```

NN result:

| Model | Test Accuracy | Precision | Recall | F1 Hallucinated | Generalization Gap |
|---|---:|---:|---:|---:|---:|
| Neural Network MLP | 77.51% | 72.13% | 79.92% | 75.82% | 8.12% |

Interpretation:

- The NN did not collapse, but it did not outperform simpler ML models.
- Training F1 kept improving while validation loss increased.
- This suggests overfitting on a small tabular dataset.
- Classical ML was selected because it performed better and is more interpretable.

## Failure Analysis Points

### Failure 1: More Retrieval Chunks Did Not Help

Expectation:

- Increasing `top_k` should retrieve more useful evidence.

Observed:

- `top_k=10` reduced performance.

Reason:

- More chunks introduced irrelevant or noisy evidence.
- NLI can misclassify irrelevant evidence as contradiction.

Final decision:

- Use `top_k=5`.

### Failure 2: Neural Network Did Not Generalize Better

Expectation:

- A deeper neural network might learn better nonlinear patterns.

Observed:

- NN test performance was below Random Forest and Logistic Regression.
- Validation loss increased while training continued improving.

Reason:

- Dataset is small and tabular.
- NN has more parameters and lower interpretability.

Final decision:

- Use classical ML models.

### Failure 3: Summary-Based Evidence Limits Coverage

Expectation:

- Wikipedia summaries should provide enough evidence for many FEVER claims.

Observed:

- Coverage stayed around 30% for several LLMs.
- Many claims became `True | No Evidence` or `False | No Evidence`.

Reason:

- FEVER claims often require facts outside article intros.

Final decision:

- Keep summaries for efficiency and explain this as a limitation.

## Streamlit Dashboard

App location:

```text
scripts/app.py
```

Run from:

```powershell
cd C:\Users\User\Desktop\HN_AI_Proj\TruthLens-LLM-Hallucination-Detection\scripts
streamlit run app.py
```

Dashboard tabs:

1. **TruthLens Metrics**
   - Shows rule-based system metrics for each LLM.

2. **ML Classifier**
   - Shows trained model comparison.
   - Shows confusion matrices and feature importance.

3. **Failure Analysis**
   - Shows NN results and training/validation curve.
   - Shows top-k retrieval experiment.

4. **Inspect Evidence**
   - Lets user inspect saved claims, LLM answers, NLI verdicts, final verdicts, and retrieved evidence.

5. **Predict Saved Claim**
   - Uses the saved Random Forest model to predict hallucination on already-scored claims.

## Why The App Cannot Predict Arbitrary New Claims Directly

The ML model does not take raw claim text as input.

It needs pipeline features:

- LLM label
- NLI label
- NLI confidence
- retrieved evidence scores
- entailment/contradiction/neutral scores
- LLM model identity

For a brand-new claim, those features do not exist yet.

To predict an arbitrary claim, the system would need to run:

1. LLM answer generation
2. Wikipedia retrieval
3. FAISS embedding search
4. NLI scoring
5. feature extraction
6. ML prediction

This is slow, so the dashboard only predicts on already-scored saved claims.

## Good Dashboard Test Claims

Use these full claims in the dashboard:

```text
Roman Atwood is a content creator.
Adrienne Bailon is an accountant.
Stranger than Fiction is a film.
Chris Hemsworth appeared in A Perfect Getaway.
The Silence of the Lambs was a film starring Scott Glenn.
Tetris has sold millions of physical copies.
The Ten Commandments is an epic film.
Homeland is an American television spy thriller based on the Israeli television series Prisoners of War.
Stranger Things is set in Bloomington, Indiana.
The Boston Celtics play their home games at TD Garden.
```

Use shorter search terms if needed:

```text
Roman Atwood
Adrienne Bailon
Chris Hemsworth
Stranger Things
Boston Celtics
```

## Ethical Reflection And Limitations

Limitations:

- The system depends on Wikipedia evidence.
- Wikipedia can be incomplete, outdated, biased, or ambiguous.
- Summary-based retrieval misses many specific facts.
- NLI can misinterpret irrelevant evidence as contradiction.
- The system abstains often due to no evidence.
- The ML classifier depends on features produced by the pipeline and cannot verify raw claims by itself.

Ethical implications:

- The system should not be treated as a final truth authority.
- False hallucination flags could reduce trust in correct LLM answers.
- Missed hallucinations could allow false information to pass.
- It should be used as an assistive verification tool, not an automatic judge.

## Final Recommended Report Story

TruthLens has two layers:

1. **Explainable rule-based hallucination detector**
   - Strong selective performance when evidence is available.
   - Abstains when evidence is insufficient.

2. **Trained full-coverage ML classifier**
   - Learns from retrieval and NLI features.
   - Improves over the rule-based baseline on all parsed rows.
   - Classical ML is selected over NN because the data is tabular and limited.

