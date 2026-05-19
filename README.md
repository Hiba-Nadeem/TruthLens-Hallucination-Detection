# TruthLens: Evidence-Based Hallucination Detection in LLMs

TruthLens is an AI course project for detecting hallucinated LLM responses using retrieval, natural language inference, and a trained meta-classifier.

The project has two layers:

1. **Rule-based TruthLens verifier**
   - Generates a True/False LLM answer for a FEVER claim.
   - Retrieves Wikipedia evidence using FAISS and direct entity lookup.
   - Uses an NLI model to classify evidence as entailment, contradiction, or neutral.
   - Combines the LLM stance and NLI evidence using deterministic rules.

2. **Trained ML meta-classifier**
   - Learns from TruthLens pipeline features.
   - Predicts whether the LLM hallucinated even on rows where the rule-based system abstains.
   - Final selected model after tuning: **Gradient Boosting**.

---

## Project Motivation

LLMs can produce confident but factually incorrect answers. TruthLens tests whether an LLM response is supported by external evidence instead of trusting the response directly.

The project focuses on this question:

> Given a factual claim and an LLM's True/False response, can we detect whether the LLM hallucinated?

---

## Dataset

The project uses the **FEVER** claim verification dataset.

Final experiment file:

```text
data/fever_1000.csv
```

Each row contains:

| Column | Meaning |
|---|---|
| `claim` | Factual statement to verify |
| `label` | FEVER ground truth: `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO` |
| `evidence` | Wikipedia evidence metadata |

For hallucination detection:

```text
SUPPORTS -> TRUE
REFUTES  -> FALSE
```

Rows with unclear LLM output are marked `UNKNOWN` and excluded from selective rule-based evaluation.

---

## Evidence Source

TruthLens uses Wikipedia summaries rather than full Wikipedia articles.

This was a deliberate tradeoff:

- summaries are cheaper to embed,
- retrieval is faster,
- less irrelevant article text is introduced,
- but coverage is lower because some FEVER facts are not in article summaries.

Saved evidence/index files:

```text
data/wiki_summaries.json
data/wiki_pages/
data/chunk_metadata.json
vectorstore_wiki/index.faiss
vectorstore_wiki/index.pkl
```

---

## Models Used

| Model | Purpose |
|---|---|
| `google/flan-t5-base` | LLM answer generation |
| `google/flan-t5-small` | LLM answer generation |
| `declare-lab/flan-alpaca-base` | LLM answer generation |
| `Qwen/Qwen2.5-0.5B-Instruct` | LLM answer generation |
| `BAAI/bge-large-en-v1.5` | Embedding model for FAISS retrieval |
| `cross-encoder/nli-deberta-v3-base` | NLI evidence scoring |
| Logistic Regression | ML meta-classifier baseline |
| Random Forest | ML meta-classifier baseline |
| Gradient Boosting | Final tuned ML meta-classifier |
| MLP Neural Network | Failure-analysis experiment |

---

## Rule-Based TruthLens Pipeline

```text
FEVER claim
   |
   v
LLM generates True/False answer
   |
   v
Parse LLM stance: TRUE / FALSE / UNKNOWN
   |
   v
Retrieve Wikipedia evidence
   |-- direct entity lookup
   |-- FAISS semantic retrieval
   |
   v
Run NLI on evidence chunks
   |
   v
Aggregate NLI labels
   |
   v
Combine LLM stance + NLI label
   |
   v
TruthLens verdict
```

### Final Verdict Rules

| LLM Label | NLI Label | Verdict |
|---|---|---|
| TRUE | ENTAILMENT | Not Hallucination |
| TRUE | CONTRADICTION | Hallucination |
| FALSE | CONTRADICTION | Not Hallucination |
| FALSE | ENTAILMENT | Hallucination |
| TRUE | NEUTRAL | True \| No Evidence |
| FALSE | NEUTRAL | False \| No Evidence |
| UNKNOWN | any | Not Enough Evidence |

The rule-based system is intentionally selective. It only makes binary hallucination decisions when evidence is strong enough.

---

## Multi-LLM TruthLens Results

These results evaluate the rule-based TruthLens system only on binary verdict rows:

```text
Hallucination
Not Hallucination
```

No-evidence rows are excluded from this selective evaluation.

| LLM | Coverage | LLM Hallucination Rate | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| `google/flan-t5-base` | 30.20% | 39.74% | 79.14% | 68.39% | 88.33% | 77.09% |
| `google/flan-t5-small` | 23.60% | 48.31% | 76.27% | 75.89% | 74.56% | 75.22% |
| `declare-lab/flan-alpaca-base` | 44.70% | 38.70% | 78.08% | 68.29% | 80.92% | 74.07% |
| `Qwen/Qwen2.5-0.5B-Instruct` | 28.70% | 56.45% | 75.61% | 92.59% | 61.73% | 74.07% |

Key findings:

- `google/flan-t5-base` gives the best named-model F1.
- `declare-lab/flan-alpaca-base` gives the best coverage.
- `Qwen/Qwen2.5-0.5B-Instruct` gives the highest precision but lower recall.
- Coverage is the main limitation because Wikipedia summaries often do not contain enough evidence.

Saved comparison:

```text
data/llm_compare.csv
```

---

## Retrieval Depth Experiment

Different retrieval depths were tested for FLAN-T5-base.

| Top-K | Accuracy | Coverage | Outcome |
|---:|---:|---:|---|
| 3 | 79.28% | 30.40% | Slight improvement |
| 5 | 79.14% | 30.20% | Selected balanced baseline |
| 10 | 79.00% | 30.00% | No meaningful improvement; more retrieval cost/noise |

Final retrieval setting:

```text
top_k = 5
```

Reason: `top_k=10` did not improve results enough to justify extra retrieval cost or added evidence noise.

---

## Trained ML Meta-Classifier

The ML classifier is trained on features extracted from the TruthLens pipeline.

Unlike the rule-based system, the ML model uses all parsed rows:

- `Hallucination`
- `Not Hallucination`
- `True | No Evidence`
- `False | No Evidence`

This gives the ML model a broader task:

> Predict hallucination even when the rule-based system would abstain.

### Training Data

The model combines scored outputs from all four LLMs.

Initial rows:

```text
1000 rows x 4 LLMs = 4000 rows
```

Usable rows after filtering valid FEVER labels and parseable LLM labels:

```text
2919 rows
```

Rows by LLM:

| LLM | Usable Rows |
|---|---:|
| `declare-lab/flan-alpaca-base` | 998 |
| `google/flan-t5-base` | 691 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 647 |
| `google/flan-t5-small` | 583 |

The train/test split is grouped by claim ID so the same claim does not appear in both train and test sets.

| Split | Rows | Unique Claims |
|---|---:|---:|
| Train | 2332 | 800 |
| Test | 587 | 200 |

### ML Features

Examples of features:

- `llm_label_enc`
- `nli_label_enc`
- `nli_confidence`
- `llm_nli_agree`
- `claim_length`
- `answer_length`
- `max_entailment`
- `max_contradiction`
- `avg_entailment`
- `avg_contradiction`
- `avg_neutral`
- `entail_contra_diff`
- `n_chunks`
- `n_direct_chunks`
- `n_faiss_chunks`
- `avg_faiss_score`
- one-hot encoded LLM model identity

---

## Untuned ML Model Comparison

Script:

```text
scripts/model.py
```

Saved results:

```text
data/model_hallucination_results.csv
data/model_hallucination_per_llm_results.csv
data/model_hallucination_confusion_matrices.csv
data/confusion_matrices.png
data/feature_importance.png
```

| Model | Accuracy | Precision Hallucinated | Recall Hallucinated | F1 Hallucinated | F1 Not Hallucinated | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 79.39% | 76.54% | 76.83% | 76.69% | 81.53% | 79.11% |
| Logistic Regression | 79.05% | 73.45% | 82.24% | 77.60% | 80.32% | 78.96% |
| Gradient Boosting | 78.71% | 76.59% | 74.52% | 75.54% | 81.15% | 78.34% |
| Rule-based baseline | 66.61% | 73.68% | 37.84% | 50.00% | 74.94% | 62.47% |

Key finding:

The trained ML models outperform the rule-based baseline on the full-coverage task because they learn from soft NLI/retrieval signals instead of relying only on fixed rules.

---

## Hyperparameter Tuning

Script:

```text
scripts/model_tuning.py
```

Saved outputs:

```text
data/hyperparameter_tuning_results.csv
data/hyperparameter_tuning_best.csv
data/hyperparameter_tuning_plot.png
data/model_hallucination_tuned_gb.pkl
```

Best configurations:

| Model | Best Config | Test Accuracy | Test F1 Hallucinated | Test Macro F1 |
|---|---|---:|---:|---:|
| Gradient Boosting | `n_estimators=150, learning_rate=0.03, max_depth=3` | 81.60% | 79.23% | 81.36% |
| Logistic Regression | `C=1.0, class_weight=None` | 79.05% | 77.60% | 78.96% |
| Random Forest | `n_estimators=100, max_depth=None, min_samples_leaf=5, class_weight=None` | 79.22% | 76.45% | 78.93% |

Final selected ML classifier:

```text
Tuned Gradient Boosting
```

Reason:

- Best held-out test accuracy.
- Best hallucination F1.
- Best macro F1.

---

## Neural Network Failure Analysis

Script:

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
| MLP Neural Network | 77.51% | 72.13% | 79.92% | 75.82% | 8.12% |

Finding:

The neural network did not outperform classical ML models. Training performance improved while validation behavior became less stable, showing that the NN was not the best fit for this small tabular feature dataset.

---

## Architecture Selection

The final selected model family is classical ML on tabular features.

Why not CNN?

- CNNs are designed for images or spatial/local sequence patterns.
- TruthLens features are structured numeric/categorical values.

Why not final NN?

- The dataset is small for a neural network.
- The NN did not outperform Gradient Boosting or Logistic Regression.
- Classical ML provides better interpretability.

Why Gradient Boosting?

- It performed best after tuning.
- It handles nonlinear feature interactions.
- It works well on tabular data.

---

## Streamlit Dashboard

Dashboard file:

```text
scripts/app.py
```

Run:

```powershell
cd C:\Users\User\Desktop\HN_AI_Proj\TruthLens-LLM-Hallucination-Detection\scripts
streamlit run app.py
```

Dashboard tabs:

1. **TruthLens Metrics**
   - Multi-LLM rule-based system metrics.

2. **ML Classifier**
   - Untuned ML comparison.
   - Hyperparameter tuning results.
   - Confusion matrices.

3. **Failure Analysis**
   - NN experiment.
   - Retrieval depth experiment.

4. **Inspect Evidence**
   - View saved claims, raw LLM output, NLI label, verdict, and retrieved evidence.

5. **Predict Saved Claim**
   - Uses the tuned Gradient Boosting model on already-scored claims.

Important:

The dashboard does not predict arbitrary raw claims directly. A raw claim must first pass through LLM generation, retrieval, NLI, and feature extraction before the ML model can classify it.

---

## Example Dashboard Test Claims

Use these full claims in Inspect Evidence or Predict Saved Claim:

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

If exact search does not match, use shorter phrases such as:

```text
Roman Atwood
Adrienne Bailon
Chris Hemsworth
Stranger Things
Boston Celtics
```

---

## Important Note About LLM Outputs

Some generated LLM responses contain noisy or contradictory explanations, such as prompt echoing or awkward text.

TruthLens handles this by extracting only the first clear stance:

```text
TRUE
FALSE
UNKNOWN
```

The raw generated answer is shown in the dashboard for transparency, but the rule-based system and ML classifier use the parsed stance and evidence features.

---

## How To Run Key Scripts

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run from the `scripts/` directory:

```powershell
cd scripts
```

Generate LLM answers for a model:

```powershell
python generate_answers.py --model-name google/flan-t5-base --input ../data/fever_1000.csv --output ../data/llm_answers_googleflan-t5-base.csv
```

Score claims with retrieval + NLI:

```powershell
python score_claims.py --input ../data/llm_answers_googleflan-t5-base.csv --output ../data/scored_claims_googleflan-t5-base.csv --top-k 5
```

Compare saved LLM runs:

```powershell
python compare_llms.py ../data/scored_claims_googleflan-t5-base.csv ../data/scored_claims_googleflan-t5-small.csv ../data/scored_claims_declare-lab-flan-alpaca-base.csv ../data/scored_claims_Qwen2.5-0.5B-Instruct.csv --output ../data/llm_compare.csv --print
```

Train untuned ML classifiers:

```powershell
python model.py
```

Run hyperparameter tuning:

```powershell
python model_tuning.py
```

Run NN failure-analysis experiment:

```powershell
python model_nn.py
```

Launch dashboard:

```powershell
streamlit run app.py
```

---

## Folder Structure

```text
TruthLens-LLM-Hallucination-Detection/
|
|-- data/
|   |-- fever_1000.csv
|   |-- fever_filtered.csv
|   |-- wiki_summaries.json
|   |-- chunk_metadata.json
|   |-- llm_answers_*.csv
|   |-- scored_claims_*.csv
|   |-- llm_compare.csv
|   |-- model_hallucination_*.pkl
|   |-- model_hallucination_tuned_gb.pkl
|   |-- hyperparameter_tuning_*.csv
|   |-- nn_results.csv
|   |-- *.png
|
|-- scripts/
|   |-- app.py
|   |-- generate_answers.py
|   |-- score_claims.py
|   |-- evaluation.py
|   |-- compare_llms.py
|   |-- model.py
|   |-- model_tuning.py
|   |-- model_nn.py
|   |-- build_index.py
|   |-- fetch_wiki.py
|   |-- test_retrieval.py
|
|-- vectorstore_wiki/
|   |-- index.faiss
|   |-- index.pkl
|
|-- requirements.txt
|-- PROJECT_CONTEXT_SUMMARY.md
|-- README.md
```

---

## Limitations And Ethics

Limitations:

- Coverage is limited by summary-based Wikipedia evidence.
- Wikipedia can be incomplete, outdated, or biased.
- NLI can misclassify irrelevant evidence as contradiction.
- Small LLMs can produce malformed or contradictory answers.
- The ML classifier depends on features generated by the TruthLens pipeline.
- The dashboard cannot verify arbitrary raw claims without rerunning the full pipeline.

Ethical reflection:

- TruthLens should be used as an assistive verification tool, not a final truth authority.
- False hallucination flags can reduce trust in correct answers.
- Missed hallucinations can allow false information to pass.
- Any real-world deployment should include human review, source transparency, and uncertainty reporting.

---

## Final Project Finding

The main finding is:

> Rule-based evidence verification is strong when evidence is available, but coverage is limited. A trained ML meta-classifier can extend the system to uncertain/no-evidence cases by learning from soft retrieval and NLI signals. Classical ML, especially tuned Gradient Boosting, is better suited than neural networks for this small tabular feature dataset.

