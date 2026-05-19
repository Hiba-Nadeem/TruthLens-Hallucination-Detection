import ast
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# -------------------------------------------------------
# Config
# -------------------------------------------------------
INPUT_PATH      = "../data/scored_claims.csv"
OUTPUT_LR       = "../data/model_lr.pkl"
OUTPUT_RF       = "../data/model_rf.pkl"
OUTPUT_XGB      = "../data/model_xgb.pkl"
OUTPUT_CM       = "../data/confusion_matrices.png"
OUTPUT_FI       = "../data/feature_importance.png"

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
print("Loading data...")
df = pd.read_csv(INPUT_PATH)

# Keep only SUPPORTS and REFUTES
df = df[df['label'].isin(['SUPPORTS', 'REFUTES'])].copy()

# Drop rows where LLM couldn't parse an answer
df = df[df['llm_label'] != 'UNKNOWN'].copy()

# Reset index — prevents iloc/loc mismatch bugs
df = df.reset_index(drop=True)

print(f"Working set: {len(df)} rows")
print(df['label'].value_counts())
supports_count = (df['label'] == 'SUPPORTS').sum()
refutes_count  = (df['label'] == 'REFUTES').sum()

# -------------------------------------------------------
# Feature engineering — basic features
# -------------------------------------------------------
df['llm_label_enc'] = (df['llm_label'] == 'TRUE').astype(int)
df['nli_label_enc'] = df['nli_label'].map({
    'ENTAILMENT':    2,
    'NEUTRAL':       1,
    'CONTRADICTION': 0
})

df['llm_nli_agree'] = (
    ((df['llm_label'] == 'TRUE')  & (df['nli_label'] == 'ENTAILMENT')) |
    ((df['llm_label'] == 'FALSE') & (df['nli_label'] == 'CONTRADICTION'))
).astype(int)

df['claim_length']  = df['claim'].str.split().str.len()
df['answer_length'] = df['llm_answer'].str.split().str.len()

# -------------------------------------------------------
# Feature engineering — extract from all_retrieved_evidence
# -------------------------------------------------------
def extract_evidence_features(row):
    """
    Pull per-chunk NLI scores out of the all_retrieved_evidence column
    and compute aggregate features that are more informative than a
    single nli_confidence number.
    """
    try:
        evidence = ast.literal_eval(row['all_retrieved_evidence'])
    except Exception:
        return pd.Series({
            'max_entailment':    0.0,
            'max_contradiction': 0.0,
            'avg_entailment':    0.0,
            'avg_contradiction': 0.0,
            'n_chunks':          0,
            'entail_contra_diff': 0.0,
        })

    entailments    = [e.get('entailment',    0.0) for e in evidence]
    contradictions = [e.get('contradiction', 0.0) for e in evidence]

    max_e = max(entailments)    if entailments    else 0.0
    max_c = max(contradictions) if contradictions else 0.0

    return pd.Series({
        'max_entailment':     max_e,
        'max_contradiction':  max_c,
        'avg_entailment':     float(np.mean(entailments)),
        'avg_contradiction':  float(np.mean(contradictions)),
        'n_chunks':           len(evidence),
        # How decisively does entailment beat contradiction?
        # Positive → evidence supports claim, Negative → evidence refutes it
        'entail_contra_diff': max_e - max_c,
    })

print("Extracting evidence features...")
extra = df.apply(extract_evidence_features, axis=1)
df    = pd.concat([df, extra], axis=1)

# -------------------------------------------------------
# Final feature list
# -------------------------------------------------------
FEATURES = [
    # NLI core signals
    'nli_confidence',
    'nli_label_enc',
    # LLM signals
    'llm_label_enc',
    'llm_nli_agree',
    # Text length
    'claim_length',
    'answer_length',
    # Evidence aggregate features (new)
    'max_entailment',
    'max_contradiction',
    'avg_entailment',
    'avg_contradiction',
    'entail_contra_diff',
    'n_chunks',
]

X = df[FEATURES]
y = (df['label'] == 'SUPPORTS').astype(int)  # 1 = SUPPORTS, 0 = REFUTES

print(f"\nFeatures: {FEATURES}")
print(f"Class distribution — SUPPORTS: {y.sum()}, REFUTES: {(1 - y).sum()}")

# -------------------------------------------------------
# Train / test split — stratified to preserve class ratio
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# -------------------------------------------------------
# Baseline: rule-based system (your original pipeline)
# -------------------------------------------------------
def rule_based_predict(row):
    """Replicates compute_verdict() logic from score_claims.py."""
    llm = row['llm_label']
    nli = row['nli_label']
    if llm == 'TRUE'  and nli == 'ENTAILMENT':    return 1  # Supported
    if llm == 'FALSE' and nli == 'CONTRADICTION':  return 1  # Supported
    if llm == 'TRUE'  and nli == 'CONTRADICTION':  return 0  # Hallucinated
    if llm == 'FALSE' and nli == 'ENTAILMENT':     return 0  # Hallucinated
    return 1  # NEUTRAL defaults to SUPPORTS

test_df        = df.loc[X_test.index]
baseline_preds = test_df.apply(rule_based_predict, axis=1).values

print("\n" + "="*55)
print("BASELINE: Rule-based system (original pipeline)")
print("="*55)
print(classification_report(y_test, baseline_preds,
                             target_names=['REFUTES', 'SUPPORTS']))

# -------------------------------------------------------
# Model 1: Logistic Regression
# class_weight='balanced' → fixes class imbalance
# -------------------------------------------------------
lr = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'   # FIX: penalise misclassifying minority class more
)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

print("="*55)
print("LOGISTIC REGRESSION (balanced)")
print("="*55)
print(classification_report(y_test, lr_preds,
                             target_names=['REFUTES', 'SUPPORTS']))

# -------------------------------------------------------
# Model 2: Random Forest
# class_weight='balanced' → same fix
# -------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight='balanced'   # FIX: upweights REFUTES during training
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("="*55)
print("RANDOM FOREST (balanced, 200 trees)")
print("="*55)
print(classification_report(y_test, rf_preds,
                             target_names=['REFUTES', 'SUPPORTS']))

# -------------------------------------------------------
# Model 3: XGBoost
# scale_pos_weight → XGBoost's equivalent of class_weight='balanced'
# ratio = majority / minority = SUPPORTS / REFUTES
# -------------------------------------------------------
scale = supports_count / refutes_count  # e.g. 475/179 ≈ 2.65

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale,   # FIX: tells XGBoost REFUTES is the rare class
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

print("="*55)
print(f"XGBOOST (scale_pos_weight={scale:.2f})")
print("="*55)
print(classification_report(y_test, xgb_preds,
                             target_names=['REFUTES', 'SUPPORTS']))

# -------------------------------------------------------
# Feature importance plot (Random Forest)
# -------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=FEATURES)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(importances.index, importances.values, color='steelblue')
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importances — Random Forest')
ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_FI, dpi=150)
plt.close()
print(f"\nFeature importance plot saved to {OUTPUT_FI}")

print("\n=== Feature Importances (Random Forest) ===")
print(importances.sort_values(ascending=False).to_string())

# -------------------------------------------------------
# Confusion matrices — all 4 systems side by side
# -------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

all_preds  = [baseline_preds, lr_preds, rf_preds, xgb_preds]
all_titles = [
    'Baseline\n(Rule-based)',
    'Logistic Regression\n(balanced)',
    'Random Forest\n(balanced)',
    f'XGBoost\n(scale={scale:.2f})'
]

for ax, preds, title in zip(axes, all_preds, all_titles):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(
        cm, annot=True, fmt='d', ax=ax,
        xticklabels=['REFUTES', 'SUPPORTS'],
        yticklabels=['REFUTES', 'SUPPORTS'],
        cmap='Blues', cbar=False
    )
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.suptitle('Confusion Matrices: Baseline vs Trained Classifiers', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_CM, dpi=150, bbox_inches='tight')
plt.close()
print(f"Confusion matrices saved to {OUTPUT_CM}")

# -------------------------------------------------------
# Summary table
# -------------------------------------------------------
from sklearn.metrics import f1_score, accuracy_score

print("\n" + "="*55)
print("SUMMARY")
print("="*55)

rows = []
for name, preds in zip(
    ['Baseline (Rule-based)', 'Logistic Regression', 'Random Forest', 'XGBoost'],
    [baseline_preds, lr_preds, rf_preds, xgb_preds]
):
    rows.append({
        'Model':           name,
        'Accuracy':        round(accuracy_score(y_test, preds), 3),
        'F1 (REFUTES)':    round(f1_score(y_test, preds, pos_label=0), 3),
        'F1 (SUPPORTS)':   round(f1_score(y_test, preds, pos_label=1), 3),
        'F1 (macro avg)':  round(f1_score(y_test, preds, average='macro'), 3),
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

# -------------------------------------------------------
# Save all models
# -------------------------------------------------------
joblib.dump(lr,  OUTPUT_LR)
joblib.dump(rf,  OUTPUT_RF)
joblib.dump(xgb, OUTPUT_XGB)
print(f"\nModels saved:")
print(f"  {OUTPUT_LR}")
print(f"  {OUTPUT_RF}")
print(f"  {OUTPUT_XGB}")