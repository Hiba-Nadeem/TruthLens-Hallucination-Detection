import ast
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------
# Config
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_PATHS = [
    DATA_DIR / "scored_claims_googleflan-t5-base.csv",
    DATA_DIR / "scored_claims_googleflan-t5-small.csv",
    DATA_DIR / "scored_claims_declare-lab-flan-alpaca-base.csv",
    DATA_DIR / "scored_claims_Qwen2.5-0.5B-Instruct.csv",
]

OUTPUT_LR = DATA_DIR / "model_hallucination_lr.pkl"
OUTPUT_RF = DATA_DIR / "model_hallucination_rf.pkl"
OUTPUT_GB = DATA_DIR / "model_hallucination_gb.pkl"
OUTPUT_SUMMARY = DATA_DIR / "model_hallucination_results.csv"
OUTPUT_PER_LLM = DATA_DIR / "model_hallucination_per_llm_results.csv"
OUTPUT_CONFUSION = DATA_DIR / "model_hallucination_confusion_matrices.csv"
OUTPUT_FEATURES = DATA_DIR / "model_hallucination_feature_columns.csv"
OUTPUT_CM = DATA_DIR / "confusion_matrices.png"
OUTPUT_FI = DATA_DIR / "feature_importance.png"

RANDOM_STATE = 42


# -------------------------------------------------------
# Load and combine scored LLM runs
# -------------------------------------------------------
def infer_model_name(df, path):
    if "llm_model" in df.columns:
        non_null = df["llm_model"].dropna()
        if not non_null.empty:
            return str(non_null.iloc[0])
    return path.stem.replace("scored_claims_", "").replace("__", "/")


def load_combined_data(paths):
    frames = []

    for path in paths:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        df = pd.read_csv(path)
        df["source_file"] = path.name
        df["llm_model"] = infer_model_name(df, path)
        frames.append(df)
        print(f"Loaded {len(df)} rows from {path.name}")

    if not frames:
        raise FileNotFoundError("No scored_claims CSV files were found.")

    return pd.concat(frames, ignore_index=True)


# -------------------------------------------------------
# Target: did the LLM hallucinate?
# -------------------------------------------------------
def compute_true_hallucination(row):
    llm = row["llm_label"]
    fever = row["label"]

    if llm == "TRUE" and fever == "SUPPORTS":
        return 0
    if llm == "TRUE" and fever == "REFUTES":
        return 1
    if llm == "FALSE" and fever == "REFUTES":
        return 0
    if llm == "FALSE" and fever == "SUPPORTS":
        return 1

    return np.nan


# -------------------------------------------------------
# Evidence/NLI feature extraction
# -------------------------------------------------------
def extract_evidence_features(row):
    try:
        evidence = ast.literal_eval(str(row["all_retrieved_evidence"]))
    except Exception:
        evidence = []

    entailments = [float(e.get("entailment", 0.0)) for e in evidence]
    contradictions = [float(e.get("contradiction", 0.0)) for e in evidence]
    neutrals = [float(e.get("neutral", 0.0)) for e in evidence]
    faiss_scores = [
        float(e.get("store_score", 0.0))
        for e in evidence
        if str(e.get("source", "")).lower() == "faiss"
    ]

    max_e = max(entailments) if entailments else 0.0
    max_c = max(contradictions) if contradictions else 0.0
    max_n = max(neutrals) if neutrals else 0.0

    return pd.Series({
        "max_entailment": max_e,
        "max_contradiction": max_c,
        "max_neutral": max_n,
        "avg_entailment": float(np.mean(entailments)) if entailments else 0.0,
        "avg_contradiction": float(np.mean(contradictions)) if contradictions else 0.0,
        "avg_neutral": float(np.mean(neutrals)) if neutrals else 0.0,
        "entail_contra_diff": max_e - max_c,
        "n_chunks": len(evidence),
        "n_direct_chunks": sum(str(e.get("source", "")).lower() == "direct" for e in evidence),
        "n_faiss_chunks": sum(str(e.get("source", "")).lower() == "faiss" for e in evidence),
        "avg_faiss_score": float(np.mean(faiss_scores)) if faiss_scores else 0.0,
    })


def prepare_features(df):
    df = df[df["label"].isin(["SUPPORTS", "REFUTES"])].copy()
    df = df[df["llm_label"].isin(["TRUE", "FALSE"])].copy()
    df = df.reset_index(drop=True)

    df["true_hallucination"] = df.apply(compute_true_hallucination, axis=1)
    df = df.dropna(subset=["true_hallucination"]).reset_index(drop=True)
    df["true_hallucination"] = df["true_hallucination"].astype(int)

    df["llm_label_enc"] = (df["llm_label"] == "TRUE").astype(int)
    df["nli_label_enc"] = df["nli_label"].map({
        "CONTRADICTION": 0,
        "NEUTRAL": 1,
        "ENTAILMENT": 2,
    }).fillna(1)

    df["llm_nli_agree"] = (
        ((df["llm_label"] == "TRUE") & (df["nli_label"] == "ENTAILMENT"))
        | ((df["llm_label"] == "FALSE") & (df["nli_label"] == "CONTRADICTION"))
    ).astype(int)

    df["claim_length"] = df["claim"].fillna("").astype(str).str.split().str.len()
    df["answer_length"] = df["llm_answer"].fillna("").astype(str).str.split().str.len()

    print("Extracting evidence features...")
    evidence_features = df.apply(extract_evidence_features, axis=1)
    df = pd.concat([df, evidence_features], axis=1)

    numeric_features = [
        "nli_confidence",
        "nli_label_enc",
        "llm_label_enc",
        "llm_nli_agree",
        "claim_length",
        "answer_length",
        "max_entailment",
        "max_contradiction",
        "max_neutral",
        "avg_entailment",
        "avg_contradiction",
        "avg_neutral",
        "entail_contra_diff",
        "n_chunks",
        "n_direct_chunks",
        "n_faiss_chunks",
        "avg_faiss_score",
    ]

    model_features = pd.get_dummies(df["llm_model"], prefix="model", dtype=int)
    X = pd.concat([df[numeric_features], model_features], axis=1).fillna(0)
    y = df["true_hallucination"]

    return df, X, y, list(X.columns)


# -------------------------------------------------------
# Baseline: original rule-based TruthLens decision logic
# -------------------------------------------------------
def rule_based_predict(row):
    llm = row["llm_label"]
    nli = row["nli_label"]

    if llm == "TRUE" and nli == "CONTRADICTION":
        return 1
    if llm == "FALSE" and nli == "ENTAILMENT":
        return 1
    if llm == "TRUE" and nli == "ENTAILMENT":
        return 0
    if llm == "FALSE" and nli == "CONTRADICTION":
        return 0

    # If evidence is neutral, the rule system abstains in the main pipeline.
    # For this full-coverage classifier comparison, default neutral to not hallucinated.
    return 0


def summarize_predictions(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_hallucinated": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_hallucinated": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_hallucinated": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_not_hallucinated": round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def plot_confusion_matrices(y_test, prediction_map):
    fig, axes = plt.subplots(1, len(prediction_map), figsize=(5 * len(prediction_map), 5))
    if len(prediction_map) == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, prediction_map.items()):
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            ax=ax,
            xticklabels=["Not Hallucinated", "Hallucinated"],
            yticklabels=["Not Hallucinated", "Hallucinated"],
            cmap="Reds",
            cbar=False,
        )
        ax.set_title(name)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.suptitle("Confusion Matrices: Full-Coverage Hallucination Detection", y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_CM, dpi=150, bbox_inches="tight")
    plt.close()


def build_confusion_matrix_rows(y_test, prediction_map):
    rows = []
    for name, preds in prediction_map.items():
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        rows.append({
            "model": name,
            "true_not_hallucinated_pred_not_hallucinated": int(cm[0, 0]),
            "true_not_hallucinated_pred_hallucinated": int(cm[0, 1]),
            "true_hallucinated_pred_not_hallucinated": int(cm[1, 0]),
            "true_hallucinated_pred_hallucinated": int(cm[1, 1]),
        })
    return pd.DataFrame(rows)


def plot_feature_importance(rf, feature_columns):
    importances = pd.Series(rf.feature_importances_, index=feature_columns)
    top_importances = importances.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top_importances.index, top_importances.values, color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Random Forest Feature Importances")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_FI, dpi=150)
    plt.close()

    return importances.sort_values(ascending=False)


def main():
    print("Loading combined scored claims...")
    df = load_combined_data(INPUT_PATHS)
    df, X, y, feature_columns = prepare_features(df)

    print(f"\nCombined usable rows: {len(df)}")
    print("Rows by LLM:")
    print(df["llm_model"].value_counts().to_string())
    print("\nTarget distribution:")
    print(y.value_counts().rename({0: "Not Hallucinated", 1: "Hallucinated"}).to_string())

    groups = df["id"].astype(str) if "id" in df.columns else df["claim"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    test_df = df.iloc[test_idx]

    print(f"\nTrain rows: {len(X_train)}")
    print(f"Test rows : {len(X_test)}")
    print(f"Train unique claims: {groups.iloc[train_idx].nunique()}")
    print(f"Test unique claims : {groups.iloc[test_idx].nunique()}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    baseline_preds = test_df.apply(rule_based_predict, axis=1).values

    lr = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=2000,
        class_weight="balanced",
    )
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)

    prediction_map = {
        "Rule-based": baseline_preds,
        "Logistic Regression": lr_preds,
        "Random Forest": rf_preds,
        "Gradient Boosting": gb_preds,
    }

    print("\n" + "=" * 72)
    print("CLASSIFICATION REPORTS")
    print("=" * 72)
    for name, preds in prediction_map.items():
        print(f"\n{name}")
        print(classification_report(
            y_test,
            preds,
            target_names=["Not Hallucinated", "Hallucinated"],
            zero_division=0,
        ))

    summary = pd.DataFrame([
        summarize_predictions(name, y_test, preds)
        for name, preds in prediction_map.items()
    ]).sort_values("f1_macro", ascending=False)

    print("\n" + "=" * 72)
    print("SUMMARY: Full-coverage hallucination detection")
    print("=" * 72)
    print(summary.to_string(index=False))

    print("\nPer-LLM test performance for the best model:")
    best_name = str(summary.iloc[0]["model"])
    best_preds = prediction_map[best_name]
    per_model_rows = []
    for llm_model, idx in test_df.groupby("llm_model").groups.items():
        idx_positions = [test_df.index.get_loc(i) for i in idx]
        per_model_rows.append(summarize_predictions(
            llm_model,
            y_test.iloc[idx_positions],
            best_preds[idx_positions],
        ))
    per_model_summary = pd.DataFrame(per_model_rows).sort_values("f1_macro", ascending=False)
    print(per_model_summary.to_string(index=False))

    feature_importances = plot_feature_importance(rf, feature_columns)
    plot_confusion_matrices(y_test, prediction_map)
    confusion_summary = build_confusion_matrix_rows(y_test, prediction_map)

    print("\nTop Random Forest features:")
    print(feature_importances.head(15).round(4).to_string())

    coef_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": lr.coef_[0],
    }).sort_values("coefficient", ascending=False)
    print("\nTop Logistic Regression coefficients toward hallucination:")
    print(coef_df.head(12).round(4).to_string(index=False))
    print("\nTop Logistic Regression coefficients away from hallucination:")
    print(coef_df.tail(12).round(4).to_string(index=False))

    summary.to_csv(OUTPUT_SUMMARY, index=False)
    per_model_summary.to_csv(OUTPUT_PER_LLM, index=False)
    confusion_summary.to_csv(OUTPUT_CONFUSION, index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(OUTPUT_FEATURES, index=False)

    joblib.dump({"model": lr, "scaler": scaler, "features": feature_columns}, OUTPUT_LR)
    joblib.dump({"model": rf, "features": feature_columns}, OUTPUT_RF)
    joblib.dump({"model": gb, "features": feature_columns}, OUTPUT_GB)

    print("\nSaved artifacts:")
    print(f"  {OUTPUT_LR}")
    print(f"  {OUTPUT_RF}")
    print(f"  {OUTPUT_GB}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_PER_LLM}")
    print(f"  {OUTPUT_CONFUSION}")
    print(f"  {OUTPUT_FEATURES}")
    print(f"  {OUTPUT_CM}")
    print(f"  {OUTPUT_FI}")


if __name__ == "__main__":
    main()
