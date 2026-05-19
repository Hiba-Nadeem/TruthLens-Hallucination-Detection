import argparse

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def map_fever_to_truth(label):
    if label == "SUPPORTS":
        return "TRUE"
    if label == "REFUTES":
        return "FALSE"
    return "UNKNOWN"


def map_verdict_to_binary(verdict):
    return 1 if verdict == "Hallucination" else 0


def evaluate_dataframe(df):
    total_rows = len(df)
    df = df[df["final_verdict"].isin(["Hallucination", "Not Hallucination"])].copy()
    binary_rows = len(df)

    print(f"Rows with binary verdict (evaluated): {binary_rows}")
    print(f"Rows excluded (no evidence / unknown): {total_rows - binary_rows}")
    print(f"Coverage  : {binary_rows / total_rows:.4f}" if total_rows else "Coverage  : 0.0000")

    if df.empty:
        print("\nNo binary predictions found. Nothing to evaluate.")
        return

    df["true_answer"] = df["label"].apply(map_fever_to_truth)
    df["true_hallucination"] = (df["llm_label"] != df["true_answer"]).astype(int)
    df["predicted_hallucination"] = df["final_verdict"].apply(map_verdict_to_binary)

    y_true = df["true_hallucination"]
    y_pred = df["predicted_hallucination"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    llm_hallucination_rate = y_true.mean()
    predicted_hallucination_rate = y_pred.mean()

    print("=== Hallucination Detection Evaluation ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"LLM Hallucination Rate (evaluated rows): {llm_hallucination_rate:.4f}")
    print(f"System Flag Rate (evaluated rows)      : {predicted_hallucination_rate:.4f}")

    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("\n--- Debug Info ---")
    print("True hallucination counts:\n", df["true_hallucination"].value_counts())
    print("Predicted hallucination counts:\n", df["predicted_hallucination"].value_counts())

    print("\n--- Sample Errors ---")
    errors = df[df["true_hallucination"] != df["predicted_hallucination"]]
    print(errors[["claim", "llm_answer", "llm_label", "true_answer", "final_verdict"]].head(10))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a scored claims CSV.")
    parser.add_argument("--input", default="../data/scored_claims_googleflan-t5-base_top3.csv", help="Scored CSV path.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    evaluate_dataframe(df)


if __name__ == "__main__":
    main()
