import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DEFAULT_OUTPUT = "../data/llm_compare.csv"


def map_fever_to_truth(label):
    if label == "SUPPORTS":
        return "TRUE"
    if label == "REFUTES":
        return "FALSE"
    return "UNKNOWN"


def map_verdict_to_binary(verdict):
    return 1 if verdict == "Hallucination" else 0


def infer_model_name(df, path):
    if "llm_model" in df.columns:
        non_null = df["llm_model"].dropna()
        if not non_null.empty:
            return str(non_null.iloc[0])
    return Path(path).stem.replace("scored_claims_", "").replace("__", "/")


def summarize_scored_file(path):
    df = pd.read_csv(path)
    total_rows = len(df)
    binary_df = df[df["final_verdict"].isin(["Hallucination", "Not Hallucination"])].copy()
    binary_rows = len(binary_df)
    excluded_rows = total_rows - binary_rows

    summary = {
        "run": Path(path).stem,
        "model": infer_model_name(df, path),
        "input_file": Path(path).name,
        "total_rows": total_rows,
        "binary_rows": binary_rows,
        "excluded_rows": excluded_rows,
        "coverage": round(binary_rows / total_rows, 4) if total_rows else 0.0,
        "llm_hallucination_rate": 0.0,
        "system_flag_rate": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    if binary_df.empty:
        return summary

    binary_df["true_answer"] = binary_df["label"].apply(map_fever_to_truth)
    binary_df["true_hallucination"] = (binary_df["llm_label"] != binary_df["true_answer"]).astype(int)
    binary_df["predicted_hallucination"] = binary_df["final_verdict"].apply(map_verdict_to_binary)

    y_true = binary_df["true_hallucination"]
    y_pred = binary_df["predicted_hallucination"]

    summary.update({
        "llm_hallucination_rate": round(float(y_true.mean()), 4),
        "system_flag_rate": round(float(y_pred.mean()), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare multiple scored LLM runs.")
    parser.add_argument("inputs", nargs="+", help="One or more scored_claims CSV paths.")
    parser.add_argument(
        "--sort-by",
        default="f1",
        choices=[
            "f1",
            "accuracy",
            "coverage",
            "precision",
            "recall",
            "llm_hallucination_rate",
            "system_flag_rate",
        ],
        help="Metric to sort descending by.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to save the comparison CSV.")
    parser.add_argument("--print", action="store_true", dest="print_table", help="Print table in terminal.")
    args = parser.parse_args()

    rows = [summarize_scored_file(path) for path in args.inputs]
    summary_df = pd.DataFrame(rows).sort_values(by=args.sort_by, ascending=False)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_df.columns))
        writer.writeheader()
        writer.writerows(summary_df.to_dict(orient="records"))

    if args.print_table:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(summary_df.to_string(index=False))
    else:
        print("LLM hallucination rate comparison:")
        for _, row in summary_df.iterrows():
            print(f"{row['model']}: {row['llm_hallucination_rate'] * 100:.2f}%")
        print(f"\nSaved comparison CSV to: {output_path}")


if __name__ == "__main__":
    main()
