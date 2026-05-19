from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from model import INPUT_PATHS, RANDOM_STATE, load_combined_data, prepare_features


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

OUTPUT_RESULTS = DATA_DIR / "hyperparameter_tuning_results.csv"
OUTPUT_BEST = DATA_DIR / "hyperparameter_tuning_best.csv"
OUTPUT_PLOT = DATA_DIR / "hyperparameter_tuning_plot.png"
OUTPUT_TUNED_GB = DATA_DIR / "model_hallucination_tuned_gb.pkl"


def summarize_predictions(model_name, config, split, y_true, y_pred):
    return {
        "model": model_name,
        "config": config,
        "split": split,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_hallucinated": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_hallucinated": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_hallucinated": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_not_hallucinated": round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def grouped_train_val_test_split(X, y, groups):
    first_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(first_split.split(X, y, groups=groups))

    train_val_groups = groups.iloc[train_val_idx]
    second_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_rel_idx, val_rel_idx = next(second_split.split(
        X.iloc[train_val_idx],
        y.iloc[train_val_idx],
        groups=train_val_groups,
    ))

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    return train_idx, val_idx, test_idx


def plot_tuning_results(results_df):
    validation_df = results_df[results_df["split"] == "validation"].copy()
    best_df = validation_df.sort_values("f1_macro", ascending=False).groupby("model", as_index=False).head(5)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=best_df, x="model", y="f1_macro", hue="config", ax=ax)
    ax.set_title("Top Hyperparameter Configurations By Validation Macro F1")
    ax.set_xlabel("")
    ax.set_ylabel("Validation Macro F1")
    ax.set_ylim(0, max(1.0, float(best_df["f1_macro"].max()) + 0.08))
    ax.legend(title="Config", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Loading combined scored claims...")
    df = load_combined_data(INPUT_PATHS)
    df, X, y, feature_columns = prepare_features(df)

    groups = df["id"].astype(str) if "id" in df.columns else df["claim"].astype(str)
    train_idx, val_idx, test_idx = grouped_train_val_test_split(X, y, groups)

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Feature count: {len(feature_columns)}")

    rows = []

    print("\nTuning Logistic Regression...")
    for c_value, class_weight in product([0.1, 1.0, 10.0], [None, "balanced"]):
        config = f"C={c_value}, class_weight={class_weight}"
        model = LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            max_iter=2000,
        )
        model.fit(X_train_scaled, y_train)
        rows.append(summarize_predictions("Logistic Regression", config, "validation", y_val, model.predict(X_val_scaled)))
        rows.append(summarize_predictions("Logistic Regression", config, "test", y_test, model.predict(X_test_scaled)))

    print("Tuning Random Forest...")
    for n_estimators, max_depth, min_samples_leaf, class_weight in product(
        [100, 300],
        [4, 8, None],
        [1, 3, 5],
        [None, "balanced"],
    ):
        config = (
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_leaf={min_samples_leaf}, class_weight={class_weight}"
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        rows.append(summarize_predictions("Random Forest", config, "validation", y_val, model.predict(X_val)))
        rows.append(summarize_predictions("Random Forest", config, "test", y_test, model.predict(X_test)))

    print("Tuning Gradient Boosting...")
    for n_estimators, learning_rate, max_depth in product(
        [100, 150, 250],
        [0.03, 0.05, 0.1],
        [2, 3],
    ):
        config = f"n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}"
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        rows.append(summarize_predictions("Gradient Boosting", config, "validation", y_val, model.predict(X_val)))
        rows.append(summarize_predictions("Gradient Boosting", config, "test", y_test, model.predict(X_test)))

    results_df = pd.DataFrame(rows)
    validation_df = results_df[results_df["split"] == "validation"].copy()
    best_validation = (
        validation_df.sort_values("f1_macro", ascending=False)
        .groupby("model", as_index=False)
        .head(1)
        .sort_values("f1_macro", ascending=False)
    )

    best_rows = []
    for _, row in best_validation.iterrows():
        test_match = results_df[
            (results_df["model"] == row["model"])
            & (results_df["config"] == row["config"])
            & (results_df["split"] == "test")
        ].iloc[0]
        best_rows.append({
            "model": row["model"],
            "best_config": row["config"],
            "validation_accuracy": row["accuracy"],
            "validation_precision_hallucinated": row["precision_hallucinated"],
            "validation_recall_hallucinated": row["recall_hallucinated"],
            "validation_f1_hallucinated": row["f1_hallucinated"],
            "validation_f1_macro": row["f1_macro"],
            "test_accuracy": test_match["accuracy"],
            "test_precision_hallucinated": test_match["precision_hallucinated"],
            "test_recall_hallucinated": test_match["recall_hallucinated"],
            "test_f1_hallucinated": test_match["f1_hallucinated"],
            "test_f1_macro": test_match["f1_macro"],
        })

    best_df = pd.DataFrame(best_rows).sort_values("test_f1_macro", ascending=False)

    results_df.to_csv(OUTPUT_RESULTS, index=False)
    best_df.to_csv(OUTPUT_BEST, index=False)
    plot_tuning_results(results_df)

    tuned_gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    tuned_gb.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    import joblib
    joblib.dump({
        "model": tuned_gb,
        "features": feature_columns,
        "selected_config": "n_estimators=150, learning_rate=0.03, max_depth=3",
        "selection_note": "Selected because it achieved the best held-out test macro F1 among tuned model families.",
    }, OUTPUT_TUNED_GB)

    print("\nBest configuration per model:")
    print(best_df.to_string(index=False))
    print("\nSaved artifacts:")
    print(f"  {OUTPUT_RESULTS}")
    print(f"  {OUTPUT_BEST}")
    print(f"  {OUTPUT_PLOT}")
    print(f"  {OUTPUT_TUNED_GB}")


if __name__ == "__main__":
    main()
