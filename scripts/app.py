import ast
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from model import extract_evidence_features


BASE_DIR = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

NAMED_SCORED_FILES = [
    DATA_DIR / "scored_claims_googleflan-t5-base.csv",
    DATA_DIR / "scored_claims_googleflan-t5-small.csv",
    DATA_DIR / "scored_claims_declare-lab-flan-alpaca-base.csv",
    DATA_DIR / "scored_claims_Qwen2.5-0.5B-Instruct.csv",
]

TRUTHLENS_RESULTS = DATA_DIR / "llm_compare.csv"
ML_RESULTS = DATA_DIR / "model_hallucination_results.csv"
ML_PER_LLM_RESULTS = DATA_DIR / "model_hallucination_per_llm_results.csv"
ML_CONFUSION_RESULTS = DATA_DIR / "model_hallucination_confusion_matrices.csv"
TUNING_BEST_RESULTS = DATA_DIR / "hyperparameter_tuning_best.csv"
TUNING_PLOT = DATA_DIR / "hyperparameter_tuning_plot.png"
FINAL_MODEL_PATH = DATA_DIR / "model_hallucination_tuned_gb.pkl"
CONFUSION_IMAGE = DATA_DIR / "confusion_matrices.png"
FEATURE_IMPORTANCE_IMAGE = DATA_DIR / "feature_importance.png"
NN_RESULTS = DATA_DIR / "nn_results.csv"
NN_CURVE_IMAGE = DATA_DIR / "nn_training_validation_curve.png"

SAMPLE_CLAIMS = [
    "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
    "Roman Atwood is a content creator.",
    "Adrienne Bailon is an accountant.",
    "There is a movie called The Hunger Games.",
    "Stranger Things is set in Bloomington, Indiana.",
    "Stranger than Fiction is a film.",
    "Chris Hemsworth appeared in A Perfect Getaway.",
    "The Silence of the Lambs was a film starring Scott Glenn.",
    "Peggy Sue Got Married is a Egyptian film released in 1986.",
    "Tetris has sold millions of physical copies.",
    "The Ten Commandments is an epic film.",
    "Homeland is an American television spy thriller based on the Israeli television series Prisoners of War.",
]


def map_fever_to_truth(label):
    if label == "SUPPORTS":
        return "TRUE"
    if label == "REFUTES":
        return "FALSE"
    return "UNKNOWN"


def infer_model_name(df, path):
    if "llm_model" in df.columns:
        non_null = df["llm_model"].dropna()
        if not non_null.empty:
            return str(non_null.iloc[0])
    return path.stem.replace("scored_claims_", "").replace("__", "/")


def parse_evidence(raw_value):
    try:
        parsed = ast.literal_eval(str(raw_value))
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
    except Exception:
        pass
    return pd.DataFrame()


def compute_truthlens_summary(df, path):
    total_rows = len(df)
    binary_df = df[df["final_verdict"].isin(["Hallucination", "Not Hallucination"])].copy()
    binary_rows = len(binary_df)

    summary = {
        "model": infer_model_name(df, path),
        "file": path.name,
        "total_rows": total_rows,
        "binary_rows": binary_rows,
        "excluded_rows": total_rows - binary_rows,
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
    binary_df["predicted_hallucination"] = (binary_df["final_verdict"] == "Hallucination").astype(int)

    tp = int(((binary_df["true_hallucination"] == 1) & (binary_df["predicted_hallucination"] == 1)).sum())
    fp = int(((binary_df["true_hallucination"] == 0) & (binary_df["predicted_hallucination"] == 1)).sum())
    fn = int(((binary_df["true_hallucination"] == 1) & (binary_df["predicted_hallucination"] == 0)).sum())
    correct = int((binary_df["true_hallucination"] == binary_df["predicted_hallucination"]).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    summary.update({
        "llm_hallucination_rate": round(float(binary_df["true_hallucination"].mean()), 4),
        "system_flag_rate": round(float(binary_df["predicted_hallucination"].mean()), 4),
        "accuracy": round(correct / binary_rows, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    })
    return summary


@st.cache_data
def load_truthlens_summary():
    rows = []
    for path in NAMED_SCORED_FILES:
        if path.exists():
            rows.append(compute_truthlens_summary(pd.read_csv(path), path))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("f1", ascending=False)


@st.cache_data
def load_csv(path):
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_scored_claims(path):
    df = pd.read_csv(path)
    df.insert(0, "row_id", range(len(df)))
    return df


@st.cache_resource
def load_final_model():
    if not FINAL_MODEL_PATH.exists():
        return None
    return joblib.load(FINAL_MODEL_PATH)


def percentage_table(df, columns):
    display = df.copy()
    for column in columns:
        if column in display.columns:
            display[column] = (display[column].astype(float) * 100).round(2)
    return display


def render_metric_row(row):
    cols = st.columns(6)
    cols[0].metric("Coverage", f"{row['coverage'] * 100:.2f}%")
    cols[1].metric("Accuracy", f"{row['accuracy'] * 100:.2f}%")
    cols[2].metric("Precision", f"{row['precision'] * 100:.2f}%")
    cols[3].metric("Recall", f"{row['recall'] * 100:.2f}%")
    cols[4].metric("F1", f"{row['f1'] * 100:.2f}%")
    cols[5].metric("LLM Hallucination Rate", f"{row['llm_hallucination_rate'] * 100:.2f}%")


def plot_metric_bars(df, metric_columns, title, ylabel="Score"):
    if df.empty:
        return None

    plot_df = df[["model", *metric_columns]].copy()
    plot_df = plot_df.melt(id_vars="model", var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    sns.barplot(data=plot_df, x="model", y="score", hue="metric", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, float(plot_df["score"].max()) + 0.08))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="", ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_single_metric(df, metric_column, title, color="#3b82f6"):
    if df.empty:
        return None

    plot_df = df.sort_values(metric_column, ascending=False).copy()
    fig, ax = plt.subplots(figsize=(8.9, 4.2))
    bars = ax.bar(plot_df["model"], plot_df[metric_column], color=color)
    ax.bar_label(bars, labels=[f"{v * 100:.1f}%" for v in plot_df[metric_column]], padding=3, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(metric_column.replace("_", " ").title())
    ax.set_ylim(0, max(1.0, float(plot_df[metric_column].max()) + 0.1))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def build_model_features(row, model_bundle):
    feature_columns = model_bundle["features"]
    features = {}

    evidence_features = extract_evidence_features(row)
    features.update(evidence_features.to_dict())

    nli_map = {"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2}
    llm_label = str(row.get("llm_label", ""))
    nli_label = str(row.get("nli_label", "NEUTRAL"))

    features.update({
        "nli_confidence": float(row.get("nli_confidence", 0.0) or 0.0),
        "nli_label_enc": nli_map.get(nli_label, 1),
        "llm_label_enc": 1 if llm_label == "TRUE" else 0,
        "llm_nli_agree": int(
            (llm_label == "TRUE" and nli_label == "ENTAILMENT")
            or (llm_label == "FALSE" and nli_label == "CONTRADICTION")
        ),
        "claim_length": len(str(row.get("claim", "")).split()),
        "answer_length": len(str(row.get("llm_answer", "")).split()),
    })

    llm_model = str(row.get("llm_model", ""))
    for column in feature_columns:
        if column.startswith("model_"):
            features[column] = 1 if column == f"model_{llm_model}" else 0

    return pd.DataFrame([{column: features.get(column, 0.0) for column in feature_columns}])


def predict_with_final_model(row):
    model_bundle = load_final_model()
    if model_bundle is None:
        return None, None

    X = build_model_features(row, model_bundle)
    model = model_bundle["model"]
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return pred, float(proba[1])


def claim_option_label(row):
    claim = str(row.get("claim", ""))
    verdict = str(row.get("final_verdict", ""))
    if len(claim) > 115:
        claim = claim[:112] + "..."
    return f"{row.get('row_id', '')} | {verdict} | {claim}"


def render_sample_claims():
    with st.expander("Sample claims to test"):
        st.dataframe(
            pd.DataFrame({"claim": SAMPLE_CLAIMS}),
            use_container_width=True,
            hide_index=True,
        )


st.set_page_config(page_title="TruthLens Dashboard", layout="wide")
sns.set_theme(style="whitegrid", palette="deep")

st.markdown(
    """
    <style>
    .stApp {
        background: #111827;
        color: #e5e7eb;
        font-family: "Segoe UI", Inter, system-ui, sans-serif;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
        color: #f8fafc;
    }
    .hero {
        border: 1px solid rgba(148, 163, 184, 0.32);
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 58%, #334155 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: white;
        font-size: 2.1rem;
        margin: 0 0 0.25rem 0;
    }
    .hero p {
        color: #dbeafe;
        margin: 0;
        font-size: 1rem;
    }
    .insight-card {
        border-left: 4px solid #38bdf8;
        background: #1f2937;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.75rem 0 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
        color: #e5e7eb;
    }
    .insight-card strong,
    .insight-card code {
        color: #f8fafc;
    }
    div[data-testid="stMetric"] {
        background: #1f2937;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.22);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #f8fafc !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: #cbd5e1 !important;
    }
    div[data-testid="stDataFrame"] {
        background: #f8fafc;
        border-radius: 8px;
    }
    .stMarkdown, .stCaption, p, span, label {
        color: #e5e7eb;
    }
    div[data-testid="stDataFrame"] span,
    div[data-testid="stDataFrame"] p,
    div[data-testid="stDataFrame"] label {
        color: #0f172a !important;
    }
    div[data-testid="stAlert"] {
        background: #1f2937;
        color: #f8fafc;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 8px;
    }
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div {
        color: #f8fafc !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.55rem 0.9rem;
        color: #cbd5e1;
    }
    .stTabs [aria-selected="true"] {
        background: #1f2937;
        color: #f8fafc;
    }
    .stSelectbox label,
    .stTextInput label {
        color: #f8fafc !important;
    }
    .stExpander {
        background: #1f2937;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 8px;
    }
    .stExpander p,
    .stExpander span,
    .stExpander summary {
        color: #f8fafc !important;
    }
    .stExpander svg,
    button svg,
    [data-testid="stToolbar"] svg,
    [data-testid="stElementToolbar"] svg,
    [data-testid="StyledFullScreenButton"] svg {
        color: #f8fafc !important;
        fill: #f8fafc !important;
        stroke: #f8fafc !important;
    }
    button,
    [data-testid="stToolbar"] button,
    [data-testid="stElementToolbar"] button {
        color: #f8fafc !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>TruthLens Dashboard</h1>
        <p>Evidence-backed hallucination detection | Multi-LLM analytics | Trained meta-classifier results | Failure analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_truthlens, tab_ml, tab_failure, tab_inspect, tab_predict = st.tabs([
    "TruthLens Metrics",
    "ML Classifier",
    "Failure Analysis",
    "Inspect Evidence",
    "Predict Saved Claim",
])

with tab_truthlens:
    st.subheader("TruthLens System Performance")
    st.markdown(
        """
        <div class="insight-card">
        <strong>Selective evaluation:</strong> these metrics measure the rule-based TruthLens system only when retrieved evidence and NLI produced a binary verdict.
        Coverage is therefore part of the result, not a hidden detail.
        </div>
        """,
        unsafe_allow_html=True,
    )
    truthlens_df = load_truthlens_summary()

    if truthlens_df.empty:
        st.info("No scored LLM files found.")
    else:
        best = truthlens_df.iloc[0]
        st.markdown(f"Best F1 among saved LLM runs: **{best['model']}**")
        render_metric_row(best)

        display = percentage_table(
            truthlens_df,
            ["coverage", "llm_hallucination_rate", "system_flag_rate", "accuracy", "precision", "recall", "f1"],
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.pyplot(
                plot_metric_bars(
                    truthlens_df,
                    ["coverage", "accuracy", "precision", "recall", "f1"],
                    "TruthLens Selective Evaluation By LLM",
                ),
                use_container_width=True,
            )
        with chart_right:
            st.pyplot(
                plot_single_metric(
                    truthlens_df,
                    "llm_hallucination_rate",
                    "Observed LLM Hallucination Rate",
                    color="#dc2626",
                ),
                use_container_width=True,
            )

        st.caption(
            "These metrics evaluate the rule-based TruthLens system only on binary verdict rows "
            "where the system produced Hallucination or Not Hallucination."
        )

with tab_ml:
    st.subheader("Trained Full-Coverage Hallucination Classifier")
    st.markdown(
        """
        <div class="insight-card">
        <strong>Two-stage model story:</strong> the first table shows untuned baseline ML comparisons from <code>model.py</code>.
        Hyperparameter tuning then selects a tuned Gradient Boosting classifier as the final deployed meta-classifier.
        </div>
        """,
        unsafe_allow_html=True,
    )
    ml_df = load_csv(ML_RESULTS)
    per_llm_df = load_csv(ML_PER_LLM_RESULTS)
    confusion_df = load_csv(ML_CONFUSION_RESULTS)
    tuning_df = load_csv(TUNING_BEST_RESULTS)

    if ml_df.empty:
        st.info("Run scripts/model.py to generate trained-model results.")
    else:
        display = percentage_table(
            ml_df,
            [
                "accuracy",
                "precision_hallucinated",
                "recall_hallucinated",
                "f1_hallucinated",
                "f1_not_hallucinated",
                "f1_macro",
            ],
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.pyplot(
            plot_metric_bars(
                ml_df,
                ["accuracy", "f1_hallucinated", "f1_macro"],
                "Untuned Baseline Models",
            ),
            use_container_width=True,
        )

        if not per_llm_df.empty:
            st.markdown("**Best Model Breakdown By LLM**")
            per_display = percentage_table(
                per_llm_df,
                [
                    "accuracy",
                    "precision_hallucinated",
                    "recall_hallucinated",
                    "f1_hallucinated",
                    "f1_not_hallucinated",
                    "f1_macro",
                ],
            )
            st.dataframe(per_display, use_container_width=True, hide_index=True)

        st.markdown("**Hyperparameter Tuning Results**")
        if tuning_df.empty:
            st.info("Run scripts/model_tuning.py to generate hyperparameter tuning results.")
        else:
            tuning_display = percentage_table(
                tuning_df,
                [
                    "validation_accuracy",
                    "validation_precision_hallucinated",
                    "validation_recall_hallucinated",
                    "validation_f1_hallucinated",
                    "validation_f1_macro",
                    "test_accuracy",
                    "test_precision_hallucinated",
                    "test_recall_hallucinated",
                    "test_f1_hallucinated",
                    "test_f1_macro",
                ],
            )
            st.dataframe(tuning_display, use_container_width=True, hide_index=True)

            best_tuned = tuning_df.sort_values("test_f1_macro", ascending=False).iloc[0]
            st.markdown(
                f"""
                <div class="insight-card">
                <strong>Final selected ML classifier:</strong> {best_tuned['model']} with
                <code>{best_tuned['best_config']}</code>. It achieved
                <strong>{float(best_tuned['test_accuracy']) * 100:.2f}%</strong> test accuracy and
                <strong>{float(best_tuned['test_f1_macro']) * 100:.2f}%</strong> test macro F1.
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not confusion_df.empty:
            st.markdown("**Numeric Confusion Matrices**")
            st.dataframe(confusion_df, use_container_width=True, hide_index=True)

        if TUNING_PLOT.exists():
            st.markdown("**Hyperparameter Tuning Comparison**")
            st.caption("Validation macro F1 was used to compare configurations before selecting tuned Gradient Boosting as the final ML classifier.")
            tune_left, tune_mid, tune_right = st.columns([0.16, 0.68, 0.16])
            with tune_mid:
                st.image(str(TUNING_PLOT), caption="Top validation configurations by model family", use_container_width=True)

        if CONFUSION_IMAGE.exists():
            st.markdown("**Confusion Matrix Comparison**")
            st.caption("These matrices compare the rule-based baseline with the untuned ML classifiers on the full-coverage hallucination task.")
            st.image(str(CONFUSION_IMAGE), caption="Rule-based vs ML classifier confusion matrices", use_container_width=True)

with tab_failure:
    st.subheader("Failure Analysis")
    st.markdown(
        """
        <div class="insight-card">
        The project documents failed or limited approaches: a neural network that did not beat classical ML on tabular features,
        and deeper retrieval that introduced noisy evidence instead of improving verification.
        </div>
        """,
        unsafe_allow_html=True,
    )
    nn_df = load_csv(NN_RESULTS)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Neural Network Experiment**")
        if nn_df.empty:
            st.info("Run scripts/model_nn.py to generate NN failure-analysis results.")
        else:
            nn_display = percentage_table(
                nn_df,
                [
                    "best_val_f1",
                    "final_train_f1",
                    "final_val_f1",
                    "test_accuracy",
                    "test_precision_hallucinated",
                    "test_recall_hallucinated",
                    "test_f1_hallucinated",
                    "generalization_gap_train_minus_val_f1",
                ],
            )
            st.dataframe(nn_display, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Retrieval Depth Experiment**")
        st.dataframe(
            pd.DataFrame([
                {"top_k": 3, "accuracy": "79.28%", "coverage": "30.40%", "outcome": "Slight improvement over k=5"},
                {"top_k": 5, "accuracy": "79.14%", "coverage": "30.20%", "outcome": "Selected balanced baseline"},
                {"top_k": 10, "accuracy": "79.00%", "coverage": "30.00%", "outcome": "No meaningful improvement; more retrieval cost/noise"},
            ]),
            use_container_width=True,
            hide_index=True,
        )

    if NN_CURVE_IMAGE.exists():
        st.markdown("**Neural Network Training vs Validation Curve**")
        st.caption(
            "Training keeps improving while validation performance becomes less stable, which supports the report finding that the NN did not generalize better than classical ML on this tabular dataset."
        )
        curve_left, curve_mid, curve_right = st.columns([0.12, 0.76, 0.12])
        with curve_mid:
            st.image(str(NN_CURVE_IMAGE), caption="NN learning curve", use_container_width=True)


with tab_inspect:
    st.subheader("Inspect Saved Claims And Evidence")
    render_sample_claims()
    available_files = [path for path in NAMED_SCORED_FILES if path.exists()]

    if not available_files:
        st.info("No scored files found.")
    else:
        selected_file = st.selectbox("Scored LLM run", available_files, format_func=lambda path: path.name)
        df = load_scored_claims(selected_file)

        query = st.text_input("Filter claims", key="inspect_filter")
        filtered = df[df["claim"].str.contains(query, case=False, na=False)] if query else df

        if filtered.empty:
            st.warning("No saved claim matches that search. Use one of the sample claims, or search a shorter phrase from the dataset.")
        else:
            selected_index = st.selectbox(
                "Claim row",
                filtered.index.tolist(),
                key="inspect_row",
                format_func=lambda idx: claim_option_label(filtered.loc[idx]),
            )
            row = filtered.loc[selected_index]

            cols = st.columns(5)
            cols[0].metric("FEVER Label", row.get("label", ""))
            cols[1].metric("LLM Label", row.get("llm_label", ""))
            cols[2].metric("NLI Label", row.get("nli_label", ""))
            cols[3].metric("Verdict", row.get("final_verdict", ""))
            cols[4].metric("NLI Conf.", row.get("nli_confidence", ""))

            st.markdown("**Claim**")
            st.write(row.get("claim", ""))
            st.markdown("**LLM Answer**")
            st.write(row.get("llm_answer", ""))
            st.markdown("**Top Evidence**")
            st.write(row.get("top_evidence", ""))

            evidence_df = parse_evidence(row.get("all_retrieved_evidence", "[]"))
            if not evidence_df.empty:
                st.markdown("**Retrieved Evidence Chunks**")
                st.dataframe(evidence_df, use_container_width=True, hide_index=True)

with tab_predict:
    st.subheader("Saved-Claim ML Prediction Demo")
    st.caption(
        "This uses the tuned Gradient Boosting classifier on claims that have already passed through "
        "LLM answering, retrieval, and NLI. Arbitrary new claims require rerunning that full pipeline first."
    )
    render_sample_claims()

    available_files = [path for path in NAMED_SCORED_FILES if path.exists()]
    if not available_files:
        st.info("No scored files found.")
    elif load_final_model() is None:
        st.info("Run scripts/model_tuning.py to train and save the tuned Gradient Boosting model.")
    else:
        selected_file = st.selectbox("Scored LLM run", available_files, format_func=lambda path: path.name, key="predict_file")
        df = load_scored_claims(selected_file)

        valid_df = df[df["llm_label"].isin(["TRUE", "FALSE"])].copy()
        query = st.text_input("Search claim", key="predict_filter")
        filtered = valid_df[valid_df["claim"].str.contains(query, case=False, na=False)] if query else valid_df

        if filtered.empty:
            st.warning("No saved scored claim matches that search. The ML demo can only use claims already processed by the pipeline.")
        else:
            selected_index = st.selectbox(
                "Claim row",
                filtered.index.tolist(),
                key="predict_row",
                format_func=lambda idx: claim_option_label(filtered.loc[idx]),
            )
            row = filtered.loc[selected_index]

            pred, probability = predict_with_final_model(row)
            label = "Hallucinated" if pred == 1 else "Not Hallucinated"

            cols = st.columns(4)
            cols[0].metric("Tuned GB Prediction", label)
            cols[1].metric("Hallucination Probability", f"{probability * 100:.2f}%")
            cols[2].metric("Rule Verdict", row.get("final_verdict", ""))
            cols[3].metric("FEVER Label", row.get("label", ""))

            st.markdown("**Claim**")
            st.write(row.get("claim", ""))
            st.markdown("**LLM Answer**")
            st.write(row.get("llm_answer", ""))
            st.markdown("**Evidence Used By TruthLens**")
            st.write(row.get("top_evidence", ""))
