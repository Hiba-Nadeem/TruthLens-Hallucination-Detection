from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import INPUT_PATHS, RANDOM_STATE, load_combined_data, prepare_features


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

OUTPUT_CURVE = DATA_DIR / "nn_training_validation_curve.png"
OUTPUT_RESULTS = DATA_DIR / "nn_results.csv"
OUTPUT_HISTORY = DATA_DIR / "nn_training_history.csv"
OUTPUT_MODEL = DATA_DIR / "model_hallucination_nn.pt"


class HallucinationMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def make_loader(X, y, batch_size=64, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y.to_numpy(), dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, loss_fn):
    model.eval()
    losses = []
    probs = []
    labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            losses.append(loss.item())
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch_y.cpu().numpy().tolist())

    preds = (np.array(probs) >= 0.5).astype(int)
    labels = np.array(labels).astype(int)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_f1"], label="Train F1")
    axes[1].plot(history["epoch"], history["val_f1"], label="Validation F1")
    axes[1].set_title("F1 Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 hallucinated")
    axes[1].legend()

    plt.suptitle("Neural Network Training vs Validation")
    plt.tight_layout()
    plt.savefig(OUTPUT_CURVE, dpi=150)
    plt.close()


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("Loading combined scored claims...")
    df = load_combined_data(INPUT_PATHS)
    df, X, y, feature_columns = prepare_features(df)

    groups = df["id"].astype(str) if "id" in df.columns else df["claim"].astype(str)

    test_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(test_splitter.split(X, y, groups=groups))

    train_val_groups = groups.iloc[train_val_idx]
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_rel_idx, val_rel_idx = next(val_splitter.split(
        X.iloc[train_val_idx],
        y.iloc[train_val_idx],
        groups=train_val_groups,
    ))

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

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

    train_loader = make_loader(X_train_scaled, y_train, shuffle=True)
    val_loader = make_loader(X_val_scaled, y_val)
    test_loader = make_loader(X_test_scaled, y_test)

    model = HallucinationMLP(input_dim=len(feature_columns))
    positive_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

    history_rows = []
    epochs = 120

    print(f"Training NN on {len(X_train)} rows, validating on {len(X_val)}, testing on {len(X_test)}")
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_metrics = evaluate(model, train_loader, loss_fn)
        val_metrics = evaluate(model, val_loader, loss_fn)

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
        })

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"train_f1={train_metrics['f1']:.4f}, "
                f"val_f1={val_metrics['f1']:.4f}"
            )

    history = pd.DataFrame(history_rows)
    best_epoch_row = history.sort_values("val_f1", ascending=False).iloc[0]
    test_metrics = evaluate(model, test_loader, loss_fn)

    result = pd.DataFrame([{
        "model": "Neural Network MLP",
        "epochs": epochs,
        "best_val_epoch": int(best_epoch_row["epoch"]),
        "best_val_f1": round(float(best_epoch_row["val_f1"]), 4),
        "final_train_f1": round(float(history.iloc[-1]["train_f1"]), 4),
        "final_val_f1": round(float(history.iloc[-1]["val_f1"]), 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_precision_hallucinated": round(test_metrics["precision"], 4),
        "test_recall_hallucinated": round(test_metrics["recall"], 4),
        "test_f1_hallucinated": round(test_metrics["f1"], 4),
        "generalization_gap_train_minus_val_f1": round(
            float(history.iloc[-1]["train_f1"] - history.iloc[-1]["val_f1"]),
            4,
        ),
    }])

    history.to_csv(OUTPUT_HISTORY, index=False)
    result.to_csv(OUTPUT_RESULTS, index=False)
    plot_history(history)
    torch.save({
        "model_state_dict": model.state_dict(),
        "features": feature_columns,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }, OUTPUT_MODEL)

    print("\nNeural network failure-analysis result:")
    print(result.to_string(index=False))
    print("\nSaved artifacts:")
    print(f"  {OUTPUT_RESULTS}")
    print(f"  {OUTPUT_HISTORY}")
    print(f"  {OUTPUT_CURVE}")
    print(f"  {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
