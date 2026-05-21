import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
np.random.seed(42)

# ════════════════════════════════════════════════════════
# 1. DATASET
# ════════════════════════════════════════════════════════


class TicketDataset(Dataset):
    """
    PyTorch Dataset wraps your numpy arrays.
    DataLoader uses this to fetch mini-batches automatically.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_data(n=1000):
    """Simulate support ticket features — replace with real data in production."""
    np.random.seed(42)
    wait = np.random.exponential(30, n).clip(1, 300)
    sent = np.random.uniform(-1, 1, n)
    repeat = np.random.randint(0, 5, n)
    bill = np.random.randint(0, 2, n)
    msgs = np.random.randint(1, 20, n)
    hour = np.random.randint(0, 24, n)

    X = np.column_stack([wait, sent, repeat, bill, msgs, hour])

    y = []
    for i in range(n):
        score = (wait[i] > 60) * 2 + (sent[i] < -0.5) * 2 + (repeat[i] > 2) * 1 + (bill[i] == 1) * 1
        y.append(2 if score >= 4 else 1 if score >= 2 else 0)

    return X, np.array(y)


# ════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════


class TicketClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden_dims=None, num_classes=3, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


# ════════════════════════════════════════════════════════
# 3. TRAINING LOOP
# ════════════════════════════════════════════════════════


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(X_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)

        total_loss += loss.item() * len(X_batch)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(X_batch)

    return total_loss / total, correct / total


# ════════════════════════════════════════════════════════
# 4. MAIN TRAINING PIPELINE
# ════════════════════════════════════════════════════════


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y = generate_data(1000)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(TicketDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TicketDataset(X_val, y_val), batch_size=64)
    test_loader = DataLoader(TicketDataset(X_test, y_test), batch_size=64)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    model = TicketClassifier(input_dim=6, hidden_dims=[64, 32, 16], num_classes=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_loss = float("inf")
    best_state = None
    patience = 15
    patience_count = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n{'Epoch':>5} {'Train Loss':>10} {'Val Loss':>10} {'Val Acc':>8} {'LR':>10}")
    print("-" * 50)

    for epoch in range(1, 101):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"{epoch:>5} {train_loss:>10.4f} {val_loss:>10.4f} {val_acc:>8.3f} {lr:>10.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
                break

    model.load_state_dict(best_state)

    print("\n-- Final evaluation on test set --")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.tolist())

    report = classification_report(
        all_labels, all_preds, target_names=["Low", "Medium", "High"], output_dict=True
    )
    print(
        classification_report(all_labels, all_preds, target_names=["Low", "Medium", "High"])
    )

    os.makedirs("saved_model", exist_ok=True)

    torch.save(best_state, "saved_model/model_weights.pt")
    joblib.dump(scaler, "saved_model/scaler.pkl")

    config = {"input_dim": 6, "hidden_dims": [64, 32, 16], "num_classes": 3}
    with open("saved_model/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
            for name in ["Low", "Medium", "High"]
        },
        "epochs_trained": len(history["train_loss"]),
        "parameters": total_params,
    }
    with open("saved_model/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: saved_model/model_weights.pt")
    print("Saved: saved_model/scaler.pkl")
    print("Saved: saved_model/config.json")
    print("Saved: saved_model/metrics.json")

    return model, scaler, config


# ════════════════════════════════════════════════════════
# 5. PRODUCTION INFERENCE
# ════════════════════════════════════════════════════════


def load_and_predict(ticket_features: dict):
    """
    Production inference — loads weights, scaler, and config from disk.
    Call this from a FastAPI endpoint or batch job.
    """
    with open("saved_model/config.json", encoding="utf-8") as f:
        config = json.load(f)

    model = TicketClassifier(**config)
    model.load_state_dict(
        torch.load("saved_model/model_weights.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    scaler = joblib.load("saved_model/scaler.pkl")

    features = np.array(
        [
            [
                ticket_features["wait_time"],
                ticket_features["sentiment"],
                ticket_features["repeat_contacts"],
                ticket_features["is_billing"],
                ticket_features["message_count"],
                ticket_features["hour_of_day"],
            ]
        ]
    )
    features_scaled = scaler.transform(features)
    x_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1)[0]

    labels = ["Low", "Medium", "High"]
    pred_idx = probs.argmax().item()

    return {
        "priority": labels[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "all_probs": {label: round(p.item(), 4) for label, p in zip(labels, probs)},
    }


if __name__ == "__main__":
    model, scaler, config = main()

    print("\n-- Test inference --")
    result = load_and_predict(
        {
            "wait_time": 95,
            "sentiment": -0.8,
            "repeat_contacts": 3,
            "is_billing": 1,
            "message_count": 8,
            "hour_of_day": 14,
        }
    )
    print(f"Priority:   {result['priority']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"All probs:  {result['all_probs']}")
