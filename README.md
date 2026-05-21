# Ticket Classifier — Neural Network From Scratch

A PyTorch feedforward network that classifies support ticket priority (Low / Medium / High) from six numeric features — built for WhatsApp chatbot routing at Deepija, with no `model.fit()` shortcuts.

## Architecture

```
Input (6 features)
    |
    v
[Linear 6->64] -> GELU -> Dropout(0.3)
    |
    v
[Linear 64->32] -> GELU -> Dropout(0.3)
    |
    v
[Linear 32->16] -> GELU -> Dropout(0.3)
    |
    v
[Linear 16->3]  -> logits (CrossEntropyLoss)
```

**Features:** wait time, sentiment, repeat contacts, billing flag, message count, hour of day.

**Training:** AdamW (lr=3e-3, weight_decay=1e-4), cosine LR schedule, early stopping (patience=15), stratified 70/15/15 split, `StandardScaler` fit on train only.

## Quick start

```bash
cd ticket-classifier
pip install -r requirements.txt
python train.py
```

Artifacts are written to `saved_model/`:

| File | Purpose |
|------|---------|
| `model_weights.pt` | Best checkpoint (early stopping) |
| `scaler.pkl` | Feature scaling for new tickets |
| `config.json` | Architecture hyperparameters |
| `metrics.json` | Test-set evaluation summary |

## Inference

After training, call `load_and_predict()` from `train.py` (or import it in FastAPI):

```python
from train import load_and_predict

result = load_and_predict({
    "wait_time": 95,
    "sentiment": -0.8,
    "repeat_contacts": 3,
    "is_billing": 1,
    "message_count": 8,
    "hour_of_day": 14,
})

print(result["priority"])    # High
print(result["confidence"])  # e.g. 0.9288
print(result["all_probs"])    # {'Low': ..., 'Medium': ..., 'High': ...}
```

## Results (test set, n=150)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Low | 0.99 | 0.97 | 0.98 | 77 |
| Medium | 0.95 | 0.97 | 0.96 | 58 |
| High | 0.93 | 0.93 | 0.93 | 15 |
| **Overall** | — | — | **accuracy 0.97** | 150 |

Macro F1: **0.96**

> Re-run `python train.py` to refresh metrics; numbers depend on the synthetic seed-42 dataset.

## Month 1 concepts in this repo

| Concept | Where |
|---------|--------|
| Gradient descent | `loss.backward()` + `optimizer.step()` |
| Learning rate | `lr=3e-3` + `CosineAnnealingLR` |
| Loss | `F.cross_entropy` |
| Overfitting | `Dropout(0.3)` + `weight_decay=1e-4` |
| Early stopping | `patience_count` + `best_state` restore |
| Data leakage | Split before `scaler.fit_transform()` |
| Activations | `nn.GELU()` |
| Deployment | `load_and_predict()` + scaler + config |

## Project layout

```
ticket-classifier/
├── train.py
├── requirements.txt
├── README.md
└── saved_model/
    ├── model_weights.pt
    ├── scaler.pkl
    ├── config.json
    └── metrics.json
```

## Production notes

Replace `generate_data()` with real ticket features from your CRM or chat logs. You must ship **weights**, **scaler**, and **config** together — weights alone cannot reproduce preprocessing or layer shapes.
