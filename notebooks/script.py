import os, random, time, math, json, numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 19980311
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR      = Path('../data/')          # edit if your files are elsewhere
TRAIN_NPZ     = DATA_DIR / 'training_data.npz' #à modifier
TRAIN_LABEL_CSV = DATA_DIR / 'training_labels.csv' #à modifier
EVAL_NPZ      = DATA_DIR / 'evaluation_data.npz' #à modifier
SUBMISSION_CSV = DATA_DIR / 'submissionV2.csv'
N_SPLITS      = 7

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
class TensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = torch.from_numpy(x)
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx]) if self.y is not None else self.x[idx]

class CNN1D(nn.Module):
    def __init__(self, n_feat: int = 77, n_filt: int = 96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, n_filt, 3, padding=1),
            nn.BatchNorm1d(n_filt), nn.ReLU(),
            nn.Conv1d(n_filt, n_filt, 3, padding=1),
            nn.BatchNorm1d(n_filt), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(n_filt, 1)
    def forward(self, x):              # x: (B, T, F)
        x = x.permute(0, 2, 1)         # -> (B, F, T)
        x = self.conv(x).squeeze(-1)   # -> (B, n_filt)
        return self.fc(x).squeeze(-1)  # logits

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def summary_stats(x: np.ndarray) -> np.ndarray:
    """Return mean, std, min, max, and slope along time axis (axis=1)."""
    mean  = x.mean(1)
    std   = x.std(1)
    mn    = x.min(1)
    mx    = x.max(1)
    slope = x[:, -1] - x[:, 0]
    return np.hstack([mean, std, mn, mx, slope])


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Grid‑search threshold ∈ (0.05,0.95) for max F1."""
    thresholds = np.linspace(0.05, 0.95, 37)
    f1_scores = [f1_score(y_true, y_prob >= t) for t in thresholds]
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def train_cnn_fold(x_train: np.ndarray, y_train: np.ndarray,
                   x_val: np.ndarray,   y_val: np.ndarray,
                   epochs: int = 25, batch: int = 256) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = CNN1D().to(device)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=3e-4)

    tr_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=batch, shuffle=True)
    va_dl = DataLoader(TensorDataset(x_val,   y_val),   batch_size=batch)

    best_auc = 0.
    best_weights = model.state_dict()  # ensure dict‑like even if no epoch improves
    patience = 5; bad_epochs = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            optimizer.zero_grad()
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

        # ----- validation -----
        model.eval(); y_prob = []
        with torch.no_grad():
            for xb, _ in va_dl:
                p = torch.sigmoid(model(xb.to(device))).cpu()
                y_prob.append(p)
        y_prob = torch.cat(y_prob).numpy()
        auc = roc_auc_score(y_val, y_prob)
        if auc > best_auc:
            best_auc = auc; bad_epochs = 0
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    # load best weights
    model.load_state_dict(best_weights)

    # final predictions
    model.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(model(torch.from_numpy(x_val).to(device))).cpu().numpy()
        train_pred = torch.sigmoid(model(torch.from_numpy(x_train).to(device))).cpu().numpy()
    return model, val_pred, train_pred


def train_lightgbm(x_tr: np.ndarray, y_tr: np.ndarray) -> lgb.Booster:
    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    params = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
        'num_leaves': 64, 'min_data_in_leaf': 50,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'seed': SEED,
        'scale_pos_weight': pos_weight,
    }
    dtrain = lgb.Dataset(x_tr, y_tr)
    gbm = lgb.train(params, dtrain, num_boost_round=800)
    return gbm


def train_catboost(x_tr: np.ndarray, y_tr: np.ndarray) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function='Logloss', eval_metric='AUC', learning_rate=0.05,
        depth=6, l2_leaf_reg=3.0, random_seed=SEED, verbose=False,
        iterations=600, class_weights=[1.0, (len(y_tr)/y_tr.sum())])
    model.fit(x_tr, y_tr)
    return model

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # ----- load data -----
    with np.load(TRAIN_NPZ, allow_pickle=True) as f:
        X = f['data'].astype(np.float32)
        feat_labels = f['feature_labels']
    y = pd.read_csv(TRAIN_LABEL_CSV)['Label'].values

    with np.load(EVAL_NPZ, allow_pickle=True) as f:
        X_eval = f['data'].astype(np.float32)

    print(f"Loaded train shape {X.shape}  | positives {y.sum()/len(y):.2%}")

    # ----- imputation + scaling -----
    # 1. median‑impute each biomarker (handles missing lab values)
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()

    X_2d        = X.reshape(-1, X.shape[-1])   # (N*T, 77)
    X_eval_2d   = X_eval.reshape(-1, X_eval.shape[-1])

    X_2d        = imputer.fit_transform(X_2d)
    X_eval_2d   = imputer.transform(X_eval_2d)

    X_2d        = scaler.fit_transform(X_2d)
    X_eval_2d   = scaler.transform(X_eval_2d)

    # Replace any NaN / ±Inf produced by constant biomarkers (std=0)
    X_2d      = np.nan_to_num(X_2d, nan=0.0, posinf=0.0, neginf=0.0)
    X_eval_2d = np.nan_to_num(X_eval_2d, nan=0.0, posinf=0.0, neginf=0.0)

    X       = X_2d.reshape(X.shape)
    X_eval  = X_eval_2d.reshape(X_eval.shape)

    # ----- CV setup -----
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_prob = np.zeros(len(y), dtype=float)
    eval_prob = np.zeros(len(X_eval), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{N_SPLITS}  (train {len(train_idx)}, val {len(val_idx)})")

        # split
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # ---------------- CNN ----------------
        cnn_model, val_cnn, _ = train_cnn_fold(X_tr, y_tr, X_val, y_val)
        with torch.no_grad():
            device = next(cnn_model.parameters()).device
            eval_cnn = []
            for i in range(0, len(X_eval), 512):
                xb = torch.from_numpy(X_eval[i:i+512]).to(device)
                eval_cnn.append(torch.sigmoid(cnn_model(xb)).cpu())
            eval_cnn = torch.cat(eval_cnn).numpy()

        # ---------------- LightGBM -----------
        X_tr_stat, X_val_stat, X_eval_stat = summary_stats(X_tr), summary_stats(X_val), summary_stats(X_eval)
        gbm = train_lightgbm(X_tr_stat, y_tr)
        val_gbm = gbm.predict(X_val_stat, num_iteration=gbm.best_iteration)
        eval_gbm = gbm.predict(X_eval_stat, num_iteration=gbm.best_iteration)

        # ---------------- CatBoost -----------
        X_tr_flat = X_tr.reshape(len(X_tr), -1); X_val_flat = X_val.reshape(len(X_val), -1)
        X_eval_flat = X_eval.reshape(len(X_eval), -1)
        cb = train_catboost(X_tr_flat, y_tr)
        val_cb = cb.predict_proba(X_val_flat)[:, 1]
        eval_cb = cb.predict_proba(X_eval_flat)[:, 1]

        # ---------------- Blend --------------
        val_blend = (val_cnn + val_gbm + val_cb) / 3
        eval_blend_fold = (eval_cnn + eval_gbm + eval_cb) / 3

        oof_prob[val_idx] = val_blend
        eval_prob += eval_blend_fold / N_SPLITS

        fold_auc = roc_auc_score(y_val, val_blend)
        fold_thr, fold_f1 = best_threshold(y_val, val_blend)
        print(f"Fold AUC={fold_auc:.4f} | best F1={fold_f1:.4f} @thr={fold_thr:.3f}")

    # ----- global threshold & metrics -----
    thr_star, f1_star = best_threshold(y, oof_prob)
    auc_star = roc_auc_score(y, oof_prob)
    print(f"\nOOF AUC={auc_star:.4f} | OOF F1={f1_star:.4f} @thr={thr_star:.3f}")

    # ----- submission -----
    submission = pd.DataFrame({
        'Id': np.arange(len(eval_prob)),
        'Label': (eval_prob >= thr_star).astype(int)   # binary as per competition spec
        # For probabilistic submission replace by: eval_prob
    })
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"Wrote {SUBMISSION_CSV}  (rows {len(submission)})")

