import os, random, time, math, json, numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

SEED = 19980311
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR      = Path('../data/')
TRAIN_NPZ     = DATA_DIR / 'training_data.npz'
TRAIN_LABEL_CSV = DATA_DIR / 'training_labels.csv'
EVAL_NPZ      = DATA_DIR / 'evaluation_data.npz'
SUBMISSION_CSV = DATA_DIR / 'submissionV3_optimized.csv'
N_SPLITS      = 3  # Augmenté pour plus de robustesse

# -----------------------------------------------------------------------------
# Enhanced Dataset and Models
# -----------------------------------------------------------------------------
class TensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx]) if self.y is not None else self.x[idx]

class EnhancedCNN1D(nn.Module):
    def __init__(self, n_feat: int = 77, n_filt: int = 128):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            self._conv_block(n_feat, n_filt//2, 3),
            self._conv_block(n_filt//2, n_filt, 3),
            self._conv_block(n_filt, n_filt, 5),  # Kernel plus large
        ])

        # Attention mechanism simple
        self.attention = nn.MultiheadAttention(n_filt, num_heads=8, batch_first=True)

        # Classification head avec dropout
        self.classifier = nn.Sequential(
            nn.Linear(n_filt, n_filt//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_filt//2, n_filt//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_filt//4, 1)
        )

    def _conv_block(self, in_ch, out_ch, kernel_size):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout1d(0.1)
        )

    def forward(self, x):  # x: (B, T, F)
        x = x.permute(0, 2, 1)  # -> (B, F, T)

        # Convolutions séquentielles avec connexions résiduelles
        for i, block in enumerate(self.conv_blocks):
            if i == 0:
                x = block(x)
            else:
                residual = x
                x = block(x)
                if x.shape == residual.shape:
                    x = x + residual

        # Global average pooling + max pooling
        avg_pool = nn.AdaptiveAvgPool1d(1)(x).squeeze(-1)
        max_pool = nn.AdaptiveMaxPool1d(1)(x).squeeze(-1)
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Réduire la dimension pour l'attention si nécessaire
        if x.shape[1] != self.attention.embed_dim:
            x = nn.Linear(x.shape[1], self.attention.embed_dim).to(x.device)(x)

        # Attention (nécessite une dimension de séquence)
        x = x.unsqueeze(1)  # (B, 1, n_filt)
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(1)  # (B, n_filt)

        return self.classifier(x).squeeze(-1)

# -----------------------------------------------------------------------------
# Enhanced Feature Engineering
# -----------------------------------------------------------------------------
def advanced_stats(x: np.ndarray) -> np.ndarray:
    """Statistiques avancées pour améliorer les features."""
    # Stats de base
    mean = x.mean(1)
    std = x.std(1)
    mn = x.min(1)
    mx = x.max(1)
    median = np.median(x, axis=1)

    # Stats avancées
    q25 = np.percentile(x, 25, axis=1)
    q75 = np.percentile(x, 75, axis=1)
    iqr = q75 - q25

    # Tendances temporelles
    slope = x[:, -1] - x[:, 0]
    trend = np.array([np.polyfit(range(len(row)), row, 1)[0] for row in x])

    # Variabilité
    range_val = mx - mn
    cv = std / (mean + 1e-8)  # Coefficient de variation

    # Asymétrie et aplatissement (approximations)
    skew = np.array([((row - row.mean()) ** 3).mean() / (row.std() ** 3 + 1e-8) for row in x])
    kurt = np.array([((row - row.mean()) ** 4).mean() / (row.std() ** 4 + 1e-8) - 3 for row in x])

    return np.hstack([mean, std, mn, mx, median, q25, q75, iqr,
                     slope, trend, range_val, cv, skew, kurt])

def create_interaction_features(X_stat: np.ndarray, top_k: int = 50) -> np.ndarray:
    """Créer des features d'interaction entre les plus importantes."""
    # Sélectionner les top features (approximation rapide)
    n_features = X_stat.shape[1]
    interactions = []

    # Interactions multiplicatives des top features
    for i in range(min(top_k, n_features)):
        for j in range(i+1, min(top_k, n_features)):
            interactions.append(X_stat[:, i] * X_stat[:, j])

    if interactions:
        return np.column_stack(interactions)
    return np.empty((X_stat.shape[0], 0))

def optimal_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Trouve le seuil optimal pour maximiser F1 en utilisant precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Éviter les cas extrêmes
    valid_idx = (precision > 0.01) & (recall > 0.01)
    if valid_idx.sum() == 0:
        return 0.5, 0.0

    best_idx = np.argmax(f1_scores[valid_idx])
    valid_thresholds = thresholds[valid_idx[:-1]]  # thresholds has one less element

    if len(valid_thresholds) > best_idx:
        best_threshold = float(valid_thresholds[best_idx])
    else:
        best_threshold = 0.5

    best_f1 = float(f1_scores[valid_idx][best_idx])
    return best_threshold, best_f1

# -----------------------------------------------------------------------------
# Enhanced Training Functions
# -----------------------------------------------------------------------------
def train_enhanced_cnn(x_train: np.ndarray, y_train: np.ndarray,
                      x_val: np.ndarray, y_val: np.ndarray,
                      epochs: int = 40, batch: int = 32) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EnhancedCNN1D().to(device)

    # Focal loss pour les données déséquilibrées
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer avec weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    tr_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=batch, shuffle=True)
    va_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=batch)

    best_f1 = 0.
    best_weights = model.state_dict()
    patience = 8; bad_epochs = 0

    for ep in range(epochs):
        model.train()
        train_loss = 0

        for xb, yb in tr_dl:
            optimizer.zero_grad()
            logits = model(xb.to(device))
            loss = criterion(logits, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        y_prob = []
        with torch.no_grad():
            for xb, _ in va_dl:
                p = torch.sigmoid(model(xb.to(device))).cpu()
                y_prob.append(p)
        y_prob = torch.cat(y_prob).numpy()

        # Utiliser F1 comme métrique principale
        _, f1 = optimal_threshold_f1(y_val, y_prob)
        scheduler.step(train_loss)

        if f1 > best_f1:
            best_f1 = f1
            bad_epochs = 0
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    model.load_state_dict(best_weights)

    # Prédictions finales
    model.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(model(torch.from_numpy(x_val).to(device))).cpu().numpy()
        train_pred = torch.sigmoid(model(torch.from_numpy(x_train).to(device))).cpu().numpy()

    return model, val_pred, train_pred

def train_optimized_lightgbm(x_tr: np.ndarray, y_tr: np.ndarray) -> lgb.Booster:
    """LightGBM optimisé pour F1 score."""
    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()

    params = {
        'objective': 'binary',
        'metric': 'None',  # Utiliser une métrique custom
        'learning_rate': 0.03,
        'num_leaves': 64,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'seed': SEED,
        'scale_pos_weight': pos_weight,
        'max_depth': 8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    dtrain = lgb.Dataset(x_tr, y_tr)

    def f1_eval(y_pred, dtrain):
        y_true = dtrain.get_label()
        _, f1 = optimal_threshold_f1(y_true, y_pred)
        return 'f1', f1, True  # True pour "higher_better"

    gbm = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        feval=f1_eval
    )
    return gbm

def train_optimized_catboost(x_tr: np.ndarray, y_tr: np.ndarray) -> CatBoostClassifier:
    """CatBoost optimisé."""
    model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='F1',
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=SEED,
        verbose=False,
        iterations=800,
        early_stopping_rounds=50,
        class_weights=[1.0, (len(y_tr)/y_tr.sum())],
        bootstrap_type='Bernoulli',
        subsample=0.8,
        rsm=0.8,
    )
    model.fit(x_tr, y_tr)
    return model

# -----------------------------------------------------------------------------
# Ensemble avec stacking
# -----------------------------------------------------------------------------
def create_meta_features(cnn_pred, gbm_pred, cb_pred):
    """Créer des meta-features pour le stacking."""
    return np.column_stack([
        cnn_pred, gbm_pred, cb_pred,
        cnn_pred * gbm_pred, cnn_pred * cb_pred, gbm_pred * cb_pred,
        (cnn_pred + gbm_pred + cb_pred) / 3,
        np.maximum(np.maximum(cnn_pred, gbm_pred), cb_pred),
        np.minimum(np.minimum(cnn_pred, gbm_pred), cb_pred)
    ])

# -----------------------------------------------------------------------------
# Main Optimized Pipeline
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Load data
    with np.load(TRAIN_NPZ, allow_pickle=True) as f:
        X = f['data'].astype(np.float32)
        feat_labels = f['feature_labels']
    y = pd.read_csv(TRAIN_LABEL_CSV)['Label'].values

    with np.load(EVAL_NPZ, allow_pickle=True) as f:
        X_eval = f['data'].astype(np.float32)

    print(f"Loaded train shape {X.shape} | positives {y.sum()/len(y):.2%}")

    # Enhanced preprocessing
    # 1. KNN imputation (plus sophistiqué que median)
    imputer = KNNImputer(n_neighbors=5)
    scaler = RobustScaler()  # Plus robuste aux outliers

    X_2d = X.reshape(-1, X.shape[-1])
    X_eval_2d = X_eval.reshape(-1, X_eval.shape[-1])

    X_2d = imputer.fit_transform(X_2d)
    X_eval_2d = imputer.transform(X_eval_2d)

    X_2d = scaler.fit_transform(X_2d)
    X_eval_2d = scaler.transform(X_eval_2d)

    X_2d = np.nan_to_num(X_2d, nan=0.0, posinf=0.0, neginf=0.0)
    X_eval_2d = np.nan_to_num(X_eval_2d, nan=0.0, posinf=0.0, neginf=0.0)

    X = X_2d.reshape(X.shape)
    X_eval = X_eval_2d.reshape(X_eval.shape)

    # CV avec plus de folds
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_prob = np.zeros(len(y), dtype=float)
    eval_prob = np.zeros(len(X_eval), dtype=float)

    # Stockage pour le stacking
    oof_meta = np.zeros((len(y), 3), dtype=float)
    eval_meta = np.zeros((len(X_eval), 3), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{N_SPLITS} (train {len(train_idx)}, val {len(val_idx)})")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Enhanced CNN
        cnn_model, val_cnn, _ = train_enhanced_cnn(X_tr, y_tr, X_val, y_val)
        with torch.no_grad():
            device = next(cnn_model.parameters()).device
            eval_cnn = []
            for i in range(0, len(X_eval), 256):
                xb = torch.from_numpy(X_eval[i:i+256]).to(device)
                eval_cnn.append(torch.sigmoid(cnn_model(xb)).cpu())
            eval_cnn = torch.cat(eval_cnn).numpy()

        # Enhanced statistical features
        X_tr_stat = advanced_stats(X_tr)
        X_val_stat = advanced_stats(X_val)
        X_eval_stat = advanced_stats(X_eval)

        # Feature interactions
        X_tr_interact = create_interaction_features(X_tr_stat, top_k=20)
        X_val_interact = create_interaction_features(X_val_stat, top_k=20)
        X_eval_interact = create_interaction_features(X_eval_stat, top_k=20)

        # Combine features
        if X_tr_interact.shape[1] > 0:
            X_tr_combined = np.hstack([X_tr_stat, X_tr_interact])
            X_val_combined = np.hstack([X_val_stat, X_val_interact])
            X_eval_combined = np.hstack([X_eval_stat, X_eval_interact])
        else:
            X_tr_combined = X_tr_stat
            X_val_combined = X_val_stat
            X_eval_combined = X_eval_stat

        # Optimized LightGBM
        gbm = train_optimized_lightgbm(X_tr_combined, y_tr)
        val_gbm = gbm.predict(X_val_combined, num_iteration=gbm.best_iteration)
        eval_gbm = gbm.predict(X_eval_combined, num_iteration=gbm.best_iteration)

        # Enhanced CatBoost
        X_tr_flat = X_tr.reshape(len(X_tr), -1)
        X_val_flat = X_val.reshape(len(X_val), -1)
        X_eval_flat = X_eval.reshape(len(X_eval), -1)

        cb = train_optimized_catboost(X_tr_flat, y_tr)
        val_cb = cb.predict_proba(X_val_flat)[:, 1]
        eval_cb = cb.predict_proba(X_eval_flat)[:, 1]

        # Store individual predictions for stacking
        oof_meta[val_idx, 0] = val_cnn
        oof_meta[val_idx, 1] = val_gbm
        oof_meta[val_idx, 2] = val_cb

        eval_meta[:, 0] += eval_cnn / N_SPLITS
        eval_meta[:, 1] += eval_gbm / N_SPLITS
        eval_meta[:, 2] += eval_cb / N_SPLITS

        # Ensemble optimisé (pondération basée sur les performances individuelles)
        cnn_f1 = optimal_threshold_f1(y_val, val_cnn)[1]
        gbm_f1 = optimal_threshold_f1(y_val, val_gbm)[1]
        cb_f1 = optimal_threshold_f1(y_val, val_cb)[1]

        # Poids basés sur les performances F1
        total_f1 = cnn_f1 + gbm_f1 + cb_f1 + 1e-8
        w_cnn = cnn_f1 / total_f1
        w_gbm = gbm_f1 / total_f1
        w_cb = cb_f1 / total_f1

        val_blend = w_cnn * val_cnn + w_gbm * val_gbm + w_cb * val_cb
        eval_blend_fold = w_cnn * eval_cnn + w_gbm * eval_gbm + w_cb * eval_cb

        oof_prob[val_idx] = val_blend
        eval_prob += eval_blend_fold / N_SPLITS

        fold_thr, fold_f1 = optimal_threshold_f1(y_val, val_blend)
        fold_auc = roc_auc_score(y_val, val_blend)
        print(f"Fold F1={fold_f1:.4f} | AUC={fold_auc:.4f} | thr={fold_thr:.3f}")
        print(f"Weights: CNN={w_cnn:.3f}, GBM={w_gbm:.3f}, CB={w_cb:.3f}")

    # Final meta-learning (simple linear combination)
    from sklearn.linear_model import LogisticRegression
    meta_model = LogisticRegression(random_state=SEED, class_weight='balanced')
    meta_features_train = create_meta_features(oof_meta[:, 0], oof_meta[:, 1], oof_meta[:, 2])
    meta_features_eval = create_meta_features(eval_meta[:, 0], eval_meta[:, 1], eval_meta[:, 2])

    meta_model.fit(meta_features_train, y)
    oof_meta_pred = meta_model.predict_proba(meta_features_train)[:, 1]
    eval_meta_pred = meta_model.predict_proba(meta_features_eval)[:, 1]

    # Combiner les prédictions de base et méta
    alpha = 0.7  # Poids pour l'ensemble de base
    final_oof = alpha * oof_prob + (1 - alpha) * oof_meta_pred
    final_eval = alpha * eval_prob + (1 - alpha) * eval_meta_pred

    # Métriques finales
    thr_star, f1_star = optimal_threshold_f1(y, final_oof)
    auc_star = roc_auc_score(y, final_oof)

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"OOF F1={f1_star:.4f} | OOF AUC={auc_star:.4f} | thr={thr_star:.3f}")
    print(f"{'='*50}")

    # Submission avec seuil optimisé
    submission = pd.DataFrame({
        'Id': np.arange(len(final_eval)),
        'Label': (final_eval >= thr_star).astype(int)
    })
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"Wrote {SUBMISSION_CSV} (rows {len(submission)})")

    # Statistiques sur les prédictions
    print(f"Predicted positives: {submission['Label'].sum()}/{len(submission)} ({submission['Label'].mean():.2%})")