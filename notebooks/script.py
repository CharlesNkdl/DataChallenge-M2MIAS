import os
import random
import time
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
# KNNImputer removed, SimpleImputer might be used if ffill/bfill isn't enough, but aiming to avoid
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 98
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path('../data/')
TRAIN_NPZ = DATA_DIR / 'training_data.npz'
TRAIN_LABEL_CSV = DATA_DIR / 'training_labels.csv'
EVAL_NPZ = DATA_DIR / 'evaluation_data.npz'
SUBMISSION_CSV = DATA_DIR / 'submissionV2.csv'
N_SPLITS = 5

MY_BIOCHEMICAL_GROUPS = {
    "glucose_markers": ["glucose", "hba1c"],
    "lipid_markers": ["chol", "trig", "hdl", "ldl"],
    "liver_enzymes": ["alt", "ast", "ggt", "alp"],
    "kidney_markers": ["creat", "urea", "egfr"],
}

class TensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx]) if self.y is not None else self.x[idx]

def extract_custom_features(
    X_3d: np.ndarray,
    feature_labels: List[str],
    biochemical_groups: Dict[str, List[str]] = None,
    include_basic_stats: bool = True
) -> Tuple[np.ndarray, List[str]]:
    n_samples, n_timesteps, n_original_features = X_3d.shape
    extracted_features_list = []
    extracted_feature_names = []

    mean_vals = np.nanmean(X_3d, axis=1)

    if include_basic_stats:
        std_vals = np.nanstd(X_3d, axis=1)
        min_vals = np.nanmin(X_3d, axis=1)
        max_vals = np.nanmax(X_3d, axis=1)

        if n_timesteps > 1:
            slope_vals = X_3d[:, -1, :] - X_3d[:, 0, :]
        else:
            slope_vals = np.zeros((n_samples, n_original_features))

        for i, label in enumerate(feature_labels):
            extracted_features_list.append(mean_vals[:, i])
            extracted_feature_names.append(f"{label}_mean")
            extracted_features_list.append(std_vals[:, i])
            extracted_feature_names.append(f"{label}_std")
            extracted_features_list.append(min_vals[:, i])
            extracted_feature_names.append(f"{label}_min")
            extracted_features_list.append(max_vals[:, i])
            extracted_feature_names.append(f"{label}_max")
            extracted_features_list.append(slope_vals[:, i])
            extracted_feature_names.append(f"{label}_slope")

    if biochemical_groups:
        group_means_dict = {}
        for group_name, marker_keywords in biochemical_groups.items():
            group_indices = [
                i for i, label in enumerate(feature_labels)
                if any(keyword.lower() in label.lower() for keyword in marker_keywords)
            ]

            if not group_indices:
                print(f"Warning: No features found for group '{group_name}'. Skipping.")
                continue

            current_group_means = np.nanmean(mean_vals[:, group_indices], axis=1)
            group_means_dict[group_name] = current_group_means

            extracted_features_list.append(current_group_means)
            extracted_feature_names.append(f"group_{group_name}_mean")

        if "glucose_markers" in group_means_dict and "lipid_markers" in group_means_dict:
            g_l_ratio = group_means_dict["glucose_markers"] / (np.abs(group_means_dict["lipid_markers"]) + 1e-9)
            extracted_features_list.append(g_l_ratio)
            extracted_feature_names.append("ratio_glucose_lipid")

            metabolic_score = (group_means_dict["glucose_markers"] + group_means_dict["lipid_markers"]) / 2
            extracted_features_list.append(metabolic_score)
            extracted_feature_names.append("score_metabolic")

        if "liver_enzymes" in group_means_dict and "kidney_markers" in group_means_dict:
            liv_kid_ratio = group_means_dict["liver_enzymes"] / (np.abs(group_means_dict["kidney_markers"]) + 1e-9)
            extracted_features_list.append(liv_kid_ratio)
            extracted_feature_names.append("ratio_liver_kidney")

    if not extracted_features_list:
        return np.array([]).reshape(n_samples, 0), []

    return np.column_stack(extracted_features_list), extracted_feature_names

class CNN1D(nn.Module):
    def __init__(self, n_feat: int = 77, n_filt: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, n_filt, 3, padding=1),
            nn.BatchNorm1d(n_filt),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(n_filt, n_filt, 3, padding=1),
            nn.BatchNorm1d(n_filt),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(n_filt, n_filt//2, 3, padding=1),
            nn.BatchNorm1d(n_filt//2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(n_filt//2, n_filt//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_filt//4, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

def summary_stats(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    slope = x[:, -1] - x[:, 0]
    p25 = np.percentile(x, 25, axis=1)
    p75 = np.percentile(x, 75, axis=1)
    iqr = p75 - p25
    cv = std / (mean + 1e-8)
    trend = np.array([np.polyfit(range(x.shape[1]), row, 1)[0] for row in x])
    return np.column_stack([mean, std, mn, mx, slope, p25, p75, iqr, cv, trend])

def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    y_true_bool = y_true.astype(bool)
    predictions = y_prob[:, None] >= thresholds[None, :]
    tp = (predictions & y_true_bool[:, None]).sum(axis=0)
    fp = (predictions & ~y_true_bool[:, None]).sum(axis=0)
    fn = (~predictions & y_true_bool[:, None]).sum(axis=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])

def train_cnn_fold(x_train: np.ndarray, y_train: np.ndarray,
                           x_val: np.ndarray, y_val: np.ndarray,
                           epochs: int = 20, batch: int = 512) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN1D().to(device)
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    tr_dl = DataLoader(TensorDataset(x_train, y_train),
                      batch_size=batch, shuffle=True,
                      num_workers=2, pin_memory=True if device == 'cuda' else False)
    va_dl = DataLoader(TensorDataset(x_val, y_val),
                      batch_size=batch,
                      num_workers=2, pin_memory=True if device == 'cuda' else False)
    best_auc = 0.
    best_weights = None
    patience = 4
    bad_epochs = 0

    for ep in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in tr_dl:
            optimizer.zero_grad()
            logits = model(xb.to(device, non_blocking=True))
            loss = criterion(logits, yb.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        y_prob_val = []
        with torch.no_grad():
            for xb, _ in va_dl:
                p = torch.sigmoid(model(xb.to(device, non_blocking=True))).cpu()
                y_prob_val.append(p)
        y_prob_val = torch.cat(y_prob_val).numpy()
        val_auc = roc_auc_score(y_val, y_prob_val)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            bad_epochs = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        val_pred = torch.sigmoid(model(torch.from_numpy(x_val).to(device))).cpu().numpy()
        train_pred = torch.sigmoid(model(torch.from_numpy(x_train).to(device))).cpu().numpy()
    return model, val_pred, train_pred

def train_lightgbm(x_tr: np.ndarray, y_tr: np.ndarray,
                           x_val: np.ndarray = None, y_val: np.ndarray = None) -> lgb.Booster:
    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    params = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05,
        'num_leaves': 50, 'min_data_in_leaf': 20, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0.1,
        'lambda_l2': 0.1, 'seed': SEED, 'scale_pos_weight': pos_weight, 'verbose': -1
    }
    dtrain = lgb.Dataset(x_tr, y_tr)
    if x_val is not None and y_val is not None:
        dval = lgb.Dataset(x_val, y_val, reference=dtrain)
        gbm = lgb.train(params, dtrain, valid_sets=[dval],
                       num_boost_round=1000,
                       callbacks=[lgb.early_stopping(50, verbose=-1), lgb.log_evaluation(0)])
    else:
        gbm = lgb.train(params, dtrain, num_boost_round=500)
    return gbm

def train_catboost(x_tr: np.ndarray, y_tr: np.ndarray,
                           x_val: np.ndarray = None, y_val: np.ndarray = None) -> CatBoostClassifier:
    params = {
        'loss_function': 'Logloss', 'eval_metric': 'AUC', 'learning_rate': 0.05,
        'depth': 6, 'l2_leaf_reg': 3.0, 'random_seed': SEED, 'verbose': False,
        'iterations': 800, 'class_weights': [1.0, (len(y_tr) / y_tr.sum())],
        'early_stopping_rounds': 50
    }
    model = CatBoostClassifier(**params)
    if x_val is not None and y_val is not None:
        model.fit(x_tr, y_tr, eval_set=(x_val, y_val), verbose=False)
    else:
        model.fit(x_tr, y_tr)
    return model

if __name__ == '__main__':
    with np.load(TRAIN_NPZ, allow_pickle=True) as f:
        X = f['data'].astype(np.float32)
        feat_labels = f['feature_labels']
    y = pd.read_csv(TRAIN_LABEL_CSV)['Label'].values
    with np.load(EVAL_NPZ, allow_pickle=True) as f:
        X_eval = f['data'].astype(np.float32)

    print(f"Loaded train shape {X.shape}  | positives {y.sum()/len(y):.2%}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_prob = np.zeros(len(y), dtype=float)
    eval_prob_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{N_SPLITS}  (train {len(train_idx)}, val {len(val_idx)})")

        X_tr_fold, X_val_fold = X[train_idx], X[val_idx]
        y_tr_fold, y_val_fold = y[train_idx], y[val_idx]

        X_tr_fold_ti = X_tr_fold.copy()
        for f_idx in range(X_tr_fold_ti.shape[2]):
            feature_slice_df = pd.DataFrame(X_tr_fold_ti[:, :, f_idx])
            feature_slice_df.ffill(axis=1, inplace=True)
            feature_slice_df.bfill(axis=1, inplace=True)
            X_tr_fold_ti[:, :, f_idx] = feature_slice_df.to_numpy()

        X_val_fold_ti = X_val_fold.copy()
        for f_idx in range(X_val_fold_ti.shape[2]):
            feature_slice_df = pd.DataFrame(X_val_fold_ti[:, :, f_idx])
            feature_slice_df.ffill(axis=1, inplace=True)
            feature_slice_df.bfill(axis=1, inplace=True)
            X_val_fold_ti[:, :, f_idx] = feature_slice_df.to_numpy()

        X_eval_fold_ti = X_eval.copy()
        for f_idx in range(X_eval_fold_ti.shape[2]):
            feature_slice_df = pd.DataFrame(X_eval_fold_ti[:, :, f_idx])
            feature_slice_df.ffill(axis=1, inplace=True)
            feature_slice_df.bfill(axis=1, inplace=True)
            X_eval_fold_ti[:, :, f_idx] = feature_slice_df.to_numpy()

        scaler = StandardScaler()

        X_tr_fold_2d = X_tr_fold_ti.reshape(-1, X_tr_fold_ti.shape[-1])
        X_val_fold_2d = X_val_fold_ti.reshape(-1, X_val_fold_ti.shape[-1])
        X_eval_fold_2d = X_eval_fold_ti.reshape(-1, X_eval_fold_ti.shape[-1])

        X_tr_fold_2d = scaler.fit_transform(X_tr_fold_2d)
        X_val_fold_2d = scaler.transform(X_val_fold_2d)
        X_eval_fold_2d = scaler.transform(X_eval_fold_2d)

        X_tr_fold_2d = np.nan_to_num(X_tr_fold_2d, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_fold_2d = np.nan_to_num(X_val_fold_2d, nan=0.0, posinf=0.0, neginf=0.0)
        X_eval_fold_2d = np.nan_to_num(X_eval_fold_2d, nan=0.0, posinf=0.0, neginf=0.0)

        X_tr_processed = X_tr_fold_2d.reshape(X_tr_fold.shape)
        X_val_processed = X_val_fold_2d.reshape(X_val_fold.shape)
        X_eval_processed_fold = X_eval_fold_2d.reshape(X_eval.shape)

        X_tr_custom_feats, custom_feat_names = extract_custom_features(
            X_tr_processed, feat_labels,
            biochemical_groups=MY_BIOCHEMICAL_GROUPS, include_basic_stats=True
        )
        X_val_custom_feats, _ = extract_custom_features(
            X_val_processed, feat_labels,
            biochemical_groups=MY_BIOCHEMICAL_GROUPS, include_basic_stats=True
        )
        X_eval_custom_feats_fold, _ = extract_custom_features(
            X_eval_processed_fold, feat_labels,
            biochemical_groups=MY_BIOCHEMICAL_GROUPS, include_basic_stats=True
        )

        cnn_model, val_cnn, _ = train_cnn_fold(X_tr_processed, y_tr_fold, X_val_processed, y_val_fold)
        with torch.no_grad():
            device = next(cnn_model.parameters()).device
            eval_cnn_fold_preds = []
            for i in range(0, len(X_eval_processed_fold), 512):
                xb = torch.from_numpy(X_eval_processed_fold[i:i+512]).to(device)
                eval_cnn_fold_preds.append(torch.sigmoid(cnn_model(xb)).cpu())
            eval_cnn_fold = torch.cat(eval_cnn_fold_preds).numpy()

        gbm = train_lightgbm(X_tr_custom_feats, y_tr_fold, X_val_custom_feats, y_val_fold)
        val_gbm = gbm.predict(X_val_custom_feats, num_iteration=gbm.best_iteration if hasattr(gbm, 'best_iteration') else -1)
        eval_gbm_fold = gbm.predict(X_eval_custom_feats_fold, num_iteration=gbm.best_iteration if hasattr(gbm, 'best_iteration') else -1)


        X_tr_flat = X_tr_processed.reshape(len(X_tr_processed), -1)
        X_val_flat = X_val_processed.reshape(len(X_val_processed), -1)
        X_eval_flat_fold = X_eval_processed_fold.reshape(len(X_eval_processed_fold), -1)

        X_tr_catboost_input = np.hstack([X_tr_flat, X_tr_custom_feats])
        X_val_catboost_input = np.hstack([X_val_flat, X_val_custom_feats])
        X_eval_catboost_input_fold = np.hstack([X_eval_flat_fold, X_eval_custom_feats_fold])

        cb = train_catboost(X_tr_catboost_input, y_tr_fold, X_val_catboost_input, y_val_fold)
        val_cb = cb.predict_proba(X_val_catboost_input)[:, 1]
        eval_cb_fold = cb.predict_proba(X_eval_catboost_input_fold)[:, 1]

        val_blend = (val_cnn + val_gbm + val_cb) / 3
        eval_blend_fold = (eval_cnn_fold + eval_gbm_fold + eval_cb_fold) / 3

        oof_prob[val_idx] = val_blend
        eval_prob_list.append(eval_blend_fold)

        fold_auc = roc_auc_score(y_val_fold, val_blend)
        fold_thr, fold_f1 = best_threshold(y_val_fold, val_blend)
        print(f"Fold AUC={fold_auc:.4f} | best F1={fold_f1:.4f} @thr={fold_thr:.3f}")

    thr_star, f1_star = best_threshold(y, oof_prob)
    auc_star = roc_auc_score(y, oof_prob)
    print(f"\nOOF AUC={auc_star:.4f} | OOF F1={f1_star:.4f} @thr={thr_star:.3f}")
    eval_prob = np.mean(eval_prob_list, axis=0)

    submission = pd.DataFrame({
        'Id': np.arange(len(eval_prob)),
        'Label': (eval_prob >= thr_star).astype(int)
    })
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"Wrote {SUBMISSION_CSV}  (rows {len(submission)})")