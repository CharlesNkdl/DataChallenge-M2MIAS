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
import torch.nn.functional as F # Import F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
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
    "glucose_markers": ["glucose", "hba1c", "fasting_glucose"],
    "lipid_markers": ["chol", "trig", "hdl", "ldl"],
    "liver_enzymes": ["alt", "ast", "ggt", "alp", "total_bilirubin"], # Added total_bilirubin
    "kidney_markers": ["creat", "urea", "egfr", "uric_acid"], # Added uric_acid
    "inflammation_markers": ["c_reactive_protein", "ferritin", "white_blood_cells"],
    "hematology_markers": ["hematocrit", "platelets", "hemoglobin", "red_blood_cells", "mch", "mcv", "rdw"]
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
            # Handle cases where all values in a timeseries for a feature are NaN for a patient
            # slope_vals = np.where(np.isnan(X_3d[:, -1, :]) | np.isnan(X_3d[:, 0, :]), 0, slope_vals)
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

        try:
            trig_idx = -1
            hdl_idx = -1
            for i, label in enumerate(feature_labels):
                if "triglycerides" in label.lower():
                    trig_idx = i
                if "hdl_cholesterol" in label.lower(): # Make sure this matches your hdl feature name
                    hdl_idx = i

            if trig_idx != -1 and hdl_idx != -1:
                trig_mean_vals = mean_vals[:, trig_idx]
                hdl_mean_vals = mean_vals[:, hdl_idx]
                trig_hdl_ratio = trig_mean_vals / (hdl_mean_vals + 1e-9)
                extracted_features_list.append(trig_hdl_ratio)
                extracted_feature_names.append("ratio_trig_hdl")
            else:
                print("Warning: Could not find 'triglycerides' or 'hdl_cholesterol' for ratio.")
        except Exception as e:
            print(f"Error calculating trig/hdl ratio: {e}")


    if not extracted_features_list: # Should not happen if include_basic_stats is True
        return np.array([]).reshape(n_samples, 0), []

    return np.column_stack(extracted_features_list), extracted_feature_names

class CNN1D(nn.Module):
    def __init__(self, n_feat: int = 77, n_filt: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, n_filt, 3, padding=1), nn.BatchNorm1d(n_filt), nn.ReLU(), nn.Dropout1d(dropout),
            nn.Conv1d(n_filt, n_filt, 3, padding=1), nn.BatchNorm1d(n_filt), nn.ReLU(), nn.Dropout1d(dropout),
            nn.Conv1d(n_filt, n_filt//2, 3, padding=1), nn.BatchNorm1d(n_filt//2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(n_filt//2, n_filt//4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_filt//4, 1)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=77, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x): # x shape: (batch, seq_len, features)
        x = F.relu(self.embed(x))
        x = self.transformer_encoder(x) # x shape: (batch, seq_len, d_model)
        return self.fc(x.mean(dim=1)).squeeze(-1) # mean over seq_len


def summary_stats(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=1) # Use nanmean
    std = np.nanstd(x, axis=1)   # Use nanstd
    mn = np.nanmin(x, axis=1)    # Use nanmin
    mx = np.nanmax(x, axis=1)    # Use nanmax

    # Slope: ensure there are at least 2 non-NaN points to calculate slope
    slope_vals = np.zeros((x.shape[0], x.shape[2]))
    if x.shape[1] > 1:
        slope_vals = x[:, -1, :] - x[:, 0, :]
        # If first or last is NaN, slope becomes NaN. We might want to set to 0 or use other points.
        # For now, nan_to_num will handle it later in the main pipeline.

    p25 = np.nanpercentile(x, 25, axis=1)
    p75 = np.nanpercentile(x, 75, axis=1)
    iqr = p75 - p25
    cv = std / (mean + 1e-8)

    trend = np.zeros((x.shape[0], x.shape[2]))
    if x.shape[1] > 1:
        for i in range(x.shape[0]): # Iterate over samples
            for j in range(x.shape[2]): # Iterate over features
                valid_indices = ~np.isnan(x[i, :, j])
                if np.sum(valid_indices) >= 2: # Need at least 2 points for polyfit
                    trend[i,j] = np.polyfit(np.arange(x.shape[1])[valid_indices], x[i, valid_indices, j], 1)[0]
                # else, trend remains 0

    return np.column_stack([mean, std, mn, mx, slope_vals, p25, p75, iqr, cv, trend])


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19*2) # More granularity
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

def _train_pytorch_model_fold(
    model_class, x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 25, batch: int = 256, model_params: dict = None
) -> Tuple[nn.Module, np.ndarray, np.ndarray]:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_params is None: model_params = {}
    model = model_class(**model_params).to(device)

    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.2, min_lr=1e-6) # Changed to max for AUC

    tr_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=batch, shuffle=True, num_workers=2, pin_memory=device=='cuda')
    va_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=batch*2, num_workers=2, pin_memory=device=='cuda')

    best_metric = -1.0 # AUC starts at 0
    best_weights = None
    patience_counter = 0
    early_stopping_patience = 7

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            optimizer.zero_grad()
            logits = model(xb.to(device, non_blocking=True))
            loss = criterion(logits, yb.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        y_prob_val_list = []
        with torch.no_grad():
            for xb, _ in va_dl:
                p = torch.sigmoid(model(xb.to(device, non_blocking=True))).cpu()
                y_prob_val_list.append(p)
        y_prob_val = torch.cat(y_prob_val_list).numpy()

        current_metric = roc_auc_score(y_val, y_prob_val) # Use AUC for scheduler and early stopping
        scheduler.step(current_metric)

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {ep+1}")
            break

    if best_weights:
        model.load_state_dict(best_weights)

    model.eval()
    val_preds_list = []
    train_preds_list = []
    with torch.no_grad():
        # Val preds (already computed for last best model essentially)
        # Recompute with best weights loaded
        for xb, _ in va_dl:
             val_preds_list.append(torch.sigmoid(model(xb.to(device, non_blocking=True))).cpu())
        val_pred = torch.cat(val_preds_list).numpy()

        # Train preds (for OOF consistency, not used for training itself)
        # This can be slow, consider removing if not strictly needed for analysis
        # For OOF, we only need val_pred
        # train_dl_no_shuffle = DataLoader(TensorDataset(x_train, y_train), batch_size=batch*2, shuffle=False)
        # for xb, _ in train_dl_no_shuffle:
        #     train_preds_list.append(torch.sigmoid(model(xb.to(device, non_blocking=True))).cpu())
        # train_pred = torch.cat(train_preds_list).numpy() if train_preds_list else np.array([])
        train_pred = np.array([]) # Placeholder, train_pred is not used in current pipeline

    return model, val_pred, train_pred


def train_cnn_fold(*args, **kwargs):
    n_feat = kwargs.pop('n_feat', 77) # Get n_feat from kwargs or default
    return _train_pytorch_model_fold(CNN1D, *args, model_params={'n_feat': n_feat, 'n_filt': 64, 'dropout': 0.3}, **kwargs)

def train_transformer_fold(*args, **kwargs):
    input_dim = kwargs.pop('input_dim', 77)
    return _train_pytorch_model_fold(TemporalTransformer, *args, model_params={'input_dim': input_dim, 'd_model': 64, 'nhead': 4}, **kwargs)


def train_lightgbm(x_tr: np.ndarray, y_tr: np.ndarray,
                           x_val: np.ndarray = None, y_val: np.ndarray = None) -> lgb.Booster:
    pos_weight = (len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-8)
    params = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.03, # Slightly lower LR
        'num_leaves': 31, 'min_data_in_leaf': 30, 'feature_fraction': 0.7, # More regularization
        'bagging_fraction': 0.7, 'bagging_freq': 1, 'lambda_l1': 0.5, 'lambda_l2': 0.5,
        'seed': SEED, 'scale_pos_weight': pos_weight, 'verbose': -1, 'n_estimators': 2000
    }
    dtrain = lgb.Dataset(x_tr, y_tr)
    callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)] if x_val is not None else []

    if x_val is not None and y_val is not None:
        dval = lgb.Dataset(x_val, y_val, reference=dtrain)
        gbm = lgb.train(params, dtrain, valid_sets=[dtrain, dval], callbacks=callbacks)
    else:
        gbm = lgb.train(params, dtrain, num_boost_round=params.get('n_estimators', 500)) # Use n_estimators if no early stopping
    return gbm

def train_catboost(x_tr: np.ndarray, y_tr: np.ndarray,
                           x_val: np.ndarray = None, y_val: np.ndarray = None) -> CatBoostClassifier:
    params = {
        'loss_function': 'Logloss', 'eval_metric': 'AUC', 'learning_rate': 0.03,
        'depth': 5, 'l2_leaf_reg': 5.0, 'random_seed': SEED, 'verbose': False, # More regularization
        'iterations': 2000, 'class_weights': [1.0, (len(y_tr) / (y_tr.sum() + 1e-8))],
        'early_stopping_rounds': 100
    }
    model = CatBoostClassifier(**params)
    if x_val is not None and y_val is not None:
        model.fit(x_tr, y_tr, eval_set=(x_val, y_val), verbose=False)
    else:
        model.fit(x_tr, y_tr, verbose=False)
    return model

if __name__ == '__main__':
    with np.load(TRAIN_NPZ, allow_pickle=True) as f:
        X = f['data'].astype(np.float32)
        feat_labels = f['feature_labels'].tolist()
    y = pd.read_csv(TRAIN_LABEL_CSV)['Label'].values
    with np.load(EVAL_NPZ, allow_pickle=True) as f:
        X_eval = f['data'].astype(np.float32)

    N_FEATURES_ORIG = X.shape[-1]
    print(f"Original feature names (example): {feat_labels[:5]}")
    print(f"Number of original features: {N_FEATURES_ORIG}")
    print(f"Train data: {X.shape}, Train labels: {y.shape}, Eval data: {X_eval.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Positive case percentage: {y.mean()*100:.2f}%")


    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_cnn_prob = np.zeros(len(y), dtype=float)
    oof_transformer_prob = np.zeros(len(y), dtype=float)
    oof_gbm_prob = np.zeros(len(y), dtype=float)
    oof_cb_prob = np.zeros(len(y), dtype=float)

    eval_cnn_prob_list = []
    eval_transformer_prob_list = []
    eval_gbm_prob_list = []
    eval_cb_prob_list = []


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
        X_tr_fold_2d = X_tr_fold_ti.reshape(-1, N_FEATURES_ORIG)
        X_val_fold_2d = X_val_fold_ti.reshape(-1, N_FEATURES_ORIG)
        X_eval_fold_2d = X_eval_fold_ti.reshape(-1, N_FEATURES_ORIG)

        X_tr_fold_2d = scaler.fit_transform(X_tr_fold_2d)
        X_val_fold_2d = scaler.transform(X_val_fold_2d)
        X_eval_fold_2d = scaler.transform(X_eval_fold_2d)

        X_tr_processed = np.nan_to_num(X_tr_fold_2d.reshape(X_tr_fold.shape), nan=0.0, posinf=0.0, neginf=0.0)
        X_val_processed = np.nan_to_num(X_val_fold_2d.reshape(X_val_fold.shape), nan=0.0, posinf=0.0, neginf=0.0)
        X_eval_processed_fold = np.nan_to_num(X_eval_fold_2d.reshape(X_eval.shape), nan=0.0, posinf=0.0, neginf=0.0)

        X_tr_custom_feats, custom_feat_names = extract_custom_features(
            X_tr_processed, feat_labels, biochemical_groups=MY_BIOCHEMICAL_GROUPS, include_basic_stats=True)
        X_val_custom_feats, _ = extract_custom_features(
            X_val_processed, feat_labels, biochemical_groups=MY_BIOCHEMICAL_GROUPS, include_basic_stats=True)
        X_eval_custom_feats_fold, _ = extract_custom_features(
            X_eval_processed_fold, feat_labels, biochemical_groups=MY_BIOCHEMICAL_GROUPS, include_basic_stats=True)

        X_tr_custom_feats = np.nan_to_num(X_tr_custom_feats, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_custom_feats = np.nan_to_num(X_val_custom_feats, nan=0.0, posinf=0.0, neginf=0.0)
        X_eval_custom_feats_fold = np.nan_to_num(X_eval_custom_feats_fold, nan=0.0, posinf=0.0, neginf=0.0)


        # CNN
        cnn_model, val_cnn, _ = train_cnn_fold(X_tr_processed, y_tr_fold, X_val_processed, y_val_fold, n_feat=N_FEATURES_ORIG)
        oof_cnn_prob[val_idx] = val_cnn
        with torch.no_grad():
            device = next(cnn_model.parameters()).device; eval_cnn_preds_list = []
            for i in range(0, len(X_eval_processed_fold), 512):
                xb = torch.from_numpy(X_eval_processed_fold[i:i+512]).to(device)
                eval_cnn_preds_list.append(torch.sigmoid(cnn_model(xb)).cpu())
            eval_cnn_prob_list.append(torch.cat(eval_cnn_preds_list).numpy())

        # Transformer
        transformer_model, val_transformer, _ = train_transformer_fold(
            X_tr_processed, y_tr_fold, X_val_processed, y_val_fold, input_dim=N_FEATURES_ORIG)
        oof_transformer_prob[val_idx] = val_transformer
        with torch.no_grad():
            device = next(transformer_model.parameters()).device; eval_trans_preds_list = []
            for i in range(0, len(X_eval_processed_fold), 512):
                xb = torch.from_numpy(X_eval_processed_fold[i:i+512]).to(device)
                eval_trans_preds_list.append(torch.sigmoid(transformer_model(xb)).cpu())
            eval_transformer_prob_list.append(torch.cat(eval_trans_preds_list).numpy())

        # LightGBM
        gbm = train_lightgbm(X_tr_custom_feats, y_tr_fold, X_val_custom_feats, y_val_fold)
        val_gbm = gbm.predict(X_val_custom_feats, num_iteration=gbm.best_iteration)
        oof_gbm_prob[val_idx] = val_gbm
        eval_gbm_prob_list.append(gbm.predict(X_eval_custom_feats_fold, num_iteration=gbm.best_iteration))

        # CatBoost
        X_tr_flat = X_tr_processed.reshape(len(X_tr_processed), -1)
        X_val_flat = X_val_processed.reshape(len(X_val_processed), -1)
        X_eval_flat_fold = X_eval_processed_fold.reshape(len(X_eval_processed_fold), -1)

        X_tr_catboost_input = np.hstack([X_tr_flat, X_tr_custom_feats])
        X_val_catboost_input = np.hstack([X_val_flat, X_val_custom_feats])
        X_eval_catboost_input_fold = np.hstack([X_eval_flat_fold, X_eval_custom_feats_fold])

        X_tr_catboost_input = np.nan_to_num(X_tr_catboost_input, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_catboost_input = np.nan_to_num(X_val_catboost_input, nan=0.0, posinf=0.0, neginf=0.0)
        X_eval_catboost_input_fold = np.nan_to_num(X_eval_catboost_input_fold, nan=0.0, posinf=0.0, neginf=0.0)

        cb = train_catboost(X_tr_catboost_input, y_tr_fold, X_val_catboost_input, y_val_fold)
        val_cb = cb.predict_proba(X_val_catboost_input)[:, 1]
        oof_cb_prob[val_idx] = val_cb
        eval_cb_prob_list.append(cb.predict_proba(X_eval_catboost_input_fold)[:, 1])

        # Fold blend for monitoring
        val_blend_fold = (val_cnn + val_transformer + val_gbm + val_cb) / 4.0
        fold_auc = roc_auc_score(y_val_fold, val_blend_fold)
        fold_thr, fold_f1 = best_threshold(y_val_fold, val_blend_fold)
        print(f"Fold AUC (CNN):{roc_auc_score(y_val_fold, val_cnn):.4f} (Transformer):{roc_auc_score(y_val_fold, val_transformer):.4f} (LGBM):{roc_auc_score(y_val_fold, val_gbm):.4f} (CB):{roc_auc_score(y_val_fold, val_cb):.4f}")
        print(f"Fold Blend AUC={fold_auc:.4f} | best F1={fold_f1:.4f} @thr={fold_thr:.3f}")

    oof_blend_prob = (oof_cnn_prob + oof_transformer_prob + oof_gbm_prob + oof_cb_prob) / 4.0
    thr_star, f1_star = best_threshold(y, oof_blend_prob)
    auc_star = roc_auc_score(y, oof_blend_prob)
    print(f"\nOOF CNN AUC: {roc_auc_score(y, oof_cnn_prob):.4f}")
    print(f"OOF Transformer AUC: {roc_auc_score(y, oof_transformer_prob):.4f}")
    print(f"OOF LGBM AUC: {roc_auc_score(y, oof_gbm_prob):.4f}")
    print(f"OOF CB AUC: {roc_auc_score(y, oof_cb_prob):.4f}")
    print(f"OOF BLEND AUC={auc_star:.4f} | OOF F1={f1_star:.4f} @thr={thr_star:.3f}")

    eval_cnn_prob = np.mean(eval_cnn_prob_list, axis=0)
    eval_transformer_prob = np.mean(eval_transformer_prob_list, axis=0)
    eval_gbm_prob = np.mean(eval_gbm_prob_list, axis=0)
    eval_cb_prob = np.mean(eval_cb_prob_list, axis=0)
    eval_blend_prob = (eval_cnn_prob + eval_transformer_prob + eval_gbm_prob + eval_cb_prob) / 4.0

    submission = pd.DataFrame({'Id': np.arange(len(eval_blend_prob)), 'Label': (eval_blend_prob >= thr_star).astype(int)})
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"Wrote {SUBMISSION_CSV}  (rows {len(submission)})")