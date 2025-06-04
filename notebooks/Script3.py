import os
import random
import time
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 98
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA_DIR = Path('../data/')
TRAIN_NPZ = DATA_DIR / 'training_data.npz'
TRAIN_LABEL_CSV = DATA_DIR / 'training_labels.csv'
EVAL_NPZ = DATA_DIR / 'evaluation_data.npz'
SUBMISSION_CSV = DATA_DIR / 'submissionV4_rnn.csv' # Changed submission name
N_SPLITS = 5

# Groupes biochimiques étendus (inchangé)
MY_BIOCHEMICAL_GROUPS = {
    "glucose_markers": ["glucose", "hba1c", "fasting_glucose", "glycemia"],
    "lipid_markers": ["chol", "trig", "hdl", "ldl", "lipid"],
    "liver_enzymes": ["alt", "ast", "ggt", "alp", "total_bilirubin", "direct_bilirubin"],
    "kidney_markers": ["creat", "urea", "egfr", "uric_acid", "albumin"],
    "inflammation_markers": ["c_reactive_protein", "ferritin", "white_blood_cells", "neutrophil"],
    "hematology_markers": ["hematocrit", "platelets", "hemoglobin", "red_blood_cells", "mch", "mcv", "rdw"],
    "protein_markers": ["protein", "albumin", "globulin"],
    "electrolyte_markers": ["sodium", "potassium", "chloride"]
}

# TensorDataset, robust_imputation, extract_enhanced_features (inchangés)
class TensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray = None):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(y.astype(np.float32))
    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return (self.x[idx], self.y[idx]) if self.y is not None else self.x[idx]

def robust_imputation(X_3d: np.ndarray) -> np.ndarray:
    X_imputed = X_3d.copy()
    n_samples, n_timesteps, n_features = X_3d.shape
    for sample_idx in range(n_samples):
        for feat_idx in range(n_features):
            series = X_imputed[sample_idx, :, feat_idx]
            mask = ~np.isnan(series)
            if np.any(mask):
                last_valid = np.nan
                for i in range(len(series)):
                    if not np.isnan(series[i]): last_valid = series[i]
                    elif not np.isnan(last_valid): series[i] = last_valid
                last_valid = np.nan
                for i in range(len(series)-1, -1, -1):
                    if not np.isnan(series[i]): last_valid = series[i]
                    elif not np.isnan(last_valid): series[i] = last_valid
                X_imputed[sample_idx, :, feat_idx] = series
    return X_imputed

def extract_enhanced_features(
    X_3d: np.ndarray, feature_labels: List[str], biochemical_groups: Dict[str, List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    n_samples, n_timesteps, n_original_features_in_X_3d = X_3d.shape
    if len(feature_labels) != n_original_features_in_X_3d:
        print(f"Warning: Mismatch in extract_enhanced_features: len(feature_labels)={len(feature_labels)} vs X_3d.shape[-1]={n_original_features_in_X_3d}")
    extracted_features_list = []
    extracted_feature_names = []
    mean_vals = np.nanmean(X_3d, axis=1); std_vals = np.nanstd(X_3d, axis=1)
    min_vals = np.nanmin(X_3d, axis=1); max_vals = np.nanmax(X_3d, axis=1)
    median_vals = np.nanmedian(X_3d, axis=1); p25_vals = np.nanpercentile(X_3d, 25, axis=1)
    p75_vals = np.nanpercentile(X_3d, 75, axis=1); iqr_vals = p75_vals - p25_vals
    cv_vals = std_vals / (np.abs(mean_vals) + 1e-8); range_vals = max_vals - min_vals
    if n_timesteps > 1:
        slope_vals = X_3d[:, -1, :] - X_3d[:, 0, :]
        trend_vals = np.zeros((n_samples, n_original_features_in_X_3d))
        for i in range(n_samples):
            for j in range(n_original_features_in_X_3d):
                valid_mask = ~np.isnan(X_3d[i, :, j])
                if np.sum(valid_mask) >= 2:
                    x_vals = np.arange(n_timesteps)[valid_mask]; y_vals = X_3d[i, valid_mask, j]
                    if len(x_vals) >= 2: trend_vals[i, j] = np.polyfit(x_vals, y_vals, 1)[0]
    else: slope_vals = np.zeros((n_samples, n_original_features_in_X_3d)); trend_vals = np.zeros((n_samples, n_original_features_in_X_3d))
    stability_vals = np.zeros((n_samples, n_original_features_in_X_3d))
    if n_timesteps > 1:
        for i in range(n_samples):
            for j in range(n_original_features_in_X_3d):
                series = X_3d[i, :, j]; valid_mask = ~np.isnan(series)
                if np.sum(valid_mask) >= 2:
                    valid_series = series[valid_mask]
                    if len(valid_series) >= 2:
                        diffs = np.diff(valid_series)
                        stability_vals[i, j] = np.std(diffs) if len(diffs) > 0 else 0
    for i, label in enumerate(feature_labels):
        if i >= n_original_features_in_X_3d: continue
        extracted_features_list.extend([
            mean_vals[:, i], std_vals[:, i], min_vals[:, i], max_vals[:, i],
            median_vals[:, i], p25_vals[:, i], p75_vals[:, i], iqr_vals[:, i],
            cv_vals[:, i], range_vals[:, i], slope_vals[:, i], trend_vals[:, i],
            stability_vals[:, i]
        ])
        extracted_feature_names.extend([
            f"{label}_mean", f"{label}_std", f"{label}_min", f"{label}_max",
            f"{label}_median", f"{label}_p25", f"{label}_p75", f"{label}_iqr",
            f"{label}_cv", f"{label}_range", f"{label}_slope", f"{label}_trend",
            f"{label}_stability"
        ])
    if biochemical_groups:
        group_stats = {}
        for group_name, marker_keywords in biochemical_groups.items():
            group_indices_in_X_3d = [
                idx for idx, lbl in enumerate(feature_labels)
                if any(keyword.lower() in lbl.lower() for keyword in marker_keywords) and idx < n_original_features_in_X_3d
            ]
            if not group_indices_in_X_3d: continue
            group_mean = np.nanmean(mean_vals[:, group_indices_in_X_3d], axis=1)
            group_std = np.nanmean(std_vals[:, group_indices_in_X_3d], axis=1)
            group_max = np.nanmax(max_vals[:, group_indices_in_X_3d], axis=1)
            group_min = np.nanmin(min_vals[:, group_indices_in_X_3d], axis=1)
            group_stats[group_name] = {'mean': group_mean, 'std': group_std, 'max': group_max, 'min': group_min}
            extracted_features_list.extend([group_mean, group_std, group_max, group_min])
            extracted_feature_names.extend([
                f"group_{group_name}_mean", f"group_{group_name}_std",
                f"group_{group_name}_max", f"group_{group_name}_min"
            ])
        group_pairs = [
            ("glucose_markers", "lipid_markers", "glucose_lipid"),
            ("liver_enzymes", "kidney_markers", "liver_kidney"),
            ("inflammation_markers", "hematology_markers", "inflam_hemato"),
        ]
        for group1, group2, ratio_name in group_pairs:
            if group1 in group_stats and group2 in group_stats:
                ratio = group_stats[group1]['mean'] / (np.abs(group_stats[group2]['mean']) + 1e-9)
                extracted_features_list.append(ratio)
                extracted_feature_names.append(f"ratio_{ratio_name}")
        if len(group_stats) >= 3:
            health_score = np.zeros(n_samples)
            weights = {'glucose_markers': 0.3, 'lipid_markers': 0.25, 'liver_enzymes': 0.2,
                      'kidney_markers': 0.15, 'inflammation_markers': 0.1}
            for group_name, weight in weights.items():
                if group_name in group_stats: health_score += weight * group_stats[group_name]['mean']
            extracted_features_list.append(health_score)
            extracted_feature_names.append("health_score_weighted")
    if not extracted_features_list: return np.zeros((n_samples, 0)), []
    return np.column_stack(extracted_features_list), extracted_feature_names

# EnhancedCNN1D (inchangé)
class EnhancedCNN1D(nn.Module):
    def __init__(self, n_feat: int = 77, n_filt: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(n_feat, n_filt, 3, padding=1), nn.BatchNorm1d(n_filt), nn.ReLU(), nn.Dropout1d(dropout * 0.5))
        self.conv2 = nn.Sequential(nn.Conv1d(n_filt, n_filt, 3, padding=1), nn.BatchNorm1d(n_filt), nn.ReLU(), nn.Dropout1d(dropout * 0.5))
        self.conv3 = nn.Sequential(nn.Conv1d(n_filt, n_filt, 5, padding=2), nn.BatchNorm1d(n_filt), nn.ReLU(), nn.Dropout1d(dropout * 0.5))
        self.conv4 = nn.Sequential(nn.Conv1d(n_filt, n_filt//2, 3, padding=1), nn.BatchNorm1d(n_filt//2), nn.ReLU())
        self.attention = nn.Sequential(nn.Conv1d(n_filt//2, n_filt//4, 1), nn.ReLU(), nn.Conv1d(n_filt//4, 1, 1), nn.Sigmoid())
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1); self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Linear(n_filt, n_filt//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(n_filt//2, n_filt//4), nn.ReLU(), nn.Dropout(dropout * 0.5), nn.Linear(n_filt//4, 1))
    def forward(self, x):
        x = x.permute(0, 2, 1); x1 = self.conv1(x); x2 = self.conv2(x1); x2 = x2 + x1
        x3 = self.conv3(x2); x4 = self.conv4(x3); att_weights = self.attention(x4); x4_att = x4 * att_weights
        avg_pool = self.global_avg_pool(x4_att).squeeze(-1); max_pool = self.global_max_pool(x4_att).squeeze(-1)
        x_combined = torch.cat([avg_pool, max_pool], dim=1); return self.fc(x_combined).squeeze(-1)

# --- NOUVEAU: Modèle RNN Simple (LSTM ou GRU) ---
class SimpleRNN(nn.Module):
    def __init__(self, input_dim=77, hidden_dim=128, num_layers=2, rnn_type='LSTM', dropout=0.3, bidirectional=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        rnn_dropout = dropout if num_layers > 1 else 0 # Dropout only between RNN layers

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=rnn_dropout, bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=rnn_dropout, bidirectional=bidirectional)
        else:
            raise ValueError("Unsupported RNN type. Choose 'LSTM' or 'GRU'.")

        # Calculate fc input size based on bidirectionality
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) - Pas besoin de permuter pour RNN batch_first=True

        # Initialiser les états cachés (et de cellule pour LSTM)
        # h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        # if isinstance(self.rnn, nn.LSTM):
        #     c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim).to(x.device)
        #     rnn_out, _ = self.rnn(x, (h0, c0))
        # else: # GRU
        #     rnn_out, _ = self.rnn(x, h0)

        # Pour LSTM/GRU, si les états initiaux ne sont pas fournis, ils sont initialisés à zéro par défaut.
        rnn_out, _ = self.rnn(x)

        # Prendre la sortie du dernier pas de temps
        # Si bidirectionnel, rnn_out est (batch, seq_len, hidden_dim * 2)
        # On veut le dernier état de la direction forward et le premier état de la direction backward
        if self.bidirectional:
            # Concaténer le dernier état de la direction avant et le premier état de la direction arrière
            # rnn_out[:, -1, :self.hidden_dim] (dernier état forward)
            # rnn_out[:, 0, self.hidden_dim:] (premier état backward - qui est le dernier vu par la couche backward)
            out = torch.cat((rnn_out[:, -1, :self.hidden_dim], rnn_out[:, 0, self.hidden_dim:]), dim=1)
        else:
            out = rnn_out[:, -1, :]  # (batch, hidden_dim)

        return self.fc(out).squeeze(-1)


# _train_pytorch_model_fold (inchangé, mais j'ai simplifié le print d'avertissement)
def _train_pytorch_model_fold(
    model_class, x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 40, batch: int = 256, model_params: dict = None
) -> Tuple[nn.Module, np.ndarray, np.ndarray]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_params is None: model_params = {}

    current_input_dim_key = 'n_feat' if model_class == EnhancedCNN1D else 'input_dim'
    if model_params.get(current_input_dim_key, 1) == 0 :
        print(f"Warning: {current_input_dim_key} is 0 for {model_class.__name__}. Skipping PyTorch model.")
        dummy_model_params = model_params.copy()
        dummy_model_params[current_input_dim_key] = 1 # Ensure it's at least 1 for instantiation
        model = model_class(**dummy_model_params).to(device)
        return model, np.full(y_val.shape, 0.5), np.array([])

    model = model_class(**model_params).to(device)
    pos_weight_val = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': 2e-3, 'weight_decay': 1e-4}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    tr_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=batch, shuffle=True, num_workers=2, pin_memory=device=='cuda', drop_last=True)
    va_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=batch*2, num_workers=2, pin_memory=device=='cuda')
    best_metric = -1.0; best_weights = None; patience_counter = 0; early_stopping_patience = 12
    for ep in range(epochs):
        model.train(); total_loss = 0
        for xb, yb in tr_dl:
            optimizer.zero_grad(); logits = model(xb.to(device, non_blocking=True))
            loss = criterion(logits, yb.to(device, non_blocking=True)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total_loss += loss.item()
        model.eval(); y_prob_val_list = []
        with torch.no_grad():
            for xb_val, _ in va_dl:
                p = torch.sigmoid(model(xb_val.to(device, non_blocking=True))).cpu()
                y_prob_val_list.append(p)
        y_prob_val = torch.cat(y_prob_val_list).numpy(); current_metric = roc_auc_score(y_val, y_prob_val); scheduler.step()
        if current_metric > best_metric:
            best_metric = current_metric; patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else: patience_counter += 1
        if patience_counter >= early_stopping_patience: print(f"Early stopping at epoch {ep+1}"); break
    if best_weights: model.load_state_dict(best_weights)
    model.eval(); val_preds_list = []
    with torch.no_grad():
        for xb_val, _ in va_dl: val_preds_list.append(torch.sigmoid(model(xb_val.to(device, non_blocking=True))).cpu())
    val_pred = torch.cat(val_preds_list).numpy(); return model, val_pred, np.array([])

# train_enhanced_cnn_fold (inchangé)
def train_enhanced_cnn_fold(*args, **kwargs):
    # n_feat should be passed in model_params by the main loop
    n_feat = kwargs.get('model_params', {}).get('n_feat', args[0].shape[-1] if args and hasattr(args[0], 'shape') else 77)
    kwargs['model_params'] = {'n_feat': n_feat, 'n_filt': 128, 'dropout': 0.3}
    return _train_pytorch_model_fold(EnhancedCNN1D, *args, **kwargs)

# --- NOUVEAU: Fonction d'entraînement pour SimpleRNN ---
def train_simple_rnn_fold(*args, **kwargs):
    # input_dim should be passed in model_params by the main loop
    input_dim = kwargs.get('model_params', {}).get('input_dim', args[0].shape[-1] if args and hasattr(args[0], 'shape') else 77)
    # Vous pouvez configurer rnn_type, hidden_dim, etc. ici ou les passer via kwargs
    kwargs['model_params'] = {
        'input_dim': input_dim,
        'hidden_dim': 64, # Plus petit que le Transformer
        'num_layers': 1,  # Moins de couches
        'rnn_type': 'GRU', # GRU est souvent un peu plus léger que LSTM
        'dropout': 0.2,
        'bidirectional': True
    }
    return _train_pytorch_model_fold(SimpleRNN, *args, **kwargs)


# Les fonctions train_enhanced_lightgbm, train_enhanced_catboost, best_threshold (inchangées)
def train_enhanced_lightgbm(x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray = None, y_val: np.ndarray = None) -> lgb.Booster:
    pos_weight = (len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-8)
    params = {'objective': 'binary','metric': 'auc','boosting_type': 'gbdt','learning_rate': 0.02,'num_leaves': 63,'max_depth': 7,
        'min_data_in_leaf': 25,'feature_fraction': 0.8,'bagging_fraction': 0.8,'bagging_freq': 1,'lambda_l1': 1.0,'lambda_l2': 1.0,
        'min_gain_to_split': 0.1,'seed': SEED,'scale_pos_weight': pos_weight,'verbose': -1,'n_estimators': 3000}
    dtrain = lgb.Dataset(x_tr, y_tr)
    callbacks = [lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)] if x_val is not None else []
    if x_val is not None and y_val is not None:
        dval = lgb.Dataset(x_val, y_val, reference=dtrain)
        gbm = lgb.train(params, dtrain, valid_sets=[dtrain, dval], callbacks=callbacks)
    else: gbm = lgb.train(params, dtrain, num_boost_round=params.get('n_estimators', 1000))
    return gbm

def train_enhanced_catboost(x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray = None, y_val: np.ndarray = None) -> CatBoostClassifier:
    pos_weight_cat = (len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-8)
    params = {'loss_function': 'Logloss','eval_metric': 'AUC','learning_rate': 0.02,'depth': 6,'l2_leaf_reg': 3.0,
        'bootstrap_type': 'Bayesian','bagging_temperature': 1.0,'od_type': 'Iter','od_wait': 150,'random_seed': SEED,
        'verbose': False,'iterations': 3000,'class_weights': [1.0, pos_weight_cat],'early_stopping_rounds': 150}
    model = CatBoostClassifier(**params)
    if x_val is not None and y_val is not None: model.fit(x_tr, y_tr, eval_set=(x_val, y_val), verbose=False)
    else: model.fit(x_tr, y_tr, verbose=False)
    return model

def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    if len(y_prob) == 0 or np.all(np.isnan(y_prob)) or len(np.unique(y_prob)) <=1 : return 0.5, 0.0
    thresholds = np.linspace(0.05, 0.95, 91); y_true_bool = y_true.astype(bool)
    predictions = y_prob[:, None] >= thresholds[None, :]; tp = (predictions & y_true_bool[:, None]).sum(axis=0)
    fp = (predictions & ~y_true_bool[:, None]).sum(axis=0); fn = (~predictions & y_true_bool[:, None]).sum(axis=0)
    precision = tp / (tp + fp + 1e-8); recall = tp / (tp + fn + 1e-8)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    if np.all(np.isnan(f1_scores)) or len(f1_scores)==0 : return 0.5,0.0
    best_idx = np.nanargmax(f1_scores) # Use nanargmax
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


if __name__ == '__main__':
    with np.load(TRAIN_NPZ, allow_pickle=True) as f:
        X_orig_all_feats = f['data'].astype(np.float32)
        feat_labels_orig_all_feats = f['feature_labels'].tolist()
    y = pd.read_csv(TRAIN_LABEL_CSV)['Label'].values
    with np.load(EVAL_NPZ, allow_pickle=True) as f:
        X_eval_orig_all_feats = f['data'].astype(np.float32)

    N_FEATURES_TOTAL_ORIGINAL = X_orig_all_feats.shape[-1]
    print(f"Total original features: {N_FEATURES_TOTAL_ORIGINAL}")

    covered_feature_indices = set()
    for group_name, marker_keywords in MY_BIOCHEMICAL_GROUPS.items():
        for i, label in enumerate(feat_labels_orig_all_feats):
            if any(keyword.lower() in label.lower() for keyword in marker_keywords):
                covered_feature_indices.add(i)
    pytorch_feature_indices_to_keep = [i for i in range(N_FEATURES_TOTAL_ORIGINAL) if i not in covered_feature_indices]
    if not pytorch_feature_indices_to_keep:
        print("Warning: All original features covered. Using ALL for PyTorch.")
        pytorch_feature_indices_to_keep = list(range(N_FEATURES_TOTAL_ORIGINAL))
    feat_labels_pytorch = [feat_labels_orig_all_feats[i] for i in pytorch_feature_indices_to_keep]
    N_FEATURES_PYTORCH = len(pytorch_feature_indices_to_keep)
    print(f"Number of features for PyTorch models: {N_FEATURES_PYTORCH}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_cnn_prob = np.zeros(len(y), dtype=float)
    oof_rnn_prob = np.zeros(len(y), dtype=float) # MODIFIÉ: oof_transformer_prob -> oof_rnn_prob
    oof_gbm_prob = np.zeros(len(y), dtype=float)
    oof_cb_prob = np.zeros(len(y), dtype=float)

    eval_cnn_prob_list = []
    eval_rnn_prob_list = [] # MODIFIÉ: eval_transformer_prob_list -> eval_rnn_prob_list
    eval_gbm_prob_list = []
    eval_cb_prob_list = []
    tabular_feature_names = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_orig_all_feats, y), 1):
        print(f"\nFold {fold}/{N_SPLITS}  (train {len(train_idx)}, val {len(val_idx)})")
        X_tr_fold_all_feats, X_val_fold_all_feats = X_orig_all_feats[train_idx], X_orig_all_feats[val_idx]
        y_tr_fold, y_val_fold = y[train_idx], y[val_idx]
        X_tr_fold_imputed_all_feats = robust_imputation(X_tr_fold_all_feats)
        X_val_fold_imputed_all_feats = robust_imputation(X_val_fold_all_feats)
        X_eval_fold_imputed_all_feats = robust_imputation(X_eval_orig_all_feats.copy())

        if N_FEATURES_PYTORCH > 0:
            X_tr_fold_pytorch_subset = X_tr_fold_all_feats[:, :, pytorch_feature_indices_to_keep]
            X_val_fold_pytorch_subset = X_val_fold_all_feats[:, :, pytorch_feature_indices_to_keep]
            X_eval_pytorch_subset_for_fold = X_eval_orig_all_feats[:, :, pytorch_feature_indices_to_keep].copy()
            X_tr_fold_imputed_pytorch = robust_imputation(X_tr_fold_pytorch_subset)
            X_val_fold_imputed_pytorch = robust_imputation(X_val_fold_pytorch_subset)
            X_eval_fold_imputed_pytorch = robust_imputation(X_eval_pytorch_subset_for_fold)
            X_tr_pytorch_scaled_list = []; X_val_pytorch_scaled_list = []; X_eval_pytorch_scaled_list = []
            for i in range(N_FEATURES_PYTORCH):
                scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, X_tr_fold_imputed_pytorch.shape[0]), random_state=SEED)
                tr_feat_reshaped = X_tr_fold_imputed_pytorch[:, :, i].reshape(-1, 1)
                val_feat_reshaped_for_transform = X_val_fold_imputed_pytorch[:, :, i].reshape(-1, 1)
                eval_feat_reshaped_for_transform = X_eval_fold_imputed_pytorch[:, :, i].reshape(-1, 1)
                if np.all(tr_feat_reshaped == tr_feat_reshaped[0]):
                    tr_s = np.zeros_like(tr_feat_reshaped).reshape(X_tr_fold_imputed_pytorch.shape[0], -1)
                    val_s = np.zeros_like(val_feat_reshaped_for_transform).reshape(X_val_fold_imputed_pytorch.shape[0], -1)
                    eval_s = np.zeros_like(eval_feat_reshaped_for_transform).reshape(X_eval_fold_imputed_pytorch.shape[0], -1)
                else:
                    data_to_fit_1d = tr_feat_reshaped[~np.isnan(tr_feat_reshaped).ravel()]
                    if data_to_fit_1d.size == 0:
                        tr_s = np.zeros_like(tr_feat_reshaped).reshape(X_tr_fold_imputed_pytorch.shape[0], -1)
                        val_s = np.zeros_like(val_feat_reshaped_for_transform).reshape(X_val_fold_imputed_pytorch.shape[0], -1)
                        eval_s = np.zeros_like(eval_feat_reshaped_for_transform).reshape(X_eval_fold_imputed_pytorch.shape[0], -1)
                    else:
                        scaler.fit(data_to_fit_1d.reshape(-1, 1))
                        tr_s_flat = scaler.transform(tr_feat_reshaped); val_s_flat = scaler.transform(val_feat_reshaped_for_transform); eval_s_flat = scaler.transform(eval_feat_reshaped_for_transform)
                        tr_s = tr_s_flat.reshape(X_tr_fold_imputed_pytorch.shape[0], -1); val_s = val_s_flat.reshape(X_val_fold_imputed_pytorch.shape[0], -1); eval_s = eval_s_flat.reshape(X_eval_fold_imputed_pytorch.shape[0], -1)
                X_tr_pytorch_scaled_list.append(tr_s); X_val_pytorch_scaled_list.append(val_s); X_eval_pytorch_scaled_list.append(eval_s)
            X_tr_pytorch_processed = np.stack(X_tr_pytorch_scaled_list, axis=2)
            X_val_pytorch_processed = np.stack(X_val_pytorch_scaled_list, axis=2)
            X_eval_pytorch_processed_fold = np.stack(X_eval_pytorch_scaled_list, axis=2)
            X_tr_pytorch_processed = np.nan_to_num(X_tr_pytorch_processed, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_pytorch_processed = np.nan_to_num(X_val_pytorch_processed, nan=0.0, posinf=0.0, neginf=0.0)
            X_eval_pytorch_processed_fold = np.nan_to_num(X_eval_pytorch_processed_fold, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_tr_pytorch_processed = np.empty((X_tr_fold_all_feats.shape[0], X_tr_fold_all_feats.shape[1], 0), dtype=np.float32)
            X_val_pytorch_processed = np.empty((X_val_fold_all_feats.shape[0], X_val_fold_all_feats.shape[1], 0), dtype=np.float32)
            X_eval_pytorch_processed_fold = np.empty((X_eval_orig_all_feats.shape[0], X_eval_orig_all_feats.shape[1], 0), dtype=np.float32)
            print("Skipping PyTorch models as N_FEATURES_PYTORCH is 0.")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # CNN
        print("Training Enhanced CNN...")
        cnn_model_params = {'n_feat': N_FEATURES_PYTORCH} # n_filt et dropout sont par défaut dans train_enhanced_cnn_fold
        cnn_model, val_cnn_pred, _ = train_enhanced_cnn_fold(
            X_tr_pytorch_processed, y_tr_fold, X_val_pytorch_processed, y_val_fold, model_params=cnn_model_params)
        oof_cnn_prob[val_idx] = val_cnn_pred
        if N_FEATURES_PYTORCH > 0:
            eval_dl_cnn = DataLoader(TensorDataset(X_eval_pytorch_processed_fold), batch_size=256*2)
            eval_cnn_pred_fold_list_temp = []
            cnn_model.eval()
            with torch.no_grad():
                for xb_eval in eval_dl_cnn: eval_cnn_pred_fold_list_temp.append(torch.sigmoid(cnn_model(xb_eval.to(device, non_blocking=True))).cpu())
            eval_cnn_prob_list.append(torch.cat(eval_cnn_pred_fold_list_temp).numpy())
        else: eval_cnn_prob_list.append(np.full(X_eval_orig_all_feats.shape[0], 0.5))

        # MODIFIÉ: Remplacement du Transformer par SimpleRNN
        print("Training Simple RNN...")
        rnn_model_params = {'input_dim': N_FEATURES_PYTORCH} # Les autres params sont par défaut dans train_simple_rnn_fold
        rnn_model, val_rnn_pred, _ = train_simple_rnn_fold(
            X_tr_pytorch_processed, y_tr_fold, X_val_pytorch_processed, y_val_fold, model_params=rnn_model_params)
        oof_rnn_prob[val_idx] = val_rnn_pred # MODIFIÉ
        if N_FEATURES_PYTORCH > 0:
            eval_dl_rnn = DataLoader(TensorDataset(X_eval_pytorch_processed_fold), batch_size=256*2)
            eval_rnn_pred_fold_list_temp = []
            rnn_model.eval()
            with torch.no_grad():
                for xb_eval in eval_dl_rnn: eval_rnn_pred_fold_list_temp.append(torch.sigmoid(rnn_model(xb_eval.to(device, non_blocking=True))).cpu())
            eval_rnn_prob_list.append(torch.cat(eval_rnn_pred_fold_list_temp).numpy()) # MODIFIÉ
        else: eval_rnn_prob_list.append(np.full(X_eval_orig_all_feats.shape[0], 0.5)) # MODIFIÉ

        # Feature extraction pour modèles tabulaires
        print("Extracting features for tabular models...")
        X_tr_tab_feat, current_tab_feat_names = extract_enhanced_features(X_tr_fold_imputed_all_feats, feat_labels_orig_all_feats, MY_BIOCHEMICAL_GROUPS)
        X_val_tab_feat, _ = extract_enhanced_features(X_val_fold_imputed_all_feats, feat_labels_orig_all_feats, MY_BIOCHEMICAL_GROUPS)
        X_eval_tab_feat_fold, _ = extract_enhanced_features(X_eval_fold_imputed_all_feats, feat_labels_orig_all_feats, MY_BIOCHEMICAL_GROUPS)
        if tabular_feature_names is None and current_tab_feat_names:
            tabular_feature_names = current_tab_feat_names; print(f"Number of extracted tabular features: {len(tabular_feature_names)}")
        if X_tr_tab_feat.shape[1] == 0:
            print("Warning: No tabular features extracted. Skipping Tabular models.")
            oof_gbm_prob[val_idx] = 0.5; eval_gbm_prob_list.append(np.full(X_eval_orig_all_feats.shape[0], 0.5))
            oof_cb_prob[val_idx] = 0.5; eval_cb_prob_list.append(np.full(X_eval_orig_all_feats.shape[0], 0.5))
        else:
            if X_tr_tab_feat.size > 0:
                X_tr_tab_feat_median = np.nanmedian(X_tr_tab_feat, axis=0, keepdims=True)
                X_tr_tab_feat_median[np.isnan(X_tr_tab_feat_median)] = 0
            else: X_tr_tab_feat_median = np.zeros((1, X_tr_tab_feat.shape[1]))
            X_tr_tab_feat = np.nan_to_num(X_tr_tab_feat, nan=0.0); X_tr_tab_feat = np.where(np.isnan(X_tr_tab_feat) | ((X_tr_tab_feat == 0.0) & (np.any(X_tr_tab_feat_median != 0, axis=0, keepdims=True))), X_tr_tab_feat_median, X_tr_tab_feat)
            X_val_tab_feat = np.nan_to_num(X_val_tab_feat, nan=0.0); X_val_tab_feat = np.where(np.isnan(X_val_tab_feat) | ((X_val_tab_feat == 0.0) & (np.any(X_tr_tab_feat_median != 0, axis=0, keepdims=True))), X_tr_tab_feat_median, X_val_tab_feat)
            X_eval_tab_feat_fold = np.nan_to_num(X_eval_tab_feat_fold, nan=0.0); X_eval_tab_feat_fold = np.where(np.isnan(X_eval_tab_feat_fold) | ((X_eval_tab_feat_fold == 0.0) & (np.any(X_tr_tab_feat_median != 0, axis=0, keepdims=True))), X_tr_tab_feat_median, X_eval_tab_feat_fold)
            scaler_tab = StandardScaler()
            X_tr_tab_feat = scaler_tab.fit_transform(X_tr_tab_feat); X_val_tab_feat = scaler_tab.transform(X_val_tab_feat); X_eval_tab_feat_fold = scaler_tab.transform(X_eval_tab_feat_fold)
            print("Training LightGBM..."); lgbm_model = train_enhanced_lightgbm(X_tr_tab_feat, y_tr_fold, X_val_tab_feat, y_val_fold)
            oof_gbm_prob[val_idx] = lgbm_model.predict(X_val_tab_feat, num_iteration=lgbm_model.best_iteration if lgbm_model.best_iteration else -1)
            eval_gbm_prob_list.append(lgbm_model.predict(X_eval_tab_feat_fold, num_iteration=lgbm_model.best_iteration if lgbm_model.best_iteration else -1))
            print("Training CatBoost..."); cb_model = train_enhanced_catboost(X_tr_tab_feat, y_tr_fold, X_val_tab_feat, y_val_fold)
            oof_cb_prob[val_idx] = cb_model.predict_proba(X_val_tab_feat)[:, 1]
            eval_cb_prob_list.append(cb_model.predict_proba(X_eval_tab_feat_fold)[:, 1])
        print(f"Fold {fold} completed.")

    eval_cnn_prob = np.mean(np.array(eval_cnn_prob_list), axis=0) if eval_cnn_prob_list and np.any([arr.size > 0 for arr in eval_cnn_prob_list]) else np.full(len(y), 0.5)
    eval_rnn_prob = np.mean(np.array(eval_rnn_prob_list), axis=0) if eval_rnn_prob_list and np.any([arr.size > 0 for arr in eval_rnn_prob_list]) else np.full(len(y), 0.5) # MODIFIÉ
    eval_gbm_prob = np.mean(np.array(eval_gbm_prob_list), axis=0) if eval_gbm_prob_list and np.any([arr.size > 0 for arr in eval_gbm_prob_list]) else np.full(len(y), 0.5)
    eval_cb_prob = np.mean(np.array(eval_cb_prob_list), axis=0) if eval_cb_prob_list and np.any([arr.size > 0 for arr in eval_cb_prob_list]) else np.full(len(y), 0.5)

    print("\n--- OOF Performance ---")
    cnn_thresh, cnn_f1 = best_threshold(y, oof_cnn_prob)
    print(f"Enhanced CNN OOF F1: {cnn_f1:.4f} at threshold {cnn_thresh:.4f}, AUC: {roc_auc_score(y, oof_cnn_prob) if not np.all(oof_cnn_prob == 0.5) else 0.5:.4f}")

    # MODIFIÉ: Affichage pour RNN
    rnn_thresh, rnn_f1 = best_threshold(y, oof_rnn_prob)
    print(f"Simple RNN OOF F1: {rnn_f1:.4f} at threshold {rnn_thresh:.4f}, AUC: {roc_auc_score(y, oof_rnn_prob) if not np.all(oof_rnn_prob == 0.5) else 0.5:.4f}")

    gbm_thresh, gbm_f1 = best_threshold(y, oof_gbm_prob)
    print(f"LightGBM OOF F1: {gbm_f1:.4f} at threshold {gbm_thresh:.4f}, AUC: {roc_auc_score(y, oof_gbm_prob) if not np.all(oof_gbm_prob == 0.5) else 0.5:.4f}")
    cb_thresh, cb_f1 = best_threshold(y, oof_cb_prob)
    print(f"CatBoost OOF F1: {cb_f1:.4f} at threshold {cb_thresh:.4f}, AUC: {roc_auc_score(y, oof_cb_prob) if not np.all(oof_cb_prob == 0.5) else 0.5:.4f}")

    # MODIFIÉ: Ensemble incluant RNN
    oof_ensemble_prob = (oof_cnn_prob + oof_rnn_prob + oof_gbm_prob + oof_cb_prob) / 4.0
    eval_ensemble_prob = (eval_cnn_prob + eval_rnn_prob + eval_gbm_prob + eval_cb_prob) / 4.0
    ensemble_thresh, ensemble_f1 = best_threshold(y, oof_ensemble_prob)
    print(f"Ensemble (Average) OOF F1: {ensemble_f1:.4f} at threshold {ensemble_thresh:.4f}, AUC: {roc_auc_score(y, oof_ensemble_prob) if not np.all(oof_ensemble_prob == 0.5) else 0.5:.4f}")

    print(f"\nGenerating submission file using ensemble threshold: {ensemble_thresh:.4f}")
    eval_predictions = (eval_ensemble_prob >= ensemble_thresh).astype(int)
    submission_df = pd.DataFrame({'Id': range(len(eval_predictions)), 'Label': eval_predictions})
    submission_df.to_csv(SUBMISSION_CSV, index=False)
    print(f"Submission file saved to {SUBMISSION_CSV}")
    print(f"\nFinal OOF F1 score for the ensemble model: {ensemble_f1:.4f}")