import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score # f1_score est déjà importé
from pathlib import Path
import random
import lightgbm as lgb
from catboost import CatBoostClassifier

# --- Configuration ---
SEED = 98
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = Path('../data/')
TRAIN_NPZ = DATA_DIR / 'training_data.npz'
TRAIN_LABEL_CSV = DATA_DIR / 'training_labels.csv'
EVAL_NPZ = DATA_DIR / 'evaluation_data.npz'
SUBMISSION_CSV = DATA_DIR / 'submission_ensemble_f1.csv'

# --- Chargement des données ---
print("Chargement des données...")
with np.load(TRAIN_NPZ, allow_pickle=True) as f:
    X_train_full_original = f['data'].astype(np.float32)
    feat_labels = f['feature_labels'].tolist()
y_train_full_original = pd.read_csv(TRAIN_LABEL_CSV)['Label'].values

with np.load(EVAL_NPZ, allow_pickle=True) as f:
    X_eval_original = f['data'].astype(np.float32)
    if X_eval_original.shape[2] != X_train_full_original.shape[2]:
        raise ValueError(f"Le nombre de features de X_eval ({X_eval_original.shape[2]})"
                         f" ne correspond pas à X_train ({X_train_full_original.shape[2]})")

print(f"Données d'entraînement initiales: {X_train_full_original.shape}")
print(f"Labels d'entraînement initiaux: {y_train_full_original.shape}")
print(f"Données d'évaluation initiales: {X_eval_original.shape}")

# --- Prétraitement ---
print("\nDébut du prétraitement...")

# 1. Filtrage par âge
try:
    age_idx = feat_labels.index('patient_age')
except ValueError:
    print("ERREUR: La feature 'patient_age' n'a pas été trouvée.")
    exit()

patient_ages_all_timesteps = X_train_full_original[:, :, age_idx]
actual_ages_train = []
for i in range(X_train_full_original.shape[0]):
    patient_specific_ages = patient_ages_all_timesteps[i, :]
    valid_ages_for_patient = patient_specific_ages[ (patient_specific_ages > 0) & (~np.isnan(patient_specific_ages)) ]
    actual_ages_train.append(valid_ages_for_patient[0] if len(valid_ages_for_patient) > 0 else 0)
actual_ages_train = np.array(actual_ages_train)
age_mask_train = actual_ages_train >= 1

X_train_full = X_train_full_original[age_mask_train]
y_train_full = y_train_full_original[age_mask_train]
print(f"Nombre d'échantillons après filtrage par âge: {X_train_full.shape[0]}")
if X_train_full.shape[0] == 0: exit("Aucun patient de >= 30 ans.")

# 2. Imputation des NaN à 0
print("  Remplacement des NaNs par 0.0...")
X_train_full_imputed = np.nan_to_num(X_train_full, nan=0.0)
X_eval_imputed = np.nan_to_num(X_eval_original, nan=0.0)

""" 3. Gestion des données aberrantes (Clipping)
print("  Gestion des données aberrantes (clipping)...")
def clip_outliers(data, lower_percentile=1, upper_percentile=99, train_clips=None):
    data_clipped = data.copy()
    num_features = data.shape[2]

    if train_clips is None:
        train_clips_calculated = []
        reshaped_data = data_clipped.reshape(-1, num_features)
        for i in range(num_features):
            lower_bound = np.percentile(reshaped_data[:, i], lower_percentile)
            upper_bound = np.percentile(reshaped_data[:, i], upper_percentile)
            train_clips_calculated.append((lower_bound, upper_bound))
            data_clipped[:, :, i] = np.clip(data_clipped[:, :, i], lower_bound, upper_bound)
        return data_clipped, train_clips_calculated
    else:
        for i in range(num_features):
            lower_bound, upper_bound = train_clips[i]
            data_clipped[:, :, i] = np.clip(data_clipped[:, :, i], lower_bound, upper_bound)
        return data_clipped, None
"""
X_train_full_clipped, feature_clips_train = X_train_full_imputed, None
X_eval_clipped, _ = X_eval_imputed, None

# 4. Normalisation (StandardScaler)
print("  Normalisation (StandardScaler)...")
scaler = StandardScaler()
X_train_reshaped_for_scaler = X_train_full_clipped.reshape(-1, X_train_full_clipped.shape[2])
X_eval_reshaped_for_scaler = X_eval_clipped.reshape(-1, X_eval_clipped.shape[2])

scaler.fit(X_train_reshaped_for_scaler)
X_train_scaled_reshaped = scaler.transform(X_train_reshaped_for_scaler)
X_eval_scaled_reshaped = scaler.transform(X_eval_reshaped_for_scaler)

X_train_full_processed = X_train_scaled_reshaped.reshape(X_train_full_clipped.shape)
X_eval_processed = X_eval_scaled_reshaped.reshape(X_eval_clipped.shape)

# 5. Préparation des données pour les différents modèles
X_train_full_cnn = np.expand_dims(X_train_full_processed, axis=1)
X_eval_cnn = np.expand_dims(X_eval_processed, axis=1)

n_samples_train, n_timesteps, n_features_orig = X_train_full_processed.shape
X_train_full_2d = X_train_full_processed.reshape(n_samples_train, n_timesteps * n_features_orig)
n_samples_eval, _, _ = X_eval_processed.shape
X_eval_2d = X_eval_processed.reshape(n_samples_eval, n_timesteps * n_features_orig)

print(f"Shape X_train_full pour CNN: {X_train_full_cnn.shape}")
print(f"Shape X_train_full pour GBMs: {X_train_full_2d.shape}")
print(f"Shape X_eval pour CNN: {X_eval_cnn.shape}")
print(f"Shape X_eval pour GBMs: {X_eval_2d.shape}")

# --- Division Entraînement / Validation Stratifiée ---
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
try:
    train_idx, val_idx = next(sss.split(X_train_full_processed, y_train_full))
except ValueError as e:
    print(f"Erreur StratifiedShuffleSplit: {e}. Vérifiez la distribution des classes et la taille des données.")
    exit()

X_train_cnn, X_val_cnn = X_train_full_cnn[train_idx], X_train_full_cnn[val_idx]
y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
X_train_2d_split, X_val_2d_split = X_train_full_2d[train_idx], X_train_full_2d[val_idx]

print(f"\nCNN Train: {X_train_cnn.shape}, Val: {X_val_cnn.shape}")
print(f"GBM Train: {X_train_2d_split.shape}, Val: {X_val_2d_split.shape}")
print(f"Labels Train: {y_train.shape}, Val: {y_val.shape}")
if len(y_train) == 0 or len(y_val) == 0: exit("Ensemble train ou val vide après split.")

# --- Modèle CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, height, width, num_classes=1):
        super(SimpleCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(3, 5), padding=(1, 2)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        dummy_input = torch.randn(1, input_channels, height, width)
        dummy_output = self.conv_block(dummy_input)
        flattened_size = dummy_output.numel()
        self.fc_block = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(flattened_size, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc_block(x)

def train_and_predict_cnn(X_train_tensor, y_train_tensor, X_val_tensor, X_eval_tensor, y_val_labels):
    BATCH_SIZE_CNN = 128
    N_EPOCHS_CNN = 10
    LEARNING_RATE_CNN = 1e-4

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.float().unsqueeze(1))
    val_dataset_for_loss = TensorDataset(X_val_tensor, torch.from_numpy(y_val_labels).float().unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_CNN, shuffle=True)
    val_loader_for_loss = DataLoader(val_dataset_for_loss, batch_size=BATCH_SIZE_CNN, shuffle=False)

    input_c, input_h, input_w = X_train_tensor.shape[1], X_train_tensor.shape[2], X_train_tensor.shape[3]
    model_cnn = SimpleCNN(input_c, input_h, input_w).to(DEVICE)

    neg_count = torch.sum(y_train_tensor == 0).item()
    pos_count = torch.sum(y_train_tensor == 1).item()
    pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=DEVICE))
    optimizer = optim.Adam(model_cnn.parameters(), lr=LEARNING_RATE_CNN)

    print("Entraînement du CNN...")
    for epoch in range(N_EPOCHS_CNN):
        model_cnn.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad(); outputs = model_cnn(batch_X); loss = criterion(outputs, batch_y)
            loss.backward(); optimizer.step()
        model_cnn.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for batch_X_v, batch_y_v in val_loader_for_loss:
                outputs_v = model_cnn(batch_X_v.to(DEVICE))
                val_loss_epoch += criterion(outputs_v, batch_y_v.to(DEVICE)).item() * batch_X_v.size(0)
        print(f"CNN Epoch {epoch+1}/{N_EPOCHS_CNN}, Val Loss: {val_loss_epoch/len(val_loader_for_loss.dataset):.4f}")

    model_cnn.eval()
    cnn_val_preds_list = []
    val_pred_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=BATCH_SIZE_CNN, shuffle=False)
    with torch.no_grad():
        for batch_X in val_pred_loader:
            outputs = model_cnn(batch_X[0].to(DEVICE))
            cnn_val_preds_list.append(torch.sigmoid(outputs).cpu().numpy())
    cnn_val_probs = np.concatenate(cnn_val_preds_list).squeeze()

    cnn_eval_preds_list = []
    eval_pred_loader = DataLoader(TensorDataset(X_eval_tensor), batch_size=BATCH_SIZE_CNN, shuffle=False)
    with torch.no_grad():
        for batch_X in eval_pred_loader:
            outputs = model_cnn(batch_X[0].to(DEVICE))
            cnn_eval_preds_list.append(torch.sigmoid(outputs).cpu().numpy())
    cnn_eval_probs = np.concatenate(cnn_eval_preds_list).squeeze()

    return cnn_val_probs, cnn_eval_probs

print("\n--- Entraînement Modèle CNN ---")
cnn_val_probs, cnn_eval_probs = train_and_predict_cnn(
    torch.from_numpy(X_train_cnn).float(),
    torch.from_numpy(y_train).float(),
    torch.from_numpy(X_val_cnn).float(),
    torch.from_numpy(X_eval_cnn).float(),
    y_val
)
cnn_val_preds_binary = (cnn_val_probs > 0.5).astype(int) # Pour F1-score
print(f"CNN Val Probs Shape: {cnn_val_probs.shape}, Eval Probs Shape: {cnn_eval_probs.shape}")

# --- Modèles LightGBM et CatBoost ---
print("\n--- Entraînement Modèle LightGBM ---")
lgbm_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1,
    'seed': SEED, 'n_jobs': -1, 'verbose': -1, 'colsample_bytree': 0.7,
    'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'scale_pos_weight': np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
}
model_lgbm = lgb.LGBMClassifier(**lgbm_params)
model_lgbm.fit(X_train_2d_split, y_train, eval_set=[(X_val_2d_split, y_val)],
               callbacks=[lgb.early_stopping(10, verbose=False)])
lgbm_val_probs = model_lgbm.predict_proba(X_val_2d_split)[:, 1]
lgbm_eval_probs = model_lgbm.predict_proba(X_eval_2d)[:, 1]
lgbm_val_preds_binary = (lgbm_val_probs > 0.5).astype(int) # Pour F1-score
print(f"LGBM Val Probs Shape: {lgbm_val_probs.shape}, Eval Probs Shape: {lgbm_eval_probs.shape}")

print("\n--- Entraînement Modèle CatBoost ---")
cat_params = {
    'iterations': 200, 'learning_rate': 0.05, 'depth': 6, 'loss_function': 'Logloss',
    'eval_metric': 'AUC', 'random_seed': SEED, 'verbose': 0, 'auto_class_weights': 'Balanced'
}
model_catboost = CatBoostClassifier(**cat_params)
model_catboost.fit(X_train_2d_split, y_train, eval_set=[(X_val_2d_split, y_val)], early_stopping_rounds=10)
catboost_val_probs = model_catboost.predict_proba(X_val_2d_split)[:, 1]
catboost_eval_probs = model_catboost.predict_proba(X_eval_2d)[:, 1]
catboost_val_preds_binary = (catboost_val_probs > 0.5).astype(int) # Pour F1-score
print(f"CatBoost Val Probs Shape: {catboost_val_probs.shape}, Eval Probs Shape: {catboost_eval_probs.shape}")

# --- Méta-Modèle (Stacking) ---
print("\n--- Entraînement Méta-Modèle ---")
if not (len(cnn_val_probs) == len(lgbm_val_probs) == len(catboost_val_probs) == len(y_val)):
    print("ERREUR: Les longueurs des prédictions de validation ne correspondent pas!")
    exit()
X_meta_train = np.column_stack((cnn_val_probs, lgbm_val_probs, catboost_val_probs))

if not (len(cnn_eval_probs) == len(lgbm_eval_probs) == len(catboost_eval_probs)):
    print("ERREUR: Les longueurs des prédictions d'évaluation ne correspondent pas!")
    exit()
X_meta_eval = np.column_stack((cnn_eval_probs, lgbm_eval_probs, catboost_eval_probs))

meta_model = LogisticRegression(solver='liblinear', random_state=SEED, class_weight='balanced')
meta_model.fit(X_meta_train, y_val)

# Prédictions du méta-modèle sur l'ensemble de validation (pour évaluer le méta-modèle lui-même)
meta_val_probs = meta_model.predict_proba(X_meta_train)[:, 1]
meta_val_preds_binary = (meta_val_probs > 0.5).astype(int)

# Prédictions finales sur l'ensemble d'évaluation
ensemble_eval_probs = meta_model.predict_proba(X_meta_eval)[:, 1]
ensemble_eval_preds_binary = (ensemble_eval_probs > 0.5).astype(int)

print(f"Prédictions finales de l'ensemble (premiers 10): {ensemble_eval_preds_binary[:10]}")
print(f"Probabilités finales de l'ensemble (premiers 10): {ensemble_eval_probs[:10]}")

# --- Soumission ---
submission_df = pd.DataFrame({
    'ID': range(len(ensemble_eval_probs)),
    'Label_prob': ensemble_eval_probs,
    'Label_binary': ensemble_eval_preds_binary
})
submission_df.to_csv(SUBMISSION_CSV, index=False)
print(f"Fichier de soumission sauvegardé dans {SUBMISSION_CSV}")

# --- Évaluation F1-score et AUC sur l'ensemble de validation ---
print("\n--- Évaluation des modèles sur l'ensemble de validation (y_val) ---")
print(f"Seuil de classification pour F1-score: 0.5")

cnn_val_f1 = f1_score(y_val, cnn_val_preds_binary)
cnn_val_auc = roc_auc_score(y_val, cnn_val_probs)
print(f"CNN      | Val F1: {cnn_val_f1:.4f} | Val AUC: {cnn_val_auc:.4f}")

lgbm_val_f1 = f1_score(y_val, lgbm_val_preds_binary)
lgbm_val_auc = roc_auc_score(y_val, lgbm_val_probs)
print(f"LightGBM | Val F1: {lgbm_val_f1:.4f} | Val AUC: {lgbm_val_auc:.4f}")

catboost_val_f1 = f1_score(y_val, catboost_val_preds_binary)
catboost_val_auc = roc_auc_score(y_val, catboost_val_probs)
print(f"CatBoost | Val F1: {catboost_val_f1:.4f} | Val AUC: {catboost_val_auc:.4f}")

meta_model_val_f1 = f1_score(y_val, meta_val_preds_binary)
meta_model_val_auc = roc_auc_score(y_val, meta_val_probs)
print(f"Méta-Modèle | Val F1: {meta_model_val_f1:.4f} | Val AUC: {meta_model_val_auc:.4f} (évalué sur ses données d'entraînement)")

print("\nScript d'ensemble learning avec F1-score terminé.")