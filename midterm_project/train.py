#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import pickle

# -----------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------
depth = 10
learning_rate = 0.02
iterations = 200
n_splits = 5

params_str = f"depth_{depth}_lr_{learning_rate}_iter_{iterations}"

print(f"Parameters:\nDepth={depth}\nLearning Rate={learning_rate}\nIterations={iterations}\nSplits={n_splits}")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df_raw = pd.read_excel("Mine_Dataset.xls", sheet_name="Normalized_Data", dtype=str)
df = df_raw.apply(lambda col: col.str.replace(",", ".")).astype(float)
col_dic = {"V": "Voltage", "H": "Height", "S": "Soil_type", "M": "Mine_type"}
df = df.rename(columns=col_dic)
df["soil_type_cat"] = ((df["Soil_type"] * 5) + 1).round().astype(int)

base_features = ["Voltage", "Height", "soil_type_cat"]

# -----------------------------------------------------------
# TRAIN/VALIDATION/TEST SPLIT
# -----------------------------------------------------------
def split_df(df, target, val_ratio=0.2, test_ratio=0.2, random_state=42):

    n = len(df)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(idx)

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train+n_val]
    idx_test = idx[n_train+n_val:]

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    df_train = df_train.drop(columns=[target])
    df_val = df_val.drop(columns=[target])
    df_test = df_test.drop(columns=[target])

    return df_train, df_val, df_test, y_train, y_val, y_test


df_train, df_val, df_test, y_train, y_val, y_test = split_df(df, target="Mine_type")

# -----------------------------------------------------------
# FEATURE PREPARATION
# -----------------------------------------------------------
def prepare_X(df, features):
    df = df.copy()
    df_final = df[features]
    dicts = df_final.to_dict(orient="records")
    return dicts

# -----------------------------------------------------------
# MODEL TRAINING
# -----------------------------------------------------------
def train(df_train, y_train):
    dicts = prepare_X(df_train, base_features)
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        loss_function="MultiClass",
        verbose=0
    )
    model.fit(X_train, y_train)
    return dv, model

# -----------------------------------------------------------
# PREDICT FUNCTION
# -----------------------------------------------------------
def predict(df, dv, model):
    dicts = prepare_X(df, base_features)
    X = dv.transform(dicts)
    return model.predict_proba(X)

# -----------------------------------------------------------
# K-FOLD CROSS VALIDATION
# -----------------------------------------------------------
df_full = pd.concat([df_train, df_val], ignore_index=True)
y_full = np.concatenate([y_train, y_val])

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

auc_scores = []

for train_idx, val_idx in kfold.split(df_full):

    df_t = df_full.iloc[train_idx]
    df_v = df_full.iloc[val_idx]

    y_t = y_full[train_idx]
    y_v = y_full[val_idx]

    dv, model = train(df_t, y_t)

    y_pred_proba = predict(df_v, dv, model)

    auc = roc_auc_score(y_v, y_pred_proba, multi_class="ovr")
    auc_scores.append(auc)

print(f"KFold AUC = {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")

# -----------------------------------------------------------
# FINAL MODEL EVALUATION
# -----------------------------------------------------------
dv_final, model_final = train(df_full, y_full)
y_test_proba = predict(df_test, dv_final, model_final)

final_auc = roc_auc_score(y_test, y_test_proba, multi_class="ovr")
print(f"Final Test AUC = {final_auc:.3f}")

# -----------------------------------------------------------
# SAVE MODEL
# -----------------------------------------------------------
with open(f"model_{params_str}.bin", "wb") as f_out:
    pickle.dump((dv_final, model_final), f_out)
print(f'Model saved to file: {f_out}')
print("Model saved successfully.")

