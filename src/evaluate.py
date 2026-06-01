from pathlib import Path

import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score

from src.config import (
    TRAIN_DATA_PATH,
    TARGET,
    CATBOOST_PARAMS
)

from src.preprocess import (
    load_data,
    preprocess
)

from src.feature_engineering import (
    feature_engineering
)

df = load_data(TRAIN_DATA_PATH)

df = preprocess(df)

df = feature_engineering(df)

X = df.drop(columns=[TARGET])

y = df[TARGET]
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

cat_features = list(
    X.select_dtypes(
        include=["object", "string"]
    ).columns
)

cat_feature_indices = [
    X.columns.get_loc(col)
    for col in cat_features
]

print("\nCategorical Features:")
print(cat_features)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = []

print("\nStarting Cross Validation...\n")

for fold, (train_idx, val_idx) in enumerate(
    kf.split(X),
    start=1
):

    print(f"Fold {fold}/5")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = CatBoostRegressor(
        **CATBOOST_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        verbose=0
    )

    preds = model.predict(X_val)

    score = r2_score(
        y_val,
        preds
    )

    scores.append(score)

    print(
        f"R² = {score:.4f}"
    )
    print()

print("=" * 60)

print("Cross Validation Results")

print("=" * 60)

print(
    "Mean R²:",
    np.mean(scores)
)

print(
    "Std Dev:",
    np.std(scores)
)