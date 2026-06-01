from pathlib import Path

import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from preprocess import load_data, preprocess
from feature_engineering import feature_engineering
from config import (
    TRAIN_DATA_PATH,
    MODEL_PATH,
    TARGET,
    CATBOOST_PARAMS
)


def get_categorical_columns(df):
    return list(
        df.select_dtypes(include=["object"]).columns
    )


def train():

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = load_data(TRAIN_DATA_PATH)
    
    print(df.shape)

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    df = preprocess(df)

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = feature_engineering(df)

    print(df.shape)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    cat_features = get_categorical_columns(X)

    print("\nCategorical Features:")
    print(cat_features)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTrain Shape:", X_train.shape)
    print("Valid Shape:", X_valid.shape)

    cat_feature_indices = [
        X_train.columns.get_loc(col)
        for col in cat_features
    ]

    print("\nTraining CatBoost...")

    model = CatBoostRegressor(
    **CATBOOST_PARAMS
)

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    predictions = model.predict(X_valid)

    r2 = r2_score(
        y_valid,
        predictions
    )

    score = max(
        0,
        100 * r2
    )

    print("\nValidation R²:", r2)
    print("Competition Score:", score)

    Path("models").mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_model(MODEL_PATH)

    print("\nModel Saved:")
    print(MODEL_PATH)


if __name__ == "__main__":
    train()