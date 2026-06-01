from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from config import (
    MODEL_PATH,
    PREDICTIONS_DIR,
)

from preprocess import preprocess
from feature_engineering import feature_engineering


def load_model():
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    return model


def predict_csv(input_csv: str):

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(input_csv)

    print(df.shape)

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    df = preprocess(df)

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    df = feature_engineering(df)

    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = load_model()

    predictions = model.predict(df)

    output = pd.DataFrame()

    output["prediction"] = predictions

    output_file = (
        PREDICTIONS_DIR /
        "predictions.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )

    print("\nSaved:")
    print(output_file)

    return output


if __name__ == "__main__":

    predict_csv(
        "data/raw/test.csv"
    )