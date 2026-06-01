import pandas as pd
from pathlib import Path


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load CSV file.
    """
    print(f"\nLoading data from {filepath}...")
    return pd.read_csv(filepath)


def remove_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove non-predictive identifier columns.
    """
    print("\nRemoving identifier columns...")
    if "Index" in df.columns:
        df = df.drop(columns=["Index"])

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values.
    """

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    numerical_cols = df.select_dtypes(
        include=["number"]
    ).columns

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    for col in numerical_cols:
        if col != "demand":
            df[col] = df[col].fillna(df[col].median())

    print("\nMissing values handled.")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    """

    df = remove_identifier(df)

    df = handle_missing_values(df)

    print("\nPreprocessing completed.")
    return df