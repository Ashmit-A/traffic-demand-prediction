import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV file.
    """
    return pd.read_csv(filepath)


def remove_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove non-predictive identifier columns.
    """

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

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    """

    df = remove_identifier(df)

    df = handle_missing_values(df)

    return df