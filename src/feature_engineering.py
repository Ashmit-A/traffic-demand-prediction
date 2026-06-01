import pandas as pd


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from timestamp column.
    """

    if "timestamp" not in df.columns:
        return df

    split = (
        df["timestamp"]
        .astype(str)
        .str.split(":", expand=True)
    )

    df["hour"] = split[0].astype(int)
    df["minute"] = split[1].astype(int)

    return df


def create_peak_hour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traffic peak indicators.
    """

    if "hour" not in df.columns:
        return df

    df["is_morning_peak"] = (
        (df["hour"] >= 7) &
        (df["hour"] <= 10)
    ).astype(int)

    df["is_evening_peak"] = (
        (df["hour"] >= 16) &
        (df["hour"] <= 20)
    ).astype(int)

    df["is_night"] = (
        (df["hour"] >= 22) |
        (df["hour"] <= 5)
    ).astype(int)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature interactions.
    """

    if (
        "Temperature" in df.columns and
        "hour" in df.columns
    ):
        df["temp_hour"] = (
            df["Temperature"] *
            df["hour"]
        )

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    """

    df = extract_time_features(df)

    df = create_peak_hour_features(df)

    df = create_interaction_features(df)

    return df