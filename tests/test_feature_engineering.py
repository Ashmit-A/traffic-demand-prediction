from src.feature_engineering import feature_engineering
import pandas as pd


def test_timestamp_features():

    df = pd.DataFrame({
        "timestamp": ["8:15"]
    })

    result = feature_engineering(df)

    assert result["hour"][0] == 8
    assert result["minute"][0] == 15