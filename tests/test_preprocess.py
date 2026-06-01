from src.preprocess import preprocess
import pandas as pd


def test_preprocess():

    df = pd.DataFrame({
        "Index": [1],
        "Weather": [None],
        "Temperature": [None]
    })

    result = preprocess(df)

    assert "Index" not in result.columns
    assert result["Weather"].isnull().sum() == 0
    assert result["Temperature"].isnull().sum() == 0