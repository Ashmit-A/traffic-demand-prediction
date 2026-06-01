from pathlib import Path


def test_model_exists():

    model_path = Path(
        "models/catboost_model.cbm"
    )

    assert model_path.exists()