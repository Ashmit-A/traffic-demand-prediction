def test_reports_dir_exists():

    from src.config import REPORTS_DIR

    assert REPORTS_DIR.exists()