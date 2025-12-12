from src.utils.config import config_manager


def test_params_loads():
    params = config_manager.load_params()
    assert "training" in params
    assert "data_collection" in params


