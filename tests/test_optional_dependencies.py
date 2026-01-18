import importlib
import importlib.util
import sys

import pytest


def test_server_import_does_not_require_numpy():
    sys.modules.pop("mssp_platform_server", None)
    import mssp_platform_server

    assert mssp_platform_server is not None


def test_anomaly_feature_requires_extras(monkeypatch):
    if importlib.util.find_spec("numpy") is not None:
        pytest.skip("numpy installed; anomaly extras available")

    monkeypatch.setenv("ENABLE_ANOMALY", "1")
    sys.modules.pop("mssp_platform_server", None)
    module = importlib.import_module("mssp_platform_server")

    with pytest.raises(RuntimeError, match="Anomaly detection feature requires extra dependencies"):
        module._load_anomaly_detector()
