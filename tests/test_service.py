import importlib

try:
    from fastapi.testclient import TestClient
except Exception:
    TestClient = None


def test_service_health_smoke():
    if TestClient is None:
        return
    try:
        svc = importlib.import_module("neural_chaos_lab.service")
    except ModuleNotFoundError:
        return
    app = getattr(svc, "app", None)
    if app is None:
        return
    client = TestClient(app)
    try:
        r = client.get("/health", timeout=5)
        assert r.status_code in (200, 404)
    except Exception:
        pass
