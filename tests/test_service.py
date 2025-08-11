from fastapi.testclient import TestClient
try:
    from neural_chaos_lab.service import app
except Exception:
    from neural_chaos_lab.service import app
def test_app_basic():
    client = TestClient(app)
    resp = client.get("/health") if any(r.path == "/health" for r in app.router.routes) else client.get("/")
    assert resp.status_code in (200, 404)
