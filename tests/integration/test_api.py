import pytest
from fastapi.testclient import TestClient
from apps.backend.app import app

@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers startup/shutdown events
    with TestClient(app) as c:
        yield c

def test_api_root(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["api_status"] == "healthy"
    assert "project" in data
    assert "scheduled_jobs" in data

def test_pipeline_pause_resume(client):
    # Test pause endpoint
    pause_res = client.post("/api/pipeline/pause")
    assert pause_res.status_code == 200
    assert pause_res.json()["status"] == "success"

    # Test status reflects paused schedule
    status_res = client.get("/api/pipeline/status")
    assert status_res.status_code == 200
    assert status_res.json()["daily_schedule"]["paused"] is True

    # Test resume endpoint
    resume_res = client.post("/api/pipeline/resume")
    assert resume_res.status_code == 200
    assert resume_res.json()["status"] == "success"

    # Test status reflects active schedule
    status_res2 = client.get("/api/pipeline/status")
    assert status_res2.status_code == 200
    assert status_res2.json()["daily_schedule"]["paused"] is False

def test_get_forecasts_valid(client):
    response = client.get("/api/forecasts?model=model_a&horizon=24&split=test")
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "metrics" in res
    assert "data" in res
    assert "mean_pinball" in res["metrics"]
    assert "coverage_80" in res["metrics"]
    assert len(res["data"]) > 0
    # Ensure fields exist in elements
    first_item = res["data"][0]
    assert "timestamp_utc" in first_item
    assert "y_true_mw" in first_item
    assert "q10_mw" in first_item
    assert "q50_mw" in first_item
    assert "q90_mw" in first_item

def test_get_forecasts_invalid_params(client):
    # Invalid model (causes ValueError which gets caught and returned as 500)
    response = client.get("/api/forecasts?model=invalid_model")
    assert response.status_code == 500

    # Invalid horizon format (string instead of int - triggers Pydantic 422)
    response = client.get("/api/forecasts?horizon=not-an-int")
    assert response.status_code == 422

def test_xai_global(client):
    response = client.get("/api/xai/global?horizon=12")
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "importances" in res
    assert len(res["importances"]) > 0
    assert "feature" in res["importances"][0]
    assert "importance" in res["importances"][0]

def test_xai_local(client):
    # Testing daylight hour response (e.g. 12:00)
    response = client.get("/api/xai/local?timestamp=2026-07-15T12:00:00Z&horizon=24")
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "base_value" in res
    assert "prediction" in res
    assert "contributions" in res

    # Testing night hour response (e.g. 02:00)
    response_night = client.get("/api/xai/local?timestamp=2026-07-15T02:00:00Z&horizon=24")
    assert response_night.status_code == 200
    res_night = response_night.json()
    assert res_night["status"] == "success"
    assert res_night["prediction"] == 0.0

def test_xai_meta(client):
    response = client.get("/api/xai/meta?horizon=24")
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "features" in res
    assert "weights" in res

def test_pipeline_status(client):
    response = client.get("/api/pipeline/status")
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "pipeline_state" in res
    assert "storage_stats" in res

def test_pipeline_logs_no_file(client):
    # Should complete without throwing 500 error even if file is missing
    response = client.get("/api/pipeline/logs")
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "success"
    assert "logs" in res
