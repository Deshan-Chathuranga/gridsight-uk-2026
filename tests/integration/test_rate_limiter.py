import pytest
from fastapi.testclient import TestClient
from apps.backend.app import app
from apps.backend.middleware.rate_limiter import RateLimitMiddleware

def test_rate_limiting_triggers(monkeypatch):
    """Verifies that exceeding the limit triggers HTTP 429."""
    # Mock limits to be extremely small (2 requests per 60 seconds)
    def mock_get_limit_and_window(self, path: str):
        return 2, 60, "test_group"
        
    monkeypatch.setattr(RateLimitMiddleware, "get_limit_and_window", mock_get_limit_and_window)
    
    with TestClient(app) as client:
        # First request should succeed
        response1 = client.get("/api/health")
        assert response1.status_code == 200
        
        # Second request should succeed
        response2 = client.get("/api/health")
        assert response2.status_code == 200
        
        # Third request should fail with 429 Too Many Requests
        response3 = client.get("/api/health")
        assert response3.status_code == 429
        data = response3.json()
        assert data["status"] == "error"
        assert "Too many requests" in data["message"]
        assert "retry_after_seconds" in data
        assert "Retry-After" in response3.headers
        assert int(response3.headers["Retry-After"]) > 0

def test_rate_limiting_ip_specific(monkeypatch):
    """Verifies that rate limits are tracked per client IP."""
    # Mock limit to 1 request per 60 seconds
    def mock_get_limit_and_window(self, path: str):
        return 1, 60, "test_group_ip"
        
    monkeypatch.setattr(RateLimitMiddleware, "get_limit_and_window", mock_get_limit_and_window)
    
    with TestClient(app) as client:
        # Request from IP 1.1.1.1 should succeed
        resp1 = client.get("/api/health", headers={"X-Forwarded-For": "1.1.1.1"})
        assert resp1.status_code == 200
        
        # Subsequent request from IP 1.1.1.1 should fail
        resp2 = client.get("/api/health", headers={"X-Forwarded-For": "1.1.1.1"})
        assert resp2.status_code == 429
        
        # Request from IP 2.2.2.2 should succeed
        resp3 = client.get("/api/health", headers={"X-Forwarded-For": "2.2.2.2"})
        assert resp3.status_code == 200

def test_rate_limiting_path_exclusion(monkeypatch):
    """Verifies that non-API paths are excluded from rate limiting."""
    def mock_get_limit_and_window(self, path: str):
        return 0, 60, "test_group_exclude" # 0 limit would block all requests
        
    monkeypatch.setattr(RateLimitMiddleware, "get_limit_and_window", mock_get_limit_and_window)
    
    with TestClient(app) as client:
        # A non-API path (e.g. root or SPA catchall, even though they return 404 or index) should not return 429
        # because the middleware exits early for paths not starting with "/api"
        resp = client.get("/some-random-frontend-page")
        assert resp.status_code != 429
