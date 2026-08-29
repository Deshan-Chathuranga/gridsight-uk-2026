import pytest
from fastapi.testclient import TestClient
from apps.backend.app import app

client = TestClient(app)


def test_storage_status_endpoint():
    response = client.get("/api/storage/status")
    assert response.status_code == 200
    data = response.json()
    assert "boto3_installed" in data
    assert "connected" in data
    assert "bucket" in data
    assert data["boto3_installed"] is True
