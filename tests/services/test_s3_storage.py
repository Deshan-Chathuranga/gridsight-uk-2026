import pytest
from unittest.mock import MagicMock, patch
from gridsight.services.s3_storage import S3StorageManager


def test_s3_storage_initialization():
    storage = S3StorageManager(
        bucket_name="test-bucket",
        region_name="eu-west-2",
        endpoint_url="http://localhost:4566"
    )
    assert storage.bucket_name == "test-bucket"
    assert storage.region_name == "eu-west-2"
    assert storage.endpoint_url == "http://localhost:4566"


@patch("gridsight.services.s3_storage.boto3")
def test_s3_storage_is_connected_success(mock_boto3):
    mock_client = MagicMock()
    mock_client.list_buckets.return_value = {"Buckets": []}
    mock_boto3.client.return_value = mock_client
    mock_boto3.resource.return_value = MagicMock()

    storage = S3StorageManager(bucket_name="test-bucket")
    storage._init_client()

    assert storage.is_connected() is True


@patch("gridsight.services.s3_storage.boto3")
def test_s3_storage_upload_file(mock_boto3, tmp_path):
    mock_client = MagicMock()
    mock_client.list_buckets.return_value = {"Buckets": []}
    mock_client.head_bucket.return_value = {}
    mock_boto3.client.return_value = mock_client
    mock_boto3.resource.return_value = MagicMock()

    test_file = tmp_path / "model_weights.pt"
    test_file.write_text("fake weights")

    storage = S3StorageManager(bucket_name="test-bucket")
    storage._init_client()

    success = storage.upload_file(str(test_file), "checkpoints/model_weights.pt")
    assert success is True
    mock_client.upload_file.assert_called_once_with(
        str(test_file), "test-bucket", "checkpoints/model_weights.pt"
    )


def test_s3_storage_get_status_disconnected():
    storage = S3StorageManager(bucket_name="gridsight-test")

    with patch.object(storage, "is_connected", return_value=False):
        status = storage.get_status()
        assert status["connected"] is False
        assert status["bucket"] == "gridsight-test"
