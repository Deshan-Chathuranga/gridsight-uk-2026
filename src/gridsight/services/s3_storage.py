import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger("gridsight.services.s3_storage")


class S3StorageManager:
    """
    AWS S3 Cloud Storage & LocalStack Emulator Manager for GridSight UK.
    Handles uploading/downloading model checkpoints, Silver/Gold parquet feature tables,
    and fan plot artifacts.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        self.bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET", "gridsight-uk-storage")
        self.region_name = region_name or os.getenv("AWS_REGION", "eu-west-2")
        self.endpoint_url = endpoint_url or os.getenv("AWS_ENDPOINT_URL", None)

        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        self.client = None
        self.resource = None

        if BOTO3_AVAILABLE:
            self._init_client()

    def _init_client(self):
        """Initializes the boto3 S3 client with optional endpoint_url (for LocalStack)."""
        try:
            kwargs: Dict[str, Any] = {"region_name": self.region_name}

            if self.aws_access_key_id and self.aws_secret_access_key:
                kwargs["aws_access_key_id"] = self.aws_access_key_id
                kwargs["aws_secret_access_key"] = self.aws_secret_access_key

            if self.endpoint_url:
                kwargs["endpoint_url"] = self.endpoint_url

            self.client = boto3.client("s3", **kwargs)
            self.resource = boto3.resource("s3", **kwargs)
        except Exception as e:
            logger.warning(f"Failed to initialize boto3 S3 client: {e}")
            self.client = None
            self.resource = None

    def is_connected(self) -> bool:
        """Checks if S3 storage (AWS or LocalStack) is reachable."""
        if not BOTO3_AVAILABLE or not self.client:
            return False
        try:
            self.client.list_buckets()
            return True
        except Exception as e:
            logger.debug(f"S3 connection check failed: {e}")
            return False

    def ensure_bucket_exists(self) -> bool:
        """Ensures that the configured S3 bucket exists, creating it if necessary."""
        if not self.is_connected():
            return False
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as err:
            error_code = err.response.get("Error", {}).get("Code")
            if error_code == "404":
                try:
                    if self.region_name == "us-east-1":
                        self.client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={"LocationConstraint": self.region_name}
                        )
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                    return True
                except Exception as create_err:
                    logger.error(f"Failed to create bucket {self.bucket_name}: {create_err}")
                    return False
            return False
        except Exception as e:
            logger.error(f"Error checking bucket {self.bucket_name}: {e}")
            return False

    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Uploads a local file to S3 storage."""
        if not self.is_connected():
            logger.warning("S3 storage is not connected. Skipping upload.")
            return False

        path = Path(local_path)
        if not path.exists():
            logger.error(f"Local file does not exist: {local_path}")
            return False

        try:
            self.ensure_bucket_exists()
            self.client.upload_file(str(path), self.bucket_name, s3_key)
            logger.info(f"Successfully uploaded {local_path} -> s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to S3: {e}")
            return False

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Downloads an S3 object to a local path."""
        if not self.is_connected():
            logger.warning("S3 storage is not connected. Skipping download.")
            return False

        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.client.download_file(self.bucket_name, s3_key, str(path))
            logger.info(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download s3://{self.bucket_name}/{s3_key}: {e}")
            return False

    def list_artifacts(self, prefix: str = "") -> List[Dict[str, Any]]:
        """Lists files and metadata stored under a prefix in the S3 bucket."""
        if not self.is_connected():
            return []

        try:
            response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            contents = response.get("Contents", [])
            return [
                {
                    "key": item["Key"],
                    "size_bytes": item["Size"],
                    "last_modified": item["LastModified"].isoformat() if hasattr(item["LastModified"], "isoformat") else str(item["LastModified"])
                }
                for item in contents
            ]
        except Exception as e:
            logger.error(f"Failed to list S3 artifacts under prefix '{prefix}': {e}")
            return []

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of S3 connection and bucket."""
        connected = self.is_connected()
        status = {
            "boto3_installed": BOTO3_AVAILABLE,
            "connected": connected,
            "bucket": self.bucket_name,
            "region": self.region_name,
            "mode": "LocalStack Emulator" if self.endpoint_url else "AWS Cloud S3",
            "endpoint_url": self.endpoint_url
        }

        if connected:
            artifacts = self.list_artifacts()
            status["object_count"] = len(artifacts)

        return status


_storage_manager_instance: Optional[S3StorageManager] = None


def get_storage_manager() -> S3StorageManager:
    """Singleton getter for S3StorageManager."""
    global _storage_manager_instance
    if _storage_manager_instance is None:
        _storage_manager_instance = S3StorageManager()
    return _storage_manager_instance
