"""
GridSight UK Services Module
Contains AWS cloud services integrations and LocalStack emulators.
"""

from .s3_storage import S3StorageManager, get_storage_manager

__all__ = ["S3StorageManager", "get_storage_manager"]
