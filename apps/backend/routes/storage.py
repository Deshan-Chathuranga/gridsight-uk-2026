from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
from gridsight.services.s3_storage import get_storage_manager

router = APIRouter(prefix="/storage", tags=["Storage & Cloud Services"])


@router.get("/status", response_model=Dict[str, Any])
def get_storage_status():
    """
    Get AWS S3 / LocalStack cloud storage connection status and configuration.
    """
    storage = get_storage_manager()
    return storage.get_status()


@router.get("/artifacts", response_model=List[Dict[str, Any]])
def list_storage_artifacts(prefix: str = Query("", description="S3 object key prefix filter")):
    """
    List model artifacts, checkpoints, or datasets stored in AWS S3 / LocalStack.
    """
    storage = get_storage_manager()
    if not storage.is_connected():
        raise HTTPException(status_code=503, detail="Cloud storage (AWS S3 / LocalStack) is not connected.")
    return storage.list_artifacts(prefix=prefix)


@router.post("/sync-checkpoint")
def sync_checkpoint_to_s3(file_path: str = Query(..., description="Local file path relative to project root")):
    """
    Upload a local checkpoint or dataset file to cloud storage.
    """
    storage = get_storage_manager()
    if not storage.is_connected():
        raise HTTPException(status_code=503, detail="Cloud storage (AWS S3 / LocalStack) is not connected.")

    s3_key = file_path.replace("\\", "/")
    success = storage.upload_file(file_path, s3_key)

    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to upload {file_path} to S3.")

    return {
        "status": "success",
        "message": f"Successfully uploaded {file_path} to s3://{storage.bucket_name}/{s3_key}"
    }
