from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
import os
from pathlib import Path
import glob
from typing import Optional
from ..pipeline_runner import trigger_pipeline_sync_thread, get_pipeline_state

router = APIRouter(prefix="/pipeline", tags=["Pipeline Ingestion"])

# Helper paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_PATH = DATA_DIR / "pipeline_run.log"

@router.post("/sync")
def trigger_sync(
    horizon_steps: int = Query(48, enum=[12, 24, 48]),
    upload: bool = Query(False)
):
    """Manually triggers the data sync pipeline (Bronze -> Silver -> Gold)."""
    # Check if already running
    state = get_pipeline_state()
    if state["status"] == "RUNNING":
        return {"status": "error", "message": "Pipeline is already running."}
        
    triggered = trigger_pipeline_sync_thread(horizon_steps=horizon_steps, upload=upload, auto_triggered=False)
    if triggered:
        return {"status": "success", "message": "Pipeline sync triggered successfully."}
    else:
        return {"status": "error", "message": "Failed to trigger pipeline sync."}

@router.get("/status")
def get_status():
    """Returns the current state of the pipeline and directory health statistics."""
    state = get_pipeline_state()
    
    # Calculate folder sizes to check ingestion health
    def get_dir_stats(path: Path) -> dict:
        if not path.exists():
            return {"exists": False, "size_mb": 0.0, "file_count": 0}
        files = glob.glob(str(path / "**" / "*.parquet"), recursive=True)
        size = sum(os.path.getsize(f) for f in files)
        return {
            "exists": True,
            "size_mb": round(size / (1024 * 1024), 2),
            "file_count": len(files)
        }
        
    stats = {
        "bronze": get_dir_stats(DATA_DIR / "bronze"),
        "silver": get_dir_stats(DATA_DIR / "silver"),
        "gold": get_dir_stats(DATA_DIR / "gold")
    }
    
    return {
        "status": "success",
        "pipeline_state": state,
        "storage_stats": stats
    }

@router.get("/logs")
def get_logs(offset: int = Query(0)):
    """Reads the pipeline execution logs. Supports fetching only new content since the last offset."""
    if not LOG_PATH.exists():
        return {"status": "success", "logs": "No log file found. Trigger the pipeline first.", "offset": 0}
        
    try:
        with open(LOG_PATH, "r") as f:
            f.seek(offset)
            content = f.read()
            new_offset = f.tell()
            
        return {
            "status": "success",
            "logs": content,
            "offset": new_offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")
