import uvicorn
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import datetime
from loguru import logger

from .routes import forecasts, xai, pipeline
from .pipeline_runner import execute_pipeline, trigger_pipeline_sync_thread
from .middleware.rate_limiter import RateLimitMiddleware

# Initialize FastAPI application
app = FastAPI(
    title="GridSight UK Solar Forecasting API",
    description="Backend service providing probabilistic solar forecasts, Explainable AI diagnostics, and pipeline ingestion control.",
    version="1.0.0"
)

# Enable rate limiting to prevent abuse
app.add_middleware(RateLimitMiddleware)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(forecasts.router, prefix="/api")
app.include_router(xai.router, prefix="/api")
app.include_router(pipeline.router, prefix="/api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TS = "timestamp_utc"
frontend_dist_path = PROJECT_ROOT / "apps" / "frontend" / "dist"

# Initialize Background Scheduler
scheduler = BackgroundScheduler()

def scheduled_sync_job():
    """Triggered daily. Syncs data and automatically pushes updates back to HuggingFace."""
    logger.info("Scheduler: Initiating scheduled daily ingestion pipeline sync...")
    # Schedule with upload=True as confirmed by the user
    trigger_pipeline_sync_thread(horizon_steps=48, upload=True, auto_triggered=True)

@app.on_event("startup")
def start_scheduler():
    """Starts the APScheduler on app startup and schedules the daily sync at 01:00 UTC unless paused."""
    scheduler.start()
    pause_daily = os.getenv("PAUSE_DAILY_PIPELINE", "true").lower() in ("true", "1", "yes")
    
    if pause_daily:
        logger.info("Daily ingestion pipeline schedule is currently PAUSED (PAUSE_DAILY_PIPELINE=true).")
    else:
        # Schedule daily at 01:00 UTC
        scheduler.add_job(
            scheduled_sync_job,
            trigger=CronTrigger(hour=1, minute=0, timezone="UTC"),
            id="daily_pipeline_sync",
            replace_existing=True
        )
        logger.success("Scheduler started. Scheduled daily sync job at 01:00 UTC.")

@app.on_event("shutdown")
def stop_scheduler():
    """Gracefully shuts down the scheduler on app termination."""
    scheduler.shutdown()
    logger.info("Scheduler shut down successfully.")

@app.get("/")
def read_root():
    if frontend_dist_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(frontend_dist_path / "index.html"))
    job = scheduler.get_job("daily_pipeline_sync")
    return {
        "project": "GridSight UK Solar Energy Forecasting",
        "api_status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "scheduled_jobs": [
            {
                "id": "daily_pipeline_sync",
                "trigger": "cron[hour=1, minute=0]",
                "timezone": "UTC",
                "status": "active" if job else "paused",
                "next_run_time": str(job.next_run_time) if job else "Paused"
            }
        ]
    }


@app.get("/api/health")
def health_check():
    job = scheduler.get_job("daily_pipeline_sync")
    return {
        "project": "GridSight UK Solar Energy Forecasting",
        "api_status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "scheduled_jobs": [
            {
                "id": "daily_pipeline_sync",
                "trigger": "cron[hour=1, minute=0]",
                "timezone": "UTC",
                "status": "active" if job else "paused",
                "next_run_time": str(job.next_run_time) if job else "Paused"
            }
        ]
    }


@app.get("/api/debug_import")
def debug_import():
    import sys
    import os
    import traceback
    
    debug_info = {
        "sys.path": sys.path,
        "cwd": os.getcwd(),
        "environ_pythonpath": os.environ.get("PYTHONPATH"),
        "exists_src": os.path.exists("src"),
        "exists_gridsight": os.path.exists("src/gridsight"),
        "exists_gridsight_data": os.path.exists("src/gridsight/data"),
        "dir_contents_root": os.listdir("."),
        "dir_contents_src": os.listdir("src") if os.path.exists("src") else []
    }
    
    try:
        import gridsight
        debug_info["import_gridsight"] = "success"
    except Exception as e:
        debug_info["import_gridsight"] = traceback.format_exc()
        
    try:
        import gridsight.data
        debug_info["import_gridsight_data"] = "success"
    except Exception as e:
        debug_info["import_gridsight_data"] = traceback.format_exc()
        
    try:
        import gridsight.data.bronze
        debug_info["import_gridsight_data_bronze"] = "success"
    except Exception as e:
        debug_info["import_gridsight_data_bronze"] = traceback.format_exc()
        
    return debug_info


@app.get("/api/debug_xai")
def debug_xai(horizon: int = 24):
    import traceback
    import joblib
    
    steps = {6: 12, 12: 24, 24: 48}[horizon]
    folder = "model" if steps == 48 else f"model_h{steps}"
    stack_path = ARTIFACTS_DIR / folder / "stack.joblib"
    
    info = {
        "stack_path": str(stack_path),
        "exists": stack_path.exists(),
    }
    
    if not stack_path.exists():
        return {"info": info, "status": "stack.joblib does not exist"}
        
    try:
        art = joblib.load(stack_path)
        info["keys"] = list(art.keys()) if isinstance(art, dict) else str(type(art))
        
        # Test global importance logic
        try:
            features = art["features"]
            lgbm = art["lgbm"]
            info["lgbm_type"] = str(type(lgbm))
            info["lgbm_dir"] = dir(lgbm)
            
            # Check feature importance
            if hasattr(lgbm, "feature_importance"):
                importances_dict = lgbm.feature_importance()
                info["global_test"] = f"success: {importances_dict}"
            elif hasattr(lgbm, "feature_importances_"):
                info["global_test"] = f"feature_importances_ exists: {lgbm.feature_importances_}"
            else:
                info["global_test"] = "neither method exists"
        except Exception as e1:
            info["global_test"] = traceback.format_exc()
            
        # Test meta weight logic
        try:
            meta = art["meta"]
            info["meta_type"] = str(type(meta))
            info["meta_dir"] = dir(meta)
            
            # Check model fields
            if hasattr(meta, "models_"):
                weights = {}
                for q, model in meta.models_.items():
                    coefs = model.coef_.tolist()
                    weights[float(q)] = [float(c) for c in coefs]
                info["meta_test"] = f"success: {weights}"
            else:
                info["meta_test"] = "models_ attribute missing"
        except Exception as e2:
            info["meta_test"] = traceback.format_exc()
            
    except Exception as e:
        info["load_error"] = traceback.format_exc()
        
    return info






# Serve React static assets in production (registered after root "/" so it doesn't shadow it)
if frontend_dist_path.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from fastapi import HTTPException
    
    app.mount("/assets", StaticFiles(directory=str(frontend_dist_path / "assets")), name="assets")
    
    @app.get("/{catchall:path}")
    def serve_react_app(catchall: str):
        # Prevent API routing conflicts
        if catchall.startswith("api"):
            raise HTTPException(status_code=404, detail="API route not found")
            
        # Serve public files (favicon, etc.) if they exist in frontend/dist
        public_file = frontend_dist_path / catchall
        if public_file.exists() and public_file.is_file():
            return FileResponse(str(public_file))
            
        # Fallback to index.html for React SPA client-side routing
        return FileResponse(str(frontend_dist_path / "index.html"))

if __name__ == "__main__":
    uvicorn.run("apps.backend.app:app", host="0.0.0.0", port=8000, reload=True)
