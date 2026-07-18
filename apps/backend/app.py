import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import datetime
from loguru import logger

from .routes import forecasts, xai, pipeline
from .pipeline_runner import execute_pipeline, trigger_pipeline_sync_thread

# Initialize FastAPI application
app = FastAPI(
    title="GridSight UK Solar Forecasting API",
    description="Backend service providing probabilistic solar forecasts, Explainable AI diagnostics, and pipeline ingestion control.",
    version="1.0.0"
)

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
    """Starts the APScheduler on app startup and schedules the daily sync at 01:00 UTC."""
    scheduler.start()
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
    return {
        "project": "GridSight UK Solar Energy Forecasting",
        "api_status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "scheduled_jobs": [
            {
                "id": "daily_pipeline_sync",
                "trigger": "cron[hour=1, minute=0]",
                "timezone": "UTC",
                "next_run_time": str(scheduler.get_job("daily_pipeline_sync").next_run_time) if scheduler.get_job("daily_pipeline_sync") else "None"
            }
        ]
    }


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
