import asyncio
import subprocess
from pathlib import Path
import datetime
from loguru import logger
import threading

# Global state tracker for pipeline runs
pipeline_state = {
    "status": "IDLE",  # IDLE, RUNNING, SUCCESS, FAILED
    "current_stage": "",
    "started_at": None,
    "completed_at": None,
    "last_error": None,
    "auto_triggered": False
}

state_lock = threading.Lock()

def get_pipeline_state():
    with state_lock:
        return pipeline_state.copy()

def update_pipeline_state(status=None, current_stage=None, completed=False, error=None, auto_triggered=None):
    with state_lock:
        if status is not None:
            pipeline_state["status"] = status
        if current_stage is not None:
            pipeline_state["current_stage"] = current_stage
        if error is not None:
            pipeline_state["last_error"] = error
        if auto_triggered is not None:
            pipeline_state["auto_triggered"] = auto_triggered
        
        if status == "RUNNING" and pipeline_state["started_at"] is None:
            pipeline_state["started_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            pipeline_state["completed_at"] = None
            pipeline_state["last_error"] = None
        
        if completed:
            pipeline_state["completed_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            pipeline_state["current_stage"] = "Done"

def run_command_sync(args, log_file):
    """Executes a terminal command and logs stdout/stderr to a file."""
    import os
    cmd_str = " ".join(args)
    log_file.write(f"\n--- [{datetime.datetime.utcnow().isoformat()}] EXECUTING: {cmd_str} ---\n")
    log_file.flush()
    
    logger.info(f"Pipeline runner: Executing {cmd_str}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:apps:{env.get('PYTHONPATH', '')}"
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    for line in process.stdout:
        log_file.write(line)
        log_file.flush()
        
    process.wait()
    if process.returncode != 0:
        log_file.write(f"\n--- ERROR: Command failed with exit code {process.returncode} ---\n")
        log_file.flush()
        raise subprocess.CalledProcessError(process.returncode, args)
        
    log_file.write(f"--- SUCCESS ---\n")
    log_file.flush()

def execute_pipeline(horizon_steps: int = 48, upload: bool = False, auto_triggered: bool = False):
    """Runs the full data ingestion and feature generation pipeline sequentially."""
    update_pipeline_state(status="RUNNING", current_stage="Initializing", auto_triggered=auto_triggered)
    
    log_dir = Path("data")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline_run.log"
    
    current_year = datetime.datetime.now().year
    import sys
    python_bin = sys.executable
    
    # 1. Prepare commands
    bronze_args = [python_bin, "-m", "gridsight.data.bronze", "--source", "all", "--years", str(current_year)]
    if upload:
        bronze_args.append("--upload")
        
    silver_args = [python_bin, "-m", "gridsight.data.silver", "--source", "all"]
    if upload:
        silver_args.append("--upload")
        
    gold_args = [python_bin, "-m", "gridsight.data.gold", "--horizon-steps", str(horizon_steps)]
    if upload:
        gold_args.append("--upload")
        
    # We build gold features for other horizons as well to sync everything
    other_horizons = [12, 24, 48]
    other_gold_commands = []
    for h in other_horizons:
        if h != horizon_steps:
            args = [python_bin, "-m", "gridsight.data.gold", "--horizon-steps", str(h)]
            if upload:
                args.append("--upload")
            other_gold_commands.append(args)

    try:
        with open(log_path, "w") as log_file:
            log_file.write(f"GridSight UK Ingestion Pipeline Run\n")
            log_file.write(f"Started at: {datetime.datetime.utcnow().isoformat()} UTC\n")
            log_file.write(f"Auto-triggered: {auto_triggered}\n")
            log_file.write(f"Upload to HF: {upload}\n")
            log_file.write(f"Primary Horizon Steps: {horizon_steps}\n")
            log_file.write(f"=========================================\n")
            
            # STAGE 1: Bronze Ingestion
            update_pipeline_state(current_stage="Bronze Ingestion")
            run_command_sync(bronze_args, log_file)
            
            # STAGE 2: Silver Alignment
            update_pipeline_state(current_stage="Silver Cleaning & Alignment")
            run_command_sync(silver_args, log_file)
            
            # STAGE 3: Gold Features (Primary Horizon)
            update_pipeline_state(current_stage=f"Gold Feature Generation (H-{horizon_steps})")
            run_command_sync(gold_args, log_file)
            
            # STAGE 4: Gold Features (Other Horizons)
            for args in other_gold_commands:
                h_val = args[args.index("--horizon-steps") + 1]
                update_pipeline_state(current_stage=f"Gold Feature Generation (H-{h_val})")
                run_command_sync(args, log_file)
                
            # STAGE 5: Live Inference Generation
            update_pipeline_state(current_stage="Live Inference Generation")
            inference_args = [python_bin, "-m", "apps.backend.run_live_inference"]
            run_command_sync(inference_args, log_file)
                
            log_file.write(f"\n=========================================\n")
            log_file.write(f"Pipeline completed successfully at {datetime.datetime.utcnow().isoformat()} UTC\n")
            
        update_pipeline_state(status="SUCCESS", completed=True)
        logger.info("Pipeline runner: Full pipeline sync and live inference completed successfully.")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Pipeline runner: Pipeline failed. Error: {error_msg}")
        try:
            with open(log_path, "a") as log_file:
                log_file.write(f"\n=========================================\n")
                log_file.write(f"PIPELINE FAILED: {error_msg}\n")
        except Exception:
            pass
        update_pipeline_state(status="FAILED", error=error_msg, completed=True)

def trigger_pipeline_sync_thread(horizon_steps: int = 48, upload: bool = False, auto_triggered: bool = False):
    """Spawns a background thread to run the sync pipeline to prevent blockages."""
    state = get_pipeline_state()
    if state["status"] == "RUNNING":
        logger.warning("Pipeline is already running.")
        return False
        
    thread = threading.Thread(
        target=execute_pipeline,
        args=(horizon_steps, upload, auto_triggered),
        daemon=True
    )
    thread.start()
    return True
