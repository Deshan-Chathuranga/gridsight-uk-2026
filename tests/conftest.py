import sys
import os
from pathlib import Path

# Provide default env vars for test runs when .env file is not present (e.g. CI runners)
os.environ.setdefault("HF_TOKEN", "dummy_test_token")
os.environ.setdefault("BRONZE_HF_REPO", "gridsight-team/gridsight-bronze-data")

# Add project root to sys.path so tests can import backend, lstm_q, data_ingestion etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
