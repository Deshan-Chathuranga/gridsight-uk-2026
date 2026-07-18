import sys
from pathlib import Path

# Add project root to sys.path so tests can import backend, lstm_q, data_ingestion etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "apps"))
