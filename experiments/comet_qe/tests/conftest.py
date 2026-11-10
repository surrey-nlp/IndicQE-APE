"""Put the framework's `src/` on the import path for tests.

The training scripts run as `python src/train.py` from the repo root, so modules
import each other by bare name (`from qe_data import ...`). Tests need the same
path setup.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
