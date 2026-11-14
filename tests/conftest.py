import os
import sys
from pathlib import Path

os.environ["DEBIAS_FAKE"] = "1"
os.environ.setdefault("DEBIAS_ARTIFACTS", str(Path(__file__).parent / ".artifacts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
