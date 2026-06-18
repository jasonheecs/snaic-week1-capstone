import sys
from pathlib import Path

# src/ modules import each other with bare names (e.g. `from preprocessor import ...`),
# so put src/ on sys.path for the test suite.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
