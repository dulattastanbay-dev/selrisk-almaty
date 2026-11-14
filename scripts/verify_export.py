"""Compare generated web/data with git HEAD without masking data changes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from selrisk.reproducibility import verify_export

if __name__=="__main__":
    raise SystemExit(verify_export())
