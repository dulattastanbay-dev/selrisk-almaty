"""Train and save the active real-data Random Forest + boosting map."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from selrisk.two_model_map import main
if __name__ == "__main__":
    main()
