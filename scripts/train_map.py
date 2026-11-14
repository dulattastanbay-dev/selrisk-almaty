"""Train the separate 300-assumed-negative map experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from selrisk.map_model import main
if __name__ == "__main__":
    main()
