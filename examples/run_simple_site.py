"""Run the Exergy Kernel v0 simple-site simulation from source or install."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eie.simulation.simple_site import main


if __name__ == "__main__":
    main()
