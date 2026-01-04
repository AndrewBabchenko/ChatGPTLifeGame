"""
Life Game Demo Launcher
Run the modular simulation demo application
"""

import sys
from pathlib import Path

# Ensure demo package is importable
sys.path.insert(0, str(Path(__file__).parent))

from demo.app import main

if __name__ == "__main__":
    main()
