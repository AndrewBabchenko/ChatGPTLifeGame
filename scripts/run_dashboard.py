#!/usr/bin/env python3
"""
Dashboard Launcher - Entry point for the modular training dashboard
"""
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Add project root to path
project_root = scripts_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Launch the training dashboard"""
    from dashboard import main as run_dashboard
    run_dashboard()


if __name__ == "__main__":
    main()
