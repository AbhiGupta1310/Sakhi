#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the root project directory is in the PYTHONPATH
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.core.rag import main

if __name__ == "__main__":
    main()
