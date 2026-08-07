#!/usr/bin/env python3
"""
run_demo.py — Pipeline entrypoint.

Usage:
    python run_demo.py

Regenerates artifacts/ from scratch, running the complete
end-to-end LLM pretraining simulation pipeline.

This file delegates entirely to main.py so all logic lives in one place.
"""

import os
import sys

# Ensure the project root is on the path regardless of where this is invoked from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run main
from main import main

if __name__ == "__main__":
    sys.exit(main())
