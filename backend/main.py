"""
Scholarship Voice Assistant - Main Entry Point
===============================================
Starts the voice assistant agent.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.livekit_agent import main

if __name__ == "__main__":
    main()
