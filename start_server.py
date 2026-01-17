#!/usr/bin/env python3
"""
Entry point for Railway deployment
Handles port assignment and starts the Streamlit application
"""

import os
import sys
from pathlib import Path
import subprocess

# Add the sports_form_analysis directory to the path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

if __name__ == "__main__":
    # Get the port from environment, default to 8501
    port = os.environ.get("PORT", "8501")
    
    # Run the Streamlit app as a subprocess
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "sports_form_analysis/app/app.py",
        "--server.port",
        port,
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false"
    ]
    
    # Execute the Streamlit command
    os.execv(sys.executable, cmd)