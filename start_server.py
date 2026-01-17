#!/usr/bin/env python3
"""
Entry point for Railway deployment
Handles port assignment and starts the Streamlit application
"""

import os
import sys
from pathlib import Path

# Add the sports_form_analysis directory to the path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and run the Streamlit app
if __name__ == "__main__":
    import streamlit.web.bootstrap as bootstrap
    
    # Get the port from environment, default to 8501
    port = int(os.environ.get("PORT", 8501))
    
    # Set Streamlit server configuration
    sys.argv = [
        "streamlit", 
        "run", 
        "sports_form_analysis/app/app.py",
        "--server.port", 
        str(port),
        "--server.address", 
        "0.0.0.0",
        "--server.enableCORS", 
        "false",
        "--server.enableXsrfProtection", 
        "false"
    ]
    
    # Start the Streamlit app
    bootstrap.main()