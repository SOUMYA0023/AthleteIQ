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

# Import the main function from the app
sys.path.insert(0, str(current_dir / 'sports_form_analysis'))

if __name__ == "__main__":
    # Set environment variables for Streamlit before importing
    port = os.environ.get("PORT", "8501")
    os.environ["STREAMLIT_SERVER_PORT"] = port
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_XSRFPROTECTION"] = "false"
    
    # Now import and run the app
    import streamlit as st
    from sports_form_analysis.app.app import main
    
    # Run the main function
    main()