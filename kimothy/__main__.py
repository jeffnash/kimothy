#!/usr/bin/env python3.11
"""Module entry point for Kimothy proxy."""

import uvicorn
from .main import app

if __name__ == "__main__":
    import sys
    
    # Check if running as module (python -m)
    if any('gunicorn' in x for x in sys.argv):
        # Won't run uvicorn, gunicorn runs the app
        pass
    else:
        # Run with uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8928,
            log_level="info",
            access_log=True
        )
