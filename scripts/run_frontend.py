#!/usr/bin/env python
"""
Entry point for Streamlit frontend
Usage: python scripts/run_frontend.py or streamlit run scripts/run_frontend.py
"""

import subprocess
import sys

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting Streamlit Frontend...")
    print("=" * 60)
    print("Frontend running at: http://localhost:8501")
    print("=" * 60 + "\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/frontend/app.py",
            "--logger.level=error"
        ])
    except KeyboardInterrupt:
        print("\nFrontend stopped")
        sys.exit(0)
    except Exception as e:
        print(f"Failed to start frontend: {e}")
        sys.exit(1)
