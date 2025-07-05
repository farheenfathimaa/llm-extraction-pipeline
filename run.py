"""
Application runner for LLM Information Extraction Pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def check_setup():
    """Check if the application is properly set up"""
    # Check if required files exist
    required_files = [
        "app.py",
        "requirements.txt", 
        "config.py",
        ".env"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run 'python setup.py' first")
        return False
    
    # Check if API keys are set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY not found in environment")
        print("Please set your API key in the .env file")
        return False
    
    return True

def run_streamlit():
    """Run the Streamlit application"""
    if not check_setup():
        sys.exit(1)
    
    print("Starting LLM Information Extraction Pipeline...")
    print("Open your browser at http://localhost:8501")
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit()