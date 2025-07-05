"""
Setup script for LLM Information Extraction Pipeline
Run this script to set up the project environment and dependencies
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "data/uploads",
        "data/outputs", 
        "data/cache",
        "data/logs",
        "data/sample_documents"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Created project directories")

def create_env_file():
    """Create .env file template"""
    env_content = """# LLM Information Extraction Pipeline Environment Variables
# Add your API keys here

# OpenAI API Key (required for GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (optional, for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Hugging Face API Key (optional, for open-source models)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Application Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✓ Created .env file template")
        print("  Please edit .env file and add your API keys")
    else:
        print("✓ .env file already exists")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Environment variables
.env
.env.local
.env.*.local

# API keys and secrets
*.key
secrets.json

# Data files
data/uploads/*
data/outputs/*
data/cache/*
data/logs/*
!data/sample_documents/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("✓ Created .gitignore file")
    else:
        print("✓ .gitignore file already exists")

def setup_sample_data():
    """Set up sample data"""
    print("Setting up sample data...")
    try:
        from data.sample_documents import create_sample_documents, create_sample_config
        create_sample_documents()
        create_sample_config()
        print("✓ Sample data created successfully")
    except Exception as e:
        print(f"✗ Failed to create sample data: {e}")

def validate_setup():
    """Validate that setup was successful"""
    print("\nValidating setup...")
    
    # Check if required files exist
    required_files = [
        "app.py",
        "requirements.txt",
        "config.py",
        "pipeline/__init__.py",
        "pipeline/document_processor.py",
        "pipeline/llm_chains.py",
        "pipeline/evaluation.py",
        "pipeline/utils.py"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    # Check if directories exist
    required_dirs = [
        "data",
        "data/uploads",
        "data/outputs",
        "data/cache",
        "data/logs"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ missing")
            return False
    
    return True

def main():
    """Main setup function"""
    print("=" * 50)
    print("LLM Information Extraction Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Create configuration files
    create_env_file()
    create_gitignore()
    
    # Setup sample data
    setup_sample_data()
    
    # Validate setup
    if validate_setup():
        print("\n" + "=" * 50)
        print("✓ Setup completed successfully!")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Edit .env file and add your API keys")
        print("2. Run the application: streamlit run app.py")
        print("3. Open browser at http://localhost:8501")
        print("\nFor development:")
        print("- Use VS Code for editing")
        print("- Install Python extensions")
        print("- Configure your API keys in .env file")
    else:
        print("\n✗ Setup validation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()