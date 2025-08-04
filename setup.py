#!/usr/bin/env python3
"""
Setup script for Geothermal Digital Twin AI
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Ensure Python 3.8+ is being used."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_requirements():
    """Install Python dependencies."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def setup_tesseract():
    """Provide instructions for Tesseract OCR setup."""
    print("\n📋 Tesseract OCR Setup Required:")
    print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("   Linux: sudo apt-get install tesseract-ocr")
    print("   macOS: brew install tesseract")
    print("   After installation, ensure 'tesseract' is in your PATH")

def create_env_template():
    """Create environment variables template."""
    env_template = """# Geothermal Digital Twin AI - Environment Variables
# Copy this file to .env and fill in your values

# OpenAI API Key (required for text embeddings)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom model configurations
# OPENAI_MODEL=gpt-4
# EMBEDDING_MODEL=text-embedding-ada-002

# Optional: Tesseract path (if not in system PATH)
# TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
    
    print("✅ Created .env.template file")
    print("   → Copy to .env and add your OpenAI API key")

def verify_installation():
    """Verify key components can be imported."""
    print("\n🧪 Verifying installation...")
    
    modules = [
        ("PyPDF2", "PyPDF2"),
        ("PyMuPDF", "fitz"),
        ("tiktoken", "tiktoken"),
        ("openai", "openai"),
        ("chromadb", "chromadb"),
        ("open_clip", "open_clip"),
        ("torch", "torch"),
        ("PIL", "PIL"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pyvista", "pyvista"),
        ("fastapi", "fastapi"),
        ("watchdog", "watchdog"),
    ]
    
    failed_imports = []
    
    for display_name, module_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            failed_imports.append(display_name)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("   Try: pip install -r requirements.txt")
        return False
    
    print("\n✅ All core modules imported successfully!")
    return True

def main():
    """Main setup function."""
    print("=== Geothermal Digital Twin AI Setup ===\n")
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Setup environment template
    create_env_template()
    
    # Tesseract instructions
    setup_tesseract()
    
    # Verify installation
    if not verify_installation():
        print("❌ Setup completed with errors")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and add your OpenAI API key")
    print("2. Install Tesseract OCR (see instructions above)")
    print("3. Add some PDFs to knowledge/corpus/")
    print("4. Run: python src/file_watcher.py")

if __name__ == "__main__":
    main()