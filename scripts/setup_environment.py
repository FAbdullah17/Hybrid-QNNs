#!/usr/bin/env python3
"""
Setup Environment Script
Verifies and sets up the development environment for Hybrid-QNNs
"""

import sys
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: not installed")
        return False


def install_requirements():
    """Install requirements from requirements.txt."""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False


def install_dev_requirements():
    """Install development requirements."""
    print("\nInstalling development requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"
        ])
        print("✅ Development requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install development requirements")
        return False


def setup_pre_commit():
    """Setup pre-commit hooks."""
    print("\nSetting up pre-commit hooks...")
    try:
        subprocess.check_call(["pre-commit", "install"])
        print("✅ Pre-commit hooks installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Pre-commit not available (optional)")
        return False


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    dirs = ['results', 'data', 'logs', 'checkpoints']
    
    for dir_name in dirs:
        path = Path(dir_name)
        path.mkdir(exist_ok=True)
        (path / '.gitkeep').touch(exist_ok=True)
        print(f"✅ Created {dir_name}/")
    
    return True


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        elif torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
        else:
            print("⚠️  No GPU detected, will use CPU")
    except ImportError:
        print("⚠️  PyTorch not installed yet")


def main():
    """Main setup function."""
    print("="*60)
    print("Hybrid-QNNs Environment Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check core packages
    print("\nChecking core packages...")
    packages_to_check = [
        ("pennylane", "pennylane"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("wandb", "wandb"),
    ]
    
    missing_packages = []
    for pkg, import_name in packages_to_check:
        if not check_package(pkg, import_name):
            missing_packages.append(pkg)
    
    # Install if missing
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        response = input("Install missing packages? (y/n): ")
        if response.lower() == 'y':
            install_requirements()
    
    # Optional: Install dev requirements
    print("\n" + "="*60)
    response = input("Install development dependencies? (y/n): ")
    if response.lower() == 'y':
        install_dev_requirements()
        setup_pre_commit()
    
    # Create directories
    create_directories()
    
    # Check GPU
    check_gpu()
    
    # Final summary
    print("\n" + "="*60)
    print("✅ Environment setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run quick test: python test_setup.py")
    print("2. Read documentation: docs/getting_started/quickstart.md")
    print("3. Run experiments: python run_all_experiments.py")
    print("\nFor help: https://github.com/FAbdullah17/Hybrid-QNNs")


if __name__ == "__main__":
    main()
