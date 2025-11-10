#!/usr/bin/env python3
"""
Setup Verification Script

Run this script to verify that all dependencies are correctly installed
and the environment is ready for distributed training.

Usage:
    python verify_setup.py
"""

import sys
import os

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_section(title):
    """Print a section header."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def check_passed(message):
    """Print a passed check."""
    print(f"{GREEN}[PASS]{RESET} {message}")

def check_failed(message):
    """Print a failed check."""
    print(f"{RED}[FAIL]{RESET} {message}")

def check_warning(message):
    """Print a warning."""
    print(f"{YELLOW}[WARN]{RESET} {message}")

def check_python_version():
    """Check Python version."""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 9:
        check_passed("Python version is 3.9 or higher")
        return True
    else:
        check_warning("Python 3.9+ recommended (you have {}.{})".format(
            version.major, version.minor))
        return False

def check_pytorch():
    """Check PyTorch installation."""
    print_section("PyTorch")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        check_passed("PyTorch is installed")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            check_passed("CUDA is available")
        else:
            check_warning("CUDA not available (CPU-only mode)")
        
        return True
    except ImportError:
        check_failed("PyTorch is not installed")
        print("  Install: pip install torch torchvision")
        return False

def check_transformers():
    """Check Transformers library."""
    print_section("Transformers & Datasets")
    success = True
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        check_passed("Transformers is installed")
    except ImportError:
        check_failed("Transformers is not installed")
        print("  Install: pip install transformers")
        success = False
    
    try:
        import datasets
        print(f"Datasets version: {datasets.__version__}")
        check_passed("Datasets is installed")
    except ImportError:
        check_failed("Datasets is not installed")
        print("  Install: pip install datasets")
        success = False
    
    return success

def check_horovod():
    """Check Horovod installation."""
    print_section("Horovod")
    try:
        import horovod
        import horovod.torch as hvd
        print(f"Horovod version: {horovod.__version__}")
        check_passed("Horovod is installed")
        
        # Check if built with GPU support
        try:
            print(f"Horovod built with NCCL: {hvd.nccl_built()}")
            if hvd.nccl_built():
                check_passed("Horovod has GPU support (NCCL)")
            else:
                check_warning("Horovod built without NCCL (CPU-only)")
        except:
            check_warning("Could not check Horovod GPU support")
        
        return True
    except ImportError:
        check_failed("Horovod is not installed")
        print("  Install: HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[pytorch]")
        print("  Note: Requires MPI and NCCL")
        return False

def check_other_dependencies():
    """Check other dependencies."""
    print_section("Other Dependencies")
    
    deps = {
        'yaml': 'pyyaml',
        'tensorboard': 'tensorboard',
        'numpy': 'numpy',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for module, package in deps.items():
        try:
            __import__(module)
            check_passed(f"{package} is installed")
        except ImportError:
            check_failed(f"{package} is not installed")
            print(f"  Install: pip install {package}")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Check project structure."""
    print_section("Project Structure")
    
    required_dirs = [
        'src',
        'src/models',
        'src/utils',
        'configs',
        'scripts',
        'notebooks'
    ]
    
    required_files = [
        'src/train.py',
        'src/data.py',
        'src/generate.py',
        'src/models/transformer_lm.py',
        'src/models/hf_wrapper.py',
        'configs/base.yaml',
        'scripts/train_local.sh',
        'scripts/train_zaratan.sh',
        'requirements.txt',
        'README.md'
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            check_passed(f"Directory exists: {dir_path}")
        else:
            check_failed(f"Directory missing: {dir_path}")
            all_ok = False
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            check_passed(f"File exists: {file_path}")
        else:
            check_failed(f"File missing: {file_path}")
            all_ok = False
    
    return all_ok

def check_imports():
    """Check if project modules can be imported."""
    print_section("Project Modules")
    
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    modules = [
        'src.models.transformer_lm',
        'src.models.hf_wrapper',
        'src.data',
        'src.utils.distributed',
        'src.utils.config',
        'src.utils.logging',
        'src.metrics'
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            check_passed(f"Can import {module}")
        except Exception as e:
            check_failed(f"Cannot import {module}")
            print(f"  Error: {str(e)}")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Horovod Distributed Training - Setup Verification{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch", check_pytorch),
        ("Transformers & Datasets", check_transformers),
        ("Horovod", check_horovod),
        ("Other Dependencies", check_other_dependencies),
        ("Project Structure", check_project_structure),
        ("Project Modules", check_imports)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            check_failed(f"Error checking {name}: {str(e)}")
            results.append((name, False))
    
    # Summary
    print_section("Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Passed: {passed}/{total} checks\n")
    
    if passed == total:
        print(f"{GREEN}[SUCCESS] All checks passed! Environment is ready for training.{RESET}")
        print(f"\nNext steps:")
        print(f"  1. Run notebooks: jupyter notebook notebooks/")
        print(f"  2. Local training: bash scripts/train_local.sh")
        print(f"  3. Cluster training: sbatch scripts/train_zaratan.sh")
        return 0
    else:
        print(f"{YELLOW}[WARNING] Some checks failed. Please install missing dependencies.{RESET}")
        print(f"\nSee README.md for installation instructions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

