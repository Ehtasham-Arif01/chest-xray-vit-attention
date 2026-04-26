#!/usr/bin/env python3
"""Setup Kaggle API and download datasets."""

import os
import sys
from pathlib import Path
import subprocess
import json

def setup_kaggle():
    """Setup Kaggle credentials."""
    print("🔧 Setting up Kaggle API...")
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Instructions
    print("\n📋 To set up Kaggle API:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download kaggle.json")
    print("5. Move it to ~/.kaggle/kaggle.json")
    
    # Check if exists
    kaggle_json = kaggle_dir / "kaggle.json"
    if kaggle_json.exists():
        print(f"✅ Kaggle API found at {kaggle_json}")
        # Set permissions
        os.chmod(kaggle_json, 0o600)
    else:
        print(f"❌ Please place kaggle.json in {kaggle_json}")
        return False
    
    return True

def download_nih_dataset(output_dir: str):
    """Download NIH Chest X-ray dataset."""
    print("\n📥 Downloading NIH Chest X-ray dataset...")
    
    # Kaggle dataset: nih-chest-xrays
    cmd = f"kaggle datasets download -d nih-chest-xrays/data -p {output_dir} --unzip"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Dataset downloaded to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading: {e}")
        return False

def download_chexpert_dataset(output_dir: str):
    """Download CheXpert dataset."""
    print("\n📥 Downloading CheXpert dataset...")
    
    # CheXpert dataset requires authentication
    cmd = f"kaggle datasets download -d chexpert/chexpert -p {output_dir} --unzip"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ CheXpert downloaded to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading: {e}")
        return False

if __name__ == "__main__":
    if setup_kaggle():
        # Download datasets
        data_dir = Path("data")
        download_nih_dataset(data_dir / "nih")
        # download_chexpert_dataset(data_dir / "chexpert")
    else:
        print("Please setup Kaggle API first")
        sys.exit(1)
