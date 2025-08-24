#!/usr/bin/env python3
"""
Script to download FashionMNIST dataset for quantum machine learning experiments.
"""

import torchvision
import os

def download_fashionmnist():
    """Download FashionMNIST dataset to ./data directory."""
    print("Downloading FashionMNIST dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Download training data
    print("Downloading training data...")
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True
    )
    
    # Download test data
    print("Downloading test data...")
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True
    )
    
    print(f"Dataset downloaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Data stored in: {os.path.abspath('./data')}")

if __name__ == "__main__":
    download_fashionmnist()
