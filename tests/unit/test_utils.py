"""Unit tests for utility functions."""

import pytest
import torch
from pathlib import Path
from src.utils import (
    load_fashion_mnist,
    load_config,
    save_config,
    create_experiment_directory,
)


@pytest.mark.unit
class TestDataLoading:
    """Test data loading functions."""
    
    @pytest.mark.slow
    def test_load_fashion_mnist(self):
        """Test FashionMNIST data loading."""
        train_loader, val_loader, test_loader = load_fashion_mnist(
            batch_size=32, train_split=0.8
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Check batch shape
        data, target = next(iter(train_loader))
        assert data.shape[1] == 784  # Flattened 28x28
        assert target.shape[0] <= 32


@pytest.mark.unit
class TestConfiguration:
    """Test configuration functions."""
    
    def test_save_and_load_config(self, temp_dir):
        """Test saving and loading configuration."""
        config = {
            'n_qubits': 4,
            'n_layers': 3,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        config_path = temp_dir / "test_config.yaml"
        save_config(config, str(config_path))
        
        loaded_config = load_config(str(config_path))
        
        assert loaded_config == config
    
    def test_create_experiment_directory(self, temp_dir):
        """Test experiment directory creation."""
        base_dir = str(temp_dir)
        exp_name = "test_experiment"
        
        exp_dir = create_experiment_directory(base_dir, exp_name)
        
        assert Path(exp_dir).exists()
        assert exp_name in exp_dir
