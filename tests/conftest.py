"""Pytest configuration and shared fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def random_seed():
    """Fix random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Create a sample quantum configuration."""
    from src.quantum.models import QuantumConfig
    
    return QuantumConfig(
        n_qubits=4,
        n_layers=2,
        n_rotations=3,
        entanglement_type="none",
        entanglement_strength=0.0,
        measurement_type="expval",
        backend="default.qubit",
        shots=None
    )


@pytest.fixture
def sample_batch():
    """Create a sample batch of data."""
    batch_size = 8
    input_size = 784
    num_classes = 10
    
    data = torch.randn(batch_size, input_size)
    target = torch.randint(0, num_classes, (batch_size,))
    
    return data, target


@pytest.fixture
def quantum_model(sample_config):
    """Create a quantum neural network model."""
    from src.quantum.models import QuantumNeuralNetwork
    
    return QuantumNeuralNetwork(sample_config)


@pytest.fixture
def mock_dataloader(sample_batch):
    """Create a mock data loader."""
    from torch.utils.data import TensorDataset, DataLoader
    
    data, target = sample_batch
    dataset = TensorDataset(data, target)
    
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data files."""
    data_dir = Path(__file__).parent / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "quantum: marks tests requiring quantum simulation"
    )
