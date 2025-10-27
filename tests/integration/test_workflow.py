"""Integration tests for full experiment workflow."""

import pytest
import torch
from pathlib import Path
from src.quantum.models import QuantumConfig
from src.classical.trainer import QuantumTrainer
from src.utils import load_fashion_mnist


@pytest.mark.integration
@pytest.mark.slow
class TestFullExperimentWorkflow:
    """Test complete experiment workflow."""
    
    def test_end_to_end_training(self, temp_dir):
        """Test end-to-end training workflow."""
        # Setup configuration
        config = QuantumConfig(
            n_qubits=4,
            n_layers=2,
            entanglement_type="none"
        )
        
        # Create small dataset
        from torch.utils.data import TensorDataset, DataLoader
        
        train_data = torch.randn(100, 784)
        train_targets = torch.randint(0, 10, (100,))
        val_data = torch.randn(20, 784)
        val_targets = torch.randint(0, 10, (20,))
        
        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_data, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=10)
        val_loader = DataLoader(val_dataset, batch_size=10)
        
        # Initialize trainer
        trainer = QuantumTrainer(config, device='cpu')
        
        # Run training
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            lr=0.01,
            patience=5,
            save_dir=str(temp_dir)
        )
        
        # Verify results
        assert 'final_train_loss' in results
        assert 'final_val_loss' in results
        assert results['epochs_trained'] <= 2
        
        # Check saved model
        model_path = temp_dir / 'best_model.pth'
        assert model_path.exists()
    
    def test_barren_plateau_detection(self, temp_dir):
        """Test barren plateau detection in workflow."""
        config = QuantumConfig(
            n_qubits=4,
            n_layers=2,
            entanglement_type="full"
        )
        
        # Create small dataset
        from torch.utils.data import TensorDataset, DataLoader
        
        data = torch.randn(50, 784)
        targets = torch.randint(0, 10, (50,))
        dataset = TensorDataset(data, targets)
        loader = DataLoader(dataset, batch_size=10)
        
        # Initialize and train
        trainer = QuantumTrainer(config, device='cpu')
        results = trainer.train(
            train_loader=loader,
            val_loader=loader,
            epochs=2,
            lr=0.001,
            save_dir=str(temp_dir)
        )
        
        # Check barren plateau analysis
        assert 'barren_plateau_risk' in results
        assert results['barren_plateau_risk'] in ['high', 'low']
