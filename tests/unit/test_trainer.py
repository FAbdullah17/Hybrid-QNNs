"""Unit tests for classical trainer."""

import pytest
import torch
from src.classical.trainer import QuantumTrainer, AutomatedExperimentRunner
from src.quantum.models import QuantumConfig


@pytest.mark.unit
class TestQuantumTrainer:
    """Test QuantumTrainer class."""
    
    def test_trainer_initialization(self, sample_config):
        """Test trainer initialization."""
        trainer = QuantumTrainer(sample_config, device='cpu')
        
        assert trainer.config == sample_config
        assert trainer.device == 'cpu'
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'analyzer')
        assert len(trainer.train_losses) == 0
        assert len(trainer.val_losses) == 0
    
    def test_device_setup_auto(self, sample_config):
        """Test automatic device selection."""
        trainer = QuantumTrainer(sample_config, device='auto')
        
        assert trainer.device in ['cpu', 'cuda', 'mps']
    
    @pytest.mark.slow
    def test_train_epoch(self, sample_config, mock_dataloader):
        """Test single epoch training."""
        trainer = QuantumTrainer(sample_config, device='cpu')
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)
        
        train_loss, train_acc = trainer.train_epoch(
            mock_dataloader, criterion, optimizer, epoch=0
        )
        
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss >= 0
        assert 0 <= train_acc <= 100
        assert len(trainer.gradient_norms) > 0
    
    @pytest.mark.slow
    def test_validate_epoch(self, sample_config, mock_dataloader):
        """Test validation epoch."""
        trainer = QuantumTrainer(sample_config, device='cpu')
        criterion = torch.nn.CrossEntropyLoss()
        
        val_loss, val_acc = trainer.validate_epoch(mock_dataloader, criterion)
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss >= 0
        assert 0 <= val_acc <= 100
    
    @pytest.mark.slow
    def test_train_method(self, sample_config, mock_dataloader, temp_dir):
        """Test full training method."""
        trainer = QuantumTrainer(sample_config, device='cpu')
        
        results = trainer.train(
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            epochs=2,
            lr=0.001,
            patience=5,
            save_dir=str(temp_dir)
        )
        
        assert 'final_train_loss' in results
        assert 'final_val_loss' in results
        assert 'final_train_acc' in results
        assert 'final_val_acc' in results
        assert 'training_time' in results
        assert 'barren_plateau_risk' in results


@pytest.mark.unit
class TestAutomatedExperimentRunner:
    """Test AutomatedExperimentRunner class."""
    
    def test_runner_initialization(self, sample_config):
        """Test runner initialization."""
        runner = AutomatedExperimentRunner(sample_config)
        
        assert runner.base_config == sample_config
        assert len(runner.results) == 0
