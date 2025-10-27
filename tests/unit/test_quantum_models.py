"""Unit tests for quantum models."""

import pytest
import torch
from src.quantum.models import QuantumConfig, QuantumNeuralNetwork, BarrenPlateauAnalyzer


@pytest.mark.unit
class TestQuantumConfig:
    """Test QuantumConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QuantumConfig()
        
        assert config.n_qubits == 4
        assert config.n_layers == 3
        assert config.n_rotations == 3
        assert config.entanglement_type == "none"
        assert config.entanglement_strength == 1.0
        assert config.measurement_type == "expval"
        assert config.backend == "default.qubit"
        assert config.shots is None
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = QuantumConfig(
            n_qubits=8,
            n_layers=4,
            entanglement_type="full",
            entanglement_strength=0.5
        )
        
        assert config.n_qubits == 8
        assert config.n_layers == 4
        assert config.entanglement_type == "full"
        assert config.entanglement_strength == 0.5


@pytest.mark.unit
@pytest.mark.quantum
class TestQuantumNeuralNetwork:
    """Test QuantumNeuralNetwork class."""
    
    def test_model_initialization(self, sample_config):
        """Test model initialization."""
        model = QuantumNeuralNetwork(sample_config)
        
        assert model.config == sample_config
        assert hasattr(model, 'input_layer')
        assert hasattr(model, 'output_layer')
        assert hasattr(model, 'quantum_layers')
        assert hasattr(model, 'qnode')
    
    def test_forward_pass_shape(self, sample_config, sample_batch):
        """Test forward pass returns correct shape."""
        model = QuantumNeuralNetwork(sample_config)
        data, _ = sample_batch
        
        output = model(data)
        
        assert output.shape == (data.shape[0], 10)
    
    def test_parameter_count(self, sample_config):
        """Test model has trainable parameters."""
        model = QuantumNeuralNetwork(sample_config)
        
        param_count = sum(p.numel() for p in model.parameters())
        
        assert param_count > 0
    
    def test_different_entanglement_types(self, random_seed):
        """Test model with different entanglement types."""
        entanglement_types = ["none", "full", "varied"]
        
        for ent_type in entanglement_types:
            config = QuantumConfig(
                n_qubits=4,
                n_layers=2,
                entanglement_type=ent_type
            )
            model = QuantumNeuralNetwork(config)
            
            data = torch.randn(4, 784)
            output = model(data)
            
            assert output.shape == (4, 10)


@pytest.mark.unit
class TestBarrenPlateauAnalyzer:
    """Test BarrenPlateauAnalyzer class."""
    
    def test_analyzer_initialization(self, quantum_model):
        """Test analyzer initialization."""
        analyzer = BarrenPlateauAnalyzer(quantum_model)
        
        assert analyzer.model == quantum_model
        assert len(analyzer.gradient_history) == 0
        assert len(analyzer.loss_history) == 0
    
    def test_gradient_norm_computation(self, quantum_model, sample_batch):
        """Test gradient norm computation."""
        analyzer = BarrenPlateauAnalyzer(quantum_model)
        data, target = sample_batch
        
        loss_fn = torch.nn.CrossEntropyLoss()
        grad_norm = analyzer.compute_gradient_norm(loss_fn, data, target)
        
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0
        assert len(analyzer.gradient_history) == 1
    
    def test_generate_analysis_report(self, quantum_model):
        """Test analysis report generation."""
        analyzer = BarrenPlateauAnalyzer(quantum_model)
        
        # Add some fake history
        analyzer.gradient_history = [0.1, 0.05, 0.02, 0.01]
        analyzer.loss_history = [2.0, 1.5, 1.2, 1.0]
        
        report = analyzer.generate_analysis_report()
        
        assert 'gradient_statistics' in report
        assert 'loss_statistics' in report
        assert 'barren_plateau_risk' in report['gradient_statistics']
