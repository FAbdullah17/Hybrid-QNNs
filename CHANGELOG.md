# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- GPU acceleration support for quantum simulations
- Advanced entanglement metrics and analysis
- Interactive web dashboard for results visualization
- Multi-dataset support (CIFAR-10, MNIST, etc.)
- Distributed training capabilities
- Hyperparameter optimization with Optuna
- Quantum circuit visualization tools

## [1.0.0] - 2025-10-27

### Added
- Initial release of Hybrid-QNNs framework
- Three entanglement strategies implementation:
  - No entanglement baseline
  - Full entanglement analysis
  - Varied entanglement exploration
- Quantum neural network models with PennyLane
- Classical-quantum hybrid architecture
- Automated barren plateau detection and analysis
- Comprehensive training infrastructure with QuantumTrainer
- Real-time gradient norm monitoring
- FashionMNIST dataset integration
- Automated experiment runner for all strategies
- Rich visualization suite:
  - Training curves (loss and accuracy)
  - Gradient distributions
  - Barren plateau indicators
  - Experiment comparison plots
- Configuration management with YAML files
- Weights & Biases (W&B) integration for experiment tracking
- Extensive utility functions for data loading and analysis
- Google Colab support and templates
- Comprehensive documentation:
  - README with quick start guide
  - API documentation
  - Contribution guidelines
  - Code of conduct
- Testing infrastructure with pytest
- Type hints throughout codebase
- Professional logging system

### Features

#### Quantum Models
- Parameterized quantum circuits with rotation gates
- Configurable entanglement strategies
- Multiple measurement types (expectation values, probabilities)
- Barren plateau analyzer with statistical reporting
- Quantum layer composition

#### Training Infrastructure
- Automated training loops with early stopping
- Learning rate scheduling
- Gradient norm tracking and analysis
- Validation and test evaluation
- Model checkpointing
- Training history persistence

#### Analysis Tools
- Comprehensive barren plateau detection
- Gradient flow analysis
- Training dynamics visualization
- Cross-experiment comparison reports
- Statistical significance testing
- Performance metrics collection

#### Experiment Automation
- Master script for running all experiments
- Individual experiment scripts for each strategy
- Configurable hyperparameters via YAML
- Automatic result organization and archiving
- Research insights generation

### Documentation
- Complete README with installation and usage instructions
- API documentation for all modules
- Tutorial notebooks for getting started
- Configuration file documentation
- Contributing guidelines
- Citation file (CFF format)
- License (MIT)
- Authors and acknowledgments

### Configuration
- YAML-based experiment configuration
- Separate configs for each entanglement strategy
- Flexible parameter specification
- Environment-specific settings

### Dependencies
- PennyLane >= 0.30.0 for quantum computing
- PyTorch >= 1.12.0 for neural networks
- torchvision for FashionMNIST dataset
- wandb for experiment tracking
- scikit-learn for additional metrics
- matplotlib and seaborn for visualization
- PyYAML for configuration management
- numpy for numerical operations

## [0.1.0] - 2025-10-01

### Added
- Initial project structure
- Basic quantum circuit implementation
- Simple training loop
- Preliminary experiments

---

## Version History Summary

- **1.0.0**: First major release with complete framework
- **0.1.0**: Initial prototype and proof of concept

## Migration Guide

### Upgrading to 1.0.0 from 0.1.0

If you were using an earlier version:

1. **Update dependencies**: `pip install -r requirements.txt --upgrade`
2. **Update configuration files**: New YAML format with additional parameters
3. **Update import statements**: Module structure has been reorganized
4. **Review API changes**: Some function signatures have been updated

Example migration:

```python
# Old (0.1.0)
from models import QuantumNet
model = QuantumNet(qubits=4)

# New (1.0.0)
from src.quantum.models import QuantumNeuralNetwork, QuantumConfig
config = QuantumConfig(n_qubits=4, n_layers=3)
model = QuantumNeuralNetwork(config)
```

## Contributing

For contributions, please open an issue or pull request on GitHub. Follow the existing code style and add tests for new features.

## Links

- [Repository](https://github.com/FAbdullah17/Hybrid-QNNs)
- [Issue Tracker](https://github.com/FAbdullah17/Hybrid-QNNs/issues)
- [Documentation](https://fabdullah17.github.io/Hybrid-QNNs/)
