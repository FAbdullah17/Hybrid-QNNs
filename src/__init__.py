"""
Hybrid Quantum-Classical Neural Networks
==========================================

A comprehensive research framework for studying barren plateaus in 
quantum-classical hybrid neural networks through systematic entanglement analysis.

Main modules:
- quantum: Quantum neural network models and circuits
- classical: Classical training infrastructure
- metrics: Evaluation metrics and analysis tools
- utils: Utility functions for data and experiments
"""

__version__ = "1.0.0"
__author__ = "Hybrid-QNNs Research Team"
__email__ = "contact via GitHub"

from src.quantum.models import QuantumConfig, QuantumNeuralNetwork, BarrenPlateauAnalyzer
from src.classical.trainer import QuantumTrainer, AutomatedExperimentRunner
from src.utils import (
    load_fashion_mnist,
    load_config,
    save_config,
    setup_wandb,
    create_experiment_directory,
)

__all__ = [
    # Core classes
    "QuantumConfig",
    "QuantumNeuralNetwork",
    "BarrenPlateauAnalyzer",
    "QuantumTrainer",
    "AutomatedExperimentRunner",
    # Utility functions
    "load_fashion_mnist",
    "load_config",
    "save_config",
    "setup_wandb",
    "create_experiment_directory",
]
