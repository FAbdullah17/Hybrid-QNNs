# Hybrid Quantum-Classical Neural Networks (Hybrid-QNNs)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PennyLane](https://img.shields.io/badge/PennyLane-quantum-blueviolet.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-red.svg)](https://pytorch.org/)

> **A comprehensive research framework for studying barren plateaus in quantum-classical hybrid neural networks through systematic entanglement analysis.**

## 🎯 Overview

Hybrid-QNNs is a collaborative research project focused on investigating optimization challenges in quantum machine learning, particularly the **barren plateau phenomenon**. The project provides a systematic framework for analyzing how different entanglement strategies affect gradient flow and model trainability in quantum neural networks.

### Key Features

✨ **Three Entanglement Strategies**: No entanglement, full entanglement, and varied entanglement  
🔬 **Automated Barren Plateau Detection**: Real-time gradient norm analysis and visualization  
📊 **Comprehensive Metrics**: Training curves, gradient distributions, and performance comparisons  
🚀 **Automated Experiment Pipeline**: Run all experiments with a single command  
📈 **Rich Visualization**: Training dynamics, quantum circuit analysis, and result comparison  
🎓 **Research-Ready**: Publication-quality plots and detailed experimental reports  

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Experiments](#-experiments)
- [Configuration](#-configuration)
- [Results](#-results)
- [Development](#-development)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- CUDA-compatible GPU (optional, for accelerated training)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/FAbdullah17/Hybrid-QNNs.git
cd Hybrid-QNNs

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Google Colab Setup

For running experiments in Google Colab:

```python
# Clone repository in Colab
!git clone https://github.com/FAbdullah17/Hybrid-QNNs.git
%cd Hybrid-QNNs

# Install dependencies
!pip install -r requirements.txt

# Run experiments
!python run_all_experiments.py
```

## ⚡ Quick Start

### Run All Experiments

```bash
# Run comprehensive experiment suite
python run_all_experiments.py

# Results will be saved in results/ directory with timestamp
```

### Run Individual Experiment

```bash
# No entanglement strategy
python experiments/no_entanglement.py

# Full entanglement strategy
python experiments/with_entanglement.py

# Varied entanglement strategy
python experiments/varied_entanglement.py
```

### Run Specific Experiment from Master Script

```bash
# Run only one experiment type
python run_all_experiments.py --experiment no_entanglement
python run_all_experiments.py --experiment with_entanglement
python run_all_experiments.py --experiment varied_entanglement
```

### Using Python API

```python
from src.quantum.models import QuantumConfig, QuantumNeuralNetwork
from src.classical.trainer import QuantumTrainer
from src.utils import load_fashion_mnist

# Create quantum configuration
config = QuantumConfig(
    n_qubits=4,
    n_layers=3,
    entanglement_type="full",
    entanglement_strength=1.0
)

# Load data
train_loader, val_loader, test_loader = load_fashion_mnist(batch_size=32)

# Initialize trainer and run
trainer = QuantumTrainer(config)
results = trainer.train(train_loader, val_loader, epochs=50)
```

## 📁 Project Structure

```
Hybrid-QNNs/
├── src/                          # Source code
│   ├── __init__.py
│   ├── utils.py                  # Utility functions
│   ├── classical/               # Classical neural network components
│   │   ├── __init__.py
│   │   └── trainer.py           # Training and experiment automation
│   ├── quantum/                 # Quantum circuit components
│   │   ├── __init__.py
│   │   └── models.py            # Quantum neural networks
│   └── metrics/                 # Evaluation metrics
│       └── __init__.py
├── experiments/                  # Experiment scripts
│   ├── no_entanglement.py       # No entanglement experiments (Asma)
│   ├── with_entanglement.py     # Full entanglement experiments (Farhan)
│   └── varied_entanglement.py   # Varied entanglement experiments (Fahad)
├── configs/                      # Configuration files
│   ├── no_entanglement.yaml
│   ├── with_entanglement.yaml
│   └── varied_entanglement.yaml
├── results/                      # Experimental results (gitignored)
│   └── .gitkeep
├── tests/                        # Unit and integration tests
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── docs/                         # Documentation
│   ├── api/
│   ├── tutorials/
│   └── conf.py
├── scripts/                      # Utility scripts
│   ├── setup_environment.py
│   ├── analyze_results.py
│   └── visualize.py
├── notebooks/                    # Jupyter notebooks
│   ├── tutorials/
│   └── analysis/
├── .github/                      # GitHub configuration
│   └── workflows/               # CI/CD workflows
│       ├── tests.yml
│       ├── lint.yml
│       └── docs.yml
├── run_all_experiments.py       # Master automation script
├── test_setup.py                # Setup verification
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Development dependencies
├── setup.py                     # Package installation
├── pyproject.toml              # Build system configuration
├── pytest.ini                   # Pytest configuration
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
├── CITATION.cff                # Citation information
├── AUTHORS.md                  # Authors and contributors
├── CHANGELOG.md                # Version history
└── README.md                   # This file
```

## 📖 Usage

### Configuration

Experiments are configured using YAML files in the `configs/` directory:

```yaml
# Example: configs/with_entanglement.yaml
experiment_name: "with_entanglement"
researcher: "Farhan Riaz"

# Quantum Configuration
n_qubits: 4
n_layers: 3
entanglement_type: "full"
entanglement_strength: 1.0

# Training Configuration
epochs: 50
batch_size: 32
learning_rate: 0.001
patience: 10

# Analysis Configuration
barren_plateau_threshold: 1e-6
gradient_analysis_epochs: 10
```

### Training Options

```python
# Custom training with advanced options
results = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    lr=0.001,
    patience=10,
    save_dir='./results/my_experiment'
)

# Barren plateau analysis
analysis = trainer.analyze_barren_plateaus(
    train_loader=train_loader,
    epochs=10
)

# Generate visualizations
trainer.plot_training_summary(save_path='./plots/summary.png')
```

## 🔬 Experiments

### Three Entanglement Strategies

#### 1. No Entanglement (Asma Zubair)
- **Objective**: Establish baseline performance without quantum entanglement
- **Configuration**: `configs/no_entanglement.yaml`
- **Key Insight**: Analyze if classical-quantum hybrid models benefit from quantum features

#### 2. Full Entanglement (Farhan Riaz)
- **Objective**: Study effects of maximum entanglement between all qubits
- **Configuration**: `configs/with_entanglement.yaml`
- **Key Insight**: Investigate correlation between entanglement depth and barren plateaus

#### 3. Varied Entanglement (Fahad Abdullah)
- **Objective**: Explore adaptive entanglement with probabilistic gates
- **Configuration**: `configs/varied_entanglement.yaml`
- **Key Insight**: Find optimal balance between entanglement and trainability

### Barren Plateau Detection

The framework automatically detects barren plateaus through:

- **Gradient Norm Tracking**: Monitors L2 norm of gradients during training
- **Threshold Analysis**: Flags potential barren plateaus (default: < 1e-6)
- **Statistical Analysis**: Computes gradient statistics across training
- **Visual Indicators**: Highlights problematic regions in plots

### Evaluation Metrics

- **Training/Validation Loss**: Cross-entropy loss curves
- **Classification Accuracy**: Top-1 accuracy on FashionMNIST
- **Gradient Norms**: L2 norm of model gradients
- **Convergence Speed**: Epochs to reach target accuracy
- **Barren Plateau Risk**: Classification based on gradient statistics

## 📊 Results

Results are automatically saved in the `results/` directory with timestamps:

```
results/
└── comprehensive_study_20251027_143022/
    ├── no_entanglement/
    │   ├── no_entanglement_metrics.json
    │   ├── training_summary.png
    │   ├── sample_data.png
    │   └── best_model.pth
    ├── with_entanglement/
    │   └── ...
    ├── varied_entanglement/
    │   └── ...
    ├── experiment_comparison.csv
    ├── comparison_final_val_acc.png
    ├── final_summary.txt
    └── research_insights.txt
```

### Viewing Results

```python
# Load and analyze results
from src.utils import load_experiment_results, generate_experiment_summary

# Load specific experiment
results = load_experiment_results('./results/my_exp', 'with_entanglement')

# Generate summary across all experiments
summary_df = generate_experiment_summary('./results/comprehensive_study')
print(summary_df)
```

## 🛠️ Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/unit/test_quantum_models.py -v
```

## 🤝 Contributing

We welcome contributions from the research community! For contributions, please:

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests and documentation
4. Run tests and quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Workflow

1. **Setup**: Follow installation instructions with development dependencies
2. **Branch**: Create feature branch from `main`
3. **Develop**: Write code and add tests
4. **Test**: Run pytest to ensure everything works
5. **Document**: Update relevant documentation
6. **Submit**: Open PR with clear description

## 📝 Citation

If you use this software in your research, please cite:

```bibtex
@software{hybrid_qnns_2025,
  title = {Hybrid-QNNs: Quantum-Classical Neural Networks for Barren Plateau Analysis},
  author = {Zubair, Asma and Riaz, Farhan and Abdullah, Fahad},
  year = {2025},
  url = {https://github.com/FAbdullah17/Hybrid-QNNs},
  version = {1.0.0},
  license = {MIT}
}
```

See [CITATION.cff](CITATION.cff) for more citation formats.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PennyLane Team**: For the excellent quantum computing framework
- **PyTorch Community**: For the robust machine learning infrastructure
- **FashionMNIST Dataset**: For providing accessible image classification data
- **Research Community**: For insights and inspiration in quantum ML

### Research Team

- **Asma Zubair**: No entanglement strategy and baseline analysis
- **Farhan Riaz**: Full entanglement strategy and optimization studies
- **Fahad Abdullah**: Varied entanglement strategy and framework development

See [AUTHORS.md](AUTHORS.md) for complete contributor list.

## 📮 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/FAbdullah17/Hybrid-QNNs/issues)
- **Discussions**: [Ask questions or share ideas](https://github.com/FAbdullah17/Hybrid-QNNs/discussions)
- **Email**: Available through GitHub profiles

## 🔗 Links

- [GitHub Repository](https://github.com/FAbdullah17/Hybrid-QNNs)
- [PennyLane Documentation](https://pennylane.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)

---

**Made with ❤️ by the Hybrid-QNNs Research Team**

*Last updated: October 27, 2025*