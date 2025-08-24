import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class QuantumConfig:
    """Configuration for quantum neural network"""
    n_qubits: int = 4
    n_layers: int = 3
    n_rotations: int = 3
    entanglement_type: str = "none"  # "none", "full", "varied"
    entanglement_strength: float = 1.0
    measurement_type: str = "expval"  # "expval", "probs", "samples"
    backend: str = "default.qubit"
    shots: Optional[int] = None

class QuantumLayer(nn.Module):
    """Single quantum layer with parameterized rotations"""
    
    def __init__(self, n_qubits: int, n_rotations: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_rotations = n_rotations
        self.weights = nn.Parameter(torch.randn(n_qubits, n_rotations))
        
    def forward(self, x):
        return x

class QuantumNeuralNetwork(nn.Module):
    """Hybrid Quantum-Classical Neural Network"""
    
    def __init__(self, config: QuantumConfig):
        super().__init__()
        self.config = config
        self.dev = qml.device(config.backend, wires=config.n_qubits, shots=config.shots)
        
        # Classical layers
        self.input_layer = nn.Linear(784, config.n_qubits)  # FashionMNIST: 28x28 = 784
        self.output_layer = nn.Linear(config.n_qubits, 10)  # 10 classes
        
        # Quantum layers
        self.quantum_layers = nn.ModuleList([
            QuantumLayer(config.n_qubits, config.n_rotations) 
            for _ in range(config.n_layers)
        ])
        
        # Create quantum circuit
        self.qnode = self._create_qnode()
        
    def _create_qnode(self):
        """Create quantum node with specified entanglement strategy"""
        
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # Encode classical data into quantum state
            for i in range(self.config.n_qubits):
                qml.RY(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)
            
            # Apply parameterized quantum layers
            for layer_idx in range(self.config.n_layers):
                # Single qubit rotations
                for i in range(self.config.n_qubits):
                    qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)
                
                # Entanglement strategy
                if self.config.entanglement_type == "none":
                    pass  # No entanglement
                elif self.config.entanglement_type == "full":
                    # Full entanglement between all qubits
                    for i in range(self.config.n_qubits):
                        for j in range(i + 1, self.config.n_qubits):
                            qml.CNOT(wires=[i, j])
                            qml.CRZ(weights[i % self.config.n_rotations], wires=[i, j])
                elif self.config.entanglement_type == "varied":
                    # Varied entanglement based on strength
                    for i in range(self.config.n_qubits - 1):
                        if np.random.random() < self.config.entanglement_strength:
                            qml.CNOT(wires=[i, i + 1])
                            qml.CRZ(weights[i % self.config.n_rotations], wires=[i, i + 1])
                
                # Additional single qubit operations
                for i in range(self.config.n_qubits):
                    qml.Hadamard(wires=i)
            
            # Measurement
            if self.config.measurement_type == "expval":
                return [qml.expval(qml.PauliZ(i)) for i in range(self.config.n_qubits)]
            elif self.config.measurement_type == "probs":
                return qml.probs(wires=range(self.config.n_qubits))
            else:
                return qml.sample(qml.PauliZ(0), wires=range(self.config.n_qubits))
        
        return circuit
    
    def forward(self, x):
        # Classical preprocessing
        x = torch.relu(self.input_layer(x))
        
        # Quantum processing
        for quantum_layer in self.quantum_layers:
            x = self.qnode(x, quantum_layer.weights)
            if isinstance(x, list):
                x = torch.stack(x)
        
        # Classical post-processing
        x = self.output_layer(x)
        return x

class BarrenPlateauAnalyzer:
    """Analyzer for barren plateau detection and analysis"""
    
    def __init__(self, model: QuantumNeuralNetwork):
        self.model = model
        self.gradient_history = []
        self.loss_history = []
        self.entanglement_metrics = []
        
    def compute_gradient_norm(self, loss_fn, data, target):
        """Compute gradient norm to detect barren plateaus"""
        self.model.zero_grad()
        loss = loss_fn(self.model(data), target)
        loss.backward()
        
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        self.gradient_history.append(total_norm)
        return total_norm
    
    def compute_entanglement_entropy(self, state_vector):
        """Compute entanglement entropy of quantum state"""
        # Simplified entanglement entropy calculation
        if len(state_vector.shape) == 1:
            state_vector = state_vector.unsqueeze(0)
        
        # Convert to density matrix
        rho = torch.outer(state_vector, state_vector.conj())
        
        # Partial trace over half the system
        n_qubits = int(np.log2(state_vector.shape[-1]))
        if n_qubits % 2 == 0:
            mid = n_qubits // 2
            # Simplified partial trace
            entropy = -torch.sum(rho * torch.log(rho + 1e-10))
            return entropy.item()
        return 0.0
    
    def analyze_training_dynamics(self, train_loader, loss_fn, epochs=10):
        """Analyze training dynamics for barren plateau detection"""
        print("Analyzing training dynamics for barren plateau detection...")
        
        for epoch in range(epochs):
            epoch_gradients = []
            epoch_losses = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= 10:  # Limit for analysis
                    break
                    
                grad_norm = self.compute_gradient_norm(loss_fn, data, target)
                epoch_gradients.append(grad_norm)
                
                with torch.no_grad():
                    loss = loss_fn(self.model(data), target)
                    epoch_losses.append(loss.item())
            
            # Record epoch statistics
            avg_grad = np.mean(epoch_gradients)
            avg_loss = np.mean(epoch_losses)
            
            self.gradient_history.append(avg_grad)
            self.loss_history.append(avg_loss)
            
            print(f"Epoch {epoch+1}: Avg Gradient Norm: {avg_grad:.6f}, Avg Loss: {avg_loss:.6f}")
            
            # Early barren plateau detection
            if avg_grad < 1e-6:
                print(f"⚠️  Potential barren plateau detected at epoch {epoch+1}!")
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        report = {
            "gradient_statistics": {
                "mean": np.mean(self.gradient_history),
                "std": np.std(self.gradient_history),
                "min": np.min(self.gradient_history),
                "max": np.max(self.gradient_history),
                "barren_plateau_risk": "high" if np.mean(self.gradient_history) < 1e-5 else "low"
            },
            "loss_statistics": {
                "mean": np.mean(self.loss_history),
                "std": np.std(self.loss_history),
                "convergence_rate": "slow" if np.std(self.loss_history) > 0.1 else "fast"
            },
            "entanglement_metrics": self.entanglement_metrics
        }
        return report
    
    def plot_analysis(self, save_path: str = None):
        """Generate analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gradient norm over time
        axes[0, 0].plot(self.gradient_history, 'b-', linewidth=2)
        axes[0, 0].set_title('Gradient Norm Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss over time
        axes[0, 1].plot(self.loss_history, 'r-', linewidth=2)
        axes[0, 1].set_title('Training Loss Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient distribution
        axes[1, 0].hist(self.gradient_history, bins=20, alpha=0.7, color='blue')
        axes[1, 0].set_title('Gradient Norm Distribution')
        axes[1, 0].set_xlabel('Gradient Norm')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_xscale('log')
        
        # Loss vs Gradient correlation
        axes[1, 1].scatter(self.gradient_history, self.loss_history, alpha=0.6)
        axes[1, 1].set_title('Loss vs Gradient Norm')
        axes[1, 1].set_xlabel('Gradient Norm')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
