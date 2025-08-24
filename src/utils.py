import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import yaml
import os
import wandb
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import DataLoader, random_split
import pickle
from datetime import datetime

def load_fashion_mnist(batch_size: int = 32, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load FashionMNIST dataset with train/validation/test splits
    
    Args:
        batch_size: Batch size for data loaders
        train_split: Fraction of training data to use for training (rest for validation)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading FashionMNIST dataset...")
    
    # Transform to normalize and flatten images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.flatten())  # Flatten 28x28 to 784
    ])
    
    # Load full training dataset
    full_train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Load test dataset
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Split training data into train and validation
    train_size = int(train_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def visualize_sample_data(data_loader: DataLoader, num_samples: int = 5, save_path: str = None):
    """Visualize sample data from the dataset"""
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # FashionMNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i in range(num_samples):
        # Reshape from 784 to 28x28
        img = images[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'{class_names[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def setup_wandb(project_name: str, config: Dict[str, Any], entity: str = None):
    """Setup Weights & Biases logging"""
    try:
        wandb.init(
            project=project_name,
            config=config,
            entity=entity,
            name=f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        print(f"W&B initialized for project: {project_name}")
        return True
    except Exception as e:
        print(f"W&B setup failed: {e}")
        return False

def log_metrics(metrics: Dict[str, float], step: int = None):
    """Log metrics to W&B if available"""
    try:
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except:
        pass

def save_experiment_results(results: Dict[str, Any], save_dir: str, experiment_name: str):
    """Save experiment results to files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = os.path.join(save_dir, f"{experiment_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save model state if available
    if 'model_state' in results:
        model_path = os.path.join(save_dir, f"{experiment_name}_model.pth")
        torch.save(results['model_state'], model_path)
    
    print(f"Results saved to {save_dir}")

def load_experiment_results(load_dir: str, experiment_name: str) -> Dict[str, Any]:
    """Load experiment results from files"""
    metrics_path = os.path.join(load_dir, f"{experiment_name}_metrics.json")
    
    with open(metrics_path, 'r') as f:
        results = json.load(f)
    
    # Load model state if available
    model_path = os.path.join(load_dir, f"{experiment_name}_model.pth")
    if os.path.exists(model_path):
        results['model_state'] = torch.load(model_path)
    
    return results

def create_experiment_directory(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def plot_training_curves(train_losses: list, val_losses: list, train_accs: list, val_accs: list, 
                        save_path: str = None, title: str = "Training Curves"):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def compute_entanglement_metrics(quantum_state: torch.Tensor) -> Dict[str, float]:
    """Compute various entanglement metrics for quantum state analysis"""
    metrics = {}
    
    # Von Neumann entropy
    if len(quantum_state.shape) == 1:
        # Pure state
        density_matrix = torch.outer(quantum_state, quantum_state.conj())
    else:
        density_matrix = quantum_state
    
    # Compute eigenvalues
    eigenvals = torch.linalg.eigvals(density_matrix).real
    
    # Von Neumann entropy
    entropy = -torch.sum(eigenvals * torch.log(eigenvals + 1e-10))
    metrics['von_neumann_entropy'] = entropy.item()
    
    # Purity
    purity = torch.trace(density_matrix @ density_matrix).real
    metrics['purity'] = purity.item()
    
    # Linear entropy
    linear_entropy = 1 - purity
    metrics['linear_entropy'] = linear_entropy.item()
    
    return metrics

def generate_experiment_summary(results_dir: str) -> pd.DataFrame:
    """Generate summary of all experiments for comparison"""
    summary_data = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_metrics.json'):
            experiment_name = filename.replace('_metrics.json', '')
            filepath = os.path.join(results_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    metrics = json.load(f)
                
                # Extract key metrics
                summary = {
                    'experiment_name': experiment_name,
                    'final_train_loss': metrics.get('final_train_loss', 'N/A'),
                    'final_val_loss': metrics.get('final_val_loss', 'N/A'),
                    'final_train_acc': metrics.get('final_train_acc', 'N/A'),
                    'final_val_acc': metrics.get('final_val_acc', 'N/A'),
                    'barren_plateau_risk': metrics.get('barren_plateau_risk', 'N/A'),
                    'avg_gradient_norm': metrics.get('avg_gradient_norm', 'N/A'),
                    'training_time': metrics.get('training_time', 'N/A')
                }
                
                summary_data.append(summary)
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    df = pd.DataFrame(summary_data)
    return df

def save_comparison_plot(experiment_names: list, metrics: list, metric_name: str, 
                        save_path: str = None, title: str = None):
    """Create comparison plot for different experiments"""
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(experiment_names, metrics, color=['blue', 'red', 'green'], alpha=0.7)
    plt.title(title or f'{metric_name} Comparison')
    plt.xlabel('Experiment Type')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def print_experiment_summary(experiment_name: str, config: Dict[str, Any], results: Dict[str, Any]):
    """Print a formatted summary of experiment results"""
    print("\n" + "="*60)
    print(f"EXPERIMENT SUMMARY: {experiment_name}")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Qubits: {config.get('n_qubits', 'N/A')}")
    print(f"  Layers: {config.get('n_layers', 'N/A')}")
    print(f"  Entanglement: {config.get('entanglement_type', 'N/A')}")
    print(f"  Entanglement Strength: {config.get('entanglement_strength', 'N/A')}")
    
    print(f"\nResults:")
    print(f"  Final Train Loss: {results.get('final_train_loss', 'N/A'):.6f}")
    print(f"  Final Val Loss: {results.get('final_val_loss', 'N/A'):.6f}")
    print(f"  Final Train Acc: {results.get('final_train_acc', 'N/A'):.4f}")
    print(f"  Final Val Acc: {results.get('final_val_acc', 'N/A'):.4f}")
    print(f"  Barren Plateau Risk: {results.get('barren_plateau_risk', 'N/A')}")
    print(f"  Avg Gradient Norm: {results.get('avg_gradient_norm', 'N/A'):.6f}")
    
    print("="*60 + "\n")
