
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure project root is in sys.path for Colab and local runs
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantum.models import QuantumNeuralNetwork, QuantumConfig, BarrenPlateauAnalyzer
from src.utils import log_metrics, save_experiment_results, plot_training_curves

class QuantumTrainer:
    """Automated trainer for quantum neural networks with barren plateau analysis"""
    
    def __init__(self, config: QuantumConfig, device: str = 'auto'):
        self.config = config
        self.device = self._setup_device(device)
        
        # Initialize model
        self.model = QuantumNeuralNetwork(config).to(self.device)
        
        # Initialize barren plateau analyzer
        self.analyzer = BarrenPlateauAnalyzer(self.model)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.gradient_norms = []
        
        print(f"Quantum Trainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device for training"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        print(f"Using device: {device}")
        return device
    
    def train_epoch(self, train_loader, criterion, optimizer, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Compute gradient norm for barren plateau detection
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Grad Norm': f'{total_norm:.6f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # Early barren plateau detection
            if total_norm < 1e-6:
                print(f"⚠️  Potential barren plateau detected at batch {batch_idx}!")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs: int = 50, lr: float = 0.001, 
              patience: int = 10, save_dir: str = None) -> Dict[str, Any]:
        """Complete training loop with barren plateau analysis"""
        print(f"Starting training for {epochs} epochs...")
        
        # Setup training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, epoch)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Log metrics
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr'],
                'avg_gradient_norm': np.mean(self.gradient_norms[-len(train_loader):])
            }
            log_metrics(metrics, epoch)
            
            # Print progress
            print(f'Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Barren plateau warning
            avg_grad = np.mean(self.gradient_norms[-len(train_loader):])
            if avg_grad < 1e-5:
                print(f"🚨 BARREN PLATEAU DETECTED! Avg gradient norm: {avg_grad:.2e}")
        
        training_time = time.time() - start_time
        
        # Generate final results
        results = {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_acc': self.train_accs[-1],
            'final_val_acc': self.val_accs[-1],
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'epochs_trained': len(self.train_losses),
            'avg_gradient_norm': np.mean(self.gradient_norms),
            'barren_plateau_risk': 'high' if np.mean(self.gradient_norms) < 1e-5 else 'low',
            'model_state': self.model.state_dict()
        }
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final validation accuracy: {self.val_accs[-1]:.2f}%")
        
        return results
    
    def analyze_barren_plateaus(self, train_loader, epochs: int = 10) -> Dict[str, Any]:
        """Analyze training dynamics for barren plateau detection"""
        print("Starting barren plateau analysis...")
        
        # Use the analyzer to study training dynamics
        self.analyzer.analyze_training_dynamics(train_loader, nn.CrossEntropyLoss(), epochs)
        
        # Generate analysis report
        report = self.analyzer.generate_analysis_report()
        
        # Create analysis plots
        if hasattr(self.analyzer, 'plot_analysis'):
            self.analyzer.plot_analysis()
        
        return report
    
    def plot_training_summary(self, save_path: str = None):
        """Plot comprehensive training summary"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Training curves
        plot_training_curves(
            self.train_losses, self.val_losses, 
            self.train_accs, self.val_accs,
            title="Training Summary"
        )
        
        # Gradient norm analysis
        if self.gradient_norms:
            axes[1, 0].plot(self.gradient_norms, 'g-', linewidth=2)
            axes[1, 0].set_title('Gradient Norm Over Training')
            axes[1, 0].set_xlabel('Batch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Barren plateau detection
            barren_threshold = 1e-5
            barren_batches = [i for i, grad in enumerate(self.gradient_norms) if grad < barren_threshold]
            if barren_batches:
                axes[1, 0].scatter(barren_batches, [self.gradient_norms[i] for i in barren_batches], 
                                 color='red', s=50, label='Barren Plateau')
                axes[1, 0].legend()
        
        # Gradient distribution
        if self.gradient_norms:
            axes[1, 1].hist(self.gradient_norms, bins=30, alpha=0.7, color='green')
            axes[1, 1].set_title('Gradient Norm Distribution')
            axes[1, 1].set_xlabel('Gradient Norm')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_xscale('log')
        
        # Loss vs Gradient correlation
        if self.gradient_norms and len(self.train_losses) > 1:
            # Align gradients with epochs
            gradients_per_epoch = len(self.gradient_norms) // len(self.train_losses)
            epoch_gradients = [np.mean(self.gradient_norms[i:i+gradients_per_epoch]) 
                             for i in range(0, len(self.gradient_norms), gradients_per_epoch)]
            
            if len(epoch_gradients) == len(self.train_losses):
                axes[1, 2].scatter(epoch_gradients, self.train_losses, alpha=0.6)
                axes[1, 2].set_title('Loss vs Gradient Norm')
                axes[1, 2].set_xlabel('Avg Gradient Norm per Epoch')
                axes[1, 2].set_ylabel('Training Loss')
                axes[1, 2].set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def save_training_history(self, save_path: str):
        """Save training history for later analysis"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'gradient_norms': self.gradient_norms,
            'config': self.config.__dict__
        }
        
        torch.save(history, save_path)
        print(f"Training history saved to {save_path}")
    
    def load_training_history(self, load_path: str):
        """Load training history from file"""
        history = torch.load(load_path)
        
        self.train_losses = history['train_losses']
        self.val_losses = history['val_losses']
        self.train_accs = history['train_accs']
        self.val_accs = history['val_accs']
        self.gradient_norms = history['gradient_norms']
        
        print(f"Training history loaded from {load_path}")

class AutomatedExperimentRunner:
    """Automated runner for multiple experiments with different configurations"""
    
    def __init__(self, base_config: QuantumConfig):
        self.base_config = base_config
        self.results = {}
    
    def run_experiment(self, config: QuantumConfig, train_loader, val_loader, 
                      experiment_name: str, save_dir: str = None) -> Dict[str, Any]:
        """Run a single experiment with given configuration"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*60}")
        
        # Create trainer with new config
        trainer = QuantumTrainer(config)
        
        # Run training
        results = trainer.train(train_loader, val_loader, save_dir=save_dir)
        
        # Run barren plateau analysis
        analysis = trainer.analyze_barren_plateaus(train_loader, epochs=5)
        
        # Combine results
        full_results = {**results, 'barren_plateau_analysis': analysis}
        
        # Save results
        if save_dir:
            save_experiment_results(full_results, save_dir, experiment_name)
            
            # Save training plots
            trainer.plot_training_summary(os.path.join(save_dir, f'{experiment_name}_summary.png'))
            
            # Save training history
            trainer.save_training_history(os.path.join(save_dir, f'{experiment_name}_history.pth'))
        
        self.results[experiment_name] = full_results
        return full_results
    
    def run_all_experiments(self, train_loader, val_loader, save_base_dir: str = './results'):
        """Run all three entanglement experiments automatically"""
        experiments = [
            ("no_entanglement", {"entanglement_type": "none"}),
            ("with_entanglement", {"entanglement_type": "full"}),
            ("varied_entanglement", {"entanglement_type": "varied", "entanglement_strength": 0.5})
        ]
        
        for exp_name, config_updates in experiments:
            # Create experiment-specific config
            exp_config = QuantumConfig(**{**self.base_config.__dict__, **config_updates})
            
            # Create experiment directory
            exp_dir = os.path.join(save_base_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Run experiment
            self.run_experiment(exp_config, train_loader, val_loader, exp_name, exp_dir)
        
        print("\nAll experiments completed!")
        return self.results
    
    def generate_comparison_report(self, save_dir: str = './results'):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("No results to compare. Run experiments first.")
            return
        
        # Create comparison plots
        metrics = ['final_val_acc', 'final_val_loss', 'avg_gradient_norm']
        experiment_names = list(self.results.keys())
        
        for metric in metrics:
            values = [self.results[name].get(metric, 0) for name in experiment_names]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(experiment_names, values, color=['blue', 'red', 'green'], alpha=0.7)
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xlabel('Experiment Type')
            plt.ylabel(metric.replace("_", " ").title())
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                if isinstance(value, (int, float)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'comparison_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Save comparison summary
        comparison_data = []
        for exp_name, results in self.results.items():
            comparison_data.append({
                'experiment': exp_name,
                'final_val_acc': results.get('final_val_acc', 'N/A'),
                'final_val_loss': results.get('final_val_loss', 'N/A'),
                'avg_gradient_norm': results.get('avg_gradient_norm', 'N/A'),
                'barren_plateau_risk': results.get('barren_plateau_risk', 'N/A'),
                'training_time': results.get('training_time', 'N/A')
            })
        
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        df.to_csv(os.path.join(save_dir, 'experiment_comparison.csv'), index=False)
        print(f"Comparison report saved to {save_dir}")
        
        return df
