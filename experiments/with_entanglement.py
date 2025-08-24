#!/usr/bin/env python3
"""
Automated Full Entanglement Experiment Script
Researcher: Farhan Riaz
Purpose: Study barren plateaus with full quantum entanglement
"""

import os
import sys
import yaml
import torch
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum.models import QuantumConfig, QuantumNeuralNetwork, BarrenPlateauAnalyzer
from classical.trainer import QuantumTrainer
from utils import (
    load_fashion_mnist, load_config, setup_wandb, 
    save_experiment_results, create_experiment_directory,
    print_experiment_summary, visualize_sample_data
)

def main():
    """Main experiment execution"""
    print("="*80)
    print("QUANTUM MACHINE LEARNING: FULL ENTANGLEMENT EXPERIMENT")
    print("Researcher: Farhan Riaz")
    print("Focus: Barren Plateau Analysis with Full Quantum Entanglement")
    print("="*80)
    
    # Load configuration
    config_path = "configs/with_entanglement.yaml"
    config = load_config(config_path)
    
    # Create experiment directory
    exp_dir = create_experiment_directory("results", "with_entanglement")
    print(f"Experiment directory: {exp_dir}")
    
    # Setup W&B logging
    if config.get('use_wandb', False):
        setup_wandb(
            project_name=config['wandb_project'],
            config=config,
            entity=config.get('wandb_entity')
        )
    
    # Load FashionMNIST dataset
    print("\nLoading FashionMNIST dataset...")
    train_loader, val_loader, test_loader = load_fashion_mnist(
        batch_size=config['batch_size'],
        train_split=config['train_split']
    )
    
    # Visualize sample data
    print("\nVisualizing sample data...")
    visualize_sample_data(
        train_loader, 
        num_samples=5, 
        save_path=os.path.join(exp_dir, "sample_data.png")
    )
    
    # Create quantum configuration
    quantum_config = QuantumConfig(
        n_qubits=config['n_qubits'],
        n_layers=config['n_layers'],
        n_rotations=config['n_rotations'],
        entanglement_type=config['entanglement_type'],
        entanglement_strength=config['entanglement_strength'],
        measurement_type=config['measurement_type'],
        backend=config['backend'],
        shots=config['shots']
    )
    
    print(f"\nQuantum Configuration:")
    print(f"  Qubits: {quantum_config.n_qubits}")
    print(f"  Layers: {quantum_config.n_layers}")
    print(f"  Entanglement: {quantum_config.entanglement_type}")
    print(f"  Entanglement Strength: {quantum_config.entanglement_strength}")
    print(f"  Note: Full entanglement between all qubit pairs")
    
    # Initialize trainer
    print("\nInitializing quantum trainer...")
    trainer = QuantumTrainer(quantum_config, device=config['device'])
    
    # Run training with barren plateau analysis
    print("\nStarting training with barren plateau analysis...")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        patience=config['patience'],
        save_dir=exp_dir
    )
    
    # Run additional barren plateau analysis
    print("\nRunning comprehensive barren plateau analysis...")
    analysis_results = trainer.analyze_barren_plateaus(
        train_loader, 
        epochs=config['gradient_analysis_epochs']
    )
    
    # Combine results
    final_results = {
        **results,
        'barren_plateau_analysis': analysis_results,
        'experiment_config': config,
        'quantum_config': quantum_config.__dict__,
        'timestamp': datetime.now().isoformat(),
        'researcher': config['researcher']
    }
    
    # Save results
    print(f"\nSaving experiment results...")
    save_experiment_results(final_results, exp_dir, "with_entanglement")
    
    # Generate training summary plots
    print("\nGenerating training summary plots...")
    trainer.plot_training_summary(
        save_path=os.path.join(exp_dir, "training_summary.png")
    )
    
    # Save training history
    trainer.save_training_history(
        os.path.join(exp_dir, "training_history.pth")
    )
    
    # Print final summary
    print_experiment_summary("Full Entanglement", config, final_results)
    
    # Test on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.validate_epoch(test_loader, torch.nn.CrossEntropyLoss())
    print(f"Test Loss: {test_loss:.6f}, Test Accuracy: {test_acc:.2f}%")
    
    # Final barren plateau assessment
    avg_grad = np.mean(trainer.gradient_norms)
    if avg_grad < config['barren_plateau_threshold']:
        print(f"\nBARREN PLATEAU CONFIRMED!")
        print(f"Average gradient norm: {avg_grad:.2e}")
        print(f"Threshold: {config['barren_plateau_threshold']}")
        print(f"Note: Full entanglement may contribute to gradient vanishing")
    else:
        print(f"\nNo barren plateau detected")
        print(f"Average gradient norm: {avg_grad:.2e}")
        print(f"Note: Full entanglement may help maintain gradient flow")
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {exp_dir}")
    
    return final_results

if __name__ == "__main__":
    try:
        results = main()
        print("\nFull Entanglement Experiment Completed Successfully!")
        print(f"Researcher: Farhan Riaz")
        print(f"Results saved to: {results.get('save_dir', 'results')}")
        
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
