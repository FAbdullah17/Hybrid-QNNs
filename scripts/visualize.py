#!/usr/bin/env python3
"""
Visualization Script
Create visualizations from experimental results
"""

import os
import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def plot_training_history(history_file, save_dir):
    """Plot training history from saved file."""
    import torch
    
    history = torch.load(history_file)
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Training and validation loss
    axes[0, 0].plot(history['train_losses'], label='Train', color='blue')
    axes[0, 0].plot(history['val_losses'], label='Validation', color='red')
    axes[0, 0].set_title('Loss Over Epochs')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Training and validation accuracy
    axes[0, 1].plot(history['train_accs'], label='Train', color='blue')
    axes[0, 1].plot(history['val_accs'], label='Validation', color='red')
    axes[0, 1].set_title('Accuracy Over Epochs')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Gradient norms
    axes[1, 0].plot(history['gradient_norms'], color='green', alpha=0.6)
    axes[1, 0].set_title('Gradient Norms')
    axes[1, 0].set_xlabel('Batch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)
    
    # Gradient distribution
    axes[1, 1].hist(history['gradient_norms'], bins=50, color='green', alpha=0.7)
    axes[1, 1].set_title('Gradient Norm Distribution')
    axes[1, 1].set_xlabel('Gradient Norm')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = save_path / 'training_history.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Training history plot saved to {output_file}")
    plt.close()


def plot_barren_plateau_analysis(results_dir, save_dir):
    """Create barren plateau analysis visualizations."""
    results_path = Path(results_dir)
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Collect gradient data from all experiments
    experiment_gradients = {}
    
    for json_file in results_path.rglob("*_metrics.json"):
        exp_name = json_file.stem.replace("_metrics", "")
        with open(json_file, 'r') as f:
            data = json.load(f)
            if 'avg_gradient_norm' in data:
                experiment_gradients[exp_name] = data['avg_gradient_norm']
    
    if not experiment_gradients:
        print("⚠️  No gradient data found")
        return
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    experiments = list(experiment_gradients.keys())
    gradients = list(experiment_gradients.values())
    
    bars = plt.bar(experiments, gradients, color=['blue', 'red', 'green'], alpha=0.7)
    plt.axhline(y=1e-5, color='orange', linestyle='--', 
                label='Barren Plateau Threshold')
    
    plt.title('Barren Plateau Analysis: Gradient Norms by Experiment')
    plt.xlabel('Experiment')
    plt.ylabel('Average Gradient Norm')
    plt.yscale('log')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    
    # Annotate bars
    for bar, grad in zip(bars, gradients):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{grad:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = save_path / 'barren_plateau_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Barren plateau analysis saved to {output_file}")
    plt.close()


def plot_entanglement_comparison(results_dir, save_dir):
    """Compare different entanglement strategies."""
    results_path = Path(results_dir)
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Load results
    results = {}
    for json_file in results_path.rglob("*_metrics.json"):
        exp_name = json_file.stem.replace("_metrics", "")
        with open(json_file, 'r') as f:
            results[exp_name] = json.load(f)
    
    if not results:
        print("⚠️  No results found")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Entanglement Strategy Comparison', fontsize=16, fontweight='bold')
    
    experiments = list(results.keys())
    
    # Accuracy comparison
    val_accs = [results[exp].get('final_val_acc', 0) for exp in experiments]
    axes[0].bar(experiments, val_accs, color='steelblue', alpha=0.7)
    axes[0].set_title('Validation Accuracy')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(alpha=0.3)
    
    # Loss comparison
    val_losses = [results[exp].get('final_val_loss', 0) for exp in experiments]
    axes[1].bar(experiments, val_losses, color='coral', alpha=0.7)
    axes[1].set_title('Validation Loss')
    axes[1].set_ylabel('Loss')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3)
    
    # Training time comparison
    times = [results[exp].get('training_time', 0) for exp in experiments]
    axes[2].bar(experiments, times, color='green', alpha=0.7)
    axes[2].set_title('Training Time')
    axes[2].set_ylabel('Time (seconds)')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = save_path / 'entanglement_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Entanglement comparison saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Hybrid-QNNs results")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--output", "-o", type=str, default="./visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--history", "-hist", type=str, default=None,
                       help="Path to training history file (.pth)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Hybrid-QNNs Visualization Generator")
    print("="*70)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    # Plot training history if provided
    if args.history and Path(args.history).exists():
        print(f"\nPlotting training history from {args.history}...")
        plot_training_history(args.history, args.output)
    
    # Plot barren plateau analysis
    print("\nGenerating barren plateau analysis...")
    plot_barren_plateau_analysis(args.results_dir, args.output)
    
    # Plot entanglement comparison
    print("\nGenerating entanglement comparison...")
    plot_entanglement_comparison(args.results_dir, args.output)
    
    print("\n" + "="*70)
    print("✅ Visualization complete!")
    print(f"📁 Plots saved to {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()
