#!/usr/bin/env python3
"""
Results Analysis Script
Analyze and compare experimental results
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils import generate_experiment_summary


def load_results(results_dir):
    """Load all experimental results from directory."""
    results = {}
    results_path = Path(results_dir)
    
    for json_file in results_path.rglob("*_metrics.json"):
        exp_name = json_file.stem.replace("_metrics", "")
        with open(json_file, 'r') as f:
            results[exp_name] = json.load(f)
    
    return results


def create_comparison_table(results):
    """Create comparison table of all experiments."""
    data = []
    
    for exp_name, exp_results in results.items():
        data.append({
            'Experiment': exp_name,
            'Final Val Acc': f"{exp_results.get('final_val_acc', 0):.2f}%",
            'Final Val Loss': f"{exp_results.get('final_val_loss', 0):.4f}",
            'Avg Gradient Norm': f"{exp_results.get('avg_gradient_norm', 0):.2e}",
            'Barren Plateau Risk': exp_results.get('barren_plateau_risk', 'N/A'),
            'Training Time': f"{exp_results.get('training_time', 0):.2f}s",
            'Epochs': exp_results.get('epochs_trained', 'N/A')
        })
    
    return pd.DataFrame(data)


def plot_comparison(results, save_dir):
    """Create comparison plots."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Extract metrics
    experiments = list(results.keys())
    val_accs = [results[exp].get('final_val_acc', 0) for exp in experiments]
    val_losses = [results[exp].get('final_val_loss', 0) for exp in experiments]
    grad_norms = [results[exp].get('avg_gradient_norm', 0) for exp in experiments]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    
    # Validation Accuracy
    axes[0, 0].bar(experiments, val_accs, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Final Validation Accuracy')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(alpha=0.3)
    
    # Validation Loss
    axes[0, 1].bar(experiments, val_losses, color='coral', alpha=0.7)
    axes[0, 1].set_title('Final Validation Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(alpha=0.3)
    
    # Gradient Norms (log scale)
    axes[1, 0].bar(experiments, grad_norms, color='green', alpha=0.7)
    axes[1, 0].set_title('Average Gradient Norm')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_yscale('log')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(alpha=0.3)
    
    # Training Time
    training_times = [results[exp].get('training_time', 0) for exp in experiments]
    axes[1, 1].bar(experiments, training_times, color='purple', alpha=0.7)
    axes[1, 1].set_title('Training Time')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'comparison_plots.png', dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plots saved to {save_path / 'comparison_plots.png'}")


def generate_report(results, save_dir):
    """Generate comprehensive analysis report."""
    save_path = Path(save_dir)
    report_file = save_path / 'analysis_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HYBRID-QNNS EXPERIMENT ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n")
        
        for exp_name, exp_results in results.items():
            f.write(f"\n{exp_name.upper()}\n")
            f.write(f"  Final Validation Accuracy: {exp_results.get('final_val_acc', 0):.2f}%\n")
            f.write(f"  Final Validation Loss: {exp_results.get('final_val_loss', 0):.6f}\n")
            f.write(f"  Average Gradient Norm: {exp_results.get('avg_gradient_norm', 0):.2e}\n")
            f.write(f"  Barren Plateau Risk: {exp_results.get('barren_plateau_risk', 'N/A')}\n")
            f.write(f"  Training Time: {exp_results.get('training_time', 0):.2f} seconds\n")
            f.write(f"  Epochs Trained: {exp_results.get('epochs_trained', 'N/A')}\n")
        
        # Best performance
        f.write("\n" + "="*70 + "\n")
        f.write("BEST PERFORMANCE\n")
        f.write("-"*70 + "\n")
        
        best_acc = max(results.items(), key=lambda x: x[1].get('final_val_acc', 0))
        f.write(f"Highest Accuracy: {best_acc[0]} ({best_acc[1].get('final_val_acc', 0):.2f}%)\n")
        
        best_loss = min(results.items(), key=lambda x: x[1].get('final_val_loss', float('inf')))
        f.write(f"Lowest Loss: {best_loss[0]} ({best_loss[1].get('final_val_loss', 0):.6f})\n")
        
        fastest = min(results.items(), key=lambda x: x[1].get('training_time', float('inf')))
        f.write(f"Fastest Training: {fastest[0]} ({fastest[1].get('training_time', 0):.2f}s)\n")
        
        # Barren plateau analysis
        f.write("\n" + "="*70 + "\n")
        f.write("BARREN PLATEAU ANALYSIS\n")
        f.write("-"*70 + "\n")
        
        high_risk = [name for name, res in results.items() 
                    if res.get('barren_plateau_risk') == 'high']
        
        if high_risk:
            f.write(f"High Risk Experiments: {', '.join(high_risk)}\n")
        else:
            f.write("No high-risk barren plateaus detected\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✅ Analysis report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Hybrid-QNNs experimental results")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--output", "-o", type=str, default="./analysis",
                       help="Output directory for analysis")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Hybrid-QNNs Results Analysis")
    print("="*70)
    
    # Load results
    print(f"\nLoading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"✅ Loaded {len(results)} experiments")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    # Generate comparison table
    print("\nGenerating comparison table...")
    comparison_df = create_comparison_table(results)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save table
    comparison_df.to_csv(output_path / 'comparison_table.csv', index=False)
    print(f"\n✅ Comparison table saved to {output_path / 'comparison_table.csv'}")
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_comparison(results, args.output)
    
    # Generate report
    print("\nGenerating analysis report...")
    generate_report(results, args.output)
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print(f"📁 Results saved to {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()
