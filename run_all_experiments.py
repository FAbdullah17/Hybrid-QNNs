#!/usr/bin/env python3
"""
Master Automation Script for All Quantum Machine Learning Experiments
Purpose: Run all three entanglement strategies and generate comprehensive comparison reports
"""

import os
import sys
import yaml
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from quantum.models import QuantumConfig
from classical.trainer import AutomatedExperimentRunner
from utils import (
    load_fashion_mnist, load_config, setup_wandb,
    create_experiment_directory, generate_experiment_summary
)

def main():
    """Run all experiments automatically"""
    print("="*100)
    print("QUANTUM MACHINE LEARNING: COMPREHENSIVE EXPERIMENT SUITE")
    print("Barren Plateau Analysis with Different Entanglement Strategies")
    print("="*100)
    
    print("\n🧪 Team Members:")
    print("  • Asma Zubair: No Entanglement Experiments")
    print("  • Farhan Riaz: Full Entanglement Experiments") 
    print("  • Fahad Abdullah: Varied Entanglement Experiments")
    print("\n🎯 Research Goal: Analyze barren plateau problem in quantum neural networks")
    
    # Create main results directory
    main_results_dir = create_experiment_directory("results", "comprehensive_study")
    print(f"\n📁 Main results directory: {main_results_dir}")
    
    # Load base configuration
    base_config = QuantumConfig(
        n_qubits=4,
        n_layers=3,
        n_rotations=3,
        entanglement_type="none",
        entanglement_strength=0.0,
        measurement_type="expval",
        backend="default.qubit",
        shots=None
    )
    
    # Load FashionMNIST dataset
    print("\n📊 Loading FashionMNIST dataset...")
    train_loader, val_loader, test_loader = load_fashion_mnist(
        batch_size=32,
        train_split=0.8
    )
    print("✅ Dataset loaded successfully!")
    
    # Initialize automated experiment runner
    print("\n🚀 Initializing automated experiment runner...")
    runner = AutomatedExperimentRunner(base_config)
    
    # Run all experiments
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE EXPERIMENT SUITE")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Run all experiments automatically
        results = runner.run_all_experiments(
            train_loader=train_loader,
            val_loader=val_loader,
            save_base_dir=main_results_dir
        )
        
        experiment_time = time.time() - start_time
        
        print(f"\n🎉 All experiments completed successfully!")
        print(f"⏱️  Total execution time: {experiment_time:.2f} seconds")
        
        # Generate comprehensive comparison report
        print("\n📈 Generating comprehensive comparison report...")
        comparison_df = runner.generate_comparison_report(save_dir=main_results_dir)
        
        # Print summary table
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY TABLE")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        # Save final summary
        summary_path = os.path.join(main_results_dir, "final_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("QUANTUM MACHINE LEARNING EXPERIMENT SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Execution Time: {experiment_time:.2f} seconds\n\n")
            f.write("TEAM MEMBERS:\n")
            f.write("• Asma Zubair: No Entanglement\n")
            f.write("• Farhan Riaz: Full Entanglement\n")
            f.write("• Fahad Abdullah: Varied Entanglement\n\n")
            f.write("EXPERIMENT RESULTS:\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\nBARren Plateau Analysis:\n")
            
            for exp_name, exp_results in results.items():
                risk = exp_results.get('barren_plateau_risk', 'Unknown')
                avg_grad = exp_results.get('avg_gradient_norm', 'Unknown')
                f.write(f"{exp_name}: Risk={risk}, Avg Gradient Norm={avg_grad}\n")
        
        print(f"\n📝 Final summary saved to: {summary_path}")
        
        # Generate research insights
        print("\n🔬 RESEARCH INSIGHTS:")
        insights = generate_research_insights(results)
        for insight in insights:
            print(f"  • {insight}")
        
        # Save insights
        insights_path = os.path.join(main_results_dir, "research_insights.txt")
        with open(insights_path, 'w') as f:
            f.write("RESEARCH INSIGHTS\n")
            f.write("="*20 + "\n\n")
            for insight in insights:
                f.write(f"• {insight}\n")
        
        print(f"\n💡 Research insights saved to: {insights_path}")
        
        print(f"\n🎯 COMPREHENSIVE STUDY COMPLETED!")
        print(f"📁 All results saved to: {main_results_dir}")
        print(f"📊 Comparison report: {os.path.join(main_results_dir, 'experiment_comparison.csv')}")
        print(f"📈 Plots and visualizations saved in experiment subdirectories")
        
        return results, comparison_df
        
    except Exception as e:
        print(f"\n❌ Experiment suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_research_insights(results):
    """Generate research insights from experiment results"""
    insights = []
    
    if not results:
        return ["No results available for analysis"]
    
    # Analyze barren plateau patterns
    barren_plateau_counts = sum(1 for r in results.values() 
                               if r.get('barren_plateau_risk') == 'high')
    
    if barren_plateau_counts > 0:
        insights.append(f"Barren plateaus detected in {barren_plateau_counts}/{len(results)} experiments")
    
    # Compare performance across strategies
    accuracies = [(name, r.get('final_val_acc', 0)) for name, r in results.items()]
    best_strategy = max(accuracies, key=lambda x: x[1])
    insights.append(f"Best performing strategy: {best_strategy[0]} with {best_strategy[1]:.2f}% accuracy")
    
    # Analyze gradient behavior
    gradient_norms = [(name, r.get('avg_gradient_norm', 0)) for name, r in results.items()]
    stable_gradients = [(name, norm) for name, norm in gradient_norms if norm > 1e-5]
    
    if stable_gradients:
        insights.append(f"Stable gradients maintained in {len(stable_gradients)} strategies")
    
    # Training efficiency
    training_times = [(name, r.get('training_time', 0)) for name, r in results.items()]
    fastest = min(training_times, key=lambda x: x[1])
    insights.append(f"Fastest training: {fastest[0]} in {fastest[1]:.2f} seconds")
    
    # Entanglement impact
    if 'no_entanglement' in results and 'with_entanglement' in results:
        no_ent_acc = results['no_entanglement'].get('final_val_acc', 0)
        with_ent_acc = results['with_entanglement'].get('final_val_acc', 0)
        
        if with_ent_acc > no_ent_acc:
            insights.append("Full entanglement shows improved performance over no entanglement")
        else:
            insights.append("No entanglement shows comparable or better performance")
    
    # Convergence analysis
    slow_convergence = sum(1 for r in results.values() 
                          if r.get('epochs_trained', 0) >= 40)
    if slow_convergence > 0:
        insights.append(f"Slow convergence observed in {slow_convergence} experiments")
    
    return insights

def run_single_experiment(experiment_name):
    """Run a single experiment by name"""
    print(f"\n🧪 Running single experiment: {experiment_name}")
    
    if experiment_name == "no_entanglement":
        os.system("python experiments/no_entanglement.py")
    elif experiment_name == "with_entanglement":
        os.system("python experiments/with_entanglement.py")
    elif experiment_name == "varied_entanglement":
        os.system("python experiments/varied_entanglement.py")
    else:
        print(f"❌ Unknown experiment: {experiment_name}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum ML Experiment Suite")
    parser.add_argument("--experiment", "-e", type=str, 
                       choices=["no_entanglement", "with_entanglement", "varied_entanglement", "all"],
                       default="all", help="Experiment to run")
    
    args = parser.parse_args()
    
    if args.experiment == "all":
        print("🚀 Running comprehensive experiment suite...")
        results, comparison = main()
    else:
        print(f"🧪 Running single experiment: {args.experiment}")
        success = run_single_experiment(args.experiment)
        if success:
            print(f"✅ {args.experiment} completed successfully!")
        else:
            print(f"❌ {args.experiment} failed!")
            sys.exit(1)
