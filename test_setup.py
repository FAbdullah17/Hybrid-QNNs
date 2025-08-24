#!/usr/bin/env python3
"""
Test script to verify the quantum machine learning setup
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        # Test quantum models
        from quantum.models import QuantumConfig, QuantumNeuralNetwork, BarrenPlateauAnalyzer
    print("Quantum models imported successfully")
        
        # Test classical trainer
        from classical.trainer import QuantumTrainer, AutomatedExperimentRunner
    print("Classical trainer imported successfully")
        
        # Test utilities
        from utils import load_config, create_experiment_directory
    print("Utilities imported successfully")
        
        return True
        
    except Exception as e:
    print(f"Import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from utils import load_config
        
        # Test loading each config
        configs = [
            "configs/no_entanglement.yaml",
            "configs/with_entanglement.yaml", 
            "configs/varied_entanglement.yaml"
        ]
        
        for config_path in configs:
            if os.path.exists(config_path):
                config = load_config(config_path)
            print(f"Loaded {config_path}")
                print(f"   Researcher: {config.get('researcher', 'Unknown')}")
                print(f"   Entanglement: {config.get('entanglement_type', 'Unknown')}")
            else:
            print(f"Config file not found: {config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_quantum_config():
    """Test quantum configuration creation"""
    print("\nTesting quantum configuration...")
    
    try:
        from quantum.models import QuantumConfig
        
        # Create test config
        config = QuantumConfig(
            n_qubits=4,
            n_layers=3,
            entanglement_type="none"
        )
        
    print(f"Quantum config created:")
        print(f"   Qubits: {config.n_qubits}")
        print(f"   Layers: {config.n_layers}")
        print(f"   Entanglement: {config.entanglement_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum config creation failed: {e}")
        return False

def test_model_creation():
    """Test quantum model creation"""
    print("\n🏗️  Testing model creation...")
    
    try:
        from quantum.models import QuantumConfig, QuantumNeuralNetwork
        
        # Create test config
        config = QuantumConfig(
            n_qubits=2,  # Small for testing
            n_layers=1,
            entanglement_type="none"
        )
        
        # Create model
        model = QuantumNeuralNetwork(config)
    print(f"Quantum model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 QUANTUM MACHINE LEARNING SETUP TEST")
    print("="*50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_quantum_config,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready for experiments.")
        print("\n🚀 Next steps:")
        print("   1. Run individual experiments:")
        print("      python experiments/no_entanglement.py")
        print("      python experiments/with_entanglement.py")
        print("      python experiments/varied_entanglement.py")
        print("   2. Or run all automatically:")
        print("      python run_all_experiments.py")
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
