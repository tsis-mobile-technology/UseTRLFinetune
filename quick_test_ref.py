#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/media/proidea/hdd/Programming/UseTRLFinetune')
os.chdir('/media/proidea/hdd/Programming/UseTRLFinetune')

# Import and test basic functionality
print("Testing imports...")
try:
    import torch
    print("✅ torch imported")
    import transformers
    print("✅ transformers imported")
    import trl
    print("✅ trl imported")
    import datasets
    print("✅ datasets imported")
    print("All imports successful!")
    
    print("\nTesting train.py imports...")
    # Test if train.py can be imported (this will check syntax)
    import importlib.util
    spec = importlib.util.spec_from_file_location("train", "/media/proidea/hdd/Programming/UseTRLFinetune/train.py")
    train_module = importlib.util.module_from_spec(spec)
    
    # Actually execute the module to test imports
    print("Testing train.py execution with minimal parameters...")
    
    # Mock sys.argv for testing
    original_argv = sys.argv
    sys.argv = [
        'train.py',
        '--model_name', 'EleutherAI/polyglot-ko-1.3b',
        '--dataset_name', 'maywell/korean_textbooks',
        '--output_dir', 'my_ref_model_test',
        '--batch_size', '1',
        '--mini_batch_size', '1',
        '--max_ppo_steps', '1',
        '--dataset_sample_size', '5',
        '--max_new_tokens', '8',
        '--learning_rate', '1e-5'
    ]
    
    try:
        spec.loader.exec_module(train_module)
        print("✅ train.py executed successfully!")
    except Exception as e:
        print(f"❌ Error in train.py: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv
        
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()