import sys
import os

# Set the working directory and path
sys.path.insert(0, '/media/proidea/hdd/Programming/UseTRLFinetune')
os.chdir('/media/proidea/hdd/Programming/UseTRLFinetune')

# Simple test just for PPOTrainer creation
print("Testing PPOTrainer creation with updated reference model...")

try:
    # Import and test train.py directly with minimal parameters
    import argparse
    
    # Mock sys.argv for testing
    original_argv = sys.argv
    sys.argv = [
        'train.py',
        '--model_name', 'EleutherAI/polyglot-ko-1.3b',
        '--dataset_name', 'maywell/korean_textbooks',
        '--output_dir', 'test_output',
        '--batch_size', '1',
        '--mini_batch_size', '1',
        '--max_ppo_steps', '1',
        '--dataset_sample_size', '3',
        '--max_new_tokens', '4',
        '--learning_rate', '1e-5'
    ]
    
    # Import the main function from train.py
    from train_copilot import main
    
    # Run it
    main()
    
    print("✅ Train script completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    
    # Check for specific error types
    if "'tuple' object has no attribute 'logits'" in str(e):
        print("❌ Still getting the tuple/logits AttributeError!")
    else:
        print("ℹ️  Different error detected")
    
    import traceback
    traceback.print_exc()
    
finally:
    # Restore original argv
    sys.argv = original_argv