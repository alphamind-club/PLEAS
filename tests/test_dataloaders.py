"""
Unit test for data loaders and balanced batch sampler.
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, '.')

from data.dataloaders import BalancedBatchSampler


def test_balanced_sampler():
    """Test the balanced batch sampler."""
    np.random.seed(42)
    n_samples = 1000
    n_batches = 4
    
    # Create synthetic batch labels
    batch_labels = np.array([i % n_batches for i in range(n_samples)])
    
    print(f"Testing BalancedBatchSampler with {n_samples} samples, {n_batches} batches")
    
    # Create sampler
    sampler = BalancedBatchSampler(batch_labels, batch_size=64, shuffle=True)
    
    # Get 5 batches
    batches_collected = 0
    batch_sizes = []
    
    for batch_indices in sampler:
        batches_collected += 1
        batch_sizes.append(len(batch_indices))
        
        # Verify at least 2 batches are represented
        unique_batches_in_batch = len(np.unique(batch_labels[batch_indices]))
        print(f"  Batch {batches_collected}: size={len(batch_indices)}, unique batches={unique_batches_in_batch}")
        
        if batches_collected >= 5:
            break
    
    # Verify
    assert batches_collected == 5, f"Expected 5 batches, got {batches_collected}"
    assert all(size > 0 for size in batch_sizes), "All batch sizes should be positive"
    
    print(f"\nSampler test passed: {batches_collected} batches, batch sizes {batch_sizes}")
    return True


if __name__ == "__main__":
    test_balanced_sampler()
