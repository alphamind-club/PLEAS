"""
Unit test for NT-Xent contrastive loss - verify loss decreases on toy data.
"""
import torch
import torch.nn.functional as F
import pytest


class NTXentLoss:
    """NT-Xent Loss - index i pairs with i+N"""
    
    def __init__(self, temperature=0.1):
        self.temperature = temperature
    
    def __call__(self, z, batch_size=None):
        if batch_size is None:
            batch_size = z.shape[0] // 2
        N = batch_size
        
        # Similarity matrix
        sim = torch.matmul(z, z.T)
        sim = torch.clamp(sim, -1.0, 1.0) / self.temperature
        
        # Positive indices: i -> i+N
        pos_idx = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)])
        rows = torch.arange(2 * N)
        
        # Positive similarities
        pos_sim = sim[rows, pos_idx]
        
        # Loss
        loss = -pos_sim + torch.logsumexp(sim, dim=1)
        
        return loss.mean()


def test_ntxent_loss_decreases():
    """
    Test that NT-Xent loss decreases when positives are pulled together.
    Steps:
    1. Create 16 random 64-dim vectors as 'embeddings'
    2. Duplicate to create 32-vector batch (view-a + view-b)
    3. Compute NT-Xent loss with known positive pairs
    4. Assert loss > 0 initially
    5. Optimize embeddings to pull positives together
    6. Assert loss decreases after 100 optimization steps
    """
    torch.manual_seed(42)
    
    # 1. Create 16 random 64-dim vectors
    N = 16
    dim = 64
    
    # Create embeddings with requires_grad for optimization
    embeddings = torch.randn(N, dim, requires_grad=True)
    
    # 2. Duplicate to create 32-vector batch (view-a + view-b)
    # First N are view-a, next N are view-b
    # Positive pairs: embedding[i] in view-a pairs with embedding[i] in view-b
    batch = torch.cat([embeddings, embeddings.clone()], dim=0)
    batch = F.normalize(batch, p=2, dim=1)
    
    # 3. Compute NT-Xent loss with known positive pairs
    criterion = NTXentLoss(temperature=0.1)
    loss_initial = criterion(batch, batch_size=N)
    
    # 4. Assert loss > 0 initially
    print(f"Initial loss: {loss_initial.item():.4f}")
    assert loss_initial.item() > 0, "Initial loss should be positive"
    
    # 5. Optimize embeddings to pull positives together
    optimizer = torch.optim.Adam([embeddings], lr=0.01)
    
    losses = []
    for step in range(100):
        optimizer.zero_grad()
        
        # Reconstruct batch with updated embeddings
        batch = torch.cat([embeddings, embeddings.clone()], dim=0)
        batch = F.normalize(batch, p=2, dim=1)
        
        loss = criterion(batch, batch_size=N)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
    
    final_loss = losses[-1]
    print(f"Final loss after 100 steps: {final_loss:.4f}")
    print(f"Loss reduction: {loss_initial.item() - final_loss:.4f}")
    
    # 6. Assert loss decreases
    assert final_loss < loss_initial.item(), "Loss should decrease after optimization"
    assert final_loss < 1.0, f"Final loss should be < 1.0, got {final_loss:.4f}"
    
    print("Test PASSED: Loss decreased from {:.4f} to {:.4f}".format(
        loss_initial.item(), final_loss))


if __name__ == "__main__":
    test_ntxent_loss_decreases()
