import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train(model: nn.Module, dataloader: DataLoader, prev_updates, optimizer: torch.optim.Optimizer, device, writer=None, grad_clip_threshold: float = 1.0):
    model.train()  # Set model in train mode
    
    for idx, (data, target) in enumerate(dataloader):
        n_upd = prev_updates + idx  # Calculate the current update number
        
        data = data.to(device)  # Move data to CPU or GPU
        target = target.to(device)  # If target is required, move it to the device too
        
        optimizer.zero_grad()  # Zero the gradients
        
        output = model(data)  # Forward pass
        loss = output.loss  # Assuming the model returns an object with .loss
        
        loss.backward()  # Backward pass
        
        if n_upd % 100 == 0:
            # Calculate the gradient norm for reporting purposes
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            
            print(f'Step {n_upd:,} (N samples: {n_upd * len(data):,}), '
                  f'Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, '
                  f'KL: {output.loss_kl.item():.4f}) Grad: {total_grad_norm:.4f}')
            
            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_grad_norm, global_step)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)    
        
        optimizer.step()  # Update the model parameters
    
    return prev_updates + len(dataloader)
