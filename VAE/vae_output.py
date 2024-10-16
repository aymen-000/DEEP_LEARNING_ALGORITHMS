from dataclasses import dataclass
import torch 
import torch.distributions as distro
@dataclass
class VAEOutput:
    z_dist: distro.Distribution
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor
