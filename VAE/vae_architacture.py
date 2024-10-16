import torch 
from torch.optim import adam
import torch.nn as nn 
import torch.distributions as distro
from vae_output import VAEOutput
class VAE(nn.Module):
    """
        VAE class
        args : 
            input_dims : input dimensions 
            hidden_dims : hidden network dimensions 
            latent_dims : latent variable (Z) dimensions 
    """
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(VAE, self).__init__()
        self.hidden_dims = hidden_dims
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.softplus = nn.Softplus()
        
        # Build encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims),
            nn.SiLU(),
            nn.Linear(self.hidden_dims, self.hidden_dims // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dims // 2, self.hidden_dims // 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dims // 4, self.hidden_dims // 8),
            nn.SiLU(),
            nn.Linear(self.hidden_dims // 8, 2 * self.latent_dims),
        )
        
        # Build decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dims, self.hidden_dims // 8),  # Fixed typo (self.laten_dims to self.latent_dims)
            nn.SiLU(),
            nn.Linear(self.hidden_dims // 8, self.hidden_dims // 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dims // 4, self.hidden_dims // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dims // 2, self.hidden_dims),
            nn.SiLU(),
            nn.Linear(self.hidden_dims, self.input_dims),
            nn.Sigmoid(),  # For generating values between 0 and 1
        )
    
    # Encoder function
    def encode(self, x, eps: float = 1e-8):
        """
            Encode function
            args : 
                x : input data 
                eps : small value to avoid errors 
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)  # Get mean and log-variance
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)  # Get diagonal covariance matrix
        
        return distro.MultivariateNormal(mu, scale_tril=scale_tril)  # Sample from Gaussian multivariate distribution
    
    def reparameterize(self, dist):
        """
            This function samples from our distribution
            args  : 
                dist : predefined distribution 
            return : 
                sample from distribution (torch.tensor)
        """
        return dist.rsample()
    
    def decode(self, z):
        """
            This function plays the role of decoder
            args : 
                z : torch.tensor latent space 
            return : 
                torch.tensor 
        """
        return self.decoder(z)
    
    def forward(self, x, loss: bool = True):
        """
            Forward function: encoder => sample => decode 
            args : 
                x : torch.tensor input data 
                loss : if we want to calculate loss 
            return : 
                VAE output 
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon = self.decode(z)
        
        if not loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )
        
        # Compute reconstruction loss (BCE)
        recon_loss = torch.nn.functional.binary_cross_entropy(recon, x + 0.3, reduction='none').sum(-1).mean()
        
        # Define standard normal distribution for KL divergence
        std_normal = distro.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1)
        )
        
        # Compute KL divergence between the learned distribution and the standard normal
        kl_loss = distro.kl_divergence(dist, std_normal).mean()
        
        total_loss = recon_loss + kl_loss
        
        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon,
            loss=total_loss,
            loss_recon=recon_loss,
            loss_kl=kl_loss,
        )
