import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ScoreNet(nn.Module):
    """A simple feedforward network for score matching, with two hidden layers."""

    def __init__(self, marginal_prob_std, input_dim=7, hidden_dim=256, embed_dim=256):
        """Initialize a simple feedforward network with time embedding for score matching.
        
        Args:
            marginal_prob_std: A function that takes time t and gives the standard deviation of the perturbation kernel.
            input_dim: The dimensionality of the input data (7 in your case).
            hidden_dim: The number of hidden units in each hidden layer.
            embed_dim: The dimensionality of the time embedding.
        """
        super().__init__()
 
        # Gaussian random feature embedding for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, input_dim)  # Output layer

        # Activation function (Swish)
        self.act = lambda x: x * torch.sigmoid(x)

        self.marginal_prob_std = marginal_prob_std
  
    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for time
        embed = self.act(self.embed(t))
        
        # First hidden layer
        h1 = self.fc1(x)
        h1 += embed  # Incorporate time information
        h1 = self.act(h1)

        # Second hidden layer
        h2 = self.fc2(h1)
        h2 += embed  # Incorporate time information
        h2 = self.act(h2)

        # Output layer
        h3 = self.fc3(h2)

        # Normalize output
        h = h3 / self.marginal_prob_std(t).unsqueeze(1)  # Normalizing with marginal probability
        return h
