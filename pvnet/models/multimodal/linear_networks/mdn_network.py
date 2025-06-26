import torch
import torch.nn as nn
import numpy as np

class MDN(nn.Module):
    """A Mixture Density Network layer for a GSP forecast."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features_per_timestep: int,
        num_gaussians: int,
    ):
        """
        A Mixture Density Network layer for a GSP forecast.

        Args:
            in_features: Number of input features.
            hidden_dim: Number of hidden units.
            out_features_per_timestep: Number of output features per timestep. This is the
                number of forecast horizons.
            num_gaussians: The number of gaussian distributions to use.
        """
        super().__init__()
        self.num_gaussians = num_gaussians
        self.out_features_per_timestep = out_features_per_timestep

        # Network layers
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        # The output layer will have 3 * num_gaussians outputs for each forecast horizon
        self.mdn_layer = nn.Linear(
            hidden_dim, out_features_per_timestep * num_gaussians * 3
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MDN.

        Args:
            x: The input tensor.

        Returns:
            A tuple of (pi, sigma, mu) for the GMM.
        """
        # Pass through the first fully-connected layer and ReLU
        x = self.fc1(x)
        x = self.relu(x)

        # Pass through the final MDN layer
        mdn_output = self.mdn_layer(x)

        # Reshape the output to separate the GMM parameters
        # The new shape will be (batch_size, forecast_horizons, num_gaussians, 3)
        mdn_output = mdn_output.view(
            x.shape[0], self.out_features_per_timestep, self.num_gaussians, 3
        )

        # Split the last dimension into pi, sigma, and mu
        pi, sigma, mu = torch.chunk(mdn_output, 3, dim=-1)

        # Squeeze the last dimension which is of size 1
        pi = pi.squeeze(-1)
        sigma = sigma.squeeze(-1)
        mu = mu.squeeze(-1)

        # --- Apply activations to ensure valid parameters ---
        # Use softmax for pi to ensure they sum to 1 for each forecast horizon
        pi = torch.softmax(pi, dim=-1)

        # Use exponential for sigma to ensure it's always positive
        sigma = torch.exp(sigma)

        return pi, sigma, mu
