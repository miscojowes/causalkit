"""
Bayesian Neural Network modules for causbayes.

Provides MC Dropout and Concrete Dropout layers for
epistemic uncertainty quantification in neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcreteDropout(nn.Module):
    """Concrete Dropout - learnable dropout rate.

    Uses a continuous relaxation of discrete dropout via the
    Concrete distribution (Gal et al., 2017).

    Parameters
    ----------
    temperature : float
        Temperature for Concrete distribution. Default: 0.1
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.p_logit = nn.Parameter(torch.tensor(-2.0))  # Start with ~0.12 dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Sample from Concrete distribution
            p = torch.sigmoid(self.p_logit)
            uniform = torch.rand_like(x)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
            mask = torch.sigmoid((gumbel_noise + torch.log(p / (1 - p + 1e-8) + 1e-8)) / self.temperature)
            return mask * x / (1 - p + 1e-8)
        return x

    def get_dropout_rate(self) -> float:
        with torch.no_grad():
            return float(torch.sigmoid(self.p_logit).item())


class BayesianLinear(nn.Module):
    """Bayesian Linear layer with weight uncertainty.

    Uses mean-field variational inference with Gaussian
    posterior over weights.

    Parameters
    ----------
    in_features : int
        Input dimension
    out_features : int
        Output dimension
    prior_std : float
        Prior standard deviation. Default: 0.1
    """

    def __init__(self, in_features: int, out_features: int, prior_std: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Variational parameters: mean and log variance
        self.W_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.W_log_var = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_log_var = nn.Parameter(torch.full((out_features,), -5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sampled weights.

        Args:
            x: Input tensor of shape (batch, in_features)

        Returns:
            Output tensor of shape (batch, out_features)
        """
        if self.training:
            # Sample weights using reparameterization trick
            W = self._sample_weight()
            b = self._sample_bias()
            return F.linear(x, W, b)
        else:
            # Use mean weights at test time
            return F.linear(x, self.W_mu, self.b_mu)

    def _sample_weight(self) -> torch.Tensor:
        """Sample weight using reparameterization trick."""
        std = torch.exp(0.5 * self.W_log_var)
        epsilon = torch.randn_like(std)
        return self.W_mu + epsilon * std

    def _sample_bias(self) -> torch.Tensor:
        """Sample bias using reparameterization trick."""
        std = torch.exp(0.5 * self.b_log_var)
        epsilon = torch.randn_like(std)
        return self.b_mu + epsilon * std

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence: KL(q(W|mu,sigma) || p(W)).

        Prior is Gaussian(0, prior_std).

        Returns:
            KL divergence scalar
        """
        # KL between two Gaussians
        var = torch.exp(self.W_log_var)
        kl = 0.5 * torch.sum(
            var / self.prior_std ** 2
            + self.W_mu ** 2 / self.prior_std ** 2
            - 1
            - self.W_log_var
            + 2 * torch.log(torch.tensor(self.prior_std))
        )
        # Bias KL
        b_var = torch.exp(self.b_log_var)
        kl += 0.5 * torch.sum(
            b_var / self.prior_std ** 2
            + self.b_mu ** 2 / self.prior_std ** 2
            - 1
            - self.b_log_var
            + 2 * torch.log(torch.tensor(self.prior_std))
        )
        return kl


class BayesianMLP(nn.Module):
    """Bayesian MLP with uncertainty over all weights.

    Parameters
    ----------
    layer_sizes : list of int
        Layer sizes including input and output
    prior_std : float
        Prior std for Bayesian layers. Default: 0.1
    """

    def __init__(self, layer_sizes: list, prior_std: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                BayesianLinear(layer_sizes[i], layer_sizes[i + 1], prior_std)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def kl_divergence(self) -> torch.Tensor:
        return sum(layer.kl_divergence() for layer in self.layers)
