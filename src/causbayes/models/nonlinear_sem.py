"""
Non-linear Structural Equation Model using separate MLPs per variable.

Implements the proper Neural NOTEARS approach (Lachapelle et al. 2020)
where each variable has its own MLP, and edge weights are computed
as the gradient of each decoder w.r.t. each input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonlinearSEM(nn.Module):
    """Non-linear Structural Equation Model with separate MLPs.

    Each variable X_j is modeled as f_j(X_{pa(j)}) + noise_j.
    Uses variable-specific MLPs for the non-linear functions.

    Edge weights are computed as the norm of the gradient
    of f_j w.r.t. X_i, averaged over the training data.

    Parameters
    ----------
    n_vars : int
        Number of variables
    hidden_layers : list of int
        Hidden layer sizes. Default: [64, 64]
    activation : str
        Activation function: 'relu', 'tanh', 'leaky_relu'. Default: 'relu'
    dropout_rate : float
        Dropout rate for uncertainty. Default: 0.1
    """

    def __init__(
        self,
        n_vars: int,
        hidden_layers: list = None,
        activation: str = "relu",
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_layers = hidden_layers or [64, 64]
        self.dropout_rate = dropout_rate

        # Create a separate MLP for each variable
        # Each MLP takes all d input variables and predicts variable k
        self.mlps = nn.ModuleList([
            self._build_mlp(n_vars, self.hidden_layers, activation, dropout_rate)
            for _ in range(n_vars)
        ])

        # Initialize encoder weights with smaller values
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights with small values for stability."""
        for mlp in self.mlps:
            for module in mlp:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _build_mlp(
        self,
        input_dim: int,
        hidden_layers: list,
        activation: str,
        dropout_rate: float,
    ) -> nn.Module:
        """Build an MLP for one variable."""
        act = self._get_activation(activation)
        layers = []

        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(name, nn.ReLU())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict each variable from all others.

        Args:
            X: Input tensor of shape (batch_size, n_vars)

        Returns:
            Reconstructed X of shape (batch_size, n_vars)
        """
        preds = []
        for k in range(self.n_vars):
            pred_k = self.mlps[k](X)
            preds.append(pred_k)

        out = torch.cat(preds, dim=1)
        self._weight_needs_update = True
        return out

    def compute_weight_matrix(self, normalize: bool = True, X: torch.Tensor = None) -> torch.Tensor:
        """Compute edge weight matrix from first-layer norms.

        W[i,j] = ||first_layer_weights_of_MLP_i[:,j]||_2

        Args:
            normalize: If True, scale so max weight = 1.0
            X: Optional data tensor. If provided, uses gradient-based
               weights (d f_i / d X_j) instead of first-layer norms.

        Returns:
            Weight matrix of shape (n_vars, n_vars)
            W[i,j] = strength of edge j -> i
        """
        if X is not None:
            return self.compute_weight_matrix_with_grad(X, normalize=normalize)

        d = self.n_vars
        device = next(self.parameters()).device
        weight_matrix = torch.zeros(d, d, device=device)

        for k in range(d):
            first_layer = self.mlps[k][0]
            W_first = first_layer.weight  # (hidden_dim, n_vars)

            edge_strengths = torch.norm(W_first, p=2, dim=0)  # (n_vars,)
            weight_matrix[k] = edge_strengths

        weight_matrix.fill_diagonal_(0.0)

        if normalize:
            max_val = weight_matrix.max()
            if max_val > 0:
                weight_matrix = weight_matrix / max_val

        return weight_matrix

    def compute_weight_matrix_with_grad(
        self,
        X: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Compute edge weight matrix using gradient norms.

        For each variable k, computes E[|d f_k / d X_j|] over the batch.
        This measures the sensitivity of f_k to each input X_j.
        More accurate than first-layer norms for sparsity.

        Args:
            X: Input data of shape (batch, n_vars)
            normalize: If True, scale so max weight = 1.0

        Returns:
            Weight matrix of shape (n_vars, n_vars)
        """
        d = self.n_vars
        weight_matrix = torch.zeros(d, d, device=X.device)
        n = X.shape[0]
        
        # Use a subset of data for efficiency when n is large
        batch = X[:min(n, 100)]
        
        for k in range(d):
            x_in = batch.clone().requires_grad_(True)
            pred_k = self.mlps[k](x_in)
            
            # Compute gradient for all inputs at once using torch.autograd.grad
            # with is_grads_batched=True for batch-wise gradients
            grads = torch.autograd.grad(
                pred_k.sum(), x_in, create_graph=False, retain_graph=False
            )[0]  # (batch_size, n_vars)
            
            weight_matrix[k] = grads.abs().mean(dim=0)

        weight_matrix.fill_diagonal_(0.0)

        if normalize:
            max_val = weight_matrix.max()
            if max_val > 0:
                weight_matrix = weight_matrix / max_val

        return weight_matrix


class GNN_SEM(nn.Module):
    """Graph Neural Network SEM for relational data."""

    def __init__(self, n_vars: int, hidden_dim: int = 64):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Linear(1, hidden_dim)
        self.edge_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_decoder = nn.Linear(hidden_dim, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, n_vars = X.shape
        h = self.node_encoder(X.unsqueeze(-1))

        h_i = h.unsqueeze(2).expand(-1, -1, n_vars, -1)
        h_j = h.unsqueeze(1).expand(-1, n_vars, -1, -1)
        edge_feats = torch.cat([h_i, h_j], dim=-1)
        edge_weights = torch.sigmoid(self.edge_net(edge_feats).squeeze(-1))

        eye = torch.eye(n_vars, device=X.device).unsqueeze(0)
        edge_weights = edge_weights * (1 - eye)

        h_agg = torch.bmm(edge_weights, h.squeeze(-1))
        return self.node_decoder(h_agg).squeeze(-1)

    def compute_weight_matrix(self) -> torch.Tensor:
        return torch.zeros(self.n_vars, self.n_vars)
