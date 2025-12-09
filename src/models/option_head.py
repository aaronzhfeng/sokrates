"""
Option Success Head (q̂_φ): Predicts probability that an option is solver-valid.

This is the "knowledge" component in the OaK framework - an explicit
predictive model of option success.
"""

import torch
import torch.nn as nn
from typing import Optional

from src.data.structures import OptionType, OPTION_VOCABULARY


class OptionSuccessHead(nn.Module):
    """
    Predicts P(step is solver-valid | state, option).
    
    Takes the LLM's hidden state at a decision point and an option type,
    outputs the predicted probability that applying this option will
    produce a solver-valid result.
    
    This is the q̂_φ(s, ω) predictor from the SOKRATES formulation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        option_embed_dim: int = 64,
        mlp_hidden_dim: int = 512,
        num_option_types: int = None,
        dropout: float = 0.1,
    ):
        """
        Initialize the option success head.
        
        Args:
            hidden_dim: Dimension of the LLM hidden states
            option_embed_dim: Dimension for option type embeddings
            mlp_hidden_dim: Hidden dimension for the MLP
            num_option_types: Number of option types (defaults to vocabulary size)
            dropout: Dropout probability
        """
        super().__init__()
        
        if num_option_types is None:
            num_option_types = len(OPTION_VOCABULARY)
        
        self.hidden_dim = hidden_dim
        self.option_embed_dim = option_embed_dim
        self.num_option_types = num_option_types
        
        # Option type embedding
        self.option_embed = nn.Embedding(num_option_types, option_embed_dim)
        
        # MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + option_embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 4, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        nn.init.normal_(self.option_embed.weight, mean=0, std=0.02)
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        option_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict option success probability.
        
        Args:
            hidden_state: [batch_size, hidden_dim] LLM hidden states
            option_type_ids: [batch_size] or [batch_size, 1] option type indices
            
        Returns:
            [batch_size, 1] predicted success probabilities (sigmoid applied)
        """
        # Ensure option_type_ids is 1D
        if option_type_ids.dim() > 1:
            option_type_ids = option_type_ids.squeeze(-1)
        
        # Get option embeddings
        option_emb = self.option_embed(option_type_ids)  # [batch, option_embed_dim]
        
        # Concatenate hidden state and option embedding
        combined = torch.cat([hidden_state, option_emb], dim=-1)
        
        # Predict through MLP and apply sigmoid
        logits = self.mlp(combined)
        probs = torch.sigmoid(logits)
        
        return probs
    
    def predict_all_options(
        self,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict success probability for all option types.
        
        Useful for option selection during inference.
        
        Args:
            hidden_state: [batch_size, hidden_dim] or [hidden_dim] LLM hidden state
            
        Returns:
            [batch_size, num_option_types] or [num_option_types] probabilities
        """
        # Handle single hidden state (no batch dim)
        single = hidden_state.dim() == 1
        if single:
            hidden_state = hidden_state.unsqueeze(0)
        
        batch_size = hidden_state.shape[0]
        device = hidden_state.device
        
        # Create all option type ids
        all_option_ids = torch.arange(self.num_option_types, device=device)
        all_option_ids = all_option_ids.unsqueeze(0).expand(batch_size, -1)  # [batch, num_opts]
        
        # Get embeddings for all options
        all_option_emb = self.option_embed(all_option_ids)  # [batch, num_opts, embed_dim]
        
        # Expand hidden state
        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, self.num_option_types, -1)
        
        # Concatenate
        combined = torch.cat([hidden_expanded, all_option_emb], dim=-1)  # [batch, num_opts, hidden+embed]
        
        # Reshape for MLP
        combined_flat = combined.view(-1, combined.shape[-1])
        logits_flat = self.mlp(combined_flat)
        probs = torch.sigmoid(logits_flat).view(batch_size, self.num_option_types)
        
        if single:
            probs = probs.squeeze(0)
        
        return probs
    
    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        option_type_ids: torch.Tensor,
        labels: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for training.
        
        Args:
            hidden_states: [batch_size, hidden_dim] LLM hidden states
            option_type_ids: [batch_size] option type indices
            labels: [batch_size] binary labels (1 = valid, 0 = invalid)
            reduction: "mean", "sum", or "none"
            
        Returns:
            BCE loss
        """
        probs = self.forward(hidden_states, option_type_ids)
        labels = labels.float().unsqueeze(-1)
        
        loss = nn.functional.binary_cross_entropy(
            probs, labels, reduction=reduction
        )
        
        return loss
    
    @classmethod
    def from_pretrained(cls, path: str, hidden_dim: int) -> "OptionSuccessHead":
        """Load a pretrained option success head."""
        head = cls(hidden_dim=hidden_dim)
        state_dict = torch.load(path, map_location="cpu")
        head.load_state_dict(state_dict)
        return head
    
    def save(self, path: str):
        """Save the option success head."""
        torch.save(self.state_dict(), path)


class OptionSuccessHeadWithArgs(OptionSuccessHead):
    """
    Extended option success head that also considers option arguments.
    
    This provides finer-grained predictions by encoding which specific
    formulas the option is being applied to.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        option_embed_dim: int = 64,
        arg_embed_dim: int = 32,
        max_args: int = 4,
        max_formula_idx: int = 100,
        mlp_hidden_dim: int = 512,
        num_option_types: int = None,
        dropout: float = 0.1,
    ):
        """
        Initialize with argument encoding.
        
        Args:
            hidden_dim: Dimension of LLM hidden states
            option_embed_dim: Dimension for option type embeddings
            arg_embed_dim: Dimension for argument embeddings
            max_args: Maximum number of arguments to encode
            max_formula_idx: Maximum formula index (for embedding)
            mlp_hidden_dim: Hidden dimension for MLP
            num_option_types: Number of option types
            dropout: Dropout probability
        """
        # Don't call parent __init__ yet
        nn.Module.__init__(self)
        
        if num_option_types is None:
            num_option_types = len(OPTION_VOCABULARY)
        
        self.hidden_dim = hidden_dim
        self.option_embed_dim = option_embed_dim
        self.arg_embed_dim = arg_embed_dim
        self.max_args = max_args
        self.num_option_types = num_option_types
        
        # Option type embedding
        self.option_embed = nn.Embedding(num_option_types, option_embed_dim)
        
        # Argument index embeddings
        self.arg_embed = nn.Embedding(max_formula_idx + 1, arg_embed_dim)
        
        # Total input dimension
        total_dim = hidden_dim + option_embed_dim + (max_args * arg_embed_dim)
        
        # MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 4, 1),
        )
        
        self._init_weights()
        nn.init.normal_(self.arg_embed.weight, mean=0, std=0.02)
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        option_type_ids: torch.Tensor,
        option_args: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict option success probability with arguments.
        
        Args:
            hidden_state: [batch_size, hidden_dim] LLM hidden states
            option_type_ids: [batch_size] option type indices
            option_args: [batch_size, max_args] argument indices (padded with 0)
            
        Returns:
            [batch_size, 1] predicted success probabilities
        """
        batch_size = hidden_state.shape[0]
        device = hidden_state.device
        
        # Ensure option_type_ids is 1D
        if option_type_ids.dim() > 1:
            option_type_ids = option_type_ids.squeeze(-1)
        
        # Get option embeddings
        option_emb = self.option_embed(option_type_ids)
        
        # Handle arguments
        if option_args is None:
            option_args = torch.zeros(
                batch_size, self.max_args, dtype=torch.long, device=device
            )
        
        # Pad or truncate args to max_args
        if option_args.shape[1] < self.max_args:
            padding = torch.zeros(
                batch_size, self.max_args - option_args.shape[1],
                dtype=torch.long, device=device
            )
            option_args = torch.cat([option_args, padding], dim=1)
        elif option_args.shape[1] > self.max_args:
            option_args = option_args[:, :self.max_args]
        
        # Get argument embeddings
        arg_emb = self.arg_embed(option_args)  # [batch, max_args, arg_embed_dim]
        arg_emb_flat = arg_emb.view(batch_size, -1)  # [batch, max_args * arg_embed_dim]
        
        # Concatenate all features
        combined = torch.cat([hidden_state, option_emb, arg_emb_flat], dim=-1)
        
        # Predict
        logits = self.mlp(combined)
        probs = torch.sigmoid(logits)
        
        return probs

