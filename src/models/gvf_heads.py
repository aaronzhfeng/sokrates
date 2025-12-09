"""
General Value Function (GVF) Heads for auxiliary reward-respecting subtasks.

These heads predict cumulative signals for subtasks like:
- Consistency: Will the proof remain contradiction-free?
- Goal Progress: Does this step bring us closer to the target?

These are the "reward-respecting subtasks" in OaK/STOMP terminology.
"""

import torch
import torch.nn as nn
from typing import Optional


class BaseGVFHead(nn.Module):
    """
    Base class for GVF prediction heads.
    
    A GVF predicts the expected cumulative signal under some policy,
    similar to a value function but for arbitrary signals beyond reward.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize the GVF head.
        
        Args:
            hidden_dim: Dimension of LLM hidden states
            mlp_hidden_dim: Hidden dimension for MLP
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Predict GVF value.
        
        Args:
            hidden_state: [batch_size, hidden_dim] LLM hidden states
            
        Returns:
            [batch_size, 1] predicted values (sigmoid applied for [0,1] range)
        """
        logits = self.mlp(hidden_state)
        return torch.sigmoid(logits)
    
    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute MSE loss for GVF training.
        
        Args:
            hidden_states: [batch_size, hidden_dim]
            targets: [batch_size] target values in [0, 1]
            reduction: "mean", "sum", or "none"
        """
        predictions = self.forward(hidden_states).squeeze(-1)
        loss = nn.functional.mse_loss(predictions, targets, reduction=reduction)
        return loss


class ConsistencyGVF(BaseGVFHead):
    """
    Predicts whether the proof will remain logically consistent.
    
    Subtask reward: 1 at the end if no step was invalid (no contradictions
    or illegal inferences), 0 otherwise.
    
    This GVF learns to predict this binary outcome from intermediate states,
    essentially predicting "will we avoid introducing contradictions?"
    """
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_dim, mlp_hidden_dim, dropout)
    
    @staticmethod
    def compute_targets_from_trace(trace) -> list[float]:
        """
        Compute consistency targets for each step in a trace.
        
        For each step, the target is 1 if all remaining steps (including
        this one) are valid, 0 otherwise.
        
        Args:
            trace: OptionizedTrace with solver_valid labels on steps
            
        Returns:
            List of target values, one per step
        """
        targets = []
        
        # Get validity of remaining steps at each position
        for i in range(len(trace.steps)):
            remaining_steps = trace.steps[i:]
            # Check if all remaining steps are valid
            all_valid = all(
                s.solver_valid for s in remaining_steps 
                if s.solver_valid is not None
            )
            targets.append(1.0 if all_valid else 0.0)
        
        return targets


class GoalProgressGVF(BaseGVFHead):
    """
    Predicts whether this step brings us closer to the goal.
    
    Subtask reward: 1 if the step derives a formula that is "closer"
    to the conclusion (e.g., introduces the right predicate, simplifies
    toward the goal), 0 otherwise.
    
    This is a softer notion of progress that can be computed from
    heuristics like predicate overlap with the target.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_dim, mlp_hidden_dim, dropout)
    
    @staticmethod
    def compute_targets_from_trace(trace, target_predicates: Optional[set] = None) -> list[float]:
        """
        Compute goal progress targets for each step in a trace.
        
        A step gets reward 1 if it introduces predicates that appear
        in the target conclusion.
        
        Args:
            trace: OptionizedTrace
            target_predicates: Set of predicate names in the target
            
        Returns:
            List of target values, one per step
        """
        targets = []
        
        # Extract predicates from target if not provided
        if target_predicates is None:
            target_predicates = set()
            if trace.initial_state.target_fol:
                # Simple heuristic: extract words that might be predicates
                import re
                fol_str = trace.initial_state.target_fol.fol_string
                # Match capitalized words (likely predicates in FOL notation)
                target_predicates = set(re.findall(r'\b[A-Z][a-z]*\b', fol_str))
        
        for step in trace.steps:
            # Check if this step's result contains target predicates
            if step.result_formula:
                step_text = step.result_formula.nl_text + " " + step.result_formula.fol_string
                step_predicates = set(step_text.split())
                
                # Compute overlap
                overlap = len(target_predicates & step_predicates)
                progress = min(1.0, overlap / max(1, len(target_predicates)))
            else:
                progress = 0.0
            
            targets.append(progress)
        
        return targets


class ProofLengthGVF(BaseGVFHead):
    """
    Predicts the remaining proof length.
    
    This is a auxiliary signal that can help with planning - predicting
    how many more steps are needed to complete the proof.
    
    Unlike the other GVFs, this outputs an unbounded positive value
    rather than a probability.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden_dim: int = 256,
        max_steps: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_dim, mlp_hidden_dim, dropout)
        self.max_steps = max_steps
        
        # Override the final layer to not use sigmoid
        # Instead, output normalized step count
        self.mlp[-1] = nn.Linear(mlp_hidden_dim // 2, 1)
    
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Predict remaining steps (normalized by max_steps).
        
        Returns value in [0, 1] where 1 = max_steps remaining.
        """
        logits = self.mlp(hidden_state)
        # Use softplus to ensure positive, then normalize
        return torch.softplus(logits) / self.max_steps
    
    @staticmethod
    def compute_targets_from_trace(trace, max_steps: int = 20) -> list[float]:
        """
        Compute remaining length targets for each step.
        
        Args:
            trace: OptionizedTrace
            max_steps: Maximum steps for normalization
            
        Returns:
            List of target values (remaining steps / max_steps)
        """
        total_steps = len(trace.steps)
        targets = []
        
        for i in range(total_steps):
            remaining = total_steps - i - 1
            targets.append(remaining / max_steps)
        
        return targets


class CombinedGVFHead(nn.Module):
    """
    Combines multiple GVF heads with a shared feature extractor.
    
    This is more parameter-efficient than having separate heads,
    as they can share lower-level representations.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        shared_dim: int = 256,
        head_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Individual heads
        self.consistency_head = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        
        self.goal_progress_head = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        
        self.proof_length_head = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Predict all GVF values.
        
        Args:
            hidden_state: [batch_size, hidden_dim]
            
        Returns:
            Dict with keys: "consistency", "goal_progress", "proof_length"
        """
        shared_features = self.shared(hidden_state)
        
        return {
            "consistency": torch.sigmoid(self.consistency_head(shared_features)),
            "goal_progress": torch.sigmoid(self.goal_progress_head(shared_features)),
            "proof_length": torch.softplus(self.proof_length_head(shared_features)),
        }
    
    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        targets: dict[str, torch.Tensor],
        weights: Optional[dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss for all GVFs.
        
        Args:
            hidden_states: [batch_size, hidden_dim]
            targets: Dict with target tensors for each GVF
            weights: Optional dict with loss weights for each GVF
            
        Returns:
            Combined loss
        """
        if weights is None:
            weights = {"consistency": 1.0, "goal_progress": 1.0, "proof_length": 0.5}
        
        predictions = self.forward(hidden_states)
        total_loss = 0.0
        
        for name, pred in predictions.items():
            if name in targets:
                target = targets[name]
                if target.dim() == 1:
                    target = target.unsqueeze(-1)
                loss = nn.functional.mse_loss(pred, target)
                total_loss = total_loss + weights.get(name, 1.0) * loss
        
        return total_loss

