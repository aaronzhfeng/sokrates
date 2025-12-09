"""Model architectures including base LM wrapper and auxiliary heads."""

from src.models.option_head import OptionSuccessHead
from src.models.gvf_heads import ConsistencyGVF, GoalProgressGVF

__all__ = [
    "OptionSuccessHead",
    "ConsistencyGVF", 
    "GoalProgressGVF",
]

