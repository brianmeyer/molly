"""Backward-compatibility shim. All functionality moved to evolution/ package."""
from evolution._base import PatchValidation          # canonical location
from evolution.skills import SelfImprovementEngine

__all__ = ["SelfImprovementEngine", "PatchValidation"]
