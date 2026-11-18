"""Flux.2 L1/L2 plus initial-base L3 data builder for the MLIE-CoT router.

The package is intentionally split into a pure-CPU planning/aggregation layer
and opt-in GPU executors.  Importing this package never loads torch, diffusers,
transformers, or a model checkpoint.
"""

from .schema import ACTION_LEVELS, SCHEMA_VERSION, DataValidationError

__all__ = ["ACTION_LEVELS", "SCHEMA_VERSION", "DataValidationError"]
