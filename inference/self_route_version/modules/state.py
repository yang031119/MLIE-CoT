"""State model helpers for MLIE-CoT samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

STATE_SCHEMA_VERSION = 11

@dataclass
class SampleState:
	"""Mutable execution state for one prompt and its image trajectory.

	Fields fall into four groups: immutable sample identity, current image and
	phase, pending cross-stage hand-offs, and final/audit metadata.  A pending
	compare or local review contains enough information to finish that review
	after a restart; it must therefore be persisted before the generating stage
	returns control to the scheduler.
	"""
	index: int
	prompt: str
	sample_dir: Path
	sample_id: str
	initial_generator_family: Optional[str] = None
	initial_generator_policy_digest: Optional[str] = None
	generation_prompt: Optional[str] = None
	current_image: Optional[Path] = None
	phase: str = "global"
	status: str = "pending"
	global_round: int = 0
	needs_global_analysis: bool = False
	pending_global_analysis_reason: str = "initial"
	pending_global_compare: Optional[Dict[str, Any]] = None
	pending_local_review: Optional[Dict[str, Any]] = None
	micro_plan: List[Dict[str, Any]] = field(default_factory=list)
	micro_box_review_done: bool = False
	micro_edit_index: int = 0
	micro_attempt: int = 0
	final_reason: Optional[str] = None
	final_image: Optional[Path] = None
	history: List[Dict[str, Any]] = field(default_factory=list)
	repair_memory: List[Dict[str, Any]] = field(default_factory=list)
	best_round: Optional[Dict[str, Any]] = None
	best_rank: int = 999
	error: Optional[str] = None
	done: bool = False
	started_at: float = field(default_factory=time.time)


def safe_prompt_id(index: int, prompt: str) -> str:
	"""Create a deterministic, filesystem-safe id while preserving row order."""
	slug = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt.strip()).strip("_")[:120]
	return f"{index:03d}_{slug or 'prompt'}"


def sample_round_dir(sample_dir: Path, round_index: int) -> Path:
	"""Return the single, flat artifact directory for one repair round."""
	return sample_dir / f"round{int(round_index)}"


def round_error_analysis_path(sample_dir: Path, round_index: int) -> Path:
	"""Return the canonical diagnosis artifact path for an action round."""
	return sample_round_dir(sample_dir, round_index) / "error_analysis.json"


def round_level_classification_path(sample_dir: Path, round_index: int) -> Path:
	"""Return the route-decision artifact path for an action round."""
	return sample_round_dir(sample_dir, round_index) / "level_classification.json"


def round_action_plan_path(sample_dir: Path, round_index: int) -> Path:
	"""Return the post-route executable-plan artifact path for an action round."""
	return sample_round_dir(sample_dir, round_index) / "repair_plan.json"
