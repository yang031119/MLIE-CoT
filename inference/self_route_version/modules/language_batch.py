"""Language-stage dispatch through the frozen Qwen analyzer."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Sequence

from .qwen_batch import run_qwen_batch_with_retries
from .vcot_branch import HeavyRuntimeManager


def language_model_label(_args: argparse.Namespace) -> str:
	"""Return the audit label stored for the only supported language backend."""
	return "Qwen"


def run_language_batch_with_retries(
	runtime: HeavyRuntimeManager,
	args: argparse.Namespace,
	prepared: Sequence[Dict[str, Any]],
	max_new_tokens: int,
	validator,
	retry_instruction: str,
	batch_size: int,
) -> list[Dict[str, Any]]:
	"""Run a Qwen batch and retry only items rejected by ``validator``."""
	return run_qwen_batch_with_retries(
		runtime,
		args,
		prepared,
		max_new_tokens,
		validator,
		retry_instruction,
		batch_size,
	)
