"""Immutable Flux.2 L1/L2 action-policy contract shared by collection and serving.

The route label is meaningful only relative to the editor that produced its
counterfactual utilities.  This module therefore freezes the concrete Flux.2
executor, model snapshot, generation settings, L1 crop geometry, and the
one-candidate/one-successful-call semantics for L1/L2. L3 has a separate
initial-base-bound contract in :mod:`l3_policy`. It is kept free of GPU imports
so aggregate validation and online startup checks are cheap CPU operations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .contracts import ROUTER_ACTION_POLICY_VERSION


FLUX2_POLICY_SCHEMA = "vcot_v10_flux2_l1_l2_policy_v2"
FLUX2_EXECUTOR_FAMILY = "flux2"
FLUX2_EXECUTOR_IMPLEMENTATION = "diffusers.Flux2Pipeline"
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$", flags=re.IGNORECASE)

_ACTION_SEMANTICS = {
	"L1": {
		"operation": "edit_multi_reference",
		"candidate_count": 1,
		"successful_flux_call_count": 1,
		"semantic_retry_count": 0,
		"references": "one_editable_full_image_then_one_to_three_ordered_localization_crops",
	},
	"L2": {
		"operation": "edit_single_reference",
		"candidate_count": 1,
		"successful_flux_call_count": 1,
		"semantic_retry_count": 0,
		"references": "one_editable_full_image",
	},
}


def _canonical_json(value: Any) -> str:
	"""Serialize policy data deterministically for equality and hashing."""
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def flux2_policy_digest(contract: Mapping[str, Any]) -> str:
	"""Return the content digest for an already canonical policy contract."""

	return hashlib.sha256(_canonical_json(dict(contract)).encode("utf-8")).hexdigest()


def canonical_flux_dtype(value: Any) -> str:
	"""Normalize supported PyTorch/CLI dtype aliases for stable hashing."""
	name = str(value or "").replace("torch.", "").strip().lower()
	aliases = {
		"bf16": "bfloat16",
		"bfloat16": "bfloat16",
		"fp16": "float16",
		"float16": "float16",
		"half": "float16",
	}
	if name not in aliases:
		raise ValueError(f"Unsupported Flux.2 dtype: {value!r}")
	return aliases[name]


def _make_flux2_policy_contract(
	*,
	model_id: str,
	model_revision: str,
	resolution: int,
	steps: int,
	guidance_scale: float,
	max_sequence_length: int,
	dtype: str,
	l1_roi_pad_ratio: float,
	l1_min_span: int,
	l1_max_crops: int,
	prompt_version: str,
) -> dict[str, Any]:
	"""Assemble the strict serialized shape after individual values are checked."""
	return {
		"schema_version": FLUX2_POLICY_SCHEMA,
		"action_policy_version": ROUTER_ACTION_POLICY_VERSION,
		"executor": {
			"family": FLUX2_EXECUTOR_FAMILY,
			"implementation": FLUX2_EXECUTOR_IMPLEMENTATION,
			"loader": "from_pretrained(local_files_only=True)",
			"model_id": model_id,
			"model_revision": model_revision,
			"dtype": dtype,
		},
		"generation": {
			"resolution": resolution,
			"steps": steps,
			"guidance_scale": guidance_scale,
			"max_sequence_length": max_sequence_length,
		},
		"l1_crop": {
			"roi_pad_ratio": l1_roi_pad_ratio,
			"min_span": l1_min_span,
			"max_crops": l1_max_crops,
			"crop_order": "planner_order",
			"each_crop_resized_to_generation_resolution": True,
		},
		"actions": {key: dict(value) for key, value in _ACTION_SEMANTICS.items()},
		"prompt_version": prompt_version,
		"infrastructure_retry_policy": "exact_same_request_only; no extra semantic candidate",
	}


def build_flux2_policy_contract(
	*,
	model_id: str,
	model_revision: str,
	resolution: int,
	steps: int,
	guidance_scale: float,
	max_sequence_length: int,
	dtype: str,
	l1_roi_pad_ratio: float,
	l1_min_span: int,
	l1_max_crops: int,
	prompt_version: str,
	require_immutable_revision: bool = True,
) -> dict[str, Any]:
	"""Build and validate the exact collector/deployment Flux.2 policy."""

	model_id = str(model_id or "").strip()
	revision = str(model_revision or "").strip().lower()
	prompt_version = str(prompt_version or "").strip()
	if not model_id:
		raise ValueError("Flux.2 model_id is required")
	if require_immutable_revision and IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
		raise ValueError("Flux.2 model_revision must be an immutable 40-64 character hexadecimal snapshot")
	if not revision:
		raise ValueError("Flux.2 model_revision is required")
	resolution = int(resolution)
	steps = int(steps)
	max_sequence_length = int(max_sequence_length)
	l1_min_span = int(l1_min_span)
	l1_max_crops = int(l1_max_crops)
	guidance_scale = float(guidance_scale)
	l1_roi_pad_ratio = float(l1_roi_pad_ratio)
	if resolution <= 0 or steps <= 0 or max_sequence_length <= 0:
		raise ValueError("Flux.2 resolution, steps, and max_sequence_length must be positive")
	if not math.isfinite(guidance_scale) or guidance_scale < 0.0:
		raise ValueError("Flux.2 guidance_scale must be non-negative")
	if not math.isfinite(l1_roi_pad_ratio) or l1_roi_pad_ratio < 1.0 or l1_min_span < 2 or not 1 <= l1_max_crops <= 3:
		raise ValueError("Flux.2 L1 policy requires pad>=1, min_span>=2, and 1<=max_crops<=3")
	if not prompt_version:
		raise ValueError("Flux.2 prompt_version is required")
	contract = _make_flux2_policy_contract(
		model_id=model_id,
		model_revision=revision,
		resolution=resolution,
		steps=steps,
		guidance_scale=guidance_scale,
		max_sequence_length=max_sequence_length,
		dtype=canonical_flux_dtype(dtype),
		l1_roi_pad_ratio=l1_roi_pad_ratio,
		l1_min_span=l1_min_span,
		l1_max_crops=l1_max_crops,
		prompt_version=prompt_version,
	)
	return validate_flux2_policy_contract(contract, require_immutable_revision=require_immutable_revision)


def validate_flux2_policy_contract(
	value: Mapping[str, Any],
	*,
	require_immutable_revision: bool = True,
) -> dict[str, Any]:
	"""Canonicalize a serialized policy and reject missing/extra semantics."""

	if not isinstance(value, Mapping):
		raise ValueError("Flux.2 collector policy must be an object")
	allowed = {
		"schema_version", "action_policy_version", "executor", "generation",
		"l1_crop", "actions", "prompt_version", "infrastructure_retry_policy",
	}
	if set(value) != allowed:
		raise ValueError("Flux.2 collector policy fields are incomplete or contain unknown values")
	if value.get("schema_version") != FLUX2_POLICY_SCHEMA:
		raise ValueError("Flux.2 collector policy schema mismatch")
	if value.get("action_policy_version") != ROUTER_ACTION_POLICY_VERSION:
		raise ValueError("Flux.2 collector action-policy version mismatch")
	executor = value.get("executor")
	generation = value.get("generation")
	l1_crop = value.get("l1_crop")
	actions = value.get("actions")
	if not all(isinstance(item, Mapping) for item in (executor, generation, l1_crop, actions)):
		raise ValueError("Flux.2 collector policy components must be objects")
	if set(executor) != {"family", "implementation", "loader", "model_id", "model_revision", "dtype"}:
		raise ValueError("Flux.2 executor contract fields mismatch")
	if executor.get("family") != FLUX2_EXECUTOR_FAMILY:
		raise ValueError("Flux.2 executor family mismatch")
	if executor.get("implementation") != FLUX2_EXECUTOR_IMPLEMENTATION:
		raise ValueError("Flux.2 executor implementation mismatch")
	if executor.get("loader") != "from_pretrained(local_files_only=True)":
		raise ValueError("Flux.2 executor loader mismatch")
	if set(generation) != {"resolution", "steps", "guidance_scale", "max_sequence_length"}:
		raise ValueError("Flux.2 generation contract fields mismatch")
	if set(l1_crop) != {
		"roi_pad_ratio", "min_span", "max_crops", "crop_order", "each_crop_resized_to_generation_resolution"
	}:
		raise ValueError("Flux.2 L1 crop contract fields mismatch")
	if dict(actions) != _ACTION_SEMANTICS:
		raise ValueError("Flux.2 one-call/candidate action semantics mismatch")
	if l1_crop.get("crop_order") != "planner_order" or l1_crop.get("each_crop_resized_to_generation_resolution") is not True:
		raise ValueError("Flux.2 L1 reference geometry semantics mismatch")
	if value.get("infrastructure_retry_policy") != "exact_same_request_only; no extra semantic candidate":
		raise ValueError("Flux.2 retry semantics mismatch")
	model_id = str(executor.get("model_id") or "").strip()
	revision = str(executor.get("model_revision") or "").strip().lower()
	if not model_id or not revision:
		raise ValueError("Flux.2 executor model identity is incomplete")
	if require_immutable_revision and IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
		raise ValueError("Flux.2 executor revision is not immutable")
	try:
		resolution = int(generation.get("resolution", 0))
		steps = int(generation.get("steps", 0))
		guidance_scale = float(generation.get("guidance_scale", -1))
		max_sequence_length = int(generation.get("max_sequence_length", 0))
		roi_pad_ratio = float(l1_crop.get("roi_pad_ratio", 0))
		min_span = int(l1_crop.get("min_span", 0))
		max_crops = int(l1_crop.get("max_crops", 0))
	except (TypeError, ValueError) as exc:
		raise ValueError("Flux.2 policy numeric fields are invalid") from exc
	if resolution <= 0 or steps <= 0 or max_sequence_length <= 0 or not math.isfinite(guidance_scale) or guidance_scale < 0:
		raise ValueError("Flux.2 generation settings are invalid")
	if not math.isfinite(roi_pad_ratio) or roi_pad_ratio < 1.0 or min_span < 2 or not 1 <= max_crops <= 3:
		raise ValueError("Flux.2 L1 crop settings are invalid")
	prompt_version = str(value.get("prompt_version") or "").strip()
	if not prompt_version:
		raise ValueError("Flux.2 prompt_version is required")
	rebuilt = _make_flux2_policy_contract(
		model_id=model_id,
		model_revision=revision,
		resolution=resolution,
		steps=steps,
		guidance_scale=guidance_scale,
		max_sequence_length=max_sequence_length,
		dtype=canonical_flux_dtype(executor.get("dtype")),
		l1_roi_pad_ratio=roi_pad_ratio,
		l1_min_span=min_span,
		l1_max_crops=max_crops,
		prompt_version=prompt_version,
	)
	if dict(value) != rebuilt:
		raise ValueError("Flux.2 collector policy is not canonical")
	return rebuilt


def _snapshot_repo_id(path: Path) -> str | None:
	"""Recover ``org/repo`` from a Hugging Face cache directory when possible."""
	parts = path.parts
	for part in parts:
		if part.startswith("models--"):
			segments = part[len("models--") :].split("--")
			if len(segments) >= 2 and all(segments):
				return "/".join(segments)
	return None


def resolve_online_flux2_snapshot(
	model: str,
	*,
	model_revision: str = "",
	model_id: str = "",
	require_immutable: bool = True,
) -> tuple[Path, str, str]:
	"""Resolve the exact local snapshot and its stable model identity."""

	raw_model = str(model or "").strip()
	revision = str(model_revision or "").strip().lower()
	if not raw_model:
		raise ValueError("--flux-model is required")
	path = Path(raw_model).expanduser()
	if path.exists():
		original_name = path.name.lower()
		resolved = path.resolve()
		resolved_revision = resolved.name.lower()
		if require_immutable and not revision and IMMUTABLE_REVISION_RE.fullmatch(original_name) is None:
			raise ValueError(
				"Formal route-head deployment requires --flux-model-revision when --flux-model is a mutable alias"
			)
		revision = revision or resolved_revision
		# A commit-named Hugging Face cache entry may be a symlink to a
		# materialized directory (for example ``modelscope-local``). In that
		# case the caller-visible commit name, not the physical target basename,
		# is the immutable identity that must match the declared revision.
		visible_revision = original_name if IMMUTABLE_REVISION_RE.fullmatch(original_name) else resolved_revision
		if visible_revision != revision:
			raise ValueError("Resolved Flux.2 snapshot basename differs from --flux-model-revision")
		identity = str(model_id or "").strip() or _snapshot_repo_id(resolved) or str(resolved)
	else:
		identity = str(model_id or "").strip() or raw_model
		if require_immutable and IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
			raise ValueError(
				"Formal route-head deployment with a Flux repo id requires --flux-model-revision=<immutable commit>"
			)
		roots = [
			Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub",
		]
		cache_name = "models--" + identity.replace("/", "--")
		resolved = next(
			(candidate.resolve() for root in roots if (candidate := root / cache_name / "snapshots" / revision).is_dir()),
			None,
		)
		if resolved is None:
			raise FileNotFoundError(f"Pinned Flux.2 snapshot not found: {identity}@{revision}")
	if require_immutable and IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
		raise ValueError("Resolved Flux.2 revision is not immutable")
	return resolved, identity, revision


def build_online_flux2_policy(args: Any, *, require_immutable: bool = True) -> tuple[Path, dict[str, Any]]:
	"""Construct the serving policy from parsed CLI values and resolved bytes."""

	resolved, model_id, revision = resolve_online_flux2_snapshot(
		str(getattr(args, "flux_model", "") or ""),
		model_revision=str(getattr(args, "flux_model_revision", "") or ""),
		model_id=str(getattr(args, "flux_model_id", "") or ""),
		require_immutable=require_immutable,
	)
	contract = build_flux2_policy_contract(
		model_id=model_id,
		model_revision=revision,
		resolution=int(getattr(args, "resolution", 1024)),
		steps=int(getattr(args, "flux_steps", 28)),
		guidance_scale=float(getattr(args, "flux_guidance_scale", 3.5)),
		max_sequence_length=int(getattr(args, "flux_max_sequence_length", 512)),
		dtype=str(getattr(args, "flux_dtype", "bfloat16")),
		l1_roi_pad_ratio=float(getattr(args, "flux_l1_roi_pad_ratio", 1.35)),
		l1_min_span=int(getattr(args, "flux_l1_min_span", 64)),
		l1_max_crops=int(getattr(args, "flux_l1_max_crops", 3)),
		prompt_version=str(getattr(args, "flux_prompt_version", "vcot_v10_flux2_joint_atomic_v2")),
		require_immutable_revision=require_immutable,
	)
	return resolved, contract


def validate_matching_flux2_policy(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
	"""Fail closed unless serving exactly matches the collector's L1/L2 policy."""
	expected_contract = validate_flux2_policy_contract(expected)
	actual_contract = validate_flux2_policy_contract(actual)
	if flux2_policy_digest(expected_contract) != flux2_policy_digest(actual_contract):
		raise ValueError(
			"Online Flux.2 policy differs from the route-head collector policy: "
			f"expected={_canonical_json(expected_contract)} actual={_canonical_json(actual_contract)}"
		)


__all__ = [
	"FLUX2_EXECUTOR_FAMILY",
	"FLUX2_EXECUTOR_IMPLEMENTATION",
	"FLUX2_POLICY_SCHEMA",
	"build_flux2_policy_contract",
	"build_online_flux2_policy",
	"canonical_flux_dtype",
	"flux2_policy_digest",
	"resolve_online_flux2_snapshot",
	"validate_flux2_policy_contract",
	"validate_matching_flux2_policy",
]
