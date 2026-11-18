"""Pure-data contract shared by blind evaluation and aggregation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from inference.main_method.modules.contracts import canonicalize_diagnosis, diagnosis_issue_summary

from .canonical import sha256_text, stable_id
from .schema import DataValidationError, immutable_revision, normalize_text, strict_bool


PROMPT_VERSION = "vcot_v10_source_blind_pair_evaluator_v3"
BLIND_PROTOCOL = "paired_randomized_source_blind_v3"
SUCCESS_DEFINITION = "candidate>=0.80,fix>=0.70,artifact<=0.30,no_new_error"


def evaluation_prompt(state: Mapping[str, Any], order: Sequence[str]) -> str:
	"""Build the source-blind A/B evaluation prompt for one original/candidate pair."""
	if list(sorted(order)) != ["candidate", "original"] or len(order) != 2:
		raise ValueError("evaluation order must contain original and candidate exactly once")
	diagnosis = state["diagnosis"]
	canonical = canonicalize_diagnosis(diagnosis)
	preservation = "; ".join(canonical.get("preservation_constraints") or []) or "all already-correct target facts"
	return f"""Compare Images A and B against the same target, applying exactly the same criteria to both images.

Target description: {state['target_prompt']}
Diagnosed mismatches to evaluate: {diagnosis_issue_summary(canonical)}
Facts each image should preserve: {preservation}

Judge only visible evidence. Score each image independently. The complete response is exactly one valid JSON object:
{{
  "image_a": {{
    "target_satisfaction": 0.0,
    "diagnosed_issue_resolution": 0.0,
    "preservation": 0.0,
    "artifact_severity": 0.0,
    "preservation_violation": false
  }},
  "image_b": {{
    "target_satisfaction": 0.0,
    "diagnosed_issue_resolution": 0.0,
    "preservation": 0.0,
    "artifact_severity": 0.0,
    "preservation_violation": false
  }},
  "rationale": "one concise A/B comparison based on visible evidence"
}}

For both image_a and image_b:
- target_satisfaction is a number from 0 to 1 measuring all target constraints.
- diagnosed_issue_resolution is a number from 0 to 1 measuring resolution of the listed mismatches.
- preservation is a number from 0 to 1 measuring the listed preservation facts, identities, layout, and style.
- artifact_severity is a number from 0 to 1, where 0 means no visible artifact.
- preservation_violation is true exactly when a listed preservation fact is visibly violated.
- The JSON object above is the complete response."""


def prompt_hash(state: Mapping[str, Any], order: Sequence[str]) -> str:
	# The role order is hidden from Qwen but remains part of the immutable
	# request identity so an A/B presentation cannot be replayed as its reverse.
	"""Hash the exact evaluator prompt bytes associated with a job."""
	presentation = "|".join(order)
	return sha256_text(
		PROMPT_VERSION + "\n" + BLIND_PROTOCOL + "\n" + presentation + "\n" + evaluation_prompt(state, order)
	)


def evaluator_contract(
	*,
	model: str,
	model_revision: str,
	dtype: str,
	quantization: str,
	max_new_tokens: int,
	max_image_side: int,
	allow_cpu_offload: bool,
) -> dict[str, Any]:
	"""Fingerprint evaluator prompts, image ordering, decoding, and model identity."""
	model = normalize_text(model)
	if not model:
		raise DataValidationError("evaluator model is required")
	model_revision = immutable_revision(model_revision, "evaluator model_revision")
	dtype = normalize_text(dtype).lower()
	quantization = normalize_text(quantization).lower()
	if dtype not in {"bfloat16", "float16", "float32"}:
		raise DataValidationError("invalid evaluator dtype")
	if quantization not in {"none", "8bit", "4bit", "nf4"}:
		raise DataValidationError("invalid evaluator quantization")
	if int(max_new_tokens) <= 0 or int(max_image_side) <= 0:
		raise DataValidationError("evaluator limits must be positive")
	value = {
		"prompt_version": PROMPT_VERSION,
		"blind_protocol": BLIND_PROTOCOL,
		"model": model,
		"model_revision": model_revision,
		"dtype": dtype,
		"quantization": quantization,
		"max_new_tokens": int(max_new_tokens),
		"max_image_side": int(max_image_side),
		"allow_cpu_offload": strict_bool(allow_cpu_offload, "evaluator allow_cpu_offload"),
		"decoding": "greedy_do_sample_false",
		"success_definition": SUCCESS_DEFINITION,
	}
	value["evaluator_signature"] = stable_id("evaluator", value, length=32)
	return value


def validate_evaluator_contract(value: Mapping[str, Any]) -> dict[str, Any]:
	"""Validate evaluator contract and reject contract violations."""
	provided = dict(value)
	signature = provided.get("evaluator_signature")
	unsigned = dict(provided)
	unsigned.pop("evaluator_signature", None)
	if signature != stable_id("evaluator", unsigned, length=32):
		raise ValueError("evaluator contract signature mismatch")
	if unsigned.get("prompt_version") != PROMPT_VERSION or unsigned.get("blind_protocol") != BLIND_PROTOCOL:
		raise ValueError("unsupported evaluator prompt/blind protocol")
	contract = evaluator_contract(
		model=unsigned.get("model"),
		model_revision=unsigned.get("model_revision"),
		dtype=unsigned.get("dtype"),
		quantization=unsigned.get("quantization"),
		max_new_tokens=unsigned.get("max_new_tokens"),
		max_image_side=unsigned.get("max_image_side"),
		allow_cpu_offload=unsigned.get("allow_cpu_offload"),
	)
	if contract != provided:
		raise ValueError("evaluator contract fields are non-canonical or unsupported")
	return contract
