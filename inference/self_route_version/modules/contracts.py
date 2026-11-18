"""Shared V11 diagnosis and repair-plan contracts.

The online pipeline and the router-data builder intentionally import this
module instead of maintaining two prompt/serialization implementations.  The
route encoder receives canonical diagnostic facts through
:func:`model_visible_diagnosis_text`; internal schema metadata, action
coordinates, and editor instructions are not part of its model-visible input.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence


DIAGNOSIS_SCHEMA = "vcot_v11_canonical_diagnosis_v1"
ACTION_PLAN_SCHEMA = "vcot_v11_level_conditioned_plan_v1"
ROUTER_ACTION_POLICY_VERSION = "vcot_v11_qwen_self_route_initial_base_l1_l2_l3_v2"
ROUTER_REPAIR_EXECUTOR_FAMILY = "initial_base_l1_l2_l3"
DEFAULT_MAX_REPAIR_ACTIONS_V10 = 2
QWEN_DISTRIBUTION_SCHEMA = "vcot_v10_qwen_analyzer_l1_distribution_v5_retry_and_distinct_regions"
TRIAGE_PROMPT_VERSION_V10 = "vcot_v10_canonical_diagnosis_v6_method_agnostic"
L1_PLAN_PROMPT_VERSION_V10 = "vcot_v10_l1_bounded_counterfactual_v7_distinct_regions"
MODEL_VISIBLE_DIAGNOSIS_CONTRACT = "canonical_diagnosis_without_internal_schema_metadata_v1"
L1_QUAD_MIN_SPAN = 8
L1_MAX_PAIRWISE_IOU = 0.8
L1_MAX_PAIRWISE_CONTAINMENT = 0.9
L1_DISTINCT_REGION_CONTRACT = "bbox_shorthand_overlap_merge_and_whole_image_fallback_v4"
L1_CROP_PAD_RATIO = 1.35
L1_CROP_MIN_SPAN = 64
L1_MAX_CROPS = 3

# These are the online V10 defaults for both route-free diagnosis and the
# route-conditioned L1 planner.  The offline collector imports the same values
# instead of maintaining a second preset.
QWEN_ANALYZER_L1_DTYPE_V10 = "bfloat16"
QWEN_ANALYZER_L1_QUANTIZATION_V10 = "8bit"
QWEN_ANALYZER_L1_INPUT_MAX_SIDE_V10 = 1536
QWEN_ANALYZER_L1_MAX_NEW_TOKENS_V10 = 1536
QWEN_ANALYZER_IMAGE_CONTRACT_V10 = "raw_rgb_resize_long_side_then_pad32"
QWEN_L1_PLANNER_IMAGE_CONTRACT_V10 = "ruler32_rgb_resize_long_side_then_pad32"
QWEN_DECODING_CONTRACT_V10 = {
	"do_sample": False,
}
QWEN_RETRY_DECODING_CONTRACT_V10 = {
	"do_sample": False,
	"repetition_penalty": 1.05,
	"no_repeat_ngram_size": 8,
}
QWEN_REPETITION_TRIGGER_CONTRACT_V10 = "token_ngram_n12_min_tokens256_min_occurrences4_v1"
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$", flags=re.IGNORECASE)
CONSTRAINT_AUDIT_KEYS = (
	"objects",
	"count",
	"attributes",
	"relations",
	"action_state",
	"scene",
)
ISSUE_TYPES = {
	"object_identity",
	"count",
	"attribute",
	"relation",
	"action_state",
	"scene",
	"quality",
	"other",
}


def qwen_response_has_degenerate_repetition(text: Any) -> bool:
	"""Detect long exact loops without penalizing ordinary schema mistakes."""

	tokens = re.findall(r"\w+|[^\w\s]", str(text or "").lower())
	ngram_size = 12
	if len(tokens) < 256 or len(tokens) < ngram_size:
		return False
	counts = Counter(tuple(tokens[index : index + ngram_size]) for index in range(len(tokens) - ngram_size + 1))
	return bool(counts) and max(counts.values()) >= 4


def _expand_l1_axis_span(start: int, end: int, limit: int, min_span: int) -> tuple[int, int]:
	"""Clamp an inclusive pixel span, preserving its centre where possible."""

	if limit <= 1:
		return 0, 0
	lo = max(0, min(limit - 1, min(start, end)))
	hi = max(0, min(limit - 1, max(start, end)))
	target_span = min(int(min_span), limit - 1)
	need = target_span - (hi - lo)
	if need <= 0:
		return lo, hi
	lo -= need // 2
	hi += need - need // 2
	if lo < 0:
		hi = min(limit - 1, hi - lo)
		lo = 0
	if hi > limit - 1:
		lo = max(0, lo - (hi - (limit - 1)))
		hi = limit - 1
	return lo, hi


def l1_crop_roi(
	quad_points: Sequence[Sequence[int | float]],
	width: int,
	height: int,
	pad_ratio: float = L1_CROP_PAD_RATIO,
	min_span: int = L1_CROP_MIN_SPAN,
) -> List[int]:
	"""Return the shared inclusive-exclusive square ROI for one L1 crop.

	``min_span`` is the minimum final crop size. Padding is applied to the
	diagnosed box only, so a small box is not padded twice.
	"""

	width = int(width)
	height = int(height)
	min_span = int(min_span)
	pad_ratio = float(pad_ratio)
	if width <= 0 or height <= 0:
		raise ValueError(f"image dimensions must be positive, got {width}x{height}")
	if pad_ratio < 1.0:
		raise ValueError(f"L1 crop pad_ratio must be >= 1, got {pad_ratio}")
	if min_span < 1:
		raise ValueError(f"L1 crop min_span must be >= 1, got {min_span}")
	if not isinstance(quad_points, Sequence) or len(quad_points) != 4:
		raise ValueError("L1 crop quad_points must contain exactly four points")

	points: List[tuple[int, int]] = []
	for point in quad_points:
		if not isinstance(point, Sequence) or len(point) < 2:
			raise ValueError("each L1 crop point must contain x and y coordinates")
		x = max(0, min(width - 1, int(round(float(point[0])))))
		y = max(0, min(height - 1, int(round(float(point[1])))))
		points.append((x, y))

	x0, x1_inclusive = _expand_l1_axis_span(
		min(point[0] for point in points),
		max(point[0] for point in points),
		width,
		L1_QUAD_MIN_SPAN,
	)
	y0, y1_inclusive = _expand_l1_axis_span(
		min(point[1] for point in points),
		max(point[1] for point in points),
		height,
		L1_QUAD_MIN_SPAN,
	)
	x1 = x1_inclusive + 1
	y1 = y1_inclusive + 1
	box_span = max(x1 - x0, y1 - y0)
	side = int(box_span * pad_ratio)
	side = max(min_span, min(side, min(width, height)))
	side = min(side, width, height)
	center_x = (x0 + x1) // 2
	center_y = (y0 + y1) // 2
	left = max(0, center_x - side // 2)
	top = max(0, center_y - side // 2)
	right = min(width, left + side)
	bottom = min(height, top + side)
	left = max(0, right - side)
	top = max(0, bottom - side)
	return [left, top, right, bottom]


def l1_crop_rois(
	quads: Sequence[Sequence[Sequence[int | float]]],
	width: int,
	height: int,
	pad_ratio: float = L1_CROP_PAD_RATIO,
	min_span: int = L1_CROP_MIN_SPAN,
) -> List[List[int]]:
	"""Map one to three L1 boxes to crops without changing their order."""

	if not 1 <= len(quads) <= L1_MAX_CROPS:
		raise ValueError(f"L1 requires 1-{L1_MAX_CROPS} ordered crop boxes, got {len(quads)}")
	return [
		l1_crop_roi(quad, width, height, pad_ratio=pad_ratio, min_span=min_span)
		for quad in quads
	]


# The online controller and offline collector import these exact prompts.  They
# live in this lightweight module so data construction never has to import the
# GPU/runtime stack from ``config.py``.
TRIAGE_SYSTEM_PROMPT_V10 = """You are the visual state diagnosis module.

You receive an image and its target text prompt.

Your diagnostic record has three responsibilities:
1. Decide whether the current image satisfies the target prompt's main constraints.
2. Record what is observed, what was expected, and the visual evidence for each mismatch.
3. Record which already-correct facts must be preserved.

A downstream router and action planner consume this factual record after diagnosis.

Rules:
- Keep every field diagnostic: satisfaction, observed-versus-expected evidence, and preservation facts. The complete response follows exactly the schema below.
- Use the target prompt's implicit cultural, temporal, spatial, physical, chemical, and biological knowledge. Audit object identity, count, attributes, relations, action/state, setting, and visual usability.
- In temporal prompts of the form "target place when reference place is at time T", preserve the target place. The reference place is only a clock for time-zone conversion.
- Ground every description in visible evidence. If the main constraints are already satisfied, set fully_satisfied=true even if the image could be made prettier or more specific.
- issue.constraint_type must be one of object_identity, count, attribute, relation, action_state, scene, quality, or other.
- Evidence must refer to visible content. confidence is a number from 0 to 1.
- Record every visible mismatch regardless of repair difficulty. Diagnosis is independent of editor capability.

The complete response is exactly one valid JSON object:
{
  "fully_satisfied": true | false,
  "reason": "brief factual conclusion",
  "constraint_audit": {
    "objects": "object identity/presence audit",
    "count": "count audit",
    "attributes": "color/material/size/detail audit",
    "relations": "spatial and interaction audit",
    "action_state": "action/pose/state audit",
    "scene": "setting/background/style usability audit"
  },
  "issues": [{
    "constraint_type": "attribute",
    "target_entity": "the affected entity or group",
    "observed": "what is visibly present",
    "expected": "what the target prompt requires",
    "evidence": "brief visible evidence",
    "confidence": 0.95
  }],
  "preservation_constraints": ["already-correct target facts that future repair must preserve"]
}

Rules for output:
- If the image is already correct, set fully_satisfied=true and issues=[].
- If the image is incorrect, set fully_satisfied=false and include at least one factual issue.
- The JSON object above is the complete response.
"""


TRIAGE_USER_TEMPLATE_V10 = """Target prompt:
{prompt}

Original image size: width={width}, height={height}.
Analyze the current image and return the JSON object described above.

/no_think
"""


L1_PLAN_SYSTEM_PROMPT_V10 = """You are the L1 local-action planner.

You receive the current ruler-annotated full image, its target prompt, and the canonical diagnosis. Construct one feasible L1 joint-local-edit action for this state.

Rules:
- Return exactly 1 to 3 edits targeting the highest-value localized mismatches.
- L1 is deliberately bounded. It may make useful partial progress and does not need to solve global scene, composition, viewpoint, or style mismatches, or every diagnosed issue.
- When the diagnosis is largely global, select concrete foreground objects or local regions whose correction offers the greatest partial progress. Leave whole-scene reconstruction unresolved in this bounded plan.
- Each edit has one axis-aligned bbox in original-image pixel coordinates, clockwise from top-left.
- Each bbox represents a genuine bounded object or local region, is at least 8 pixels wide and high, and occupies strictly less than the full image. Include enough context for relation/count changes, while avoiding unrelated correct content.
- Every pair of edit boxes must represent spatially distinct regions. Merge strongly overlapping regions into one edit with one combined instruction.
- The instruction describes the desired final content inside that region and explicitly preserves correct surrounding content and the background outside the selected regions.
- Express one editor-agnostic joint call. All boxes become ordered crop references for that call.

The complete response is exactly one valid JSON object:
{
  "reason": "why these regions cover the diagnosed mismatch",
  "edits": [{
    "id": "edit_1",
    "priority": 1,
    "target_entity": "localized target",
    "issue": "diagnosed local mismatch",
    "instruction": "desired final content in this region while preserving the rest",
    "quad_points": [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
  }]
}

The JSON object above is the complete response.
"""


L1_PLAN_USER_TEMPLATE_V10 = """Target prompt:
{prompt}

Canonical diagnosis JSON:
{diagnosis_json}

Original image size: width={width}, height={height}.
Create the single-call L1 plan described above.

/no_think
"""


DIAGNOSIS_RETRY_INSTRUCTION_V10 = (
	"Your previous response was invalid. Return exactly one diagnosis JSON object with fully_satisfied, reason, "
	"the six-field constraint_audit, issues, and preservation_constraints. Keep every field factual and diagnostic, "
	"using exactly the schema from the system message."
)


L1_PLAN_RETRY_INSTRUCTION_V10 = (
	"Your previous response was invalid. Return exactly one L1 plan JSON object with reason and 1 to 3 edits. "
	"Every edit needs id, target_entity, issue, instruction, and quad_points containing exactly four [x,y] pairs. "
	"Use genuine bounded object regions smaller than the full image. Every pair of boxes must be spatially distinct; "
	"merge strongly overlapping repairs into one edit with one combined instruction. If the mismatch is global, select "
	"the highest-value local object repairs and leave whole-scene reconstruction unresolved."
)


def _sha256_text(value: str) -> str:
	"""Return the lowercase SHA-256 hex digest of UTF-8 text."""
	return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
	"""Hash a mapping after canonical JSON serialization."""
	payload = json.dumps(
		value,
		ensure_ascii=False,
		sort_keys=True,
		separators=(",", ":"),
		allow_nan=False,
	)
	return _sha256_text(payload)


def canonical_qwen_dtype(value: Any) -> str:
	"""Canonicalize dtype aliases before hashing a runtime contract."""

	text = str(value or QWEN_ANALYZER_L1_DTYPE_V10).strip().lower()
	aliases = {
		"bf16": "bfloat16",
		"torch.bfloat16": "bfloat16",
		"fp16": "float16",
		"half": "float16",
		"torch.float16": "float16",
		"fp32": "float32",
		"float": "float32",
		"torch.float32": "float32",
	}
	return aliases.get(text, text)


def canonical_qwen_quantization(value: Any) -> str:
	"""Canonicalize equivalent Qwen quantization spellings."""

	text = str(value or QWEN_ANALYZER_L1_QUANTIZATION_V10).strip().lower()
	aliases = {
		"int8": "8bit",
		"int4": "4bit",
		"nf4": "4bit",
	}
	return aliases.get(text, text)


def qwen_effective_quantization(value: Any, allow_cpu_offload: Any = False) -> str:
	"""Return the actual Qwen weight mode implied by the runtime policy.

	``QwenPlanner`` intentionally loads every quantized CPU-offload request as
	int8, because bitsandbytes NF4 modules cannot be safely dispatched to CPU.
	Keeping this derivation in the shared contract prevents a requested
	``4bit`` run from being incorrectly signed as an NF4 execution.
	"""

	if not isinstance(allow_cpu_offload, bool):
		raise ValueError("Qwen allow_cpu_offload must be a boolean")
	quantization = canonical_qwen_quantization(value)
	if quantization not in {"none", "8bit", "4bit"}:
		raise ValueError(f"unsupported Qwen quantization: {quantization!r}")
	if allow_cpu_offload and quantization in {"8bit", "4bit"}:
		return "int8_cpu_offload"
	return {"none": "none", "8bit": "int8", "4bit": "nf4"}[quantization]


def build_qwen_analyzer_l1_distribution_contract(
	*,
	model: Any,
	model_revision: Any,
	dtype: Any = QWEN_ANALYZER_L1_DTYPE_V10,
	quantization: Any = QWEN_ANALYZER_L1_QUANTIZATION_V10,
	allow_cpu_offload: Any = False,
	image_shrink_retry: Any = False,
	input_max_side: Any = QWEN_ANALYZER_L1_INPUT_MAX_SIDE_V10,
	max_new_tokens: Any = QWEN_ANALYZER_L1_MAX_NEW_TOKENS_V10,
	require_immutable_revision: bool = False,
) -> Dict[str, Any]:
	"""Build the exact Qwen distribution shared by online and offline V10.

	The signature deliberately covers model identity, runtime precision, image
	preprocessing, token budget, schemas, prompt versions and prompt bytes.  An
	offline result can therefore be rejected if it was produced under a stale
	or deployment-incompatible analyzer/L1-planner distribution.
	"""

	model_text = str(model or "").strip()
	revision_text = str(model_revision or "").strip().lower()
	dtype_text = canonical_qwen_dtype(dtype)
	quantization_text = canonical_qwen_quantization(quantization)
	if not isinstance(allow_cpu_offload, bool):
		raise ValueError("Qwen allow_cpu_offload must be a boolean")
	if not isinstance(image_shrink_retry, bool):
		raise ValueError("Qwen image_shrink_retry must be a boolean")
	effective_quantization = qwen_effective_quantization(
		quantization_text,
		allow_cpu_offload,
	)
	try:
		input_side = int(input_max_side)
		token_budget = int(max_new_tokens)
	except (TypeError, ValueError) as exc:
		raise ValueError("Qwen input_max_side and max_new_tokens must be integers") from exc
	if not model_text:
		raise ValueError("Qwen model is required")
	if not revision_text:
		raise ValueError("Qwen model_revision is required")
	if require_immutable_revision and _IMMUTABLE_REVISION_RE.fullmatch(revision_text) is None:
		raise ValueError("Qwen model_revision must be an immutable 40-64 hex commit")
	if dtype_text not in {"bfloat16", "float16", "float32"}:
		raise ValueError(f"unsupported Qwen dtype: {dtype_text!r}")
	if quantization_text not in {"none", "8bit", "4bit"}:
		raise ValueError(f"unsupported Qwen quantization: {quantization_text!r}")
	if input_side <= 0 or token_budget <= 0:
		raise ValueError("Qwen input_max_side and max_new_tokens must be positive")
	contract: Dict[str, Any] = {
		"schema_version": QWEN_DISTRIBUTION_SCHEMA,
		"diagnosis_schema": DIAGNOSIS_SCHEMA,
		"model_visible_diagnosis_contract": MODEL_VISIBLE_DIAGNOSIS_CONTRACT,
		"analyzer_prompt_version": TRIAGE_PROMPT_VERSION_V10,
		"l1_planner_prompt_version": L1_PLAN_PROMPT_VERSION_V10,
		"analyzer_system_prompt_sha256": _sha256_text(TRIAGE_SYSTEM_PROMPT_V10),
		"analyzer_user_template_sha256": _sha256_text(TRIAGE_USER_TEMPLATE_V10),
		"l1_planner_system_prompt_sha256": _sha256_text(L1_PLAN_SYSTEM_PROMPT_V10),
		"l1_planner_user_template_sha256": _sha256_text(L1_PLAN_USER_TEMPLATE_V10),
		"diagnosis_retry_instruction_sha256": _sha256_text(DIAGNOSIS_RETRY_INSTRUCTION_V10),
		"l1_planner_retry_instruction_sha256": _sha256_text(L1_PLAN_RETRY_INSTRUCTION_V10),
		"l1_distinct_region_contract": L1_DISTINCT_REGION_CONTRACT,
		"model": model_text,
		"model_revision": revision_text,
		"dtype": dtype_text,
		"quantization": quantization_text,
		"allow_cpu_offload": allow_cpu_offload,
		"effective_quantization": effective_quantization,
		"image_shrink_retry": image_shrink_retry,
		"input_max_side": input_side,
		"max_new_tokens": token_budget,
		"decoding": dict(QWEN_DECODING_CONTRACT_V10),
		"retry_decoding": dict(QWEN_RETRY_DECODING_CONTRACT_V10),
		"retry_decoding_trigger": QWEN_REPETITION_TRIGGER_CONTRACT_V10,
		"analyzer_image_contract": QWEN_ANALYZER_IMAGE_CONTRACT_V10,
		"l1_planner_image_contract": QWEN_L1_PLANNER_IMAGE_CONTRACT_V10,
	}
	contract["distribution_signature"] = _sha256_json(contract)
	return contract


def validate_qwen_analyzer_l1_distribution_contract(
	value: Mapping[str, Any],
	*,
	require_immutable_revision: bool = False,
) -> Dict[str, Any]:
	"""Return a canonical contract and reject any stale/tampered field."""

	if not isinstance(value, Mapping):
		raise ValueError("Qwen analyzer/L1 distribution contract must be an object")
	canonical = build_qwen_analyzer_l1_distribution_contract(
		model=value.get("model"),
		model_revision=value.get("model_revision"),
		dtype=value.get("dtype"),
		quantization=value.get("quantization"),
		allow_cpu_offload=value.get("allow_cpu_offload"),
		image_shrink_retry=value.get("image_shrink_retry"),
		input_max_side=value.get("input_max_side"),
		max_new_tokens=value.get("max_new_tokens"),
		require_immutable_revision=require_immutable_revision,
	)
	if dict(value) != canonical:
		raise ValueError("Qwen analyzer/L1 distribution contract is stale or was modified")
	return canonical


def validate_loaded_qwen_quantization_runtime(planner: Any, contract: Mapping[str, Any]) -> None:
	"""Reject a loaded planner whose actual bnb mode differs from its lock."""

	canonical = validate_qwen_analyzer_l1_distribution_contract(contract)
	requested = canonical_qwen_quantization(getattr(planner, "requested_quantization", None))
	allow_cpu_offload = getattr(planner, "allow_cpu_offload", None)
	effective = str(getattr(planner, "effective_quantization", "") or "").strip().lower()
	if requested != canonical["quantization"]:
		raise ValueError("loaded Qwen requested quantization differs from its distribution lock")
	if allow_cpu_offload is not canonical["allow_cpu_offload"]:
		raise ValueError("loaded Qwen CPU-offload policy differs from its distribution lock")
	if effective != canonical["effective_quantization"]:
		raise ValueError(
			"loaded Qwen effective quantization differs from its distribution lock; "
			"configure the effective mode explicitly instead of accepting a runtime fallback"
		)


def _text(value: Any, default: str = "") -> str:
	"""Normalize a scalar-like value to stripped text."""
	if value is None:
		return default
	return str(value).strip()


def _truth(value: Any) -> bool:
	"""Interpret canonical boolean and common textual truth values."""
	return value is True or _text(value).lower() == "true"


def _string_list(value: Any, *, limit: int = 16) -> List[str]:
	"""Normalize, trim, and bound a model-provided string list."""
	if not isinstance(value, (list, tuple)):
		return []
	result: List[str] = []
	for item in value:
		text = _text(item)
		if text and text not in result:
			result.append(text)
		if len(result) >= limit:
			break
	return result


def canonicalize_diagnosis(payload: Any) -> Dict[str, Any]:
	"""Normalize current and legacy reflections into the router-visible form.

	The result deliberately excludes route labels, editor names, bboxes and
	executable repair instructions.  This prevents action-plan availability
	from leaking the supervision label into the MLP input.
	"""
	obj: Mapping[str, Any] = payload if isinstance(payload, Mapping) else {}
	fully_satisfied = _truth(obj.get("fully_satisfied", False))

	raw_audit = obj.get("constraint_audit")
	audit_obj: Mapping[str, Any] = raw_audit if isinstance(raw_audit, Mapping) else {}
	constraint_audit = {
		key: _text(audit_obj.get(key), "not assessed") or "not assessed"
		for key in CONSTRAINT_AUDIT_KEYS
	}
	# Accept the pre-V10 free-form audit without making it an executable hint.
	if isinstance(raw_audit, str) and raw_audit.strip():
		constraint_audit["objects"] = raw_audit.strip()

	issues: List[Dict[str, Any]] = []
	raw_issues = obj.get("issues")
	if isinstance(raw_issues, list):
		for raw_issue in raw_issues[:12]:
			if not isinstance(raw_issue, Mapping):
				continue
			constraint_type = _text(raw_issue.get("constraint_type"), "other").lower()
			if constraint_type not in ISSUE_TYPES:
				constraint_type = "other"
			try:
				confidence = float(raw_issue.get("confidence", 1.0))
			except (TypeError, ValueError):
				confidence = 1.0
			issues.append(
				{
					"constraint_type": constraint_type,
					"target_entity": _text(raw_issue.get("target_entity"), "unspecified") or "unspecified",
					"observed": _text(raw_issue.get("observed"), "unknown") or "unknown",
					"expected": _text(raw_issue.get("expected"), "unknown") or "unknown",
					"evidence": _text(raw_issue.get("evidence"), "not stated") or "not stated",
					"confidence": round(max(0.0, min(1.0, confidence)), 4),
				}
			)

	# Backward-compatible ingestion is useful when resuming V9 artifacts, but
	# serialization after this point is always the V10 schema.
	if not issues and not fully_satisfied:
		legacy_issue = _text(obj.get("issue"))
		legacy_fix = _text(obj.get("fix") or obj.get("global_edit_instruction"))
		if legacy_issue or legacy_fix:
			issues.append(
				{
					"constraint_type": "other",
					"target_entity": "unspecified",
					"observed": legacy_issue or "unknown",
					"expected": legacy_fix or "unknown",
					"evidence": _text(obj.get("reason"), "not stated") or "not stated",
					"confidence": 1.0,
				}
			)
	if fully_satisfied:
		issues = []

	return {
		"schema_version": DIAGNOSIS_SCHEMA,
		"fully_satisfied": fully_satisfied,
		"reason": _text(obj.get("reason"), "No reason supplied") or "No reason supplied",
		"constraint_audit": constraint_audit,
		"issues": issues,
		"preservation_constraints": _string_list(obj.get("preservation_constraints")),
	}


def diagnosis_payload_is_valid(payload: Any) -> bool:
	"""Return whether a native model response satisfies the V10 diagnosis API."""
	if not isinstance(payload, Mapping):
		return False
	allowed_keys = {
		"schema_version",
		"fully_satisfied",
		"reason",
		"constraint_audit",
		"issues",
		"preservation_constraints",
	}
	if set(payload) - allowed_keys:
		return False
	if "schema_version" in payload and payload.get("schema_version") != DIAGNOSIS_SCHEMA:
		return False
	if "fully_satisfied" not in payload or not isinstance(payload.get("fully_satisfied"), bool):
		return False
	if not isinstance(payload.get("reason"), str) or not payload.get("reason", "").strip():
		return False
	audit = payload.get("constraint_audit")
	if not isinstance(audit, Mapping):
		return False
	if any(not isinstance(audit.get(key), str) or not audit.get(key, "").strip() for key in CONSTRAINT_AUDIT_KEYS):
		return False
	issues = payload.get("issues")
	if not isinstance(issues, list):
		return False
	if payload["fully_satisfied"] and issues:
		return False
	if not payload["fully_satisfied"] and not issues:
		return False
	for issue in issues:
		if not isinstance(issue, Mapping):
			return False
		if _text(issue.get("constraint_type")).lower() not in ISSUE_TYPES:
			return False
		for key in ("target_entity", "observed", "expected", "evidence"):
			if not isinstance(issue.get(key), str) or not issue.get(key, "").strip():
				return False
		try:
			confidence = float(issue.get("confidence"))
		except (TypeError, ValueError):
			return False
		if not 0.0 <= confidence <= 1.0:
			return False
	return isinstance(payload.get("preservation_constraints"), list) and all(
		isinstance(item, str) for item in payload.get("preservation_constraints", [])
	)


def extract_diagnosis_payload(raw: str) -> Optional[Dict[str, Any]]:
	"""Extract the first valid JSON object without importing pipeline modules."""
	decoder = json.JSONDecoder()
	text = str(raw or "")
	for position, character in enumerate(text):
		if character != "{":
			continue
		try:
			obj, _ = decoder.raw_decode(text[position:])
		except json.JSONDecodeError:
			continue
		if isinstance(obj, dict):
			return obj
	return None


def canonical_diagnosis_text_from_raw(raw: str) -> Optional[str]:
	"""Extract and serialize a valid canonical diagnosis from raw Qwen text."""
	payload = extract_diagnosis_payload(raw)
	if not diagnosis_payload_is_valid(payload):
		return None
	return canonical_diagnosis_text(payload)


def canonical_diagnosis_text(payload: Any) -> str:
	"""Stable, whitespace-independent serialization used by every router."""
	return json.dumps(
		canonicalize_diagnosis(payload),
		ensure_ascii=False,
		sort_keys=True,
		separators=(",", ":"),
	)


def model_visible_diagnosis(payload: Any) -> Dict[str, Any]:
	"""Return canonical diagnostic facts without repository contract metadata."""
	canonical = canonicalize_diagnosis(payload)
	canonical.pop("schema_version", None)
	return canonical


def model_visible_diagnosis_text(payload: Any, *, indent: Optional[int] = None) -> str:
	"""Serialize the diagnosis exactly as it may be shown to a language model."""
	kwargs: Dict[str, Any] = {
		"ensure_ascii": False,
		"sort_keys": True,
		"indent": indent,
	}
	if indent is None:
		kwargs["separators"] = (",", ":")
	return json.dumps(model_visible_diagnosis(payload), **kwargs)


def diagnosis_issue_summary(diagnosis: Any) -> str:
	"""Produce a concise factual issue summary for logs and action prompts.

	This text is derived only from canonical diagnosis fields.  It is not a
	route recommendation and must not be used as an executable editor command.
	"""
	canonical = canonicalize_diagnosis(diagnosis)
	parts: List[str] = []
	for issue in canonical["issues"]:
		parts.append(
			f"{issue['target_entity']}: observed {issue['observed']}; expected {issue['expected']}"
		)
	return " | ".join(parts) or canonical["reason"]


def build_l1_joint_prompt(
	target_prompt: str,
	diagnosis: Any,
	edits: Sequence[Mapping[str, Any]],
	width: int,
	height: int,
) -> str:
	"""Bind one editable canvas to 1--3 crop references for one executor call."""
	lines = [
		"Edit Image 1, which is the only editable full-resolution canvas.",
		"Images 2 onward are zoomed localization references only; do not paste, tile, duplicate, or copy their borders into Image 1.",
		f"Target prompt: {target_prompt.strip()}",
		f"Diagnosed mismatch: {diagnosis_issue_summary(diagnosis)}",
	]
	for index, edit in enumerate(edits[:3], start=1):
		quad = edit.get("quad_points") or []
		box = edit.get("box_xyxy") or []
		region = box if isinstance(box, list) and len(box) == 4 else quad
		lines.append(
			f"Edit {index} (Image {index + 1}, full-canvas region {region} within {width}x{height}): "
			f"{_text(edit.get('instruction') or edit.get('edit_prompt') or edit.get('whole_edit'), 'correct the diagnosed mismatch')}."
		)
	lines.extend(
		[
			"Apply all listed edits together in this single result.",
			"Preserve every correct object, identity, count, attribute, relation, pose, composition, lighting, style, and background outside the union of the listed regions.",
			"Return one coherent full image with no annotations, rulers, boxes, crop seams, text, or watermarks.",
		]
	)
	return "\n".join(lines)


def build_level_action_plan(
	level: int,
	target_prompt: str,
	diagnosis: Any,
	*,
	edits: Sequence[Mapping[str, Any]] | None = None,
	width: int = 1024,
	height: int = 1024,
	initial_generator_family: str | None = None,
) -> Dict[str, Any]:
	"""Create the common post-route action record for online/data paths."""
	# Keep the import local: l3_policy itself imports the action-policy version
	# from this module, while contracts must remain importable without a cycle.
	from .l3_policy import normalize_initial_generator_family

	canonical = canonicalize_diagnosis(diagnosis)
	initial_family = (
		normalize_initial_generator_family(initial_generator_family)
		if initial_generator_family is not None
		else None
	)
	executor_family = initial_family or ("flux2" if int(level) in {1, 2} else None)
	if executor_family is None:
		raise ValueError(
			f"L{int(level)} requires initial_generator_family so execution cannot silently use the wrong base"
		)
	base: Dict[str, Any] = {
		"schema_version": ACTION_PLAN_SCHEMA,
		"action_policy_version": ROUTER_ACTION_POLICY_VERSION,
		"executor_family": executor_family,
		"initial_generator_family": initial_family,
		"l3_executor_binding": "same_as_initial_generator" if int(level) == 3 else None,
		"level": int(level),
		"target_prompt": target_prompt.strip(),
		"diagnosis": canonical,
	}
	if level == 1:
		normalized_edits = [dict(item) for item in (edits or [])[:3]]
		if not normalized_edits:
			raise ValueError("L1 requires one to three localized edits")
		base.update(
			{
				"action": "joint_local_edit",
				"edits": normalized_edits,
				"instruction": build_l1_joint_prompt(
					target_prompt,
					canonical,
					normalized_edits,
					width,
					height,
				),
				"candidate_count": 1,
				"editor_call_count": 1,
			}
		)
	elif level == 2:
		base.update(
			{
				"action": "global_image_edit",
				"edits": [],
				"instruction": (
					f"Edit the provided full image so it satisfies this target prompt: {target_prompt.strip()}\n"
					f"Correct these diagnosed mismatches: {diagnosis_issue_summary(canonical)}. "
					"Preserve all already-correct content, identities, counts, attributes, relations, composition, and style. "
					"Produce one clean full image without annotations or text."
				),
				"candidate_count": 1,
				"editor_call_count": 1,
			}
		)
	elif level == 3:
		base.update(
			{
				"action": "prompt_only_regeneration",
				"edits": [],
				"instruction": (
					f"Generate a new image from scratch for this target prompt: {target_prompt.strip()}\n"
					f"Resolve these diagnosed mismatches: {diagnosis_issue_summary(canonical)}. "
					"Construct the scene afresh from the complete target constraints and satisfy every requested object, count, attribute, relation, action, and scene constraint."
				),
				"candidate_count": 1,
				"editor_call_count": 1,
			}
		)
	else:
		raise ValueError(f"repair action plans only support L1/L2/L3, got {level}")
	return base


def validate_distribution_contract(plan: Mapping[str, Any]) -> None:
	"""Fast static invariant shared by collectors and online resume checks."""
	level = int(plan.get("level", -1))
	if plan.get("schema_version") != ACTION_PLAN_SCHEMA:
		raise ValueError("unexpected V11 action-plan schema")
	if plan.get("action_policy_version") != ROUTER_ACTION_POLICY_VERSION:
		raise ValueError("unexpected V11 router action policy")
	if level in {1, 2, 3}:
		from .l3_policy import normalize_initial_generator_family

		try:
			executor_family = normalize_initial_generator_family(plan.get("executor_family"))
			initial_family_value = plan.get("initial_generator_family")
			initial_family = (
				normalize_initial_generator_family(initial_family_value)
				if initial_family_value is not None
				else ("flux2" if level in {1, 2} else None)
			)
		except ValueError as exc:
			raise ValueError(f"invalid self-route initial-base binding: {exc}") from exc
		if executor_family != initial_family:
			raise ValueError("Self-route L1/L2/L3 executor must exactly follow the initial generator family")
		if level == 3 and plan.get("l3_executor_binding") != "same_as_initial_generator":
			raise ValueError("Self-route L3 executor must declare the initial-generator binding")
	if int(plan.get("candidate_count", 0)) != 1 or int(plan.get("editor_call_count", 0)) != 1:
		raise ValueError("V10 requires exactly one generated candidate and one editor call per routed action")
	if level == 1 and not (1 <= len(plan.get("edits") or []) <= 3):
		raise ValueError("V10 L1 requires one to three edits in one joint call")
	if level == 1:
		boxes: List[List[float]] = []
		for edit in plan.get("edits") or []:
			box = edit.get("box_xyxy") if isinstance(edit, Mapping) else None
			if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
				quad = edit.get("quad_points") if isinstance(edit, Mapping) else None
				if not isinstance(quad, Sequence) or len(quad) != 4:
					raise ValueError("V10 L1 edit lacks a canonical local region")
				xs = [float(point[0]) for point in quad]
				ys = [float(point[1]) for point in quad]
				box = [min(xs), min(ys), max(xs), max(ys)]
			boxes.append([float(value) for value in box])
		for left_index, left in enumerate(boxes):
			left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
			for right in boxes[left_index + 1 :]:
				right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
				intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
					0.0, min(left[3], right[3]) - max(left[1], right[1])
				)
				union = left_area + right_area - intersection
				iou = intersection / union if union > 0 else 0.0
				containment = intersection / min(left_area, right_area) if min(left_area, right_area) > 0 else 0.0
				if iou >= L1_MAX_PAIRWISE_IOU or containment >= L1_MAX_PAIRWISE_CONTAINMENT:
					raise ValueError("V10 L1 edit regions must be spatially distinct")
	if level in {2, 3} and plan.get("edits"):
		raise ValueError("V10 L2/L3 plans must not contain local edit boxes")


__all__ = [
	"ACTION_PLAN_SCHEMA",
	"CONSTRAINT_AUDIT_KEYS",
	"DEFAULT_MAX_REPAIR_ACTIONS_V10",
	"DIAGNOSIS_SCHEMA",
	"ISSUE_TYPES",
	"DIAGNOSIS_RETRY_INSTRUCTION_V10",
	"L1_DISTINCT_REGION_CONTRACT",
	"L1_MAX_PAIRWISE_CONTAINMENT",
	"L1_MAX_PAIRWISE_IOU",
	"L1_PLAN_RETRY_INSTRUCTION_V10",
	"L1_PLAN_PROMPT_VERSION_V10",
	"L1_PLAN_SYSTEM_PROMPT_V10",
	"L1_PLAN_USER_TEMPLATE_V10",
	"QWEN_ANALYZER_IMAGE_CONTRACT_V10",
	"QWEN_ANALYZER_L1_DTYPE_V10",
	"QWEN_ANALYZER_L1_INPUT_MAX_SIDE_V10",
	"QWEN_ANALYZER_L1_MAX_NEW_TOKENS_V10",
	"QWEN_ANALYZER_L1_QUANTIZATION_V10",
	"QWEN_DECODING_CONTRACT_V10",
	"QWEN_RETRY_DECODING_CONTRACT_V10",
	"QWEN_REPETITION_TRIGGER_CONTRACT_V10",
	"QWEN_DISTRIBUTION_SCHEMA",
	"QWEN_L1_PLANNER_IMAGE_CONTRACT_V10",
	"MODEL_VISIBLE_DIAGNOSIS_CONTRACT",
	"ROUTER_ACTION_POLICY_VERSION",
	"ROUTER_REPAIR_EXECUTOR_FAMILY",
	"TRIAGE_PROMPT_VERSION_V10",
	"TRIAGE_SYSTEM_PROMPT_V10",
	"TRIAGE_USER_TEMPLATE_V10",
	"build_qwen_analyzer_l1_distribution_contract",
	"build_l1_joint_prompt",
	"build_level_action_plan",
	"canonical_qwen_dtype",
	"canonical_qwen_quantization",
	"canonical_diagnosis_text",
	"model_visible_diagnosis",
	"model_visible_diagnosis_text",
	"canonicalize_diagnosis",
	"diagnosis_issue_summary",
	"diagnosis_payload_is_valid",
	"extract_diagnosis_payload",
	"qwen_effective_quantization",
	"qwen_response_has_degenerate_repetition",
	"canonical_diagnosis_text_from_raw",
	"validate_distribution_contract",
	"validate_loaded_qwen_quantization_runtime",
	"validate_qwen_analyzer_l1_distribution_contract",
]
