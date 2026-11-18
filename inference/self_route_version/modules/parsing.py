"""JSON extraction and Qwen response normalizers."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import (
	DIFFICULTY_RANK,
	MIN_QUAD_SPAN,
)
from .contracts import (
	L1_MAX_PAIRWISE_CONTAINMENT,
	L1_MAX_PAIRWISE_IOU,
	build_level_action_plan,
	canonical_diagnosis_text,
	canonicalize_diagnosis,
	diagnosis_issue_summary,
	diagnosis_payload_is_valid,
	validate_distribution_contract,
)

def normalize_json_keys(value: Any) -> Any:
	"""Recursively stringify and trim object keys after JSON decoding."""
	if isinstance(value, dict):
		return {str(key).strip(): normalize_json_keys(item) for key, item in value.items()}
	if isinstance(value, list):
		return [normalize_json_keys(item) for item in value]
	return value


def extract_json(text: str) -> Optional[Dict[str, Any]]:
	"""Extract the first decodable JSON object from raw model output.

	The search considers the complete response, fenced blocks, and broad object
	or array spans.  If an array is returned, its first object is used because
	all V10 language stages require one top-level object.
	"""
	text = text.strip()
	candidates: List[str] = []
	if text:
		candidates.append(text)
	for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
		candidates.append(match.group(1).strip())
	match = re.search(r"\{[\s\S]*\}", text)
	if match:
		candidates.append(match.group(0))
	match = re.search(r"\[[\s\S]*\]", text)
	if match:
		candidates.append(match.group(0))

	seen: set[str] = set()
	for candidate in candidates:
		candidate = candidate.strip()
		if not candidate or candidate in seen:
			continue
		seen.add(candidate)
		try:
			obj = json.loads(candidate)
		except Exception:
			continue
		if isinstance(obj, list):
			obj = next((item for item in obj if isinstance(item, dict)), None)
		if isinstance(obj, dict):
			return normalize_json_keys(obj)
	return None


def normalize_prompt_reasoning(obj: Optional[Dict[str, Any]], raw: str, original_prompt: str) -> Dict[str, Any]:
	"""Normalize optional pre-generation reasoning and preserve a safe prompt.

	Invalid output falls back to the original prompt rather than blocking image
	generation.  ``raw_output`` and ``valid_json`` keep that fallback auditable.
	"""
	payload = obj if isinstance(obj, dict) else {}
	raw_generation_prompt = str(payload.get("generation_prompt") or "").strip()
	generation_prompt = raw_generation_prompt or original_prompt.strip()

	def normalize_string_list(value: Any) -> List[str]:
		"""Keep only non-empty textual entries in a model-provided list."""
		if not isinstance(value, list):
			return []
		return [str(item).strip() for item in value if str(item).strip()]

	visual_requirements = payload.get("visual_requirements")
	return {
		"reasoning": str(payload.get("reasoning") or "").strip(),
		"implicit_logic": normalize_string_list(payload.get("implicit_logic")),
		"task_decomposition": normalize_string_list(payload.get("task_decomposition")),
		"scene_construction_plan": str(payload.get("scene_construction_plan") or "").strip(),
		"visual_requirements": normalize_string_list(visual_requirements),
		"ambiguity_resolution": str(payload.get("ambiguity_resolution") or "").strip(),
		"generation_prompt": generation_prompt,
		"raw_output": raw,
		"valid_json": isinstance(obj, dict) and bool(raw_generation_prompt),
	}


def clamp_quad(value: Any, width: int, height: int) -> Optional[List[List[int]]]:
	"""Validate a four-point region and return its canonical axis-aligned quad."""
	if not isinstance(value, Iterable):
		return None
	raw_points = list(value)
	if len(raw_points) != 4:
		return None
	quad: List[List[int]] = []
	try:
		for item in raw_points:
			coords = list(item)
			if len(coords) < 2:
				return None
			x = max(0, min(width - 1, int(round(float(coords[0])))))
			y = max(0, min(height - 1, int(round(float(coords[1])))))
			quad.append([x, y])
	except Exception:
		return None
	xs = [point[0] for point in quad]
	ys = [point[1] for point in quad]
	x0, x1 = expand_axis_span(min(xs), max(xs), width, MIN_QUAD_SPAN)
	y0, y1 = expand_axis_span(min(ys), max(ys), height, MIN_QUAD_SPAN)
	ordered = [
		[x0, y0],
		[x1, y0],
		[x1, y1],
		[x0, y1],
	]
	return ordered


def expand_axis_span(start: int, end: int, limit: int, min_span: int) -> Tuple[int, int]:
	"""Expand a clamped inclusive span around its center to a minimum size."""
	if limit <= 1:
		return 0, 0
	lo = max(0, min(limit - 1, min(start, end)))
	hi = max(0, min(limit - 1, max(start, end)))
	target_span = min(min_span, limit - 1)
	current_span = hi - lo
	if current_span >= target_span:
		return lo, hi
	need = target_span - current_span
	pad_lo = need // 2
	pad_hi = need - pad_lo
	lo -= pad_lo
	hi += pad_hi
	if lo < 0:
		hi = min(limit - 1, hi - lo)
		lo = 0
	if hi > limit - 1:
		lo = max(0, lo - (hi - (limit - 1)))
		hi = limit - 1
	return lo, hi


def quad_to_box(quad: Sequence[Sequence[int]]) -> List[int]:
	"""Convert inclusive quad corners to a half-open ``[x0,y0,x1,y1]`` box."""
	xs = [int(point[0]) for point in quad]
	ys = [int(point[1]) for point in quad]
	return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


def normalize_triage_analysis(obj: Optional[Dict[str, Any]], raw: str, width: int, height: int) -> Dict[str, Any]:
	"""Normalize canonical diagnosis facts without assigning L1/L2/L3.

	Native Qwen may declare L0 satisfaction.  Every non-satisfied valid
	diagnosis remains ``Unrouted`` until the frozen MLP (or a fixed-level
	ablation) selects a repair level.
	"""
	valid_json = diagnosis_payload_is_valid(obj)
	canonical = canonicalize_diagnosis(obj)
	fully_satisfied = bool(canonical["fully_satisfied"]) if valid_json else False
	summary = diagnosis_issue_summary(canonical) if valid_json else "Failed to parse canonical V10 diagnosis JSON."
	return {
		# category/recommended_mode are state-machine outputs, never encoder input.
		"category": "L0" if fully_satisfied else "Unrouted",
		"recommended_mode": "done" if fully_satisfied else "unrouted",
		"fully_satisfied": fully_satisfied,
		"reason": canonical["reason"],
		"constraint_audit": canonical["constraint_audit"],
		"issues": canonical["issues"],
		"preservation_constraints": canonical["preservation_constraints"],
		"canonical_diagnosis": canonical,
		"canonical_diagnosis_text": canonical_diagnosis_text(canonical),
		# Read-only summaries retain compatibility with comparison/logging code;
		# no executable instruction or bbox comes from diagnosis.
		"issue": "None" if fully_satisfied else summary,
		"fix": "None" if fully_satisfied else summary,
		"global_edit_instruction": "",
		"edits": [],
		"raw_output": raw,
		"valid_json": valid_json,
		"reflection_schema_version": 10,
		"image_size": {"width": int(width), "height": int(height)},
	}


def normalize_l1_plan(
	obj: Optional[Dict[str, Any]],
	raw: str,
	diagnosis: Dict[str, Any],
	target_prompt: str,
	width: int,
	height: int,
	initial_generator_family: str | None = None,
) -> Dict[str, Any]:
	"""Validate the post-route 1--3 region plan and build its joint prompt."""
	if not isinstance(obj, dict) or not isinstance(obj.get("edits"), list):
		return {"valid_json": False, "reason": "Missing L1 edits array.", "edits": [], "raw_output": raw}
	raw_edits = obj["edits"]
	if not 1 <= len(raw_edits) <= 3:
		return {"valid_json": False, "reason": "L1 requires one to three edits.", "edits": [], "raw_output": raw}
	edits: List[Dict[str, Any]] = []
	for index, raw_edit in enumerate(raw_edits, start=1):
		if not isinstance(raw_edit, dict):
			return {"valid_json": False, "reason": f"Edit {index} is not an object.", "edits": [], "raw_output": raw}
		raw_quad = raw_edit.get("quad_points")
		# One common schema slip is [[x0,y0,x1,y1]].  It is unambiguous,
		# so canonicalize it while keeping multi-box-in-one-edit outputs invalid.
		if isinstance(raw_quad, list) and len(raw_quad) == 1 and isinstance(raw_quad[0], list) and len(raw_quad[0]) == 4:
			x0, y0, x1, y1 = raw_quad[0]
			raw_quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
		# Qwen also occasionally emits the two diagonal corners
		# [[x0,y0],[x1,y1]].  This has exactly one rectangular interpretation.
		if (
			isinstance(raw_quad, list)
			and len(raw_quad) == 2
			and all(isinstance(point, list) and len(point) == 2 for point in raw_quad)
		):
			(x0, y0), (x1, y1) = raw_quad
			raw_quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
		quad = clamp_quad(raw_quad, width, height)
		instruction = str(raw_edit.get("instruction") or raw_edit.get("edit_prompt") or "").strip()
		if quad is None or not instruction:
			return {"valid_json": False, "reason": f"Edit {index} lacks a valid bbox/instruction.", "edits": [], "raw_output": raw}
		box = quad_to_box(quad)
		area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
		edits.append(
			{
				"id": str(raw_edit.get("id") or f"edit_{index}").strip()[:64],
				"priority": index,
				"target_entity": str(raw_edit.get("target_entity") or raw_edit.get("target_object") or "localized region").strip(),
				"target_object": str(raw_edit.get("target_entity") or raw_edit.get("target_object") or "localized region").strip(),
				"issue": str(raw_edit.get("issue") or "diagnosed mismatch").strip(),
				"instruction": instruction,
				"edit_prompt": instruction,
				"whole_edit": instruction,
				"quad_points": quad,
				"box_xyxy": box,
				"whole_image_pseudo_local": area >= int(width) * int(height) * 0.95,
			}
		)
	# A global-state counterfactual can make Qwen repeat a full-canvas box even
	# after the structured retry.  Prefer any genuine local edits and discard
	# full-image background/relation items.  If none exist, retain only the
	# highest-priority proposal and turn it into a deterministic centered inset.
	# This keeps L1 an intentionally bounded (and possibly low-value) action;
	# the blind evaluator, rather than parser compliance, decides its utility.
	localized_edits = [edit for edit in edits if not edit.pop("whole_image_pseudo_local")]
	if localized_edits:
		edits = localized_edits
	else:
		fallback = edits[0]
		pad_x = max(1, int(round(width * 0.05)))
		pad_y = max(1, int(round(height * 0.05)))
		x0, y0 = pad_x, pad_y
		x1, y1 = max(x0 + 1, width - pad_x), max(y0 + 1, height - pad_y)
		fallback["quad_points"] = [[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]]
		fallback["box_xyxy"] = [x0, y0, x1, y1]
		fallback["bbox_normalization"] = "centered_90pct_span_fallback_from_whole_image_v1"
		edits = [fallback]
	# Qwen can keep a contained sub-edit even after the structured retry (for
	# example, adding a cup inside a larger object-replacement crop).  Such
	# outputs describe one Flux crop, not two independent references.  Merge
	# them deterministically here so online inference and offline collection
	# share the same behavior and do not depend on a second model response.
	def overlap_is_strong(left_box: Sequence[int], right_box: Sequence[int]) -> bool:
		"""Detect regions that must become one executor crop."""
		left_area = max(0, left_box[2] - left_box[0]) * max(0, left_box[3] - left_box[1])
		right_area = max(0, right_box[2] - right_box[0]) * max(0, right_box[3] - right_box[1])
		intersection = max(0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0])) * max(
			0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1])
		)
		union = left_area + right_area - intersection
		iou = intersection / union if union > 0 else 0.0
		containment = intersection / min(left_area, right_area) if min(left_area, right_area) > 0 else 0.0
		return iou >= L1_MAX_PAIRWISE_IOU or containment >= L1_MAX_PAIRWISE_CONTAINMENT

	def joined_text(left: str, right: str, separator: str = "; ") -> str:
		"""Join two distinct non-empty descriptions without duplication."""
		parts: List[str] = []
		for value in (left, right):
			text = str(value or "").strip()
			if text and text not in parts:
				parts.append(text)
		return separator.join(parts)

	while True:
		merge_pair: Optional[Tuple[int, int]] = None
		for left_index, left in enumerate(edits):
			for right_index in range(left_index + 1, len(edits)):
				if overlap_is_strong(left["box_xyxy"], edits[right_index]["box_xyxy"]):
					merge_pair = (left_index, right_index)
					break
			if merge_pair is not None:
				break
		if merge_pair is None:
			break
		left_index, right_index = merge_pair
		left = edits[left_index]
		right = edits[right_index]
		left_box = left["box_xyxy"]
		right_box = right["box_xyxy"]
		box = [
			min(left_box[0], right_box[0]),
			min(left_box[1], right_box[1]),
			max(left_box[2], right_box[2]),
			max(left_box[3], right_box[3]),
		]
		area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
		if area >= int(width) * int(height) * 0.95:
			return {
				"valid_json": False,
				"reason": "Merging strongly overlapping edits produces a whole-image pseudo-local box.",
				"edits": [],
				"raw_output": raw,
			}
		instruction_parts = [left["instruction"], right["instruction"]]
		instruction = "Apply both localized repairs within this shared region: " + " ".join(
			f"({index}) {text}" for index, text in enumerate(dict.fromkeys(instruction_parts), start=1)
		)
		merged = dict(left)
		merged.update(
			{
				"target_entity": joined_text(left["target_entity"], right["target_entity"]),
				"target_object": joined_text(left["target_object"], right["target_object"]),
				"issue": joined_text(left["issue"], right["issue"]),
				"instruction": instruction,
				"edit_prompt": instruction,
				"whole_edit": instruction,
				"quad_points": [
					[box[0], box[1]],
					[box[2] - 1, box[1]],
					[box[2] - 1, box[3] - 1],
					[box[0], box[3] - 1],
				],
				"box_xyxy": box,
				"merged_edit_ids": [
					*left.get("merged_edit_ids", [left["id"]]),
					*right.get("merged_edit_ids", [right["id"]]),
				],
			}
		)
		edits[left_index] = merged
		del edits[right_index]
	for index, edit in enumerate(edits, start=1):
		edit["priority"] = index
	try:
		plan = build_level_action_plan(
			1,
			target_prompt,
			diagnosis,
			edits=edits,
			width=width,
			height=height,
			initial_generator_family=initial_generator_family,
		)
		validate_distribution_contract(plan)
	except (TypeError, ValueError) as exc:
		return {"valid_json": False, "reason": str(exc), "edits": [], "raw_output": raw}
	plan.update({"reason": str(obj.get("reason") or "").strip(), "raw_output": raw, "valid_json": True})
	return plan


def normalize_global_compare(
	obj: Optional[Dict[str, Any]],
	raw: str,
	available_winners: Sequence[str] = ("A", "B"),
) -> Dict[str, Any]:
	"""Validate a blind candidate comparison and constrain its winner labels.

	Malformed comparisons conservatively retain the first available candidate
	and request continuation, preventing invalid prose from accepting a repair.
	"""
	allowed = {
		str(label).strip().upper()
		for label in available_winners
		if str(label).strip()
	}
	fallback = next((label for label in ("A", "B") if label in allowed), next(iter(sorted(allowed)), "A"))
	required_text_fields = ("reason", "semantic_status", "constraint_audit")
	winner = obj.get("winner") if isinstance(obj, dict) else None
	decision = obj.get("decision") if isinstance(obj, dict) else None
	valid_contract = (
		isinstance(obj, dict)
		and isinstance(winner, str)
		and winner.strip().upper() in allowed
		and isinstance(decision, str)
		and decision.strip().lower() in {"satisfied", "continue"}
		and all(isinstance(obj.get(key), str) and obj[key].strip() for key in required_text_fields)
	)
	if not valid_contract:
		return {
			"winner": fallback,
			"decision": "continue",
			"reason": "Failed to parse a complete language comparison JSON contract.",
			"semantic_status": "Unknown",
			"constraint_audit": "Unknown",
			"raw_output": raw,
			"valid_json": False,
		}
	winner = winner.strip().upper()
	decision = decision.strip().lower()
	return {
		"winner": winner,
		"decision": decision,
		"reason": str(obj.get("reason", "")).strip(),
		"semantic_status": str(obj.get("semantic_status", "")).strip(),
		"constraint_audit": str(obj.get("constraint_audit", "")).strip(),
		"execution_feedback": (
			dict(obj.get("execution_feedback"))
			if isinstance(obj.get("execution_feedback"), dict)
			else {
				"progress": "unknown",
				"remaining_mismatch": str(obj.get("semantic_status", "")).strip(),
				"instruction_problem": "not supplied",
			}
		),
		"raw_output": raw,
		"valid_json": True,
	}


def normalize_box_review(
	obj: Optional[Dict[str, Any]],
	raw: str,
	current_edits: Sequence[Dict[str, Any]],
	width: int,
	height: int,
) -> Dict[str, Any]:
	"""Normalize a coordinate review while preserving edit identity and order."""
	if not isinstance(obj, dict):
		return {
			"satisfied": False,
			"reason": "Failed to parse language box-review JSON.",
			"edits": [dict(edit) for edit in current_edits],
			"raw_output": raw,
			"valid_json": False,
		}
	raw_edits = obj.get("edits")
	reviewed: List[Dict[str, Any]] = []
	valid_edits = isinstance(raw_edits, list) and len(raw_edits) == len(current_edits)
	for idx, current_edit in enumerate(current_edits):
		base = dict(current_edit)
		raw_edit = raw_edits[idx] if valid_edits and isinstance(raw_edits[idx], dict) else {}
		quad = clamp_quad(raw_edit.get("quad_points"), width, height) if raw_edit else None
		if quad is not None:
			base["quad_points"] = quad
			base["box_xyxy"] = quad_to_box(quad)
		base["box_review_satisfied"] = bool(raw_edit.get("satisfied", quad is not None)) if raw_edit else False
		base["box_review_reason"] = str(raw_edit.get("reason", "")).strip() if raw_edit else "Missing or invalid review entry."
		reviewed.append(base)
	return {
		"satisfied": bool(obj.get("satisfied", False)),
		"reason": str(obj.get("reason", "")).strip(),
		"edits": reviewed,
		"raw_output": raw,
		"valid_json": True,
		"valid_edits": valid_edits,
	}
