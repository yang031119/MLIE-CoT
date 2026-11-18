"""Implement every Qwen-controlled transition in the V10 state machine.

The module separates factual diagnosis from action selection.  Global triage
first produces canonical state-only facts; native Qwen may stop at L0, while a
frozen route head chooses L1/L2/L3 for problematic states.  L1 alone triggers a
second, ruler-annotated localization request.  L2 and L3 plans are compiled
deterministically from diagnosis plus the target prompt.

Generated candidates are never trusted directly.  Whole-image and local
results enter source-blind, deterministically permuted A/B reviews, and the
selected image is followed by a fresh diagnosis.  Role mappings remain
controller metadata and are restored only after the reviewer returns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

from .vcot_branch import HeavyRuntimeManager

from .config import (
	BOX_REVIEW_SYSTEM_PROMPT,
	BOX_REVIEW_USER_TEMPLATE,
	GLOBAL_COMPARE_USER_TEMPLATE,
	LOCAL_MODEL_ORDER,
	LOCAL_REVIEW_USER_TEMPLATE,
	L1_PLAN_SYSTEM_PROMPT,
	L1_PLAN_USER_TEMPLATE,
	PROMPT_REASONING_SYSTEM_PROMPT,
	PROMPT_REASONING_USER_TEMPLATE,
	QWEN_MULTI_IMAGE_INPUT_MAX_SIDE,
	TRIAGE_SYSTEM_PROMPT,
	TRIAGE_USER_TEMPLATE,
)
from .contracts import (
	DIAGNOSIS_RETRY_INSTRUCTION_V10,
	L1_PLAN_RETRY_INSTRUCTION_V10,
	build_level_action_plan,
	diagnosis_issue_summary,
	model_visible_diagnosis_text,
	validate_distribution_contract,
)
from .image_utils import (
	annotate_image_with_ticks_32,
	build_box_review_edit_lines,
	build_labeled_multi_image_messages,
	copy_image_generation_sized,
	draw_dynamic_contrast_boxes,
	pad_image_to_multiple,
	prepare_qwen_image,
	qwen_multi_image_ladder,
)
from .io_utils import append_event, ensure_round_initial, save_json, save_status, update_best_round
from .route_head import ROUTE_MARKER, apply_route_head_decision
from .logging_utils import describe_states_brief, log_stage
from .parsing import (
	extract_json,
	normalize_box_review,
	normalize_global_compare,
	normalize_l1_plan,
	normalize_prompt_reasoning,
	normalize_triage_analysis,
)
from .language_batch import language_model_label, run_language_batch_with_retries
from .state import (
	SampleState,
	round_action_plan_path,
	round_error_analysis_path,
	round_level_classification_path,
	sample_round_dir,
)


def apply_fixed_repair_level(args: argparse.Namespace, state: SampleState, analysis: Dict[str, Any]) -> Dict[str, Any] | None:
	"""Apply the fixed L1/L2/L3 ablation without invoking the learned V10 head."""
	level = getattr(args, "fixed_repair_level", None)
	if level not in {1, 2, 3}:
		return None
	if state.phase != "global":
		return {
			"enabled": True,
			"source": "fixed_repair_level_ablation",
			"fixed_repair_level": level,
			"enforced": False,
			"enforce_blocked_reason": "non_global_phase",
		}
	original_mode = analysis.get("recommended_mode")
	original_category = analysis.get("category")
	if bool(analysis.get("fully_satisfied")):
		level = 0
	elif level == 1:
		analysis["fully_satisfied"] = False
		analysis["recommended_mode"] = "micro"
		analysis["category"] = "L1"
		analysis["edits"] = []
	elif level == 2:
		analysis["fully_satisfied"] = False
		analysis["recommended_mode"] = "global"
		analysis["category"] = "L2"
		analysis["edits"] = []
	elif level == 3:
		analysis["fully_satisfied"] = False
		analysis["recommended_mode"] = "regenerate"
		analysis["category"] = "L3"
		analysis["edits"] = []
	decision = {
		"enabled": True,
		"source": "fixed_repair_level_ablation",
		"fixed_repair_level": int(getattr(args, "fixed_repair_level")),
		"level": level,
		"mode": {0: "done", 1: "micro", 2: "global", 3: "regenerate"}[level],
		"original_recommended_mode": original_mode,
		"original_category": original_category,
		"enforced": True,
	}
	analysis["route_head_decision"] = decision
	return decision


GLOBAL_COMPARE_SYSTEM_PROMPT = """Select between input images A and B the candidate that fully satisfies the prompt. When both candidates remain imperfect, select the one that is easiest to continue editing toward the target semantics.

The complete response is exactly one valid JSON object:
{
  "reason": "brief explanation",
  "semantic_status": "which prompt constraints are satisfied or still missing",
  "constraint_audit": "brief audit of required objects, counts, colors/attributes, and spatial relations in the winner",
  "decision": "satisfied" | "continue",
  "winner": "A" | "B"
}

Rules:
- First audit all required prompt constraints: object identities, object counts, colors/attributes, and spatial relations.
- Set decision="satisfied" when the chosen image meets the prompt well enough.
- Set decision="continue" for an imperfect winner and still select the best starting point for the next round.
- Set winner to A or B.
- Apply the same selection rule to both images.
- The JSON object above is the complete response.
"""


LOCAL_REVIEW_SYSTEM_PROMPT = """Select between input images A and B the candidate that fully satisfies the prompt. When both candidates remain imperfect, select the one that is easiest to continue editing toward the target semantics.

Judge all planned local edits together against the complete target prompt. Success requires improvement on the joint plan and preservation of every already-correct required prompt constraint outside the union of edited regions.

Before setting success=true, explicitly confirm that every required object/person, count, spatial relation, attribute, action, state, and background remains present, distinct, correctly colored, correctly positioned, and visually usable. Treat severe artifacts, text/watermarks, and facial/body/object distortions that invalidate required subjects as blocking prompt violations.

The complete response is exactly one valid JSON object:
{
  "reason": "brief explanation",
  "edit_results": [{"id": "edit_1", "satisfied": true | false, "note": "brief result"}],
  "success": true | false,
  "blocking_prompt_violation": true | false,
  "outside_regions_preserved": true | false,
  "preservation_note": "brief preservation audit",
  "winner": "A" | "B"
}

Rules:
- If both images damage global constraints, choose the cleanest image for continued editing and set success=false.
- Set winner to A or B.
- Return one edit_results entry per planned edit. Subsequent progress starts from a fresh diagnosis.
- The JSON object above is the complete response.
"""


def _blind_ab_assignment(
	state: SampleState,
	round_index: int,
	comparison_kind: str,
	role_paths: Sequence[Tuple[str, Path]],
) -> Tuple[List[Tuple[str, Path]], Dict[str, str], Dict[str, str], Dict[str, Path]]:
	"""Assign two source-blind labels with a stable per-state permutation."""
	if len(role_paths) != 2:
		raise ValueError(f"A/B blind comparison requires exactly two images, got {len(role_paths)}")
	ordered = [(str(role), Path(path)) for role, path in role_paths]
	key = f"{comparison_kind}\0{state.sample_id}\0{int(round_index)}".encode("utf-8")
	if hashlib.sha256(key).digest()[0] & 1:
		ordered.reverse()
	candidates: List[Tuple[str, Path]] = []
	label_to_role: Dict[str, str] = {}
	role_to_label: Dict[str, str] = {}
	role_to_path: Dict[str, Path] = {}
	for label, (role, path) in zip(("A", "B"), ordered):
		if role in role_to_label:
			raise ValueError(f"A/B blind comparison received duplicate role: {role!r}")
		candidates.append((label, path))
		label_to_role[label] = role
		role_to_label[role] = label
		role_to_path[role] = path
	return candidates, label_to_role, role_to_label, role_to_path


def normalize_local_review_with_roles(
	obj: Any,
	raw: str,
	label_to_role: Dict[str, str],
	expected_edit_ids: Sequence[str] = (),
) -> Dict[str, Any]:
	"""Validate a blind L1 review and map the chosen label back to source role.

	Invalid output defaults to the original image.  A claimed success is honored
	only when required content is preserved and no blocking prompt violation is
	reported.
	"""
	original_label = next(
		(label for label, role in label_to_role.items() if role == "original"),
		next(iter(label_to_role), "A"),
	)
	if not isinstance(obj, dict):
		return {
			"winner": original_label,
			"winner_role": "original",
			"success": False,
			"blocking_prompt_violation": False,
			"reason": "Failed to parse language micro-review JSON.",
			"preservation_note": "Unknown",
			"raw_output": raw,
			"valid_json": False,
		}
	winner = str(obj.get("winner", original_label)).strip().upper()
	if winner not in label_to_role:
		winner = original_label
	blocking_prompt_violation = bool(obj.get("blocking_prompt_violation", False))
	outside_regions_preserved = bool(obj.get("outside_regions_preserved", False))
	edit_results = [dict(entry) for entry in obj.get("edit_results", []) if isinstance(entry, dict)] \
		if isinstance(obj.get("edit_results"), list) else []
	returned_ids = [str(entry.get("id") or "") for entry in edit_results]
	expected_ids = [str(edit_id) for edit_id in expected_edit_ids]
	contract_valid = all(key in obj for key in ("winner", "success", "blocking_prompt_violation", "outside_regions_preserved"))
	if expected_ids:
		contract_valid = contract_valid and returned_ids == expected_ids
	return {
		"winner": winner,
		"winner_role": label_to_role.get(winner, "original"),
		"success": bool(obj.get("success", False)) and not blocking_prompt_violation and outside_regions_preserved,
		"blocking_prompt_violation": blocking_prompt_violation,
		"reason": str(obj.get("reason", "")).strip(),
		"preservation_note": str(obj.get("preservation_note", "")).strip(),
		"outside_regions_preserved": outside_regions_preserved,
		"edit_results": edit_results,
		"raw_output": raw,
		"valid_json": contract_valid,
	}


def _build_prompt_reasoning_job(state: SampleState) -> Dict[str, Any]:
	"""Build the optional text-only pre-generation prompt-expansion request."""
	user_text = PROMPT_REASONING_USER_TEMPLATE.format(prompt=state.prompt.strip())
	messages = [
		{"role": "system", "content": [{"type": "text", "text": PROMPT_REASONING_SYSTEM_PROMPT}]},
		{"role": "user", "content": [{"type": "text", "text": user_text}]},
	]
	return {
		"state": state,
		"messages": messages,
		"images": [],
		"batch_kind": "prompt-reasoning",
		"log_label": f"sample={state.sample_id} step=reason-prompt model=Qwen",
		"retry_log_label": f"sample={state.sample_id} step=reason-prompt-retry model=Qwen",
	}


def _persist_prompt_reasoning_result(state: SampleState, analysis: Dict[str, Any]) -> None:
	"""Persist valid prompt reasoning or an auditable invalid partial response."""
	if not analysis.get("valid_json"):
		save_json(state.sample_dir / "generation_prompt_partial.json", analysis)
		(state.sample_dir / "generation_prompt_partial.txt").write_text(str(analysis.get("raw_output") or "") + "\n", encoding="utf-8")
		append_event(
			state,
			{
				"event": "prompt_reasoning_partial_invalid",
				"analysis": analysis,
			},
		)
		save_status(state)
		return
	already_saved = (state.sample_dir / "generation_prompt.json").exists()
	state.generation_prompt = str(analysis.get("generation_prompt") or state.prompt).strip() or state.prompt
	save_json(state.sample_dir / "generation_prompt.json", analysis)
	(state.sample_dir / "generation_prompt.txt").write_text(state.generation_prompt + "\n", encoding="utf-8")
	if not already_saved:
		append_event(
			state,
			{
				"event": "prompt_reasoning",
				"analysis": analysis,
				"generation_prompt": state.generation_prompt,
			},
		)
	save_status(state)


def run_prompt_reasoning_batch(runtime: HeavyRuntimeManager, args: argparse.Namespace, states: Sequence[SampleState]) -> None:
	"""Populate generation prompts for fresh states without touching image budget."""
	tasks = [state for state in states if state.current_image is None and state.generation_prompt is None and not state.done and not state.error]
	if not tasks:
		return
	if bool(getattr(args, "skip_prompt_reasoning", False)):
		for state in tasks:
			state.generation_prompt = state.prompt.strip()
			analysis = {
				"reasoning": "Skipped by --skip-prompt-reasoning; using the original prompt as generation_prompt.",
				"implicit_logic": [],
				"task_decomposition": [],
				"scene_construction_plan": "",
				"visual_requirements": [],
				"ambiguity_resolution": "",
				"generation_prompt": state.generation_prompt,
				"raw_output": "",
				"valid_json": True,
			}
			save_json(state.sample_dir / "generation_prompt.json", analysis)
			(state.sample_dir / "generation_prompt.txt").write_text(state.generation_prompt + "\n", encoding="utf-8")
			append_event(state, {"event": "prompt_reasoning_skipped", "generation_prompt": state.generation_prompt})
			save_status(state)
		return
	log_stage(
		f"dispatch step=reason-prompt model={language_model_label(args)} count={len(tasks)} samples={describe_states_brief(tasks)}"
	)
	runtime.release_bagel()
	runtime.release_flux()
	prepared = [_build_prompt_reasoning_job(state) for state in tasks]
	for item in prepared:
		state: SampleState = item["state"]

		def persist_partial(analysis: Dict[str, Any], current_state: SampleState = state) -> None:
			"""Persist worker progress before fallback processing can begin."""
			_persist_prompt_reasoning_result(current_state, analysis)
			if analysis.get("valid_json"):
				log_stage(f"sample={current_state.sample_id} step=reason-prompt-save model={language_model_label(args)}")
			else:
				log_stage(f"sample={current_state.sample_id} step=reason-prompt-partial-invalid model={language_model_label(args)}")

		item["partial_result_callback"] = persist_partial
	log_stage(f"dispatch step=reason-prompt-queue model={language_model_label(args)} size={len(prepared)}")
	results = run_language_batch_with_retries(
		runtime,
		args,
		prepared,
		args.qwen_prompt_reasoning_max_new_tokens,
		lambda raw, item: normalize_prompt_reasoning(extract_json(raw), raw, item["state"].prompt),
		'Your previous response was invalid. Return exactly one JSON object whose sole key is "generation_prompt"; the JSON object is the complete response.',
		args.qwen_batch_size,
	)
	for item, analysis in zip(prepared, results):
		state: SampleState = item["state"]
		_persist_prompt_reasoning_result(state, analysis)


def _build_triage_job(state: SampleState, args: argparse.Namespace) -> Dict[str, Any]:
	"""Prepare raw-image diagnosis and route inputs for the current action round.

	The route image remains unannotated.  A separate ruler image is prepared for
	a possible L1 planning call, preserving the distinct analyzer/planner image
	distributions frozen in the checkpoint contract.
	"""
	assert state.current_image is not None
	ensure_round_initial(state)
	assert state.current_image is not None
	image = Image.open(state.current_image).convert("RGB")
	width, height = image.size
	# Diagnosis and route features use the unannotated current image.  The
	# coordinate ruler is introduced only after an L1 route, when localization
	# is actually requested.
	source_qwen_image = image.copy()
	qwen_image = prepare_qwen_image(source_qwen_image, max_side=args.qwen_input_max_side)
	qwen_image = pad_image_to_multiple(qwen_image, multiple=32)
	l1_plan_source_image = annotate_image_with_ticks_32(image)
	l1_plan_qwen_image = prepare_qwen_image(l1_plan_source_image, max_side=args.qwen_input_max_side)
	l1_plan_qwen_image = pad_image_to_multiple(l1_plan_qwen_image, multiple=32)
	round_dir = sample_round_dir(state.sample_dir, state.global_round)
	qwen_input_path = round_dir / "analysis_input.png"
	qwen_image.save(qwen_input_path)
	user_text = TRIAGE_USER_TEMPLATE.format(
		prompt=state.prompt.strip(),
		width=width,
		height=height,
	)
	messages = [
		{"role": "system", "content": [{"type": "text", "text": TRIAGE_SYSTEM_PROMPT}]},
		{
			"role": "user",
			"content": [
				{"type": "image", "image": qwen_image},
				{"type": "text", "text": user_text},
			],
		},
	]
	return {
		"state": state,
		"image": image,
		"route_image": image.copy(),
		"qwen_image": qwen_image,
		"source_qwen_image": source_qwen_image.copy(),
		"l1_plan_source_image": l1_plan_source_image.copy(),
		"l1_plan_qwen_image": l1_plan_qwen_image,
		"current_qwen_max_side": max(qwen_image.size),
		"width": width,
		"height": height,
		"messages": messages,
		"images": [qwen_image],
		"qwen_input_path": qwen_input_path,
		# At the action boundary this request is the terminal diagnosis only;
		# invoking the route head would be wasted work and could incorrectly make
		# finalization depend on a route prediction that cannot be executed.
		"route_prompt": (
			state.prompt.strip()
			if int(state.global_round) < int(args.max_global_rounds)
			else ""
		),
		"batch_kind": "global-triage",
		"log_label": (
			f"sample={state.sample_id} step=analyze model=Qwen mode=global "
			f"round={state.global_round} reason={state.pending_global_analysis_reason}"
		),
		"retry_log_label": f"sample={state.sample_id} step=analyze-retry model=Qwen mode=global",
	}


def _build_l1_plan_job(
	item: Dict[str, Any],
	analysis: Dict[str, Any],
	args: argparse.Namespace,
) -> Dict[str, Any]:
	"""Build the second, route-conditioned language request for L1 only."""
	state: SampleState = item["state"]
	diagnosis = analysis.get("canonical_diagnosis") or analysis
	diagnosis_json = model_visible_diagnosis_text(diagnosis, indent=2)
	user_text = L1_PLAN_USER_TEMPLATE.format(
		prompt=state.prompt.strip(),
		diagnosis_json=diagnosis_json,
		width=item["width"],
		height=item["height"],
	)
	messages = [
		{"role": "system", "content": [{"type": "text", "text": L1_PLAN_SYSTEM_PROMPT}]},
		{
			"role": "user",
			"content": [
				{"type": "image", "image": item["l1_plan_qwen_image"]},
				{"type": "text", "text": user_text},
			],
		},
	]
	return {
		"state": state,
		"diagnosis": diagnosis,
		"width": item["width"],
		"height": item["height"],
		"messages": messages,
		"images": [item["l1_plan_qwen_image"]],
		"qwen_image": item["l1_plan_qwen_image"],
		"source_qwen_image": item["l1_plan_source_image"],
		"current_qwen_max_side": max(item["l1_plan_qwen_image"].size),
		"route_prompt": "",
		"batch_kind": "l1-route-conditioned-plan",
		"log_label": f"sample={state.sample_id} step=plan-l1 model={language_model_label(args)} round={state.global_round}",
		"retry_log_label": f"sample={state.sample_id} step=plan-l1-json-retry model={language_model_label(args)} round={state.global_round}",
	}


def _attach_deterministic_plan(
	args: argparse.Namespace,
	state: SampleState,
	analysis: Dict[str, Any],
	level: int,
) -> Dict[str, Any]:
	"""Compile and validate a non-L1 action plan from the routed diagnosis."""
	plan = build_level_action_plan(
		level,
		state.prompt,
		analysis.get("canonical_diagnosis") or analysis,
		initial_generator_family=(
			(
				state.initial_generator_family
				or str(getattr(args, "initial_generator", "") or "")
			)
			if int(level) == 3
			else None
		),
	)
	validate_distribution_contract(plan)
	analysis["action_plan"] = plan
	analysis["global_edit_instruction"] = str(plan["instruction"])
	return plan


def _enter_micro_from_analysis(state: SampleState, analysis: Dict[str, Any], tag: str, source: str) -> None:
	"""Persist an L1 plan and move the sample to local-edit execution."""
	state.phase = "micro"
	repair_scope = str(analysis.get("category") or "").strip()
	state.micro_plan = [dict(edit, repair_scope=repair_scope) for edit in (analysis.get("edits") or [])]
	state.micro_box_review_done = False
	state.micro_edit_index = 0
	state.micro_attempt = 0
	state.pending_global_compare = None
	state.pending_local_review = None
	state.needs_global_analysis = False
	state.pending_global_analysis_reason = source
	state.status = "pending"
	state.final_reason = None
	state.final_image = None
	save_json(
		round_action_plan_path(state.sample_dir, state.global_round),
		{
			"action_plan": analysis.get("action_plan"),
			"plan": state.micro_plan,
			"source": source,
			"source_triage": tag,
			"diagnosis_path": str(round_error_analysis_path(state.sample_dir, state.global_round)),
		},
	)
	append_event(
		state,
		{
			"event": "enter_micro",
			"round": state.global_round,
			"source": source,
			"source_triage": tag,
			"plan": state.micro_plan,
		},
	)


def _enter_regenerate_from_analysis(state: SampleState, analysis: Dict[str, Any], tag: str, source: str) -> None:
	"""Persist an L3 plan and move the sample to same-base regeneration."""
	state.phase = "regenerate"
	state.micro_plan = []
	state.micro_box_review_done = False
	state.micro_edit_index = 0
	state.micro_attempt = 0
	state.pending_global_compare = None
	state.pending_local_review = None
	state.needs_global_analysis = False
	state.pending_global_analysis_reason = source
	state.status = "pending"
	state.final_reason = None
	state.final_image = None
	save_json(
		round_action_plan_path(state.sample_dir, state.global_round),
		{
			"action_plan": analysis.get("action_plan"),
			"source": source,
			"source_triage": tag,
			"diagnosis_path": str(round_error_analysis_path(state.sample_dir, state.global_round)),
		},
	)
	append_event(
		state,
		{
			"event": "enter_regenerate",
			"round": state.global_round,
			"source": source,
			"source_triage": tag,
			"analysis": analysis,
		},
	)


def _triage_tag(state: SampleState) -> str:
	"""Return the stable artifact tag for the state's current action round."""
	return f"round_{state.global_round:02d}_triage"


def _annotate_triage_analysis(item: Dict[str, Any], analysis: Dict[str, Any]) -> None:
	"""Attach audit-only image metadata without changing canonical diagnosis."""
	state: SampleState = item["state"]
	analysis.update(
		{
			"image": str(state.current_image),
			"qwen_input_image": str(item["qwen_input_path"]),
			"phase": state.phase,
			"analysis_reason": state.pending_global_analysis_reason,
		}
	)


def run_global_triage_batch(runtime: HeavyRuntimeManager, args: argparse.Namespace, states: Sequence[SampleState]) -> None:
	"""Diagnose ready images, route them, and persist the next safe transition.

	At exhausted action budget the stage performs diagnosis only.  For an L1
	route it batches a second localization request; L2/L3 plans are deterministic.
	Every result is written before phase flags make it eligible for generation.
	"""
	triage_states = [state for state in states if state.phase == "global" and state.needs_global_analysis and state.current_image and not state.done and not state.error]
	if not triage_states:
		return
	log_stage(
		f"dispatch step=analyze model={language_model_label(args)} count={len(triage_states)} samples={describe_states_brief(triage_states)}"
	)
	prepared = [_build_triage_job(state, args) for state in triage_states]
	results = run_language_batch_with_retries(
		runtime,
		args,
		prepared,
		args.qwen_max_new_tokens,
		lambda raw, item: normalize_triage_analysis(extract_json(raw), raw, item["width"], item["height"]),
		DIAGNOSIS_RETRY_INSTRUCTION_V10,
		args.qwen_batch_size,
	)

	enriched: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], str]] = []
	l1_plan_records: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
	for item, analysis in zip(prepared, results):
		state: SampleState = item["state"]
		tag = _triage_tag(state)
		_annotate_triage_analysis(item, analysis)
		has_action_budget = int(state.global_round) < int(args.max_global_rounds)
		if has_action_budget:
			level_decision = apply_fixed_repair_level(args, state, analysis) or apply_route_head_decision(state, analysis)
		else:
			level_decision = {
				"enabled": False,
				"enforced": True,
				"source": "action_budget_gate",
				"level": 0 if analysis.get("fully_satisfied") else None,
				"route": "L0_satisfied" if analysis.get("fully_satisfied") else "final_diagnosis_only",
				"action_budget_exhausted": True,
				"actions_used": int(state.global_round),
				"max_actions": int(args.max_global_rounds),
			}
			analysis["route_head_decision"] = level_decision
		slot = len(enriched)
		enriched.append((item, analysis, level_decision, tag))
		level = int(level_decision.get("level", 0) or 0)
		if analysis.get("valid_json") and not analysis.get("fully_satisfied") and level == 1 and has_action_budget:
			plan_item = _build_l1_plan_job(item, analysis, args)
			l1_plan_records.append((slot, plan_item, analysis))
		elif analysis.get("valid_json") and level in {2, 3} and has_action_budget:
			_attach_deterministic_plan(args, state, analysis, level)

	if l1_plan_records:
		plan_items = [record[1] for record in l1_plan_records]
		log_stage(
			f"dispatch step=plan-l1 model={language_model_label(args)} "
			f"count={len(plan_items)} samples={describe_states_brief([item['state'] for item in plan_items])}"
		)
		plan_results = run_language_batch_with_retries(
			runtime,
			args,
			plan_items,
			args.qwen_max_new_tokens,
			lambda raw, item: normalize_l1_plan(
				extract_json(raw),
				raw,
				item["diagnosis"],
				item["state"].prompt,
				item["width"],
				item["height"],
			),
			L1_PLAN_RETRY_INSTRUCTION_V10,
			args.qwen_batch_size,
		)
		for (slot, _plan_item, analysis), plan in zip(l1_plan_records, plan_results):
			item, _old_analysis, decision, tag = enriched[slot]
			state: SampleState = item["state"]
			plan_path = sample_round_dir(state.sample_dir, state.global_round) / "l1_action_plan.json"
			if plan.get("valid_json"):
				analysis["action_plan"] = plan
				analysis["edits"] = list(plan["edits"])
				analysis["global_edit_instruction"] = str(plan["instruction"])
				decision["planning_succeeded"] = True
			elif getattr(args, "fixed_repair_level", None) == 1:
				# Fixed-level ablations may fail, but may never execute another level.
				decision.update({
					"planning_succeeded": False,
					"planning_failure": str(plan.get("reason") or "invalid_l1_plan"),
					"level": 1,
					"fixed_level_planning_failed": True,
				})
			else:
				# A malformed localization plan is not silently treated as L1.  The
				# deterministic escalation is explicit in the trajectory record.
				decision.update(
					{
						"planning_succeeded": False,
						"planning_failure": str(plan.get("reason") or "invalid_l1_plan"),
						"original_level": 1,
						"level": 2,
						"route": "L2_global_flux2",
						"enforced_mode": "global",
					}
				)
				analysis["category"] = "L2"
				analysis["recommended_mode"] = "global"
				_attach_deterministic_plan(args, state, analysis, 2)
			save_json(plan_path, plan)
			analysis["l1_plan_path"] = str(plan_path)
			enriched[slot] = (item, analysis, decision, tag)

	for item, analysis, level_decision, tag in enriched:
		state: SampleState = item["state"]
		if not analysis.get("valid_json"):
			append_event(state, {"event": "invalid_v10_diagnosis", "tag": tag, "analysis": analysis})
			state.status = "partial"
			state.final_reason = "invalid_v10_diagnosis_after_json_retry"
			state.final_image = state.current_image
			state.needs_global_analysis = False
			save_status(state)
			continue
		has_action_budget = int(state.global_round) < int(args.max_global_rounds)
		if has_action_budget and not analysis.get("fully_satisfied") and int(level_decision.get("level", 0) or 0) not in {1, 2, 3}:
			append_event(state, {"event": "missing_v10_route_prediction", "tag": tag, "decision": level_decision})
			state.status = "partial"
			state.final_reason = "missing_v10_route_prediction"
			state.final_image = state.current_image
			state.needs_global_analysis = False
			save_status(state)
			continue
		analysis_path = round_error_analysis_path(state.sample_dir, state.global_round)
		level_path = round_level_classification_path(state.sample_dir, state.global_round)
		error_analysis = dict(analysis)
		for classification_key in (
			"category",
			"recommended_mode",
			"route_head_decision",
			"route_prediction",
			"action_plan",
			"global_edit_instruction",
			"fix",
			"edits",
		):
			error_analysis.pop(classification_key, None)
		if ROUTE_MARKER in str(error_analysis.get("raw_output") or ""):
			error_analysis["raw_output"] = str(error_analysis["raw_output"]).split(ROUTE_MARKER, 1)[0].rstrip()
		save_json(analysis_path, error_analysis)
		if isinstance(analysis.get("action_plan"), dict):
			save_json(
				round_action_plan_path(state.sample_dir, state.global_round),
				{
					"action_plan": analysis["action_plan"],
					"diagnosis_path": str(analysis_path),
					"level_classification_path": str(level_path),
				},
			)
		save_json(
			level_path,
			{
				"category": analysis.get("category"),
				"recommended_mode": analysis.get("recommended_mode"),
				"fully_satisfied": bool(analysis.get("fully_satisfied")),
				"decision": level_decision,
				"error_analysis": str(analysis_path),
			},
		)
		append_event(state, {"event": "qwen_diagnosis", "tag": tag, "analysis": error_analysis})
		if level_decision.get("enabled"):
			append_event(state, {"event": "route_head_decision", "tag": tag, "decision": level_decision})
		if level_decision.get("fixed_level_planning_failed"):
			state.status = "partial"
			state.final_reason = "fixed_l1_planning_failed_after_json_retry"
			state.final_image = state.current_image
			state.needs_global_analysis = False
			append_event(state, {"event": "fixed_level_planning_failed", "tag": tag,
				"level": 1, "retained_image": str(state.current_image),
				"reason": level_decision["planning_failure"], "action_executed": False})
			save_status(state)
			continue
		update_best_round(state, state.current_image, analysis, tag)
		state.needs_global_analysis = False
		mode = analysis.get("recommended_mode")
		if analysis.get("fully_satisfied") or mode == "done":
			save_json(
				sample_round_dir(state.sample_dir, state.global_round) / "keep_decision.json",
				{"decision": "retain_initial", "retained": True, "reason": "Initial image for this round is satisfied; no repair was needed.", "image": str(state.current_image)},
			)
			state.status = "ok"
			state.final_reason = f"{language_model_label(args).lower()}_satisfied"
			state.final_image = state.current_image
			save_status(state)
			continue
		if int(state.global_round) >= int(args.max_global_rounds):
			state.status = "partial"
			state.final_reason = "max_repair_actions_reached_after_final_diagnosis"
			state.final_image = state.current_image
			append_event(
				state,
				{
					"event": "repair_budget_exhausted",
					"actions_used": int(state.global_round),
					"max_actions": int(args.max_global_rounds),
					"final_diagnosis": analysis.get("canonical_diagnosis"),
				},
			)
			save_status(state)
			continue
		if mode == "micro":
			_enter_micro_from_analysis(state, analysis, f"round_{state.global_round:02d}_triage", "global_triage")
		elif mode == "regenerate" and bool(getattr(args, "enable_regenerate_level", False)):
			_enter_regenerate_from_analysis(state, analysis, f"round_{state.global_round:02d}_triage", "global_triage")
		else:
			state.phase = "global"
			append_event(state, {"event": "stay_global", "round": state.global_round, "analysis": analysis})
		save_status(state)


def _build_micro_box_review_job(state: SampleState, args: argparse.Namespace, review_round: int) -> Dict[str, Any]:
	"""Build a legacy optional bbox-review request with contrast-aware overlays."""
	assert state.current_image is not None
	image = Image.open(state.current_image).convert("RGB")
	width, height = image.size
	tag = f"bbox_review_{review_round:02d}"
	round_dir = sample_round_dir(state.sample_dir, state.global_round)
	round_dir.mkdir(parents=True, exist_ok=True)
	boxed_image, box_color_records = draw_dynamic_contrast_boxes(image, state.micro_plan)
	boxed_image_path = round_dir / f"{tag}_boxed.png"
	boxed_image.save(boxed_image_path)
	source_qwen_image = annotate_image_with_ticks_32(boxed_image)
	full_qwen_image = prepare_qwen_image(source_qwen_image, max_side=args.qwen_input_max_side)
	full_qwen_image = pad_image_to_multiple(full_qwen_image, multiple=32)
	qwen_input_path = round_dir / f"{tag}_input.png"
	full_qwen_image.save(qwen_input_path)
	user_text = BOX_REVIEW_USER_TEMPLATE.format(
		prompt=state.prompt.strip(),
		width=width,
		height=height,
		review_round=review_round,
		max_rounds=max(1, int(args.qwen_box_review_max_rounds)),
		edit_lines=build_box_review_edit_lines(state.micro_plan),
	)
	messages = [
		{"role": "system", "content": [{"type": "text", "text": BOX_REVIEW_SYSTEM_PROMPT}]},
		{
			"role": "user",
			"content": [
				{"type": "text", "text": "Full image with dynamically contrasting labeled boxes and coordinate ruler:"},
				{"type": "image", "image": full_qwen_image},
				{"type": "text", "text": user_text},
			],
		},
	]
	return {
		"state": state,
		"qwen_image": full_qwen_image,
		"source_qwen_image": source_qwen_image.copy(),
		"current_qwen_max_side": max(full_qwen_image.size),
		"width": width,
		"height": height,
		"messages": messages,
		"images": [full_qwen_image],
		"qwen_input_path": qwen_input_path,
		"qwen_boxed_original_path": boxed_image_path,
		"qwen_box_colors": box_color_records,
		"review_round": review_round,
		"batch_kind": "micro-box-review",
		"log_label": (
			f"sample={state.sample_id} step=review-boxes model=Qwen mode=micro "
			f"round={state.global_round} review_round={review_round}/{args.qwen_box_review_max_rounds}"
		),
		"retry_log_label": (
			f"sample={state.sample_id} step=review-boxes-retry model=Qwen mode=micro "
			f"round={state.global_round} review_round={review_round}"
		),
	}


def run_micro_box_review_batch(runtime: HeavyRuntimeManager, args: argparse.Namespace, states: Sequence[SampleState]) -> None:
	"""Review/rebuild L1 boxes when explicitly enabled by a legacy experiment.

	Production V10 sets the maximum to zero because an additional localization
	call would change the atomic L1 policy measured by router-data collection.
	"""
	max_rounds = max(0, int(args.qwen_box_review_max_rounds))
	review_states = [
		state
		for state in states
		if state.phase == "micro"
		and state.current_image
		and state.micro_plan
		and not state.micro_box_review_done
		and not state.pending_local_review
		and not state.done
		and not state.error
	]
	if not review_states:
		return
	if max_rounds <= 0:
		for state in review_states:
			state.micro_box_review_done = True
			append_event(state, {"event": "micro_box_review_skipped", "round": state.global_round, "reason": "max_rounds_zero"})
			save_status(state)
		return
	log_stage(
		f"dispatch step=review-boxes model={language_model_label(args)} count={len(review_states)} samples={describe_states_brief(review_states)}"
	)
	active_states = list(review_states)
	for review_round in range(1, max_rounds + 1):
		prepared = [_build_micro_box_review_job(state, args, review_round) for state in active_states]
		results = run_language_batch_with_retries(
			runtime,
			args,
			prepared,
			args.qwen_compare_max_new_tokens,
			lambda raw, item: normalize_box_review(extract_json(raw), raw, item["state"].micro_plan, item["width"], item["height"]),
			(
				"Your previous response was invalid. Return exactly one JSON object with satisfied, reason, "
				"and one edits entry per proposed edit, each with valid axis-aligned quad_points in the original image coordinate system."
			),
			args.qwen_batch_size,
		)
		next_active: List[SampleState] = []
		for item, review in zip(prepared, results):
			state: SampleState = item["state"]
			review.update(
				{
					"round": state.global_round,
					"review_round": review_round,
					"image": str(state.current_image),
					"qwen_input_image": str(item["qwen_input_path"]),
					"qwen_boxed_original": str(item["qwen_boxed_original_path"]),
					"qwen_box_colors": item.get("qwen_box_colors") or [],
				}
			)
			round_dir = sample_round_dir(state.sample_dir, state.global_round)
			save_json(round_dir / f"bbox_review_{review_round:02d}.json", review)
			state.micro_plan = review.get("edits") or state.micro_plan
			plan_path = round_action_plan_path(state.sample_dir, state.global_round)
			existing_plan_payload: Dict[str, Any] = {}
			if plan_path.exists():
				try:
					existing_plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
				except (OSError, json.JSONDecodeError):
					existing_plan_payload = {}
			old_action_plan = existing_plan_payload.get("action_plan") or {}
			rebuilt_action_plan = build_level_action_plan(
				1,
				state.prompt,
				old_action_plan.get("diagnosis") or {},
				edits=state.micro_plan,
				width=item["width"],
				height=item["height"],
			)
			validate_distribution_contract(rebuilt_action_plan)
			save_json(
				plan_path,
				{
					**existing_plan_payload,
					"action_plan": rebuilt_action_plan,
					"plan": state.micro_plan,
					"box_review": review,
				},
			)
			append_event(state, {"event": "micro_box_review", "round": state.global_round, "review_round": review_round, "review": review})
			if review.get("satisfied") or review_round >= max_rounds:
				state.micro_box_review_done = True
				save_json(
					round_dir / "bbox_review_final.json",
					{"review_rounds_used": review_round, "satisfied": bool(review.get("satisfied")), "plan": state.micro_plan, "last_review": review},
				)
			else:
				next_active.append(state)
			save_status(state)
		if not next_active:
			break
		active_states = next_active


def _prepare_global_compare_job(state: SampleState, compare_input_max_side: int) -> Dict[str, Any]:
	"""Prepare a deterministic blind original-versus-generated comparison."""
	pending = state.pending_global_compare
	assert pending is not None
	analysis = json.loads(Path(pending["analysis_path"]).read_text(encoding="utf-8"))
	generated = pending.get("generated_candidates")
	if not isinstance(generated, list):
		raise ValueError("A/B pending comparison must contain generated_candidates")
	if len(generated) != 1 or not isinstance(generated[0], dict):
		raise ValueError(
			"Current A/B comparison requires exactly one generated-image record, "
			f"got {len(generated)}"
		)
	candidate = generated[0]
	candidate_path = Path(str(candidate.get("path") or ""))
	if not candidate_path.exists():
		raise ValueError(f"Generated comparison image does not exist: {candidate_path}")
	role = str(candidate.get("role") or candidate.get("tool") or "repair").strip() or "repair"
	if role == "original":
		raise ValueError("Generated image role must differ from the original-image role")
	candidates, label_to_role, role_to_label, role_to_path = _blind_ab_assignment(
		state,
		int(pending["round"]),
		"global_compare",
		(
			("original", Path(pending["original"])),
			(role, candidate_path),
		),
	)
	available_winners = [label for label, _path in candidates]
	user_text = GLOBAL_COMPARE_USER_TEMPLATE.format(
		prompt=state.prompt.strip(),
		issue=analysis.get("issue") or diagnosis_issue_summary(analysis.get("canonical_diagnosis") or analysis),
		fix=diagnosis_issue_summary(analysis.get("canonical_diagnosis") or analysis),
		candidate_labels="/".join(available_winners),
	)
	side = min(QWEN_MULTI_IMAGE_INPUT_MAX_SIDE, max(1, int(compare_input_max_side)))

	def build_messages(max_side: int) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
		"""Rebuild both blind candidates at one shared OOM-retry side."""
		images: List[Image.Image] = []
		content: List[Dict[str, Any]] = []
		for label, path in candidates:
			with Image.open(path) as loaded:
				annotated = annotate_image_with_ticks_32(loaded.convert("RGB"))
				qwen_image = prepare_qwen_image(annotated, max_side=max_side)
				qwen_image = pad_image_to_multiple(qwen_image, multiple=32)
			images.append(qwen_image)
			content.append({"type": "text", "text": f"{label}:"})
			content.append({"type": "image", "image": qwen_image})
		content.append({"type": "text", "text": user_text})
		return [
			{"role": "system", "content": [{"type": "text", "text": GLOBAL_COMPARE_SYSTEM_PROMPT}]},
			{"role": "user", "content": content},
		], images

	messages, images = build_messages(side)
	return {
		"state": state,
		"qwen_image": None,
		"current_qwen_max_side": max((max(image.size) for image in images), default=side),
		"messages": messages,
		"images": images,
		"qwen_messages_builder": build_messages,
		"qwen_side_ladder": qwen_multi_image_ladder(side),
		"batch_kind": "global-compare",
		"available_winners": available_winners,
		"label_to_role": label_to_role,
		"role_to_label": role_to_label,
		"role_to_path": {role: str(path) for role, path in role_to_path.items()},
		"log_label": f"sample={state.sample_id} step=compare-global model=Qwen mode=global round={pending['round']} images={'/'.join(available_winners)} max_side={side}",
		"retry_log_label": f"sample={state.sample_id} step=compare-global-retry model=Qwen mode=global round={pending['round']}",
	}


def run_global_compare_batch(runtime: HeavyRuntimeManager, args: argparse.Namespace, states: Sequence[SampleState]) -> None:
	"""Select one whole-image candidate and schedule a fresh diagnosis.

	Malformed/unavailable winners fall back to the original.  The reviewer may
	report satisfaction, but only the subsequent canonical diagnosis can produce
	L0 and finalize the trajectory.
	"""
	compare_states = [state for state in states if state.pending_global_compare and not state.done and not state.error]
	if not compare_states:
		return
	log_stage(
		f"dispatch step=compare-global model={language_model_label(args)} count={len(compare_states)} samples={describe_states_brief(compare_states)}"
	)
	prepared = [_prepare_global_compare_job(state, args.qwen_compare_input_max_side) for state in compare_states]
	results = run_language_batch_with_retries(
		runtime,
		args,
		prepared,
		args.qwen_compare_max_new_tokens,
		lambda raw, item: normalize_global_compare(
			extract_json(raw), raw, item.get("available_winners") or ("A", "B")
		),
		"Your previous response was invalid. Return exactly one JSON object with reason first and winner last, with winner set to one of the provided image labels and decision=satisfied/continue.",
		args.qwen_batch_size,
	)

	for item, review in zip(prepared, results):
		state: SampleState = item["state"]
		pending = state.pending_global_compare
		assert pending is not None
		round_dir = Path(pending["round_dir"])
		available_winners = set(item.get("available_winners") or [])
		label_to_role = item["label_to_role"]
		role_to_label = item["role_to_label"]
		original_label = role_to_label["original"]
		if not review.get("valid_json") or review.get("winner") not in available_winners:
			review["invalid_winner_not_available"] = review.get("winner")
			review["winner"] = original_label
			review["winner_role"] = "original"
			review["decision"] = "continue"
		else:
			review["winner_role"] = label_to_role[str(review["winner"])]
		role = review.get("winner_role", "original")
		role_to_path = item["role_to_path"]
		if role not in role_to_path:
			raise ValueError(f"Blind comparison winner role has no image path: {role!r}")
		chosen_path = Path(str(role_to_path[role]))
		chosen_copy = copy_image_generation_sized(chosen_path, round_dir / f"retained_{role}.png", args)
		review["retained_repair"] = role != "original"
		review["selected_image"] = str(chosen_copy)
		save_json(round_dir / "keep_decision.json", review)
		state.current_image = chosen_copy
		state.global_round = int(pending["round"])
		append_event(state, {"event": "global_compare", "round": state.global_round, "review": review, "selected_image": str(chosen_copy)})
		state.pending_global_compare = None
		# The A/B reviewer only chooses the retained state.  L0 is always decided
		# by a fresh diagnosis, so L1/L2/L3 share the same trajectory semantics.
		state.needs_global_analysis = True
		state.pending_global_analysis_reason = (
			f"global_compare_{review.get('decision', 'continue')}_selected_{role}"
		)
		save_status(state)


def _prepare_local_review_job(state: SampleState, args: argparse.Namespace) -> Dict[str, Any]:
	"""Prepare blind original-versus-single-L1-candidate review messages."""
	pending = state.pending_local_review
	assert pending is not None
	edits = pending.get("edits") or state.micro_plan
	edit_lines = "\n".join(
		f"- {edit.get('id', f'edit_{index}')}: target={edit.get('target_entity') or edit.get('target_object') or 'Unknown'}; "
		f"issue={edit.get('issue') or 'Unknown'}; instruction={edit.get('instruction') or edit.get('edit_prompt') or edit.get('whole_edit') or 'Unknown'}; "
		f"bbox={edit.get('box_xyxy') or edit.get('quad_points')}"
		for index, edit in enumerate(edits, start=1)
	)
	user_text = LOCAL_REVIEW_USER_TEMPLATE.format(
		prompt=state.prompt.strip(),
		edit_total=len(edits),
		edit_lines=edit_lines,
	)
	attempt_dir = Path(pending["attempt_dir"])
	input_path = Path(str(pending.get("input_path") or (attempt_dir / "input.png")))
	model_paths = pending.get("model_paths") or {}
	model_roles = [model_name for model_name in LOCAL_MODEL_ORDER if model_name in model_paths]
	if len(model_roles) != 1 or set(model_paths) != set(model_roles):
		raise ValueError(
			"Current A/B local review requires exactly one supported repair image, "
			f"got roles={sorted(str(role) for role in model_paths)}"
		)
	repair_role = model_roles[0]
	repair_path = Path(model_paths[repair_role])
	if not repair_path.exists():
		raise ValueError(f"Local comparison image does not exist: {repair_path}")
	candidates, label_to_role, role_to_label, _role_to_path = _blind_ab_assignment(
		state,
		int(state.global_round),
		"local_review",
		(
			("original", input_path),
			(repair_role, repair_path),
		),
	)
	available_winners = [label for label, _path in candidates]
	user_text += (
		f"\nAvailable candidate labels: {'/'.join(available_winners)}.\n"
		"Set winner to exactly one of these labels.\n"
	)
	side = min(QWEN_MULTI_IMAGE_INPUT_MAX_SIDE, max(1, int(args.qwen_review_input_max_side)))

	def build_messages(max_side: int) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
		"""Rebuild the labeled local-review pair at a shared image side."""
		return build_labeled_multi_image_messages(LOCAL_REVIEW_SYSTEM_PROMPT, user_text, candidates, max_side)

	messages, images = build_messages(side)
	log_label = (
		f"sample={state.sample_id} step=review-micro model=Qwen mode=micro "
		f"joint_edits={len(edits)} action=single "
		f"images={'/'.join(available_winners)} max_side={side}"
	)
	return {
		"state": state,
		"available_winners": available_winners,
		"label_to_role": label_to_role,
		"role_to_label": role_to_label,
		"expected_edit_ids": [str(edit.get("id") or f"edit_{index}") for index, edit in enumerate(edits, start=1)],
		"messages": messages,
		"qwen_image": None,
		"current_qwen_max_side": max((max(image.size) for image in images), default=side),
		"images": images,
		"qwen_messages_builder": build_messages,
		"qwen_side_ladder": qwen_multi_image_ladder(side),
		"log_label": log_label,
		"batch_kind": "micro-review",
		"retry_log_label": (
			f"sample={state.sample_id} step=review-micro-retry model=Qwen mode=micro "
			f"joint_edits={len(edits)} action=single"
		),
	}


def run_local_review_batch(runtime: HeavyRuntimeManager, args: argparse.Namespace, states: Sequence[SampleState]) -> None:
	"""Retain or roll back one L1 candidate, then request fresh global diagnosis.

	The preservation guard overrides a candidate winner when required content
	was damaged.  Review and selected-image artifacts are persisted before the
	pending hand-off is cleared.
	"""
	review_states = [state for state in states if state.pending_local_review and not state.done and not state.error]
	if not review_states:
		return
	log_stage(
		f"dispatch step=review-micro model={language_model_label(args)} count={len(review_states)} samples={describe_states_brief(review_states)}"
	)
	prepared = [_prepare_local_review_job(state, args) for state in review_states]
	results = run_language_batch_with_retries(
		runtime,
		args,
		prepared,
		args.qwen_compare_max_new_tokens,
		lambda raw, item: normalize_local_review_with_roles(
			extract_json(raw), raw, item["label_to_role"], item.get("expected_edit_ids") or []
		),
		(
			"Your previous response was invalid. Return exactly one JSON object with reason, one edit_results entry in the "
			"same order for every planned edit, preservation_note, blocking_prompt_violation, outside_regions_preserved, "
			"success, and a winner chosen from the provided labels."
		),
		args.qwen_batch_size,
	)

	for item, review in zip(prepared, results):
		state: SampleState = item["state"]
		pending = state.pending_local_review
		assert pending is not None
		attempt_dir = Path(pending["attempt_dir"])
		artifact_prefix = str(pending.get("artifact_prefix") or "repair")
		input_path = Path(str(pending.get("input_path") or (attempt_dir / "input.png")))
		available_winners = set(item.get("available_winners") or [])
		original_label = item["role_to_label"]["original"]
		if not review.get("valid_json") or review.get("winner") not in available_winners:
			review["invalid_winner_not_available"] = review.get("winner")
			review["winner"] = original_label
			review["winner_role"] = "original"
			review["success"] = False
		else:
			review["winner_role"] = item["label_to_role"][str(review["winner"])]
		role = str(review.get("winner_role") or "original")
		if role != "original" and (
			review.get("blocking_prompt_violation") or not review.get("outside_regions_preserved", False)
		):
			review["retention_guard_applied"] = True
			review["retention_guard_original_winner"] = review.get("winner")
			review["retention_guard_reason"] = "candidate_damaged_required_content"
			review["winner"] = original_label
			review["winner_role"] = "original"
			review["success"] = False
			role = "original"
		if role == "original":
			chosen_src = input_path
		else:
			chosen_src = Path(pending["model_paths"][role])
		chosen_copy = copy_image_generation_sized(chosen_src, attempt_dir / f"{artifact_prefix}_retained_{role}.png", args)
		review["retained_repair"] = role != "original"
		review["selected_image"] = str(chosen_copy)
		save_json(attempt_dir / f"{artifact_prefix}_keep_decision.json", review)
		append_event(
			state,
			{
				"event": "l1_joint_review",
				"round_before_action": state.global_round,
				"edits": pending.get("edits") or state.micro_plan,
				"review": review,
				"selected": str(chosen_copy),
			},
		)
		state.current_image = chosen_copy
		state.global_round += 1
		state.phase = "global"
		state.micro_plan = []
		state.micro_box_review_done = False
		state.micro_edit_index = 0
		state.micro_attempt = 0
		state.needs_global_analysis = True
		state.pending_global_analysis_reason = (
			"l1_joint_candidate_retained" if role != "original" else "l1_joint_rollback_to_original"
		)
		state.pending_local_review = None
		save_status(state)
