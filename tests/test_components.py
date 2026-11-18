from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest


def _diagnosis() -> dict:
	return {
		"fully_satisfied": False,
		"reason": "The car color is wrong.",
		"constraint_audit": {
			"objects": "The requested car is present.",
			"count": "One car is present as requested.",
			"attributes": "The car is blue rather than red.",
			"relations": "No relation mismatch.",
			"action_state": "The parked state is correct.",
			"scene": "The road scene is correct.",
		},
		"issues": [
			{
				"constraint_type": "attribute",
				"target_entity": "car",
				"observed": "blue body paint",
				"expected": "red body paint",
				"evidence": "the visible body panels are blue",
				"confidence": 0.98,
			}
		],
		"preservation_constraints": ["one parked car", "road background"],
	}


def test_canonical_diagnosis_removes_raw_action_and_route_leakage() -> None:
	from inference.main_method.modules.contracts import (
		DIAGNOSIS_SCHEMA,
		canonical_diagnosis_text,
		model_visible_diagnosis_text,
	)

	payload = {
		**_diagnosis(),
		"raw_output": "private model chatter",
		"category": "L1",
		"route": "local",
		"fix": "make it red",
		"global_edit_instruction": "edit it",
		"edits": [{"quad_points": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
	}
	first = canonical_diagnosis_text(payload)
	second = canonical_diagnosis_text(dict(reversed(list(payload.items()))))
	assert first == second
	canonical = json.loads(first)
	assert canonical["schema_version"] == DIAGNOSIS_SCHEMA
	for forbidden in ("raw_output", "category", "route", "fix", "global_edit_instruction", "edits", "quad_points"):
		assert forbidden not in first
	visible = model_visible_diagnosis_text(payload)
	assert "schema_version" not in json.loads(visible)
	assert "vcot" not in visible.casefold()


def test_online_diagnosis_and_l1_planner_use_shared_prompt_contracts() -> None:
	from inference.main_method.modules import config, contracts

	assert config.TRIAGE_SYSTEM_PROMPT == contracts.TRIAGE_SYSTEM_PROMPT_V10
	assert config.TRIAGE_USER_TEMPLATE == contracts.TRIAGE_USER_TEMPLATE_V10
	assert config.L1_PLAN_SYSTEM_PROMPT == contracts.L1_PLAN_SYSTEM_PROMPT_V10
	assert config.L1_PLAN_USER_TEMPLATE == contracts.L1_PLAN_USER_TEMPLATE_V10


def test_online_l1_planner_hides_internal_diagnosis_metadata() -> None:
	from inference.main_method.modules.contracts import canonicalize_diagnosis
	from inference.main_method.modules.qwen_stages import _build_l1_plan_job

	image = Image.new("RGB", (32, 32), "white")
	state = SimpleNamespace(sample_id="sample", prompt="a red car", global_round=0)
	job = _build_l1_plan_job(
		{
			"state": state,
			"width": 32,
			"height": 32,
			"l1_plan_qwen_image": image,
			"l1_plan_source_image": image,
		},
		{"canonical_diagnosis": canonicalize_diagnosis(_diagnosis())},
		SimpleNamespace(language_backend="qwen"),
	)
	visible_text = _chat_text(job["messages"])
	assert "schema_version" not in visible_text
	assert "vcot" not in visible_text.casefold()


def test_system_prompt_audit_matches_runtime_positive_role_contracts() -> None:
	from inference.main_method.modules import config
	from inference.main_method.modules.contracts import L1_PLAN_SYSTEM_PROMPT_V10, TRIAGE_SYSTEM_PROMPT_V10
	from inference.main_method.modules.qwen_stages import (
		GLOBAL_COMPARE_SYSTEM_PROMPT,
		LOCAL_REVIEW_SYSTEM_PROMPT,
	)
	from inference.main_method.modules.route_head import ROUTE_ENCODER_SYSTEM_PROMPT_V10

	document = (Path(__file__).resolve().parents[1] / "inference" / "main_method" / "SYSTEM_PROMPTS.md").read_text(
		encoding="utf-8"
	)
	prompts = (
		config.PROMPT_REASONING_SYSTEM_PROMPT,
		TRIAGE_SYSTEM_PROMPT_V10,
		L1_PLAN_SYSTEM_PROMPT_V10,
		config.BOX_REVIEW_SYSTEM_PROMPT,
		GLOBAL_COMPARE_SYSTEM_PROMPT,
		LOCAL_REVIEW_SYSTEM_PROMPT,
		ROUTE_ENCODER_SYSTEM_PROMPT_V10,
	)
	for prompt in prompts:
		assert prompt.strip() in document
	combined = "\n".join(prompts).lower()
	for avoidance in (
		"not the repair-level router",
		"do not choose, recommend",
		"do not output a repair instruction",
		"do not reconsider or mention another repair level",
		"do not answer or add a route label",
	):
		assert avoidance not in combined
	for leaked_context in ("vs-cot", "vcot", "v10", "editor name", "action level", "a is the original"):
		assert leaked_context not in combined


def test_qwen_distribution_locks_oom_image_shrink_retry() -> None:
	from inference.main_method.modules.contracts import build_qwen_analyzer_l1_distribution_contract

	base = build_qwen_analyzer_l1_distribution_contract(
		model="/models/qwen/snapshots/" + "a" * 40,
		model_revision="a" * 40,
	)
	shrinking = build_qwen_analyzer_l1_distribution_contract(
		model="/models/qwen/snapshots/" + "a" * 40,
		model_revision="a" * 40,
		image_shrink_retry=True,
	)
	assert base["image_shrink_retry"] is False
	assert shrinking["image_shrink_retry"] is True
	assert base["distribution_signature"] != shrinking["distribution_signature"]


def test_qwen_multi_image_batch_keeps_every_conversation_image_in_order() -> None:
	from inference.main_method.modules.qwen_worker import _can_batch_items
	from inference.main_method.modules.vcot_branch import flatten_qwen_image_batches

	a0, a1 = Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")
	b0, b1 = Image.new("RGB", (8, 8), "green"), Image.new("RGB", (8, 8), "yellow")
	assert flatten_qwen_image_batches([[a0, a1], [b0, b1]]) == [a0, a1, b0, b1]
	assert _can_batch_items([{"image_paths": ["a0", "a1"]}, {"image_paths": ["b0", "b1"]}])
	assert not _can_batch_items([{"image_paths": ["a0"]}, {"image_paths": ["b0", "b1"]}])


def test_qwen_generate_batch_forwards_two_images_per_sample() -> None:
	import torch

	from inference.main_method.modules.vcot_branch import QwenPlanner

	class FakeProcessor:
		def __init__(self) -> None:
			self.images = None

		def apply_chat_template(self, _messages, **_kwargs):
			return "<image><image> review"

		def __call__(self, **kwargs):
			self.images = kwargs.get("images")
			return {"input_ids": torch.tensor([[1], [1]]), "attention_mask": torch.tensor([[1], [1]])}

		def batch_decode(self, _generated, **_kwargs):
			return ["first", "second"]

	class FakeModel:
		def __init__(self) -> None:
			self.kwargs = None

		def generate(self, **kwargs):
			self.kwargs = kwargs
			return torch.tensor([[1, 2], [1, 3]])

	planner = QwenPlanner.__new__(QwenPlanner)
	planner.processor = FakeProcessor()
	planner.model = FakeModel()
	planner.input_device = torch.device("cpu")
	images = [[Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")],
		[Image.new("RGB", (8, 8), "green"), Image.new("RGB", (8, 8), "yellow")]]
	outputs = planner._generate_batch([[{"role": "user", "content": []}]] * 2, images, 8)
	assert outputs == ["first", "second"]
	assert planner.processor.images == [images[0][0], images[0][1], images[1][0], images[1][1]]
	assert planner.model.kwargs["do_sample"] is False
	assert "repetition_penalty" not in planner.model.kwargs
	assert "no_repeat_ngram_size" not in planner.model.kwargs
	planner._generate_batch_retry([[{"role": "user", "content": []}]] * 2, images, 8)
	assert planner.model.kwargs["repetition_penalty"] == 1.05
	assert planner.model.kwargs["no_repeat_ngram_size"] == 8


def test_native_diagnosis_with_action_or_route_fields_is_retried() -> None:
	from inference.main_method.modules.contracts import diagnosis_payload_is_valid

	assert diagnosis_payload_is_valid(_diagnosis())
	for leaked_key, value in (
		("route", "L1"),
		("level", 1),
		("fix", "make it red"),
		("edits", []),
		("quad_points", []),
	):
		assert not diagnosis_payload_is_valid({**_diagnosis(), leaked_key: value})


def test_route_encoder_rejects_raw_reflection_text() -> None:
	from inference.main_method.modules.contracts import canonical_diagnosis_text, model_visible_diagnosis_text
	from inference.main_method.modules.route_head import build_route_encoder_messages

	image = Image.new("RGB", (32, 32), "white")
	try:
		build_route_encoder_messages("a red car", "I think the car is blue", image)
	except ValueError as exc:
		assert "canonical diagnosis JSON" in str(exc)
	else:  # pragma: no cover - defensive
		raise AssertionError("raw diagnosis text was accepted")
	messages = build_route_encoder_messages("a red car", canonical_diagnosis_text(_diagnosis()), image)
	assistant_text = messages[-1]["content"][0]["text"]
	assert assistant_text == model_visible_diagnosis_text(_diagnosis())
	assert "schema_version" not in json.loads(assistant_text)
	assert "vcot" not in assistant_text.casefold()


def test_qwen_routes_only_after_valid_json_retry_and_uses_canonical_text() -> None:
	from inference.main_method.modules.parsing import extract_json, normalize_triage_analysis
	from inference.main_method.modules.qwen_batch import run_qwen_batch
	from inference.main_method.modules.route_head import extract_route_marker

	class FakePlanner:
		route_head = object()

		def __init__(self) -> None:
			self.route_diagnoses = []
			self.route_images = []
			self.generate_calls = 0

		def _generate(self, _messages, _images, _max_new_tokens):
			self.generate_calls += 1
			if self.generate_calls == 1:
				return "not json and must never reach the router"
			return json.dumps(_diagnosis())

		def predict_routes(self, _prompts, diagnoses, images):
			self.route_diagnoses.extend(diagnoses)
			self.route_images.extend(images)
			return [{"level": 1, "route": "L1_local_flux2_full_plus_bbox_crop", "confidence": 0.8}]

	def validator(raw, _item):
		return normalize_triage_analysis(extract_json(raw), raw, 32, 32)

	planner = FakePlanner()
	route_image = Image.new("RGB", (48, 48), "red")
	item = {
		"messages": [],
		"images": [Image.new("RGB", (32, 32), "white")],
		"route_image": route_image,
		"route_prompt": "a red car",
		"batch_kind": "global-triage",
	}
	result = run_qwen_batch(planner, [item], 32, validator, "retry", 1)[0]
	assert len(planner.route_diagnoses) == 1
	encoded = json.loads(planner.route_diagnoses[0])
	assert encoded["issues"][0]["expected"] == "red body paint"
	assert "schema_version" not in encoded
	assert "vcot" not in planner.route_diagnoses[0].casefold()
	assert "not json" not in planner.route_diagnoses[0]
	assert planner.route_images == [route_image]
	assert extract_route_marker(result["raw_output"])["level"] == 1
	assert result["json_retry_count"] == 1


def test_global_compare_requires_complete_typed_contract() -> None:
	from inference.main_method.modules.parsing import normalize_global_compare

	for malformed in (
		{"winner": "B"},
		{"winner": "D", "decision": "continue", "reason": "x", "semantic_status": "x", "constraint_audit": "x"},
		{"winner": "B", "decision": True, "reason": "x", "semantic_status": "x", "constraint_audit": "x"},
		{"winner": "B", "decision": "continue", "reason": "", "semantic_status": "x", "constraint_audit": "x"},
	):
		review = normalize_global_compare(malformed, json.dumps(malformed), ("A", "B"))
		assert review["valid_json"] is False
		assert "winner_role" not in review

	valid = normalize_global_compare(
		{
			"winner": "B",
			"decision": "continue",
			"reason": "The repair is closer.",
			"semantic_status": "The color is fixed but one relation remains.",
			"constraint_audit": "Objects and counts are preserved.",
		},
		"{}",
		("A", "B"),
	)
	assert valid["valid_json"] is True
	assert valid["winner"] == "B"
	assert "winner_role" not in valid


def _chat_text(messages: list[dict]) -> str:
	return "\n".join(
		str(part.get("text") or "")
		for message in messages
		for part in message.get("content", [])
		if isinstance(part, dict) and part.get("type") == "text"
	)


def test_global_compare_blinds_source_and_stably_permutates_ab(tmp_path) -> None:
	from inference.main_method.modules.qwen_stages import _blind_ab_assignment, _prepare_global_compare_job

	original = tmp_path / "original.png"
	repair = tmp_path / "repair.png"
	Image.new("RGB", (32, 32), "white").save(original)
	Image.new("RGB", (32, 32), "blue").save(repair)
	analysis_path = tmp_path / "analysis.json"
	analysis_path.write_text(
		json.dumps({"issue": "wrong color", "canonical_diagnosis": _diagnosis()}),
		encoding="utf-8",
	)
	state = SimpleNamespace(
		sample_id="blind-global",
		prompt="a red car",
		pending_global_compare={
			"round": 1,
			"round_dir": str(tmp_path),
			"analysis_path": str(analysis_path),
			"original": str(original),
			"generated_candidates": [
				{"role": "flux2", "tool": "flux2_edit", "path": str(repair)}
			],
		},
	)
	first = _prepare_global_compare_job(state, 64)
	second = _prepare_global_compare_job(state, 64)
	assert first["label_to_role"] == second["label_to_role"]
	assert set(first["label_to_role"]) == {"A", "B"}
	assert set(first["label_to_role"].values()) == {"original", "flux2"}
	user_labels = [
		part["text"]
		for part in first["messages"][1]["content"][:-1]
		if part.get("type") == "text"
	]
	assert user_labels == ["A:", "B:"]
	visible_text = _chat_text(first["messages"]).lower()
	for leak in ("original", "flux2", "bagel", "regenerate", "editor", "action level"):
		assert leak not in visible_text

	positions = set()
	for index in range(64):
		probe = SimpleNamespace(sample_id=f"probe-{index}")
		_assignment, _labels, roles, _paths = _blind_ab_assignment(
			probe,
			1,
			"global_compare",
			(("original", original), ("repair", repair)),
		)
		positions.add(roles["original"])
	assert positions == {"A", "B"}


def test_local_review_blinds_source_and_uses_internal_mapping(tmp_path) -> None:
	from inference.main_method.modules.qwen_stages import _prepare_local_review_job

	original = tmp_path / "original.png"
	repair = tmp_path / "repair.png"
	Image.new("RGB", (32, 32), "white").save(original)
	Image.new("RGB", (32, 32), "blue").save(repair)
	edit = {
		"id": "edit_1",
		"target_entity": "car",
		"issue": "wrong color",
		"instruction": "make the car red",
		"quad_points": [[1, 1], [20, 1], [20, 20], [1, 20]],
	}
	state = SimpleNamespace(
		sample_id="blind-local",
		prompt="a red car",
		global_round=1,
		micro_plan=[edit],
		pending_local_review={
			"attempt_dir": str(tmp_path),
			"input_path": str(original),
			"edits": [edit],
			"model_paths": {"flux2": str(repair)},
		},
	)
	job = _prepare_local_review_job(state, SimpleNamespace(qwen_review_input_max_side=64))
	assert set(job["label_to_role"].values()) == {"original", "flux2"}
	assert job["role_to_label"]["original"] != job["role_to_label"]["flux2"]
	user_labels = [
		part["text"]
		for part in job["messages"][1]["content"][:-1]
		if part.get("type") == "text"
	]
	assert user_labels == ["A:", "B:"]
	visible_text = _chat_text(job["messages"]).lower()
	for leak in ("original", "flux2", "bagel", "regenerate", "editor", "action level", " l1 "):
		assert leak not in visible_text


def test_l1_route_requires_post_route_plan_not_diagnosis_bbox() -> None:
	from inference.main_method.modules.route_head import append_route_marker, apply_route_head_decision

	analysis = {
		"raw_output": append_route_marker(json.dumps(_diagnosis()), {"level": 1, "confidence": 0.9}),
		"fully_satisfied": False,
		"recommended_mode": "unrouted",
		"category": "Unrouted",
		"edits": [],
	}
	decision = apply_route_head_decision(SimpleNamespace(phase="global"), analysis)
	assert decision["enforced"] is True
	assert decision["planning_required"] == "l1_localization_plan"
	assert analysis["recommended_mode"] == "micro"
	assert analysis["edits"] == []


def test_l1_plan_accepts_three_regions_and_normalizes_whole_image_fallback() -> None:
	from inference.main_method.modules.parsing import normalize_l1_plan

	edits = [
		{
			"id": f"edit_{index}",
			"target_entity": f"object {index}",
			"issue": "wrong attribute",
			"instruction": "correct the attribute",
			"quad_points": [[x, 10], [x + 40, 10], [x + 40, 50], [x, 50]],
		}
		for index, x in enumerate((10, 70, 130), start=1)
	]
	plan = normalize_l1_plan({"reason": "three regions", "edits": edits}, "{}", _diagnosis(), "prompt", 256, 256)
	assert plan["valid_json"] is True
	assert plan["candidate_count"] == plan["editor_call_count"] == 1
	assert len(plan["edits"]) == 3
	whole = dict(edits[0], quad_points=[[0, 0], [255, 0], [255, 255], [0, 255]])
	fallback = normalize_l1_plan({"reason": "global counterfactual", "edits": [whole]}, "{}", _diagnosis(), "prompt", 256, 256)
	assert fallback["valid_json"] is True
	assert fallback["edits"][0]["box_xyxy"] == [13, 13, 243, 243]
	assert fallback["edits"][0]["bbox_normalization"] == "centered_90pct_span_fallback_from_whole_image_v1"
	local_plus_whole = normalize_l1_plan(
		{"reason": "discard global item", "edits": [edits[0], whole]},
		"{}",
		_diagnosis(),
		"prompt",
		256,
		256,
	)
	assert local_plus_whole["valid_json"] is True
	assert len(local_plus_whole["edits"]) == 1
	assert local_plus_whole["edits"][0]["id"] == "edit_1"
	shorthand = dict(edits[0], quad_points=[[10, 10, 50, 50]])
	canonicalized = normalize_l1_plan(
		{"reason": "unambiguous xyxy shorthand", "edits": [shorthand]},
		"{}",
		_diagnosis(),
		"prompt",
		256,
		256,
	)
	assert canonicalized["valid_json"] is True
	assert canonicalized["edits"][0]["quad_points"] == [[10, 10], [50, 10], [50, 50], [10, 50]]
	two_corner = dict(edits[0], quad_points=[[10, 10], [50, 50]])
	two_corner_plan = normalize_l1_plan(
		{"reason": "two diagonal corners", "edits": [two_corner]},
		"{}",
		_diagnosis(),
		"prompt",
		256,
		256,
	)
	assert two_corner_plan["valid_json"] is True
	assert two_corner_plan["edits"][0]["quad_points"] == [[10, 10], [50, 10], [50, 50], [10, 50]]
	overlapping = [
		dict(edits[0], id="edit_1", quad_points=[[10, 10], [80, 10], [80, 80], [10, 80]]),
		dict(edits[1], id="edit_2", quad_points=[[10, 10], [80, 10], [80, 80], [10, 80]]),
	]
	merged_overlap = normalize_l1_plan(
		{"reason": "duplicate crop", "edits": overlapping},
		"{}",
		_diagnosis(),
		"prompt",
		256,
		256,
	)
	assert merged_overlap["valid_json"] is True
	assert len(merged_overlap["edits"]) == 1
	assert merged_overlap["edits"][0]["merged_edit_ids"] == ["edit_1", "edit_2"]
	assert "Apply both localized repairs" in merged_overlap["edits"][0]["instruction"]


def test_every_route_action_instruction_contains_the_target_prompt() -> None:
	from inference.main_method.modules.contracts import build_level_action_plan

	target = "a red car parked to the left of a blue bicycle"
	edit = {
		"id": "edit_1",
		"target_entity": "car",
		"issue": "wrong color",
		"instruction": "make it red",
		"quad_points": [[8, 8], [48, 8], [48, 48], [8, 48]],
	}
	for level in (1, 2, 3):
		plan = build_level_action_plan(
			level,
			target,
			_diagnosis(),
			edits=[edit] if level == 1 else None,
			width=64,
			height=64,
			initial_generator_family="flux2" if level == 3 else None,
		)
		assert target in plan["instruction"]


def test_v9_route_checkpoint_schema_is_rejected_before_deployment(tmp_path) -> None:
	import torch

	from inference.main_method.modules.route_head import load_route_head

	path = tmp_path / "old.pt"
	torch.save(
		{
			"config": {
				"input_dim": 8,
				"hidden_dim": 4,
				"num_labels": 3,
				"label_levels": [1, 2, 3],
				"diagnosis_schema": "vcot_v9_reflection_only",
			},
			"state_dict": {},
		},
		path,
	)
	try:
		load_route_head(path, "cpu")
	except ValueError as exc:
		assert "schema mismatch" in str(exc)
	else:  # pragma: no cover - defensive
		raise AssertionError("V9 checkpoint was accepted by V10")


def test_v10_route_checkpoint_requires_exact_feature_contract(tmp_path) -> None:
	import copy

	import torch

	from inference.main_method.modules.route_head import (
		DIAGNOSIS_AUX_FEATURE_NAMES,
		ROUTE_CANDIDATE_QUALITY_SCHEMA,
		ROUTE_CHECKPOINT_VERSION,
		ROUTE_DIAGNOSIS_GROUP_EB_SCHEMA,
		RouteMLP,
		build_diagnosis_aux_contract,
		load_route_head,
	)
	from inference.main_method.modules.flux_policy import build_flux2_policy_contract, flux2_policy_digest
	from inference.main_method.modules.l3_policy import L3_PROMPT_VERSION, build_l3_policy_contract, l3_policy_digest
	from inference.main_method.modules.contracts import ROUTER_ACTION_POLICY_VERSION, ROUTER_REPAIR_EXECUTOR_FAMILY
	from scripts import train_route_head as trainer

	contract = trainer._static_test_feature_contract()
	flux2_policy = build_flux2_policy_contract(
		model_id="black-forest-labs/FLUX.2-dev",
		model_revision="c" * 40,
		resolution=1024,
		steps=28,
		guidance_scale=3.5,
		max_sequence_length=512,
		dtype="bfloat16",
		l1_roi_pad_ratio=1.35,
		l1_min_span=64,
		l1_max_crops=3,
		prompt_version="vcot_v10_flux2_joint_atomic_v2",
	)
	l3_policy = build_l3_policy_contract(
		family="flux2",
		model_id="black-forest-labs/FLUX.2-dev",
		model_revision="c" * 40,
		resolution=1024,
		steps=28,
		guidance_scale=3.5,
		max_sequence_length=512,
		dtype="bfloat16",
		prompt_version=L3_PROMPT_VERSION,
	)
	aux_contract = build_diagnosis_aux_contract(
		[0.0] * len(DIAGNOSIS_AUX_FEATURE_NAMES),
		[1.0] * len(DIAGNOSIS_AUX_FEATURE_NAMES),
		std_floor=0.1,
	)
	config = {
		"version": ROUTE_CHECKPOINT_VERSION,
		"input_dim": 6 + len(DIAGNOSIS_AUX_FEATURE_NAMES),
		"qwen_feature_dim": 6,
		"hidden_dim": 4,
		"num_labels": 3,
		"dropout": 0.0,
		"label_levels": [1, 2, 3],
		"diagnosis_schema": contract["diagnosis_schema"],
		"action_policy_version": ROUTER_ACTION_POLICY_VERSION,
		"repair_executor_family": ROUTER_REPAIR_EXECUTOR_FAMILY,
		"flux2_policy_contract": flux2_policy,
		"flux2_policy_digest": flux2_policy_digest(flux2_policy),
		"l3_policy_contract": l3_policy,
		"l3_policy_digest": l3_policy_digest(l3_policy),
		"feature_schema": contract["feature_schema"],
		"feature_schema_version": contract["feature_schema_version"],
		"feature_contract_digest": contract["feature_contract_digest"],
		"processor_chat_template_preprocess_pooling_contract_digest": contract[
			"processor_chat_template_preprocess_pooling_contract_digest"
		],
		"processor_contract_digest": contract["processor_contract_digest"],
		"chat_template_contract_digest": contract["chat_template_contract_digest"],
		"preprocess_contract_digest": contract["preprocess_contract_digest"],
		"pooling_contract_digest": contract["pooling_contract_digest"],
		"qwen_model": contract["qwen_model"],
		"qwen_revision": contract["qwen_revision"],
		"qwen_dtype": contract["qwen_dtype"],
		"qwen_quantization": contract["qwen_quantization"],
		"qwen_allow_cpu_offload": contract["qwen_allow_cpu_offload"],
		"qwen_effective_quantization": contract["qwen_effective_quantization"],
		"qwen_distribution_signature": contract["qwen_distribution_signature"],
		"feature_contract": contract,
		"diagnosis_aux_features_enabled": True,
		"diagnosis_aux_contract": aux_contract,
		"diagnosis_aux_contract_digest": trainer._sha256_text(trainer._canonical_json(aux_contract)),
		"dataset_digest": "d" * 64,
		"target_digest": "e" * 64,
		"target_mode": "candidate_quality",
		"train_target_shrinkage": "diagnosis_group_eb_l2",
		"target_processing": {
			"target_mode": "candidate_quality",
			"train_target_shrinkage": "diagnosis_group_eb_l2",
			"candidate_quality": {"schema": ROUTE_CANDIDATE_QUALITY_SCHEMA},
			"diagnosis_group_eb": {"schema": ROUTE_DIAGNOSIS_GROUP_EB_SCHEMA},
		},
		"class_factors": [1.0, 1.0, 1.0],
		"exact_tie_quality_weight": 0.0,
		"loss": {
			"utility_temperature": 0.5,
			"pairwise_temperature": 0.5,
			"policy_temperature": 1.0,
			"quality_tie_epsilon": 0.0,
			"soft_ce_weight": 0.5,
			"utility_alignment_weight": 0.1,
			"pairwise_weight": 0.2,
			"expected_regret_weight": 0.0,
			"efficiency_weight": 0.001,
			"uncertain_safe_prior_weight": 0.0,
			"safe_action_index": 1,
		},
		"evaluated_splits": ["train", "validation"],
	}
	model = RouteMLP(input_dim=config["input_dim"], hidden_dim=4, num_labels=3, dropout=0.0)
	valid_path = tmp_path / "valid.pt"
	torch.save({"config": config, "state_dict": model.state_dict()}, valid_path)
	loaded, loaded_config = load_route_head(valid_path, "cpu")
	assert isinstance(loaded, RouteMLP)
	assert loaded_config["feature_contract_digest"] == contract["feature_contract_digest"]

	legacy_training_config = copy.deepcopy(config)
	legacy_training_config["version"] = "vcot_v10_qwen_route_head_v4_quality_first_aux"
	legacy_training_path = tmp_path / "legacy_training.pt"
	torch.save({"config": legacy_training_config, "state_dict": model.state_dict()}, legacy_training_path)
	with pytest.raises(ValueError, match="version is not the deployable"):
		load_route_head(legacy_training_path, "cpu")

	test_selected_config = copy.deepcopy(config)
	test_selected_config["evaluated_splits"] = ["train", "validation", "test"]
	test_selected_path = tmp_path / "test_selected.pt"
	torch.save({"config": test_selected_config, "state_dict": model.state_dict()}, test_selected_path)
	with pytest.raises(ValueError, match="must not be selected using its test split"):
		load_route_head(test_selected_path, "cpu")

	tampered_config = copy.deepcopy(config)
	tampered_config["feature_contract"]["components"]["preprocess"]["annotation"] = "ruler"
	tampered_path = tmp_path / "tampered.pt"
	torch.save({"config": tampered_config, "state_dict": model.state_dict()}, tampered_path)
	try:
		load_route_head(tampered_path, "cpu")
	except ValueError as exc:
		assert "preprocess contract digest mismatch" in str(exc)
	else:  # pragma: no cover - defensive
		raise AssertionError("Tampered V10 feature contract was accepted")

	tampered_flux_config = copy.deepcopy(config)
	tampered_flux_config["flux2_policy_contract"]["generation"]["steps"] = 29
	tampered_flux_path = tmp_path / "tampered_flux.pt"
	torch.save({"config": tampered_flux_config, "state_dict": model.state_dict()}, tampered_flux_path)
	try:
		load_route_head(tampered_flux_path, "cpu")
	except ValueError as exc:
		assert "collector-policy digest mismatch" in str(exc)
	else:  # pragma: no cover - defensive
		raise AssertionError("Tampered Flux.2 collector policy was accepted")


def test_v10_route_checkpoint_requires_exact_qwen_label_distribution(tmp_path) -> None:
	from inference.main_method.modules.contracts import build_qwen_analyzer_l1_distribution_contract
	from inference.main_method.modules.route_head import validate_route_qwen_distribution
	from scripts import train_route_head as trainer

	distribution = build_qwen_analyzer_l1_distribution_contract(
		model="Qwen/Qwen3.5-9B",
		model_revision="a" * 40,
		dtype="bfloat16",
		quantization="8bit",
		allow_cpu_offload=False,
	)
	contract = trainer._static_test_feature_contract(
		qwen_distribution_signature=distribution["distribution_signature"]
	)
	config = {key: contract[key] for key in (
		"feature_schema",
		"feature_schema_version",
		"diagnosis_schema",
		"label_levels",
		"qwen_model",
		"qwen_revision",
		"qwen_dtype",
		"qwen_quantization",
		"qwen_allow_cpu_offload",
		"qwen_effective_quantization",
		"qwen_distribution_signature",
		"feature_contract_digest",
		"processor_chat_template_preprocess_pooling_contract_digest",
		"processor_contract_digest",
		"chat_template_contract_digest",
		"preprocess_contract_digest",
		"pooling_contract_digest",
	)}
	config["feature_contract"] = contract
	assert validate_route_qwen_distribution(config, distribution)["distribution_signature"] == distribution["distribution_signature"]

	different = build_qwen_analyzer_l1_distribution_contract(
		model="/test/qwen",
		model_revision="a" * 40,
		dtype="bfloat16",
		quantization="4bit",
		allow_cpu_offload=True,
	)
	with pytest.raises(ValueError, match="differs from route-head training data"):
		validate_route_qwen_distribution(config, different)

	different_shrink_policy = build_qwen_analyzer_l1_distribution_contract(
		model="/test/qwen",
		model_revision="a" * 40,
		dtype="bfloat16",
		quantization="8bit",
		allow_cpu_offload=False,
		image_shrink_retry=True,
	)
	with pytest.raises(ValueError, match="differs from route-head training data"):
		validate_route_qwen_distribution(config, different_shrink_policy)


def test_route_model_identity_accepts_commit_named_symlink_but_rejects_mutable_target(require_symlink, 
	tmp_path, monkeypatch
) -> None:
	from inference.main_method.modules import route_head

	revision = "a" * 40
	materialized = tmp_path / "main"
	materialized.mkdir()
	snapshot = tmp_path / revision
	snapshot.symlink_to(materialized, target_is_directory=True)
	monkeypatch.setattr(
		route_head,
		"validate_route_feature_config",
		lambda _config: {"qwen_revision": revision},
	)
	route_head.validate_route_model_identity({}, str(snapshot))
	with pytest.raises(ValueError, match="pinned revision"):
		route_head.validate_route_model_identity({}, str(materialized))


def test_joint_l1_review_always_ends_action_and_reenters_diagnosis(tmp_path, monkeypatch) -> None:
	from inference.main_method.modules import qwen_stages
	from inference.main_method.modules.state import SampleState

	round_dir = tmp_path / "round0"
	round_dir.mkdir()
	original = round_dir / "repair_l1_joint_initial.png"
	candidate = round_dir / "repair_l1_joint_repaired_flux2.png"
	Image.new("RGB", (64, 64), "white").save(original)
	Image.new("RGB", (64, 64), "red").save(candidate)
	edit = {
		"id": "edit_1",
		"target_entity": "car",
		"issue": "blue rather than red",
		"instruction": "make the car red",
		"quad_points": [[8, 8], [48, 8], [48, 48], [8, 48]],
	}
	state = SampleState(index=0, prompt="a red car", sample_dir=tmp_path, sample_id="sample")
	state.current_image = original
	state.phase = "micro"
	state.micro_plan = [edit]
	state.pending_local_review = {
		"action_schema": "l1_joint_multicrop_v1",
		"action_seed": 7,
		"attempt_dir": str(round_dir),
		"artifact_prefix": "repair_l1_joint",
		"input_path": str(original),
		"edits": [edit],
		"edit": edit,
		"selected_models": ["flux2"],
		"model_paths": {"flux2": str(candidate)},
	}

	def fake_language_batch(_runtime, _args, items, *_rest, **_kwargs):
		winner = items[0]["role_to_label"]["flux2"]
		return [
			{
				"winner": winner,
				"winner_role": "flux2",
				"success": True,
				"blocking_prompt_violation": False,
				"outside_regions_preserved": True,
				"edit_results": [{"id": "edit_1", "satisfied": True, "note": "fixed"}],
				"reason": "fixed",
				"preservation_note": "preserved",
				"valid_json": True,
			}
		]

	monkeypatch.setattr(qwen_stages, "run_language_batch_with_retries", fake_language_batch)
	args = SimpleNamespace(
		qwen_review_input_max_side=64,
		qwen_compare_max_new_tokens=128,
		qwen_batch_size=1,
		resolution=64,
	)
	qwen_stages.run_local_review_batch(object(), args, [state])
	assert state.global_round == 1
	assert state.phase == "global"
	assert state.needs_global_analysis is True
	assert state.pending_local_review is None
	assert state.micro_plan == []
	assert state.pending_global_analysis_reason == "l1_joint_candidate_retained"
	# The final allowed action is followed by a fresh diagnosis; budget
	# normalization must not erase that transition or its provenance.
	from inference.main_method.modules.io_utils import normalize_v10_action_budget_state

	assert normalize_v10_action_budget_state(state, 1, source="test") is False
	assert state.needs_global_analysis is True
	assert state.pending_global_analysis_reason == "l1_joint_candidate_retained"


def test_joint_l1_review_rolls_back_candidate_with_blocking_or_outside_damage(tmp_path, monkeypatch) -> None:
	from inference.main_method.modules import qwen_stages
	from inference.main_method.modules.state import SampleState

	round_dir = tmp_path / "round0"
	round_dir.mkdir()
	original = round_dir / "repair_l1_joint_initial.png"
	candidate = round_dir / "repair_l1_joint_repaired_flux2.png"
	Image.new("RGB", (64, 64), "white").save(original)
	Image.new("RGB", (64, 64), "red").save(candidate)
	edit = {
		"id": "edit_1",
		"target_entity": "car",
		"issue": "blue rather than red",
		"instruction": "make the car red",
		"quad_points": [[8, 8], [48, 8], [48, 48], [8, 48]],
	}
	state = SampleState(index=0, prompt="a red car", sample_dir=tmp_path, sample_id="sample")
	state.current_image = original
	state.phase = "micro"
	state.micro_plan = [edit]
	state.pending_local_review = {
		"action_schema": "l1_joint_multicrop_v1",
		"action_seed": 7,
		"attempt_dir": str(round_dir),
		"artifact_prefix": "repair_l1_joint",
		"input_path": str(original),
		"edits": [edit],
		"edit": edit,
		"selected_models": ["flux2"],
		"model_paths": {"flux2": str(candidate)},
	}

	selected_labels = {}

	def fake_language_batch(_runtime, _args, items, *_rest, **_kwargs):
		winner = items[0]["role_to_label"]["flux2"]
		selected_labels["repair"] = winner
		return [
			{
				"winner": winner,
				"winner_role": "flux2",
				"success": False,
				"blocking_prompt_violation": True,
				"outside_regions_preserved": False,
				"edit_results": [{"id": "edit_1", "satisfied": False, "note": "damaged background"}],
				"reason": "The requested edit damaged required content.",
				"preservation_note": "Required background was not preserved.",
				"valid_json": True,
			}
		]

	monkeypatch.setattr(qwen_stages, "run_language_batch_with_retries", fake_language_batch)
	args = SimpleNamespace(
		qwen_review_input_max_side=64,
		qwen_compare_max_new_tokens=128,
		qwen_batch_size=1,
		resolution=64,
	)
	qwen_stages.run_local_review_batch(object(), args, [state])
	assert state.pending_global_analysis_reason == "l1_joint_rollback_to_original"
	assert Image.open(state.current_image).getpixel((0, 0)) == (255, 255, 255)
	review = json.loads((round_dir / "repair_l1_joint_keep_decision.json").read_text(encoding="utf-8"))
	assert review["winner_role"] == "original"
	assert review["retention_guard_applied"] is True
	assert review["retention_guard_original_winner"] == selected_labels["repair"]


def test_resume_micro_at_action_budget_is_migrated_and_never_calls_editor(tmp_path, monkeypatch) -> None:
	from inference.main_method.modules import local_benchmark
	from inference.main_method.modules.io_utils import load_states, save_status
	from inference.main_method.modules.state import SampleState, safe_prompt_id

	prompt = "a red car"
	sample_id = safe_prompt_id(0, prompt)
	sample_dir = tmp_path / sample_id
	sample_dir.mkdir()
	image_path = sample_dir / "current.png"
	Image.new("RGB", (32, 32), "white").save(image_path)
	state = SampleState(index=0, prompt=prompt, sample_dir=sample_dir, sample_id=sample_id)
	state.current_image = image_path
	state.phase = "micro"
	state.global_round = 1
	state.micro_plan = [
		{
			"id": "edit_1",
			"target_entity": "car",
			"issue": "wrong color",
			"instruction": "make the car red",
			"quad_points": [[1, 1], [20, 1], [20, 20], [1, 20]],
		}
	]
	save_status(state)

	loaded = load_states([(0, prompt)], tmp_path, resume=True, max_actions=1)[0]
	assert loaded.phase == "global"
	assert loaded.global_round == 1
	assert loaded.needs_global_analysis is True
	assert loaded.pending_global_analysis_reason == "action_budget_exhausted_final_diagnosis"
	assert loaded.micro_plan == []

	editor_calls = []
	monkeypatch.setattr(
		local_benchmark,
		"_run_local_edit_benchmark_batch_for_models",
		lambda *_args, **_kwargs: editor_calls.append(True),
	)
	local_benchmark.run_local_edit_benchmark_batch(SimpleNamespace(max_global_rounds=1), [loaded])
	assert editor_calls == []


def test_resume_legacy_micro_final_phase_becomes_global_diagnosis(tmp_path) -> None:
	from inference.main_method.modules.io_utils import load_states, save_status
	from inference.main_method.modules.state import SampleState, safe_prompt_id

	prompt = "a red car"
	sample_id = safe_prompt_id(0, prompt)
	sample_dir = tmp_path / sample_id
	sample_dir.mkdir()
	image_path = sample_dir / "current.png"
	Image.new("RGB", (32, 32), "white").save(image_path)
	state = SampleState(index=0, prompt=prompt, sample_dir=sample_dir, sample_id=sample_id)
	state.current_image = image_path
	state.phase = "micro_final_review"
	state.global_round = 0
	save_status(state)

	loaded = load_states([(0, prompt)], tmp_path, resume=True, max_actions=1)[0]
	assert loaded.phase == "global"
	assert loaded.needs_global_analysis is True
	assert loaded.pending_global_analysis_reason == "legacy_micro_final_review_migrated"
	assert json.loads((sample_dir / "status.json").read_text(encoding="utf-8"))["phase"] == "global"


def test_zero_action_budget_blocks_l1_l2_and_l3_dispatch(tmp_path, monkeypatch) -> None:
	from inference.main_method.modules import edit_stages, local_benchmark
	from inference.main_method.modules.state import SampleState

	image_path = tmp_path / "current.png"
	Image.new("RGB", (32, 32), "white").save(image_path)

	def make_state(name: str, phase: str) -> SampleState:
		state = SampleState(index=0, prompt="a red car", sample_dir=tmp_path / name, sample_id=name)
		state.sample_dir.mkdir()
		state.current_image = image_path
		state.phase = phase
		return state

	l1 = make_state("l1", "micro")
	l1.micro_plan = [
		{
			"id": "edit_1",
			"target_entity": "car",
			"issue": "wrong color",
			"instruction": "make the car red",
			"quad_points": [[1, 1], [20, 1], [20, 20], [1, 20]],
		}
	]
	l2 = make_state("l2", "global")
	l3 = make_state("l3", "regenerate")
	args = SimpleNamespace(max_global_rounds=0)
	calls = []
	monkeypatch.setattr(
		local_benchmark,
		"_run_local_edit_benchmark_batch_for_models",
		lambda *_args, **_kwargs: calls.append("l1"),
	)
	monkeypatch.setattr(edit_stages, "_prepare_global_edit_state", lambda *_args, **_kwargs: calls.append("l2"))
	monkeypatch.setattr(
		edit_stages,
		"resolve_parallel_service_gpu_indices",
		lambda *_args, **_kwargs: calls.append("l3"),
	)

	local_benchmark.run_local_edit_benchmark_batch(args, [l1])
	edit_stages.run_global_edit_generation(object(), args, [l2])
	edit_stages.run_regenerate_generation_batch(object(), args, [l3])
	assert calls == []


def test_l3_always_uses_the_same_model_family_as_the_initial_generator(tmp_path, monkeypatch) -> None:
	import shutil

	from inference.main_method.modules import edit_stages
	from inference.main_method.modules.state import SampleState

	sample_dir = tmp_path / "sample"
	round_dir = sample_dir / "round0"
	round_dir.mkdir(parents=True)
	image_path = sample_dir / "current.png"
	Image.new("RGB", (32, 32), "white").save(image_path)
	state = SampleState(index=2, prompt="a red car", sample_dir=sample_dir, sample_id="sample")
	state.current_image = image_path
	state.phase = "regenerate"
	state.needs_global_analysis = False
	(round_dir / "error_analysis.json").write_text(json.dumps(_diagnosis()), encoding="utf-8")
	(round_dir / "repair_plan.json").write_text(
		json.dumps(
			{
				"action_plan": {
					"schema_version": "vcot_v10_level_conditioned_plan_v1",
					"level": 3,
					"instruction": "Generate a red car from scratch.",
					"executor_family": "bagel",
				}
			}
		),
		encoding="utf-8",
	)

	class FakeBagelClient:
		def generate(self, _prompt, output_path, _args, *, seed=None):
			assert seed is not None
			shutil.copy2(image_path, output_path)

	monkeypatch.setattr(edit_stages, "release_qwen_hard", lambda *_args, **_kwargs: None)
	monkeypatch.setattr(edit_stages, "BagelClient", lambda *_args, **_kwargs: FakeBagelClient())
	monkeypatch.setattr(edit_stages, "resolve_parallel_service_gpu_indices", lambda *_args, **_kwargs: [0])

	def run_pool(states, _gpus, _label, create_client, run_one, **_kwargs):
		client = create_client(0)
		for item in states:
			run_one(client, item, 0)

	monkeypatch.setattr(edit_stages, "run_state_jobs_with_client_pool", run_pool)
	runtime = SimpleNamespace(release_bagel=lambda: None, release_flux=lambda: None)
	args = SimpleNamespace(
		initial_generator="bagel",
		max_global_rounds=1,
		bagel_gpu_indices="0",
		bagel_gpu_index=0,
		bagel_parallel_workers=1,
		edit_service_min_free_gib=0,
		edit_service_memory_wait_sec=0,
		edit_service_max_memory_retries=0,
		seed=7,
	)
	edit_stages.run_regenerate_generation_batch(runtime, args, [state])
	request = json.loads((round_dir / "repair_regenerate_prompt.json").read_text(encoding="utf-8"))
	assert request["generator"] == "bagel"
	assert request["model"] == "BAGEL"
	assert state.pending_global_compare["generated_candidates"][0]["tool"] == "bagel_generate"
	assert state.pending_global_compare["generated_candidates"][0]["path"].endswith(
		"repaired_regenerate_bagel.png"
	)


def test_final_diagnosis_at_budget_does_not_require_or_schedule_router(tmp_path, monkeypatch) -> None:
	from inference.main_method.modules import qwen_stages
	from inference.main_method.modules.parsing import normalize_triage_analysis
	from inference.main_method.modules.state import SampleState

	image_path = tmp_path / "current.png"
	Image.new("RGB", (32, 32), "white").save(image_path)
	state = SampleState(index=0, prompt="a red car", sample_dir=tmp_path, sample_id="sample")
	state.current_image = image_path
	state.global_round = 0
	state.needs_global_analysis = True
	args = SimpleNamespace(
		max_global_rounds=0,
		qwen_input_max_side=32,
		qwen_max_new_tokens=128,
		qwen_batch_size=1,
		fixed_repair_level=None,
		language_backend="qwen",
	)
	job = qwen_stages._build_triage_job(state, args)
	assert job["route_prompt"] == ""

	analysis = normalize_triage_analysis(_diagnosis(), json.dumps(_diagnosis()), 32, 32)
	monkeypatch.setattr(
		qwen_stages,
		"run_language_batch_with_retries",
		lambda *_args, **_kwargs: [analysis],
	)
	qwen_stages.run_global_triage_batch(object(), args, [state])
	assert state.status == "partial"
	assert state.final_reason == "max_repair_actions_reached_after_final_diagnosis"
	assert state.needs_global_analysis is False
	decision = json.loads((tmp_path / "round0" / "level_classification.json").read_text(encoding="utf-8"))["decision"]
	assert decision["source"] == "action_budget_gate"
	assert decision["route"] == "final_diagnosis_only"


def test_pending_local_review_requires_existing_input_and_candidate_files(tmp_path) -> None:
	from inference.main_method.modules.io_utils import _pending_local_review_complete

	input_path = tmp_path / "input.png"
	missing_candidate = tmp_path / "missing.png"
	Image.new("RGB", (8, 8), "white").save(input_path)
	payload = {
		"input_path": str(input_path),
		"selected_models": ["flux2"],
		"model_paths": {"flux2": str(missing_candidate)},
	}
	assert _pending_local_review_complete(payload) is False
	Image.new("RGB", (8, 8), "red").save(missing_candidate)
	assert _pending_local_review_complete(payload) is True


def test_pending_global_compare_requires_one_canonical_candidate_record(tmp_path) -> None:
	from inference.main_method.modules.io_utils import _pending_global_compare_complete

	analysis = tmp_path / "analysis.json"
	original = tmp_path / "original.png"
	candidate = tmp_path / "candidate.png"
	for path in (analysis, original, candidate):
		path.write_bytes(b"x")
	base = {
		"round": 1,
		"round_dir": str(tmp_path),
		"analysis_path": str(analysis),
		"original": str(original),
	}
	assert _pending_global_compare_complete(
		{**base, "generated_candidates": [{"role": "flux2", "path": str(candidate)}]}
	) is True
	assert _pending_global_compare_complete({**base, "flux2": str(candidate)}) is False
	assert _pending_global_compare_complete(
		{**base, "generated_candidates": [{"path": str(candidate)}, {"path": str(candidate)}]}
	) is False
