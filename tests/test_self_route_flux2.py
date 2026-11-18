import argparse
from pathlib import Path

from inference.self_route_version.modules.contracts import (
	ROUTER_ACTION_POLICY_VERSION,
	build_level_action_plan,
	validate_distribution_contract,
)
from inference.self_route_version.modules.qwen_stages import (
	_apply_bagel_route,
	_apply_qwen_route,
	_local_review_executor,
	_normalize_l2_plan,
	_normalize_reflective_plan,
)
from inference.self_route_version.modules.state import STATE_SCHEMA_VERSION, SampleState


def diagnosis() -> dict:
	return {
		"fully_satisfied": False,
		"reason": "The chameleon is green instead of brown.",
		"constraint_audit": {
			"objects": "one chameleon and one leaf",
			"count": "correct",
			"attributes": "green chameleon, brown leaf",
			"relations": "chameleon on leaf",
			"action_state": "resting",
			"scene": "close-up",
		},
		"issues": [
			{
				"constraint_type": "attribute",
				"target_entity": "chameleon",
				"observed": "bright green body",
				"expected": "brown body matching the leaf",
				"evidence": "green scales contrast with the leaf",
				"confidence": 0.99,
			}
		],
		"preservation_constraints": ["keep the leaf and pose"],
	}


def test_self_route_l1_l2_executor_is_flux2() -> None:
	for level in (1, 2):
		kwargs = {}
		if level == 1:
			kwargs["edits"] = [
				{
					"id": "edit_1",
					"target_entity": "chameleon",
					"instruction": "Change the chameleon from green to brown.",
					"quad_points": [[0, 0], [512, 0], [512, 512], [0, 512]],
				}
			]
		plan = build_level_action_plan(level, "a brown chameleon on a leaf", diagnosis(), **kwargs)
		validate_distribution_contract(plan)
		assert plan["executor_family"] == "flux2"
		assert plan["action_policy_version"] == ROUTER_ACTION_POLICY_VERSION


def test_bagel_self_route_binds_all_action_levels_to_bagel() -> None:
	for level in (1, 2, 3):
		kwargs = {"initial_generator_family": "bagel"}
		if level == 1:
			kwargs["edits"] = [
				{
					"id": "edit_1",
					"target_entity": "chameleon",
					"instruction": "Change the chameleon from green to brown.",
					"quad_points": [[0, 0], [512, 0], [512, 512], [0, 512]],
				}
			]
		plan = build_level_action_plan(level, "a brown chameleon on a leaf", diagnosis(), **kwargs)
		validate_distribution_contract(plan)
		assert plan["executor_family"] == "bagel"
		assert plan["initial_generator_family"] == "bagel"


def test_local_l1_executor_follows_initial_generator() -> None:
	from inference.self_route_version.modules.local_benchmark import local_model_order_for_edit

	assert local_model_order_for_edit({}, argparse.Namespace(initial_generator="flux")) == ["flux2"]
	assert local_model_order_for_edit({}, argparse.Namespace(initial_generator="bagel")) == ["bagel"]


def test_legacy_bagel_semantic_route_maps_to_pipeline_modes() -> None:
	for label, expected_level, expected_mode in (
		("L1", 1, "micro"),
		("L2", 2, "global"),
		("L3", 3, "regenerate"),
	):
		analysis = diagnosis()
		decision = _apply_bagel_route(
			analysis,
			{
				"level": label,
				"reason": "scope assessment",
				"original_value": "useful",
				"required_change": "global_edit",
			},
			f'{{"level":"{label}"}}',
		)
		assert decision["level"] == expected_level
		assert analysis["recommended_mode"] == expected_mode


def test_qwen_semantic_route_maps_to_pipeline_modes_without_changing_executor() -> None:
	for label, expected_level, expected_mode in (
		("L1", 1, "micro"),
		("L2", 2, "global"),
		("L3", 3, "regenerate"),
	):
		analysis = diagnosis()
		decision = _apply_qwen_route(
			analysis,
			{
				"level": label,
				"reason": "Qwen scope assessment",
				"original_value": "useful",
				"required_change": "global_edit",
			},
			f'{{"level":"{label}"}}',
		)
		assert decision["source"] == "qwen_semantic_router"
		assert decision["level"] == expected_level
		assert analysis["recommended_mode"] == expected_mode
		kwargs = {"initial_generator_family": "flux2"}
		if expected_level == 1:
			kwargs["edits"] = [
				{
					"id": "edit_1",
					"target_entity": "chameleon",
					"instruction": "Change the chameleon from green to brown.",
					"quad_points": [[0, 0], [512, 0], [512, 512], [0, 512]],
				}
			]
		plan = build_level_action_plan(
			expected_level,
			"a brown chameleon on a leaf",
			diagnosis(),
			**kwargs,
		)
		assert plan["executor_family"] == "flux2"


def test_reflective_l2_plan_replaces_abstract_template() -> None:
	state = SampleState(
		index=0,
		prompt="A chameleon perfectly camouflaged against a brown leaf",
		sample_dir=Path("/tmp/v11-test"),
		sample_id="sample",
		repair_memory=[
			{
				"instruction": "Make the chameleon blend with the leaf.",
				"execution_feedback": {"progress": "no_progress", "remaining_mismatch": "body remains green"},
			}
		],
	)
	item = {"state": state, "diagnosis": diagnosis()}
	plan = _normalize_l2_plan(
		{
			"reason": "The previous instruction was too abstract.",
			"progress_assessment": "no_progress",
			"instruction": "Change the chameleon's entire body from green to the same brown as the leaf. Do not preserve green skin.",
			"do_not_preserve": ["green skin"],
			"preserve": ["pose", "leaf", "background"],
		},
		"{}",
		item,
	)
	assert plan["valid_json"]
	assert "green to the same brown" in plan["instruction"]
	assert plan["reflection"]["memory_items_used"] == 1
	assert plan["executor_family"] == "flux2"


def test_self_route_state_schema_isolated_from_v10() -> None:
	assert STATE_SCHEMA_VERSION == 11


def test_summary_canonicalization_keeps_latest_row_per_index(tmp_path) -> None:
	import json

	from inference.self_route_version.modules.io_utils import canonicalize_summary_jsonl

	summary_path = tmp_path / "summary.jsonl"
	rows = [
		{"index": 1, "final_reason": "old-k5"},
		{"index": 0, "final_reason": "unchanged"},
		{"index": 1, "final_reason": "new-k7"},
	]
	summary_path.write_text(
		"".join(json.dumps(row) + "\n" for row in rows),
		encoding="utf-8",
	)

	canonical = canonicalize_summary_jsonl(summary_path)

	assert [row["index"] for row in canonical] == [0, 1]
	assert canonical[1]["final_reason"] == "new-k7"
	assert [
		json.loads(line)
		for line in summary_path.read_text(encoding="utf-8").splitlines()
	] == canonical


def test_flux2_local_executor_is_supported_by_resource_maps_and_worker_cli() -> None:
	from inference.self_route_version.modules.local_edit_worker import (
		DEFAULT_MODEL_MAX_PARALLEL,
		DEFAULT_MODEL_MIN_FREE_GIB,
		SUPPORTED_MODELS,
		build_parser,
		parse_model_float_map,
		parse_model_int_map,
	)

	assert "flux2" in SUPPORTED_MODELS
	assert parse_model_float_map("flux2=55", DEFAULT_MODEL_MIN_FREE_GIB)["flux2"] == 55.0
	assert parse_model_int_map("flux2=4", DEFAULT_MODEL_MAX_PARALLEL)["flux2"] == 4
	args = build_parser().parse_args(["worker", "--model", "flux2", "--gpu-index", "0"])
	assert args.model == "flux2"


def test_l1_repair_memory_uses_actual_selected_executor_and_candidate() -> None:
	executor, candidate = _local_review_executor(
		{
			"selected_models": ["flux2"],
			"model_paths": {"flux2": "/tmp/repaired_flux2.png"},
		}
	)
	assert executor == "flux2"
	assert candidate == "/tmp/repaired_flux2.png"


def test_reflective_l3_plan_uses_flux2_and_history() -> None:
	state = SampleState(
		index=0,
		prompt="A chameleon perfectly camouflaged against a brown leaf",
		sample_dir=Path("/tmp/v11-test"),
		sample_id="sample",
		initial_generator_family="flux2",
		repair_memory=[{"instruction": "A visible green chameleon on a leaf"}],
	)
	item = {"state": state, "diagnosis": diagnosis()}
	plan = _normalize_reflective_plan(
		{
			"reason": "The prior image anchored the wrong visible green subject.",
			"progress_assessment": "no_progress",
			"instruction": "Generate a brown chameleon whose scales and markings match the brown leaf so its body outline is difficult to distinguish.",
			"do_not_preserve": ["green body", "high-contrast outline"],
			"preserve": ["one chameleon", "one brown leaf"],
		},
		"{}",
		item,
		3,
	)
	assert plan["valid_json"]
	assert plan["executor_family"] == "flux2"
	assert plan["reflection"]["memory_items_used"] == 1


def test_resume_reopens_only_budget_exhausted_state_when_budget_is_extended(tmp_path) -> None:
	from inference.self_route_version.modules.io_utils import load_states, save_status
	from inference.self_route_version.modules.state import safe_prompt_id

	prompt = "a red car"
	sample_id = safe_prompt_id(0, prompt)
	sample_dir = tmp_path / sample_id
	sample_dir.mkdir()
	current = sample_dir / "round4_retained.png"
	current.write_bytes(b"retained-k5-image")
	(sample_dir / "final.png").write_bytes(b"retained-k5-image")
	(sample_dir / "summary.json").write_text("{}\n", encoding="utf-8")
	state = SampleState(index=0, prompt=prompt, sample_dir=sample_dir, sample_id=sample_id)
	state.current_image = current
	state.final_image = current
	state.global_round = 5
	state.phase = "global"
	state.status = "partial"
	state.done = True
	state.final_reason = "max_repair_actions_reached_after_final_diagnosis"
	state.needs_global_analysis = False
	save_status(state)

	loaded = load_states([(0, prompt)], tmp_path, resume=True, max_actions=10)[0]
	assert loaded.done is False
	assert loaded.status == "pending"
	assert loaded.global_round == 5
	assert loaded.phase == "global"
	assert loaded.current_image == current
	assert loaded.final_image is None
	assert loaded.needs_global_analysis is True
	assert loaded.pending_global_analysis_reason == "resume_action_budget_extended"
	assert not (sample_dir / "final.png").exists()
	assert not (sample_dir / "summary.json").exists()
	assert any(row.get("event") == "resume_action_budget_extended" for row in loaded.history)

	# A resume at the same budget must remain terminal and leave final artifacts intact.
	state.done = True
	state.status = "partial"
	state.final_reason = "max_repair_actions_reached_after_final_diagnosis"
	state.final_image = current
	state.needs_global_analysis = False
	save_status(state)
	(sample_dir / "final.png").write_bytes(b"retained-k5-image")
	(sample_dir / "summary.json").write_text("{}\n", encoding="utf-8")
	same_budget = load_states([(0, prompt)], tmp_path, resume=True, max_actions=5)[0]
	assert same_budget.done is True
	assert same_budget.final_reason == "max_repair_actions_reached_after_final_diagnosis"
	assert (sample_dir / "final.png").is_file()
	assert (sample_dir / "summary.json").is_file()
