from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

from PIL import Image
import pytest

from inference.main_method.modules.contracts import (
	ACTION_PLAN_SCHEMA,
	DEFAULT_MAX_REPAIR_ACTIONS_V10,
	DIAGNOSIS_SCHEMA,
	L1_PLAN_PROMPT_VERSION_V10,
	L1_PLAN_SYSTEM_PROMPT_V10,
	QWEN_ANALYZER_L1_DTYPE_V10,
	QWEN_ANALYZER_L1_INPUT_MAX_SIDE_V10,
	QWEN_ANALYZER_L1_MAX_NEW_TOKENS_V10,
	QWEN_ANALYZER_L1_QUANTIZATION_V10,
	QWEN_DECODING_CONTRACT_V10,
	QWEN_DISTRIBUTION_SCHEMA,
	TRIAGE_PROMPT_VERSION_V10,
	TRIAGE_SYSTEM_PROMPT_V10,
	build_qwen_analyzer_l1_distribution_contract,
	build_level_action_plan,
	qwen_response_has_degenerate_repetition,
)
from inference.main_method.modules.cli import build_parser as build_v10_parser
from inference.router_data_collect.aggregate import aggregate_dataset
from inference.router_data_collect.bootstrap import (
	INITIAL_STAGE,
	INITIAL_PROMPT_MODE_COUNTERFACTUAL,
	INITIAL_PROMPT_MODE_UNCHANGED,
	INITIAL_STATE_ORIGIN_COUNTERFACTUAL,
	INITIAL_STATE_ORIGIN_NATURAL_FAILURE,
	L1_PLAN_STAGE,
	REFLECTION_STAGE,
	build_input_states,
	collect_l1_action_plans,
	l1_plan_prompt,
	load_reflector_config,
	plan_initial_jobs,
	plan_l1_action_jobs,
	plan_reflection_jobs,
	reflection_prompt,
)
from inference.router_data_collect.canonical import iter_jsonl, sha256_file, sha256_json, stable_seed
from inference.router_data_collect.evaluation_contract import (
	BLIND_PROTOCOL,
	PROMPT_VERSION,
	SUCCESS_DEFINITION,
	evaluation_prompt,
	evaluator_contract,
	prompt_hash,
)
from inference.router_data_collect.evaluate_qwen import _normalize_evaluation, build_evaluation_messages
from inference.router_data_collect.execute_flux2 import _resolve_pinned_model
from inference.router_data_collect.planner import plan_dataset
from inference.router_data_collect.qwen_batching import adaptive_generate_batch, generate_with_one_json_retry
from inference.router_data_collect.router_loss import utility_aware_router_loss
from inference.router_data_collect.schema import (
	ACTION_LEVELS,
	PROFILE_NAME,
	SCHEMA_VERSION,
	DataValidationError,
	canonical_feature_diagnosis,
)
from inference.router_data_collect.sharding import (
	merge_completed_shards,
	merge_training_bundles,
	partition_file,
	partition_rows,
	shard_key,
	stable_shard_id,
	validate_partition,
)
from inference.router_data_collect.validate import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_pinned_model_accepts_commit_named_symlink(require_symlink, tmp_path: Path) -> None:
	revision = "a" * 40
	materialized = tmp_path / "main"
	materialized.mkdir()
	snapshot = tmp_path / revision
	snapshot.symlink_to(materialized, target_is_directory=True)

	assert _resolve_pinned_model(str(snapshot), revision) == materialized.resolve()


def test_pinned_model_rejects_mutable_alias_even_if_it_resolves_to_snapshot(require_symlink, tmp_path: Path) -> None:
	revision = "b" * 40
	materialized = tmp_path / revision
	materialized.mkdir()
	mutable_alias = tmp_path / "main"
	mutable_alias.symlink_to(materialized, target_is_directory=True)

	with pytest.raises(DataValidationError, match="basename 'main'"):
		_resolve_pinned_model(str(mutable_alias), revision)


def _test_qwen_distribution_signature() -> str:
	return build_qwen_analyzer_l1_distribution_contract(
		model="/models/qwen/snapshots/" + "a" * 40,
		model_revision="a" * 40,
		dtype="bfloat16",
		quantization="none",
		allow_cpu_offload=False,
		input_max_side=512,
		max_new_tokens=512,
	)["distribution_signature"]


def test_reflector_defaults_are_the_shared_online_v10_distribution(tmp_path: Path) -> None:
	config_path = tmp_path / "reflector.json"
	_write_json(
		config_path,
		{
			"model": "/models/qwen/snapshots/" + "a" * 40,
			"model_revision": "a" * 40,
		},
	)
	contract = load_reflector_config(config_path)
	distribution = contract["qwen_distribution"]

	assert contract["dtype"] == QWEN_ANALYZER_L1_DTYPE_V10
	assert contract["quantization"] == QWEN_ANALYZER_L1_QUANTIZATION_V10
	assert contract["max_image_side"] == QWEN_ANALYZER_L1_INPUT_MAX_SIDE_V10
	assert contract["max_new_tokens"] == QWEN_ANALYZER_L1_MAX_NEW_TOKENS_V10
	assert distribution["schema_version"] == QWEN_DISTRIBUTION_SCHEMA
	assert distribution["diagnosis_schema"] == DIAGNOSIS_SCHEMA
	assert distribution["analyzer_prompt_version"] == TRIAGE_PROMPT_VERSION_V10
	assert distribution["l1_planner_prompt_version"] == L1_PLAN_PROMPT_VERSION_V10
	assert contract["distribution_signature"] == distribution["distribution_signature"]
	assert distribution["allow_cpu_offload"] is False
	assert distribution["effective_quantization"] == "int8"
	online = build_v10_parser().parse_args([])
	assert online.max_global_rounds == DEFAULT_MAX_REPAIR_ACTIONS_V10 == 2
	assert online.qwen_dtype == contract["dtype"]
	assert online.qwen_quantization == contract["quantization"]
	assert online.qwen_input_max_side == contract["max_image_side"]
	assert online.qwen_max_new_tokens == contract["max_new_tokens"]


def test_reflector_lock_rejects_distribution_tampering_and_canonicalizes_aliases(tmp_path: Path) -> None:
	config_path = tmp_path / "reflector.json"
	_write_json(
		config_path,
		{
			"model": "/models/qwen/snapshots/" + "b" * 40,
			"model_revision": "b" * 40,
			"dtype": "bf16",
			"quantization": "int8",
		},
	)
	contract = load_reflector_config(config_path)
	assert contract["dtype"] == "bfloat16"
	assert contract["quantization"] == "8bit"

	lock_path = tmp_path / "reflector.lock.json"
	_write_json(lock_path, contract)
	tampered = copy.deepcopy(contract)
	tampered["qwen_distribution"]["input_max_side"] = 1024
	_write_json(lock_path, tampered)
	with pytest.raises(DataValidationError, match="stale or was modified"):
		load_reflector_config(lock_path)


def test_reflector_4bit_cpu_offload_is_signed_as_effective_int8(tmp_path: Path) -> None:
	config_path = tmp_path / "reflector_offload.json"
	_write_json(
		config_path,
		{
			"model": "/models/qwen/snapshots/" + "c" * 40,
			"model_revision": "c" * 40,
			"quantization": "4bit",
			"allow_cpu_offload": True,
		},
	)
	contract = load_reflector_config(config_path)
	assert contract["quantization"] == "4bit"
	assert contract["allow_cpu_offload"] is True
	assert contract["effective_quantization"] == "int8_cpu_offload"
	assert contract["qwen_distribution"]["effective_quantization"] == "int8_cpu_offload"


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		"".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
		encoding="utf-8",
	)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
	return list(iter_jsonl(path))


def _make_image(
	path: Path,
	color: tuple[int, int, int] = (32, 64, 96),
	*,
	size: tuple[int, int] = (24, 16),
) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	Image.new("RGB", size, color).save(path)
	return path


def _config_payload(*, rollouts_per_action: int = 2, steps: int = 4) -> dict[str, Any]:
	return {
		"schema_version": SCHEMA_VERSION,
		"profile": PROFILE_NAME,
		"dataset_name": "flux2_router_unit_test",
		"dataset_seed": 20260715,
		"rollouts_per_action": rollouts_per_action,
		"require_flux2_initial_state": True,
		"flux2": {
			"model_id": "black-forest-labs/FLUX.2-dev",
			"model_revision": "0123456789abcdef0123456789abcdef01234567",
			"resolution": 32,
			"steps": steps,
			"guidance_scale": 3.5,
			"max_sequence_length": 64,
			"dtype": "bfloat16",
			"l1_roi_pad_ratio": 1.25,
			"l1_min_span": 4,
			"l1_max_edits": 3,
			"prompt_version": "unit_test_prompts_v1",
		},
		"split": {"train": 0.6, "validation": 0.2, "test": 0.2},
		# Isolate aggregation behavior: the resulting utility is exactly the
		# candidate-minus-original target satisfaction used by the fixtures.
		"utility": {
			"target_gain": 1.0,
			"targeted_fix": 0.0,
			"preservation": 0.0,
			"introduced_new_error": 0.0,
			"artifact_severity": 0.0,
			"success": 0.0,
			"cost_per_call": 0.0,
		},
		"soft_target_temperature": 0.5,
		"acceptable_epsilon": 0.01,
	}


def _bootstrap_initials(
	tmp_path: Path,
	rows: list[dict[str, Any]],
	*,
	name: str,
) -> tuple[Path, Path]:
	config_path = tmp_path / f"{name}.config.json"
	prompts_path = tmp_path / f"{name}.prompts.jsonl"
	root = tmp_path / f"{name}.bootstrap"
	_write_json(config_path, _config_payload(rollouts_per_action=1))
	_write_jsonl(prompts_path, rows)
	plan_initial_jobs(config_path, prompts_path, root)
	for index, job in enumerate(_read_jsonl(root / "initial_jobs.jsonl")):
		image_path = _make_image(
			root / job["output_image"],
			(10 + index, 20 + index, 30 + index),
			size=(32, 32),
		)
		_write_json(
			root / job["result_path"],
			{
				"schema_version": SCHEMA_VERSION,
				"stage": INITIAL_STAGE,
				"job_id": job["job_id"],
				"request_hash": job["request_hash"],
				"policy_signature": job["policy_signature"],
				"flux2_policy_digest": job["flux2_policy_digest"],
				"executor_family": job["executor_family"],
				"model_id": job["policy"]["model_id"],
				"model_revision": job["policy"]["model_revision"],
				"seed": job["seed"],
				"status": "ok",
				"output_image": job["output_image"],
				"output_sha256": sha256_file(image_path),
				"call_count": 1,
			},
		)
	reflector_path = tmp_path / f"{name}.reflector.json"
	_write_json(
		reflector_path,
		{
			"model": "/models/qwen/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"dtype": "bfloat16",
			"quantization": "none",
			"max_new_tokens": 512,
			"max_image_side": 512,
			"seed": 20260715,
		},
	)
	return root, reflector_path


def _state(
	image: Path,
	*,
	state_id: str,
	group_id: str,
	lineage_id: str | None = None,
	target_prompt: str = "a blue cube left of a red sphere",
	issue: str = "the cube is green instead of blue",
) -> dict[str, Any]:
	return {
		"state_id": state_id,
		"sample_id": state_id,
		"group_id": group_id,
		"lineage_id": lineage_id or group_id,
		"dataset": "unit_test",
		"current_image": str(image),
		"target_prompt": target_prompt,
		"initial_generator": {
			"family": "flux2",
			"model_id": "black-forest-labs/FLUX.2-dev",
			"model_revision": "0123456789abcdef0123456789abcdef01234567",
			"seed": 7,
		},
		"initial_record": {
			"status": "ok",
			"executor_family": "flux2",
			"model_id": "black-forest-labs/FLUX.2-dev",
			"model_revision": "0123456789abcdef0123456789abcdef01234567",
			"seed": 7,
			"output_sha256": sha256_file(image),
			"job_id": f"initial_{state_id}",
			"request_hash": f"request_{state_id}",
		},
		"diagnosis": {
			"schema_version": DIAGNOSIS_SCHEMA,
			"fully_satisfied": False,
			"reason": issue,
			"constraint_audit": {
				"objects": "the requested cube and sphere are both visible",
				"count": "object counts are correct",
				"attributes": issue,
				"relations": "the cube is left of the sphere",
				"action_state": "no action constraint applies",
				"scene": "the scene is otherwise correct",
			},
			"issues": [
				{
					"constraint_type": "attribute",
					"target_entity": "cube",
					"observed": issue,
					"expected": "the cube is blue",
					"evidence": "the visible cube pixels are green",
					"confidence": 0.95,
				}
			],
			"preservation_constraints": ["preserve the red sphere and its position"],
		},
		"provenance": {"qwen_distribution_signature": _test_qwen_distribution_signature()},
		"l1_action_plan": {
			"edits": [
				{
					"id": "edit_01",
					"priority": 1,
					"target_entity": "cube",
					"issue": issue,
					"instruction": "change the cube from green to blue",
					"quad_points": [[1, 1], [10, 1], [10, 12], [1, 12]],
				},
			],
		},
	}


def _plan(
	tmp_path: Path,
	states: list[dict[str, Any]],
	*,
	rollouts_per_action: int = 2,
	name: str = "dataset",
) -> tuple[Path, Path, Path]:
	config_path = tmp_path / f"{name}.config.json"
	states_path = tmp_path / f"{name}.states.jsonl"
	dataset_root = tmp_path / name
	_write_json(config_path, _config_payload(rollouts_per_action=rollouts_per_action))
	_write_jsonl(states_path, states)
	plan_dataset(config_path, states_path, dataset_root, tmp_path)
	return dataset_root, config_path, states_path


def _materialize_outcomes(
	dataset_root: Path,
	utilities: Mapping[str, Mapping[int, list[float]]],
	*,
	skip: set[tuple[str, int, int]] | None = None,
) -> None:
	"""Write tiny, contract-valid fake outputs without invoking a model."""

	skip = skip or set()
	contract = evaluator_contract(
		model="unit-test-qwen",
		model_revision="0123456789abcdef0123456789abcdef01234567",
		dtype="bfloat16",
		quantization="none",
		max_new_tokens=128,
		max_image_side=256,
		allow_cpu_offload=False,
	)
	_write_json(dataset_root / "evaluator.lock.json", contract)
	states = {row["state_id"]: row for row in _read_jsonl(dataset_root / "states.jsonl")}
	for job in _read_jsonl(dataset_root / "jobs.jsonl"):
		key = (str(job["state_id"]), int(job["action_level"]), int(job["rollout_index"]))
		if key in skip:
			continue
		value = float(utilities[key[0]][key[1]][key[2]])
		output_path = dataset_root / str(job["output_image"])
		_make_image(output_path, color=(int(job["action_level"]) * 40, int(job["rollout_index"]) * 40, 80))
		reference_records = []
		if int(job["action_level"]) == 1:
			for edit_index, edit in enumerate(job["edits"]):
				crop_path = output_path.parent / f"localization_crop_{edit_index + 1:02d}.png"
				_make_image(crop_path, color=(40, edit_index * 20, 80), size=(8, 8))
				reference_records.append(
					{
						"role": f"localization_crop_{edit_index + 1:02d}",
						"edit_id": edit["id"],
						"roi_xyxy": [0, 0, 8, 8],
						"image": str(crop_path.relative_to(dataset_root)),
						"sha256": sha256_file(crop_path),
					}
				)
		reference_images = (
			[str(job["source_image"]["path"]), *[str(item["image"]) for item in reference_records]]
			if int(job["action_level"]) == 1
			else ([str(job["source_image"]["path"])] if int(job["action_level"]) == 2 else [])
		)
		_write_json(
			dataset_root / str(job["result_path"]),
			{
				"schema_version": SCHEMA_VERSION,
				"job_id": job["job_id"],
				"state_id": job["state_id"],
				"action_level": job["action_level"],
				"rollout_index": job["rollout_index"],
				"seed": job["seed"],
				"action_seed": job["action_seed"],
				"round_index": job["round_index"],
				"request_hash": job["request_hash"],
				"policy_signature": job["policy_signature"],
				"flux2_policy_digest": job["flux2_policy_digest"],
				"l3_policy_digest": job["l3_policy_digest"],
				"executor_family": job["executor_family"],
				"status": "ok",
				"output_image": job["output_image"],
				"output_sha256": sha256_file(output_path),
				"call_count": 1,
				**(
					{"flux_call_count": 1}
					if job["executor_family"] == "flux2"
					else {"executor_call_count": 1}
				),
				"reference_images": reference_images,
				"rois_xyxy": [list(item["roi_xyxy"]) for item in reference_records],
				"reference_records": reference_records,
			},
		)
		order = ["original", "candidate"]
		if stable_seed(job["job_id"], "blind_order") % 2:
			order.reverse()
		_write_json(
			dataset_root / str(job["evaluation_path"]),
			{
				"schema_version": SCHEMA_VERSION,
				"job_id": job["job_id"],
				"request_hash": job["request_hash"],
				"status": "ok",
				"blind_protocol": BLIND_PROTOCOL,
				"evaluator_id": "unit-test-evaluator",
				"evaluator_signature": contract["evaluator_signature"],
				"prompt_version": PROMPT_VERSION,
				"prompt_hash": prompt_hash(states[job["state_id"]], order),
				"presentation_order": order,
				"source_image_sha256": job["source_image"]["sha256"],
				"candidate_image_sha256": sha256_file(output_path),
				"original_target_satisfaction": 0.0,
				"candidate_target_satisfaction": value,
				"targeted_fix": 0.0,
				"preservation": 0.0,
				"artifact_severity": 0.0,
				"introduced_new_error": False,
				"success": value >= 0.5,
				"success_definition": SUCCESS_DEFINITION,
			},
		)


def test_planner_builds_exactly_three_flux2_actions_per_rollout_with_paired_seeds(tmp_path: Path) -> None:
	states = [
		_state(_make_image(tmp_path / "source_a.png", (20, 30, 40)), state_id="state_a", group_id="group_a"),
		_state(_make_image(tmp_path / "source_b.png", (50, 60, 70)), state_id="state_b", group_id="group_b"),
	]
	dataset_root, _config_path, _states_path = _plan(tmp_path, states, rollouts_per_action=3)
	jobs = _read_jsonl(dataset_root / "jobs.jsonl")

	assert len(jobs) == len(states) * len(ACTION_LEVELS) * 3
	for state_id in ("state_a", "state_b"):
		state_jobs = [job for job in jobs if job["state_id"] == state_id]
		assert len(state_jobs) == 9
		rollout_seeds = []
		for rollout_index in range(3):
			rollout = [job for job in state_jobs if job["rollout_index"] == rollout_index]
			assert sorted(job["action_level"] for job in rollout) == [1, 2, 3]
			assert len({job["seed"] for job in rollout}) == 1
			rollout_seeds.append(rollout[0]["seed"])
		assert len(set(rollout_seeds)) == 3

	for job in jobs:
		assert job["executor_family"] == "flux2"
		assert job["policy"]["family"] == "flux2"
		assert job["policy"]["model_id"] == "black-forest-labs/FLUX.2-dev"
		assert job["action_seed"] == job["seed"]
		assert job["round_index"] == 0
		assert job["action_plan_schema"] == ACTION_PLAN_SCHEMA
		assert job["action_plan_sha256"]
		level = job["action_level"]
		if level == 1:
			assert job["operation"] == "edit_multi_reference"
			assert job["reference_roles"] == ["editable_full_image", "localization_crop_01"]
			assert job["prompt"] and len(job["edits"]) == 1
			assert "single result" in job["prompt"]
			assert "substep_seed" not in job["edits"][0]
		elif level == 2:
			assert job["operation"] == "edit_single_reference"
			assert job["reference_roles"] == ["full_image"]
			assert job["prompt"] and not job["edits"]
			assert job["target_prompt"] in job["prompt"]
		else:
			assert job["operation"] == "generate_text_only"
			assert job["reference_roles"] == []
			assert job["prompt"] and not job["edits"]

	validation = validate_dataset(dataset_root)
	assert validation["status"] == "valid"
	assert validation["job_count"] == 18
	parallel_validation = validate_dataset(dataset_root, artifact_workers=4)
	assert parallel_validation["result_counts"] == validation["result_counts"]
	assert parallel_validation["evaluation_counts"] == validation["evaluation_counts"]
	assert parallel_validation["artifact_scan"] == {"mode": "full", "workers": 4}
	structure_validation = validate_dataset(dataset_root, scan_artifacts=False)
	assert structure_validation["status"] == "valid"
	assert structure_validation["job_count"] == 18
	assert structure_validation["artifact_scan"] == {"mode": "skipped", "workers": 0}
	assert structure_validation["result_counts"]["not_scanned"] == 18


def test_router_feature_is_canonical_diagnosis_only_and_excludes_action_plan(tmp_path: Path) -> None:
	state = _state(_make_image(tmp_path / "feature.png"), state_id="feature_state", group_id="feature_group")
	dataset_root, _config_path, _states_path = _plan(tmp_path, [state], rollouts_per_action=1, name="feature_contract")
	planned_state = _read_jsonl(dataset_root / "states.jsonl")[0]
	feature = _read_jsonl(dataset_root / "features.jsonl")[0]
	diagnosis = json.loads(feature["diagnosis_text"])

	assert feature["diagnosis_text"] == canonical_feature_diagnosis(planned_state)
	assert set(diagnosis) == {
		"schema_version",
		"fully_satisfied",
		"reason",
		"constraint_audit",
		"issues",
		"preservation_constraints",
	}
	assert diagnosis["schema_version"] == DIAGNOSIS_SCHEMA
	assert not ({"edits", "bbox", "quad_points", "instruction", "fix", "global_edit_instruction"} & set(diagnosis))
	assert "change the cube from green to blue" not in feature["diagnosis_text"]
	assert planned_state["l1_action_plan"]["schema_version"] == ACTION_PLAN_SCHEMA


def test_three_local_regions_still_form_one_joint_l1_candidate(tmp_path: Path) -> None:
	state = _state(_make_image(tmp_path / "three.png", size=(32, 32)), state_id="three_state", group_id="three_group")
	state["l1_action_plan"]["edits"] = [
		{
			"id": f"edit_{index:02d}",
			"priority": index,
			"target_entity": f"object_{index}",
			"issue": f"object {index} has a wrong attribute",
			"instruction": f"correct object {index} only",
			"quad_points": [[x, 2], [x + 5, 2], [x + 5, 9], [x, 9]],
		}
		for index, x in enumerate((1, 11, 21), start=1)
	]
	dataset_root, _config_path, _states_path = _plan(
		tmp_path,
		[state],
		rollouts_per_action=1,
		name="three_joint",
	)
	l1 = next(job for job in _read_jsonl(dataset_root / "jobs.jsonl") if job["action_level"] == 1)

	assert l1["reference_roles"] == [
		"editable_full_image",
		"localization_crop_01",
		"localization_crop_02",
		"localization_crop_03",
	]
	assert len(l1["edits"]) == 3
	assert all(f"Edit {index}" in l1["prompt"] for index in (1, 2, 3))
	assert l1["action_seed"] == l1["seed"]
	assert not any("substep_seed" in edit for edit in l1["edits"])

	_materialize_outcomes(dataset_root, {"three_state": {1: [0.7], 2: [0.5], 3: [0.3]}})
	result = json.loads((dataset_root / l1["result_path"]).read_text(encoding="utf-8"))
	assert result["call_count"] == result["flux_call_count"] == 1
	assert len(result["reference_records"]) == 3
	assert result["reference_images"][0] == l1["source_image"]["path"]


def test_planning_is_idempotent_and_rejects_a_changed_contract_at_same_output(tmp_path: Path) -> None:
	image = _make_image(tmp_path / "source.png")
	state = _state(image, state_id="stable_state", group_id="stable_group")
	dataset_root, config_path, states_path = _plan(tmp_path, [state], name="immutable")
	tracked = ["config.lock.json", "states.jsonl", "jobs.jsonl", "features.jsonl", "manifest.json"]
	before = {name: (dataset_root / name).read_bytes() for name in tracked}

	plan_dataset(config_path, states_path, dataset_root, tmp_path)
	assert {name: (dataset_root / name).read_bytes() for name in tracked} == before

	changed = _config_payload(steps=5)
	_write_json(config_path, changed)
	with pytest.raises(ValueError, match="Refusing to overwrite non-identical artifact"):
		plan_dataset(config_path, states_path, dataset_root, tmp_path)
	assert {name: (dataset_root / name).read_bytes() for name in tracked} == before


def test_flux2_only_profile_rejects_a_non_flux_initial_state(tmp_path: Path) -> None:
	state = _state(_make_image(tmp_path / "bagel.png"), state_id="bagel_state", group_id="bagel_group")
	state["initial_generator"]["family"] = "bagel"
	config_path = tmp_path / "config.json"
	states_path = tmp_path / "states.jsonl"
	_write_json(config_path, _config_payload())
	_write_jsonl(states_path, [state])

	with pytest.raises(DataValidationError, match="requires Flux.2 initial states"):
		plan_dataset(config_path, states_path, tmp_path / "out", tmp_path)


def test_aggregate_rejects_state_without_qwen_distribution_signature(tmp_path: Path) -> None:
	state = _state(_make_image(tmp_path / "unsigned.png"), state_id="unsigned", group_id="unsigned_group")
	dataset_root, _config_path, _states_path = _plan(tmp_path, [state], rollouts_per_action=1, name="unsigned")
	_materialize_outcomes(dataset_root, {"unsigned": {1: [0.5], 2: [0.4], 3: [0.3]}})
	planned_states = _read_jsonl(dataset_root / "states.jsonl")
	planned_states[0]["provenance"].pop("qwen_distribution_signature")
	_write_jsonl(dataset_root / "states.jsonl", planned_states)
	with pytest.raises(DataValidationError, match="missing a valid qwen_distribution_signature"):
		aggregate_dataset(dataset_root)


def test_aggregation_uses_all_rollouts_mean_and_keeps_ties_soft(tmp_path: Path) -> None:
	states = [
		_state(_make_image(tmp_path / "mean.png", (10, 20, 30)), state_id="mean_state", group_id="mean_group"),
		_state(_make_image(tmp_path / "tie.png", (40, 50, 60)), state_id="tie_state", group_id="tie_group"),
	]
	dataset_root, _config_path, _states_path = _plan(tmp_path, states, rollouts_per_action=2, name="aggregation")
	_materialize_outcomes(
		dataset_root,
		{
			# L1 has the largest single rollout (1.0), but L2 has the best mean.
			"mean_state": {1: [1.0, 0.0], 2: [0.6, 0.6], 3: [0.2, 0.2]},
			"tie_state": {1: [0.4, 0.4], 2: [0.4, 0.4], 3: [0.4, 0.4]},
		},
	)
	summary = aggregate_dataset(dataset_root)
	targets = {row["state_id"]: row for row in _read_jsonl(dataset_root / "derived" / "state_targets.jsonl")}

	assert summary["aggregation"]["rollout_reducer"] == "arithmetic_mean"
	assert len(summary["flux2_policy_digest"]) == 64
	assert summary["trainable_state_count"] == 2
	mean_target = targets["mean_state"]
	assert mean_target["flux2_policy_contract"] == summary["flux2_policy_contract"]
	assert mean_target["flux2_policy_digest"] == summary["flux2_policy_digest"]
	assert mean_target["utility_vector"] == pytest.approx([0.5, 0.6, 0.2])
	assert mean_target["action_statistics"]["1"]["rollout_count"] == 2
	assert mean_target["hard_target"] == 2
	assert mean_target["best_levels"] == [2]

	tie_target = targets["tie_state"]
	assert tie_target["utility_vector"] == pytest.approx([0.4, 0.4, 0.4])
	assert tie_target["soft_targets"] == pytest.approx([1.0 / 3.0] * 3)
	assert sum(tie_target["soft_targets"]) == pytest.approx(1.0)
	assert tie_target["best_levels"] == [1, 2, 3]
	assert tie_target["hard_target"] is None
	assert summary["training_balance"]["soft_target_mass"].keys() == {"L1", "L2", "L3"}
	assert summary["training_balance"]["class_factors"].keys() == {"L1", "L2", "L3"}
	assert summary["training_balance"]["effective_soft_target_mass"].keys() == {"L1", "L2", "L3"}


def test_utility_aware_loss_supports_soft_targets_and_train_class_weights() -> None:
	torch = pytest.importorskip("torch")
	logits = torch.tensor([[0.2, -0.1, 0.4], [0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
	soft_targets = torch.tensor([[0.2, 0.3, 0.5], [1.0 / 3.0] * 3], dtype=torch.float32)
	utilities = torch.tensor([[-0.2, 0.0, 0.4], [0.1, 0.1, 0.1]], dtype=torch.float32)
	loss, parts = utility_aware_router_loss(
		logits,
		soft_targets,
		utilities,
		sample_weight=torch.tensor([2.0, 0.5]),
		class_weight=torch.tensor([0.5, 1.5, 2.0]),
	)
	loss.backward()

	assert torch.isfinite(loss)
	assert torch.isfinite(logits.grad).all()
	assert set(parts) == {
		"loss",
		"soft_ce",
		"utility_alignment",
		"pairwise",
		"expected_regret",
		"efficiency",
		"safe_prior",
	}


def test_quality_first_loss_masks_unreliable_pairs_and_breaks_exact_ties_by_efficiency() -> None:
	torch = pytest.importorskip("torch")
	logits = torch.zeros((2, 3), dtype=torch.float32, requires_grad=True)
	soft_targets = torch.full((2, 3), 1.0 / 3.0)
	utilities = torch.tensor([[0.0, 1.0, 0.5], [0.0, 0.0, 0.0]], dtype=torch.float32)
	regrets = utilities.max(dim=-1, keepdim=True).values - utilities
	loss, parts = utility_aware_router_loss(
		logits,
		soft_targets,
		utilities,
		regret_targets=regrets,
		pairwise_mask=torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
		efficiency_costs=torch.tensor([3.0, 2.0, 1.0]),
		soft_ce_weight=0.0,
		utility_alignment_weight=0.0,
		pairwise_weight=0.2,
		expected_regret_weight=1.0,
		efficiency_weight=0.01,
		quality_tie_epsilon=0.0,
	)
	loss.backward()

	assert parts["expected_regret"].item() == pytest.approx(0.25)
	assert parts["efficiency"].item() == pytest.approx(torch.log(torch.tensor(3.0)).item())
	assert torch.isfinite(logits.grad).all()


def test_exact_tie_quality_mask_keeps_efficiency_gradient() -> None:
	torch = pytest.importorskip("torch")
	logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
	utilities = torch.zeros((1, 3), dtype=torch.float32)
	loss, parts = utility_aware_router_loss(
		logits,
		torch.full((1, 3), 1.0 / 3.0),
		utilities,
		regret_targets=utilities,
		efficiency_regret_targets=utilities,
		quality_row_weight=torch.zeros(1),
		efficiency_costs=torch.tensor([3.0, 2.0, 1.0]),
		soft_ce_weight=1.0,
		utility_alignment_weight=1.0,
		pairwise_weight=1.0,
		expected_regret_weight=1.0,
		efficiency_weight=1.0,
	)
	loss.backward()

	assert parts["soft_ce"].item() == pytest.approx(0.0)
	assert parts["utility_alignment"].item() == pytest.approx(0.0)
	assert parts["pairwise"].item() == pytest.approx(0.0)
	assert parts["expected_regret"].item() == pytest.approx(0.0)
	assert parts["efficiency"].item() == pytest.approx(torch.log(torch.tensor(3.0)).item())
	assert logits.grad[0, 2].item() < 0.0
	assert logits.grad[0, 0].item() > 0.0


def test_efficiency_uses_observed_acceptable_set_not_shrunken_quality_target() -> None:
	torch = pytest.importorskip("torch")
	logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
	# A denoised quality target can prefer L2, while observed candidate quality
	# says L1/L2 are tied. Efficiency may choose only within that observed set.
	utilities = torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float32)
	loss, parts = utility_aware_router_loss(
		logits,
		torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float32),
		utilities,
		regret_targets=torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float32),
		efficiency_regret_targets=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
		quality_row_weight=torch.zeros(1),
		efficiency_costs=torch.tensor([3.0, 2.0, 1.0]),
		soft_ce_weight=0.0,
		utility_alignment_weight=0.0,
		pairwise_weight=0.0,
		expected_regret_weight=0.0,
		efficiency_weight=1.0,
	)
	loss.backward()

	assert parts["efficiency"].item() == pytest.approx(torch.log(torch.tensor(3.0)).item())
	assert logits.grad[0, 1].item() < 0.0
	assert logits.grad[0, 2].item() > 0.0


def test_base_sample_weight_disables_quality_and_efficiency_together() -> None:
	torch = pytest.importorskip("torch")
	logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
	zeros = torch.zeros((1, 3), dtype=torch.float32)
	loss, parts = utility_aware_router_loss(
		logits,
		torch.full((1, 3), 1.0 / 3.0),
		zeros,
		regret_targets=zeros,
		efficiency_regret_targets=zeros,
		sample_weight=torch.zeros(1),
		quality_row_weight=torch.ones(1),
		efficiency_costs=torch.tensor([3.0, 2.0, 1.0]),
		efficiency_weight=1.0,
	)
	loss.backward()

	assert loss.item() == pytest.approx(0.0)
	assert all(value.item() == pytest.approx(0.0) for value in parts.values())
	assert logits.grad.abs().sum().item() == pytest.approx(0.0)


def test_uncertain_safe_prior_turns_off_for_reliable_quality_escape() -> None:
	torch = pytest.importorskip("torch")
	logits = torch.zeros((2, 3), dtype=torch.float32, requires_grad=True)
	soft_targets = torch.full((2, 3), 1.0 / 3.0)
	utilities = torch.tensor(
		[
			[0.0, 0.0, 1.0],  # reliable L3 escape: no safe-prior gradient
			[0.0, 0.0, 1.0],  # same mean, but unreliable pair: prefer safe L2
		],
		dtype=torch.float32,
	)
	loss, parts = utility_aware_router_loss(
		logits,
		soft_targets,
		utilities,
		pairwise_mask=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
		soft_ce_weight=0.0,
		utility_alignment_weight=0.0,
		pairwise_weight=0.0,
		expected_regret_weight=0.0,
		efficiency_weight=0.0,
		uncertain_safe_prior_weight=1.0,
		safe_action_index=1,
	)
	loss.backward()

	assert parts["safe_prior"].item() == pytest.approx(torch.log(torch.tensor(3.0)).item())
	assert logits.grad[0].abs().sum().item() == pytest.approx(0.0)
	assert logits.grad[1, 1].item() < 0.0


def test_safe_prior_uses_raw_reliability_not_a_shrunken_group_pair_mask() -> None:
	torch = pytest.importorskip("torch")
	logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
	utilities = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
	loss, parts = utility_aware_router_loss(
		logits,
		torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float32),
		utilities,
		pairwise_mask=torch.tensor([[0.0, 0.0, 1.0]]),
		safe_pairwise_mask=torch.zeros((1, 3)),
		soft_ce_weight=0.0,
		utility_alignment_weight=0.0,
		pairwise_weight=0.0,
		expected_regret_weight=0.0,
		efficiency_weight=0.0,
		uncertain_safe_prior_weight=1.0,
		safe_action_index=1,
	)
	loss.backward()

	assert parts["safe_prior"].item() == pytest.approx(torch.log(torch.tensor(3.0)).item())
	assert logits.grad[0, 1].item() < 0.0


def test_uncertain_safe_prior_does_not_override_an_observed_quality_tie() -> None:
	torch = pytest.importorskip("torch")
	logits = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
	zeros = torch.zeros((1, 3), dtype=torch.float32)
	loss, parts = utility_aware_router_loss(
		logits,
		torch.full((1, 3), 1.0 / 3.0),
		zeros,
		regret_targets=zeros,
		efficiency_regret_targets=zeros,
		pairwise_mask=zeros,
		soft_ce_weight=0.0,
		utility_alignment_weight=0.0,
		pairwise_weight=0.0,
		expected_regret_weight=0.0,
		efficiency_weight=0.0,
		uncertain_safe_prior_weight=1.0,
	)
	loss.backward()

	assert parts["safe_prior"].item() == pytest.approx(0.0)
	assert logits.grad.abs().sum().item() == pytest.approx(0.0)


def test_incomplete_level_is_quarantined_instead_of_changing_the_mean_denominator(tmp_path: Path) -> None:
	state = _state(_make_image(tmp_path / "missing.png"), state_id="missing_state", group_id="missing_group")
	dataset_root, _config_path, _states_path = _plan(tmp_path, [state], rollouts_per_action=2, name="missing")
	utilities = {"missing_state": {1: [0.5, 0.5], 2: [0.6, 0.6], 3: [0.9, 0.9]}}
	_materialize_outcomes(dataset_root, utilities, skip={("missing_state", 3, 1)})

	summary = aggregate_dataset(dataset_root)
	targets = _read_jsonl(dataset_root / "derived" / "state_targets.jsonl")
	quarantine = _read_jsonl(dataset_root / "derived" / "quarantine.jsonl")

	assert summary["valid_rollout_count"] == 5
	assert summary["invalid_rollout_count"] == 1
	assert summary["trainable_state_count"] == 0
	assert summary["quarantined_state_count"] == 1
	assert targets == []
	assert quarantine[0]["valid_rollout_counts"] == {"1": 2, "2": 2, "3": 1}
	assert "L3_valid:1!=K" in quarantine[0]["reasons"]


def test_group_split_is_stable_and_never_separates_one_lineage_group(tmp_path: Path) -> None:
	states = [
		_state(
			_make_image(tmp_path / "round_0.png", (1, 2, 3)),
			state_id="trajectory_round_0",
			group_id="shared_trajectory",
			lineage_id="shared_trajectory",
		),
		_state(
			_make_image(tmp_path / "round_1.png", (4, 5, 6)),
			state_id="trajectory_round_1",
			group_id="different_legacy_group",
			lineage_id="shared_trajectory",
		),
	]
	dataset_root, _config_path, _states_path = _plan(tmp_path, states, name="grouped")
	planned_states = _read_jsonl(dataset_root / "states.jsonl")

	assert len({row["group_id"] for row in planned_states}) == 1
	assert planned_states[0]["group_id"].startswith("group_")
	assert len({row["split"] for row in planned_states}) == 1
	assert {job["split"] for job in _read_jsonl(dataset_root / "jobs.jsonl")} == {
		planned_states[0]["split"]
	}
	assert validate_dataset(dataset_root)["group_count"] == 1
	assert {row["provided_group_id"] for row in planned_states} == {
		"shared_trajectory",
		"different_legacy_group",
	}


def test_old_evaluation_cannot_be_reused_after_candidate_changes(tmp_path: Path) -> None:
	state = _state(_make_image(tmp_path / "source.png"), state_id="bound_state", group_id="legacy_group")
	dataset_root, _config_path, _states_path = _plan(tmp_path, [state], rollouts_per_action=1, name="binding")
	_materialize_outcomes(dataset_root, {"bound_state": {1: [0.2], 2: [0.6], 3: [0.4]}})
	job = next(job for job in _read_jsonl(dataset_root / "jobs.jsonl") if job["action_level"] == 2)
	output_path = dataset_root / job["output_image"]
	_make_image(output_path, color=(255, 1, 2))
	result_path = dataset_root / job["result_path"]
	result = json.loads(result_path.read_text(encoding="utf-8"))
	result["output_sha256"] = sha256_file(output_path)
	_write_json(result_path, result)

	summary = aggregate_dataset(dataset_root)
	assert summary["trainable_state_count"] == 0
	rollouts = _read_jsonl(dataset_root / "derived" / "rollout_utilities.jsonl")
	l2 = next(row for row in rollouts if row["action_level"] == 2)
	assert l2["valid"] is False
	assert l2["invalid_reason"] == "evaluation_candidate_image_hash_mismatch"


def test_aggregate_rejects_duplicate_rollout_even_when_count_is_unchanged(tmp_path: Path) -> None:
	state = _state(_make_image(tmp_path / "source.png"), state_id="duplicate_state", group_id="legacy_group")
	dataset_root, _config_path, _states_path = _plan(tmp_path, [state], rollouts_per_action=2, name="duplicate")
	_materialize_outcomes(
		dataset_root,
		{"duplicate_state": {1: [0.2, 0.2], 2: [0.4, 0.4], 3: [0.6, 0.6]}},
	)
	jobs = _read_jsonl(dataset_root / "jobs.jsonl")
	first = next(job for job in jobs if job["action_level"] == 3 and job["rollout_index"] == 0)
	index = next(
		index for index, job in enumerate(jobs) if job["action_level"] == 3 and job["rollout_index"] == 1
	)
	jobs[index] = copy.deepcopy(first)
	_write_jsonl(dataset_root / "jobs.jsonl", jobs)
	with pytest.raises(DataValidationError, match="duplicate job_id"):
		aggregate_dataset(dataset_root, dry_run=True)


def test_validate_targets_detects_modified_derived_rows(tmp_path: Path) -> None:
	state = _state(_make_image(tmp_path / "source.png"), state_id="digest_state", group_id="legacy_group")
	dataset_root, _config_path, _states_path = _plan(tmp_path, [state], rollouts_per_action=1, name="digest")
	_materialize_outcomes(dataset_root, {"digest_state": {1: [0.2], 2: [0.4], 3: [0.6]}})
	aggregate_dataset(dataset_root)
	targets_path = dataset_root / "derived" / "state_targets.jsonl"
	targets = _read_jsonl(targets_path)
	targets[0]["soft_targets"] = [1.0, 0.0, 0.0]
	_write_jsonl(targets_path, targets)
	with pytest.raises(DataValidationError, match="content is stale or modified"):
		validate_dataset(dataset_root, require_targets=True)


@pytest.mark.parametrize("leaked_key", ["category", "recommended_mode", "repair_level"])
def test_planner_rejects_legacy_route_label_leakage(tmp_path: Path, leaked_key: str) -> None:
	state = _state(_make_image(tmp_path / f"{leaked_key}.png"), state_id="leaked", group_id="leaked_group")
	state = copy.deepcopy(state)
	state["diagnosis"][leaked_key] = "L1"
	config_path = tmp_path / f"{leaked_key}.config.json"
	states_path = tmp_path / f"{leaked_key}.states.jsonl"
	_write_json(config_path, _config_payload())
	_write_jsonl(states_path, [state])

	with pytest.raises(DataValidationError, match="legacy route leakage"):
		plan_dataset(config_path, states_path, tmp_path / f"out_{leaked_key}", tmp_path)


def test_unicode_paths_ids_and_reflection_text_round_trip_without_ascii_escaping(tmp_path: Path) -> None:
	image = _make_image(tmp_path / "images_①" / "input_②.png", (90, 100, 110))
	state = _state(
		image,
		state_id="state_①",
		group_id="blue_square_②",
		target_prompt="A blue square labeled ① to the left of a red sphere.",
		issue="Square ① is green instead of blue.",
	)
	dataset_root, _config_path, _states_path = _plan(tmp_path, [state], name="unicode")
	states_text = (dataset_root / "states.jsonl").read_text(encoding="utf-8")
	features_text = (dataset_root / "features.jsonl").read_text(encoding="utf-8")
	planned = _read_jsonl(dataset_root / "states.jsonl")[0]

	assert "state_①" in states_text
	assert "images_①" in states_text
	assert "Square ① is green" in states_text and "Square ① is green" in features_text
	assert planned["state_id"] == "state_①"
	assert planned["provided_group_id"] == "blue_square_②"
	assert planned["group_id"].startswith("group_")
	assert Path(planned["current_image"]["path"]) == image.resolve()
	assert validate_dataset(dataset_root)["status"] == "valid"


def test_package_and_flux_cli_help_do_not_import_torch() -> None:
	environment = os.environ.copy()
	environment["CUDA_VISIBLE_DEVICES"] = ""
	environment["PYTHONDONTWRITEBYTECODE"] = "1"
	environment["PYTHONPATH"] = os.pathsep.join(
		[str(PROJECT_ROOT), environment.get("PYTHONPATH", "")]
	).rstrip(os.pathsep)
	import_check = subprocess.run(
		[
			sys.executable,
			"-c",
			(
				"import importlib,sys; "
				"assert 'torch' not in sys.modules; "
				"importlib.import_module('inference.router_data_collect'); "
				"importlib.import_module('inference.router_data_collect.cli'); "
				"importlib.import_module('inference.router_data_collect.execute_flux2'); "
				"importlib.import_module('inference.router_data_collect.execute_initial_flux2'); "
				"importlib.import_module('inference.router_data_collect.execute_reflections_qwen'); "
				"importlib.import_module('inference.router_data_collect.execute_l1_plans_qwen'); "
				"importlib.import_module('inference.router_data_collect.evaluate_qwen'); "
				"importlib.import_module('inference.router_data_collect.orchestrate_shard'); "
				"assert 'torch' not in sys.modules"
			),
		],
		cwd=PROJECT_ROOT,
		env=environment,
		capture_output=True,
		text=True,
		timeout=20,
		check=False,
	)
	assert import_check.returncode == 0, import_check.stderr

	cli_help = subprocess.run(
		[sys.executable, "-m", "inference.router_data_collect.cli", "--help"],
		cwd=PROJECT_ROOT,
		env=environment,
		capture_output=True,
		text=True,
		timeout=20,
		check=False,
	)
	assert cli_help.returncode == 0, cli_help.stderr
	assert "plan-initial" in cli_help.stdout
	assert "plan-reflections" in cli_help.stdout
	assert "build-states" in cli_help.stdout
	assert "plan-l1-actions" in cli_help.stdout
	assert "collect-l1-actions" in cli_help.stdout
	assert "partition-shards" in cli_help.stdout
	assert "validate-shards" in cli_help.stdout
	assert "merge-shards" in cli_help.stdout
	assert "aggregate" in cli_help.stdout

	executor_help = subprocess.run(
		[sys.executable, "-m", "inference.router_data_collect.execute_flux2", "--help"],
		cwd=PROJECT_ROOT,
		env=environment,
		capture_output=True,
		text=True,
		timeout=20,
		check=False,
	)
	assert executor_help.returncode == 0, executor_help.stderr
	assert "--dry-run" in executor_help.stdout
	assert "--dataset-root" in executor_help.stdout
	assert "--batch-size" in executor_help.stdout

	for source in (PROJECT_ROOT / "inference" / "router_data_collect").glob("*.py"):
		assert "inference.V9" not in source.read_text(encoding="utf-8"), source


def test_four_card_sharding_is_order_independent_disjoint_and_family_stable(tmp_path: Path) -> None:
	rows = [
		{
			"sample_id": f"sample_{index}",
			"dataset": "unit",
			"base_prompt_id": family,
			"prompt_family_id": f"unit:{family}",
			"target_prompt": f"target {family} variant {index}",
		}
		for index, family in enumerate(("family_a", "family_a", "family_b", "family_c", "family_d", "family_e"))
	]
	forward = partition_rows(rows, 4)
	reversed_partition = partition_rows(list(reversed(rows)), 4)

	def assignments(shards: list[list[dict[str, Any]]]) -> dict[str, int]:
		return {
			str(row["sample_id"]): shard_id
			for shard_id, shard in enumerate(shards)
			for row in shard
		}

	assert assignments(forward) == assignments(reversed_partition)
	assert assignments(forward)["sample_0"] == assignments(forward)["sample_1"]
	assert sum(len(shard) for shard in forward) == len(rows)
	assert len({row["sample_id"] for shard in forward for row in shard}) == len(rows)
	assert all(shard_key(row).startswith("prompt_family_id:") for shard in forward for row in shard)

	input_path = tmp_path / "shard_input.jsonl"
	output_root = tmp_path / "partition"
	_write_jsonl(input_path, list(reversed(rows)))
	manifest = partition_file(input_path, output_root, 4)
	assert manifest["num_shards"] == 4
	assert manifest["row_count"] == len(rows)
	assert validate_partition(output_root) == {
		"status": "valid",
		"row_count": len(rows),
		"num_shards": 4,
		"shard_policy": manifest["shard_policy"],
	}


def test_source_blind_evaluator_prompt_is_identical_for_both_hidden_orders(tmp_path: Path) -> None:
	state = _state(
		_make_image(tmp_path / "blind_source.png"),
		state_id="blind_source",
		group_id="blind_source_group",
	)
	forward_order = ["original", "candidate"]
	reverse_order = ["candidate", "original"]
	forward = evaluation_prompt(state, forward_order)
	reverse = evaluation_prompt(state, reverse_order)

	assert forward == reverse
	for leaked_source in (
		"original",
		"candidate",
		"generated",
		"generator",
		"editor",
		"model",
		"action level",
		"repair level",
		"flux",
	):
		assert leaked_source not in forward.casefold()
	assert prompt_hash(state, forward_order) != prompt_hash(state, reverse_order)


def test_source_blind_scores_map_to_internal_roles_only_after_qwen_output() -> None:
	raw = {
		"image_a": {
			"target_satisfaction": 0.2,
			"diagnosed_issue_resolution": 0.1,
			"preservation": 0.9,
			"artifact_severity": 0.1,
			"preservation_violation": False,
		},
		"image_b": {
			"target_satisfaction": 0.9,
			"diagnosed_issue_resolution": 0.8,
			"preservation": 0.7,
			"artifact_severity": 0.2,
			"preservation_violation": True,
		},
		"rationale": "B satisfies more target constraints than A.",
	}

	forward = _normalize_evaluation(raw, ["original", "candidate"])
	assert forward == {
		"original_target_satisfaction": 0.2,
		"candidate_target_satisfaction": 0.9,
		"targeted_fix": 0.8,
		"preservation": 0.7,
		"artifact_severity": 0.2,
		"introduced_new_error": True,
		"rationale": "B satisfies more target constraints than A.",
		"success": False,
	}

	reverse = _normalize_evaluation(raw, ["candidate", "original"])
	assert reverse["original_target_satisfaction"] == 0.9
	assert reverse["candidate_target_satisfaction"] == 0.2
	assert reverse["targeted_fix"] == 0.1
	assert reverse["preservation"] == 0.9
	assert reverse["artifact_severity"] == 0.1
	assert reverse["introduced_new_error"] is False
	assert reverse["success"] is False

	with pytest.raises(DataValidationError, match="evaluator keys must be exactly"):
		_normalize_evaluation(
			{
				"original_target_satisfaction": 0.2,
				"candidate_target_satisfaction": 0.9,
			},
			["original", "candidate"],
		)


def test_qwen_two_image_batch_preserves_each_samples_source_blind_order() -> None:
	class FakePlanner:
		def __init__(self) -> None:
			self.image_batches: list[list[list[str]]] = []

		def _generate_batch(self, messages_batch, images_batch, _max_new_tokens):
			self.image_batches.append([list(images) for images in images_batch])
			assert all(len(messages) == 1 for messages in messages_batch)
			return [f"response_{index}" for index in range(len(messages_batch))]

		def _generate(self, _messages, _images, _max_new_tokens):  # pragma: no cover - batch must work
			raise AssertionError("unexpected scalar fallback")

	roles_a = {"original": "a_original", "candidate": "a_candidate"}
	roles_b = {"original": "b_original", "candidate": "b_candidate"}
	messages_a, images_a = build_evaluation_messages("prompt a", ["original", "candidate"], roles_a)
	messages_b, images_b = build_evaluation_messages("prompt b", ["candidate", "original"], roles_b)
	assert messages_a[0]["content"] == [
		{"type": "text", "text": "A:"},
		{"type": "image", "image": "a_original"},
		{"type": "text", "text": "B:"},
		{"type": "image", "image": "a_candidate"},
		{"type": "text", "text": "prompt a"},
	]
	assert messages_b[0]["content"] == [
		{"type": "text", "text": "A:"},
		{"type": "image", "image": "b_candidate"},
		{"type": "text", "text": "B:"},
		{"type": "image", "image": "b_original"},
		{"type": "text", "text": "prompt b"},
	]
	planner = FakePlanner()
	outputs = adaptive_generate_batch(planner, [messages_a, messages_b], [images_a, images_b], 32)

	assert outputs == ["response_0", "response_1"]
	assert planner.image_batches == [
		[["a_original", "a_candidate"], ["b_candidate", "b_original"]]
	]


def test_completed_gpu_shards_merge_into_one_globally_balanced_training_bundle(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	families: dict[int, str] = {}
	for index in range(100):
		family = f"unit:merge_family_{index}"
		owner = stable_shard_id(f"prompt_family_id:{family}", 2)
		families.setdefault(owner, family)
		if len(families) == 2:
			break
	assert set(families) == {0, 1}
	partition_input = tmp_path / "merge_prompts.jsonl"
	partition_root = tmp_path / "merge_partition"
	_write_jsonl(
		partition_input,
		[
			{"sample_id": f"merge_{owner}", "prompt_family_id": family, "target_prompt": f"target {owner}"}
			for owner, family in sorted(families.items())
		],
	)
	partition_file(partition_input, partition_root, 2)
	config_path = tmp_path / "merge.config.json"
	_write_json(config_path, _config_payload(rollouts_per_action=1))
	for shard_id, family in sorted(families.items()):
		shard_root = partition_root / "shards" / f"shard_{shard_id:02d}"
		image = _make_image(tmp_path / f"merge_source_{shard_id}.png", (20 + shard_id, 30, 40))
		state_id = f"merge_state_{shard_id}"
		state = _state(image, state_id=state_id, group_id=f"legacy_{shard_id}")
		state["prompt_family_id"] = family
		states_path = shard_root / "states.input.jsonl"
		_write_jsonl(states_path, [state])
		dataset = shard_root / "dataset"
		plan_dataset(config_path, states_path, dataset, tmp_path)
		_materialize_outcomes(
			dataset,
			{state_id: {1: [0.2 + shard_id * 0.4], 2: [0.5], 3: [0.8 - shard_id * 0.4]}},
		)
		aggregate_dataset(dataset)

	merged_root = tmp_path / "merged_training"
	manifest = merge_completed_shards(partition_root, merged_root)
	merged_targets = _read_jsonl(merged_root / "derived" / "state_targets.jsonl")
	assert manifest["status"] == "merged_training_bundle"
	assert manifest["num_shards"] == 2
	assert manifest["trainable_state_count"] == 2
	assert len(merged_targets) == 2
	assert {row["state_id"] for row in merged_targets} == {"merge_state_0", "merge_state_1"}
	assert manifest["training_balance"]["class_factors"].keys() == {"L1", "L2", "L3"}
	assert len(manifest["flux2_policy_digest"]) == 64
	assert manifest["l3_initial_identity_binding_counts"] == {"exact_model_identity": 2}
	assert all(row["flux2_policy_digest"] == manifest["flux2_policy_digest"] for row in merged_targets)

	# Exercise merge's own mixed-policy guard independently of the per-shard
	# stale-derived validator.
	from inference.router_data_collect import validate as validate_module

	monkeypatch.setattr(
		validate_module,
		"validate_dataset",
		lambda _dataset, **_kwargs: {"state_count": 1, "job_count": 3},
	)
	tampered_summary_path = partition_root / "shards" / "shard_01" / "dataset" / "derived" / "summary.json"
	tampered_summary = json.loads(tampered_summary_path.read_text(encoding="utf-8"))
	tampered_summary["flux2_policy_digest"] = "d" * 64
	_write_json(tampered_summary_path, tampered_summary)
	with pytest.raises(DataValidationError, match="mix Flux.2 collector policies"):
		merge_completed_shards(partition_root, tmp_path / "mixed-policy")


def test_disjoint_training_bundles_merge_with_digest_and_identity_guards(tmp_path: Path) -> None:
	config = _config_payload(rollouts_per_action=1)
	evaluator = {"evaluator_signature": "evaluator_unit_test"}
	bundles = []
	for index in range(2):
		root = tmp_path / f"bundle_{index}"
		state_id = f"state_{index}"
		target = {
			"state_id": state_id,
			"group_id": f"group_{index}",
			"split": "train",
			"soft_targets": [0.6, 0.3, 0.1],
			"hard_target": 1,
			"l3_initial_identity_binding": "exact_model_identity",
		}
		rollout = {
			"state_id": state_id,
			"job_id": f"job_{index}",
			"rollout_index": 0,
			"action_level": 1,
			"valid": True,
		}
		_write_json(root / "config.lock.json", config)
		_write_json(root / "evaluator.lock.json", evaluator)
		_write_jsonl(root / "derived" / "state_targets.jsonl", [target])
		_write_jsonl(root / "derived" / "rollout_utilities.jsonl", [rollout])
		_write_jsonl(root / "derived" / "quarantine.jsonl", [])
		_write_jsonl(root / "derived" / "train.jsonl", [target])
		_write_jsonl(root / "derived" / "validation.jsonl", [])
		_write_jsonl(root / "derived" / "test.jsonl", [])
		_write_json(
			root / "derived" / "summary.json",
			{
				"status": "merged_training_bundle",
				"schema_version": SCHEMA_VERSION,
				"qwen_distribution_signature": "a" * 64,
				"flux2_policy_digest": "b" * 64,
				"flux2_policy_contract": {"policy": "flux2"},
				"l3_policy_digest": "c" * 64,
				"l3_policy_contract": {"policy": "l3"},
				"initial_generator_family": "flux2",
				"aggregation": {"rollout_reducer": "arithmetic_mean"},
				"state_count": 1,
				"valid_rollout_count": 1,
				"invalid_rollout_count": 0,
				"derived_digests": {
					"state_targets": sha256_json([target]),
					"rollout_utilities": sha256_json([rollout]),
					"quarantine": sha256_json([]),
				},
			},
		)
		bundles.append(root)

	manifest = merge_training_bundles(bundles, tmp_path / "combined")
	assert manifest["state_count"] == 2
	assert manifest["valid_rollout_count"] == 2
	assert manifest["l3_initial_identity_binding_counts"] == {"exact_model_identity": 2}
	assert manifest["split_target_counts"] == {"train": 2, "validation": 0, "test": 0}
	with pytest.raises(DataValidationError, match="inputs must be distinct"):
		merge_training_bundles([bundles[0], bundles[0]], tmp_path / "duplicate")


def test_qwen_json_contract_retries_only_invalid_rows_once() -> None:
	class FakePlanner:
		def __init__(self) -> None:
			self.scalar_messages: list[list[dict[str, Any]]] = []
			self.retry_decoding_used = False

		def _generate_batch(self, _messages_batch, _images_batch, _max_new_tokens):
			loop = " ".join(["the same twelve token phrase repeats here again and again forever now"] * 30)
			return [loop, "valid:first"]

		def _generate(self, _messages, _images, _max_new_tokens):  # pragma: no cover - retry must use its own decoder
			raise AssertionError("retry incorrectly used base decoding")

		def _generate_retry(self, messages, images, _max_new_tokens):
			self.retry_decoding_used = True
			self.scalar_messages.append(messages)
			assert images == ["image_0"]
			return "valid:retry"

	planner = FakePlanner()
	messages = [
		[{"role": "system", "content": []}, {"role": "user", "content": []}],
		[{"role": "system", "content": []}, {"role": "user", "content": []}],
	]
	records = generate_with_one_json_retry(
		planner,
		messages,
		[["image_0"], ["image_1"]],
		32,
		lambda raw, _index: {"valid_json": raw.startswith("valid:"), "value": raw},
		"return valid JSON",
	)

	assert [record["json_retry_count"] for record in records] == [1, 0]
	assert [record["valid_json"] for record in records] == [True, True]
	assert planner.retry_decoding_used is True
	assert qwen_response_has_degenerate_repetition(records[0]["initial_raw_response"])
	assert [message["role"] for message in planner.scalar_messages[0]] == [
		"system",
		"user",
		"assistant",
		"user",
	]


def test_bootstrap_keeps_zero_index_and_cross_dataset_identity_distinct(tmp_path: Path) -> None:
	config_path = tmp_path / "zero.config.json"
	prompts_path = tmp_path / "zero.prompts.jsonl"
	root = tmp_path / "zero.bootstrap"
	_write_json(config_path, _config_payload())
	_write_jsonl(
		prompts_path,
		[
			{"index": 0, "dataset": "dataset_a", "target_prompt": "a blue cube"},
			{"index": 0, "dataset": "dataset_b", "target_prompt": "a red sphere"},
		],
	)
	plan_initial_jobs(config_path, prompts_path, root)
	rows = _read_jsonl(root / "prompt_rows.jsonl")
	jobs = _read_jsonl(root / "initial_jobs.jsonl")

	assert [row["sample_id"] for row in rows] == ["0", "0"]
	assert [row["source_prompt_id"] for row in rows] == ["0", "0"]
	assert len({row["sample_key"] for row in rows}) == 2
	assert len({job["job_id"] for job in jobs}) == 2
	assert len({job["output_image"] for job in jobs}) == 2
	assert len({job["reflection_result_path"] for job in jobs}) == 2


def test_reflection_planning_rejects_a_missing_stage_row_after_manifest_reseal(tmp_path: Path) -> None:
	root, reflector_path = _bootstrap_initials(
		tmp_path,
		[
			{"index": 0, "dataset": "a", "target_prompt": "a blue cube"},
			{"index": 1, "dataset": "b", "target_prompt": "a red sphere"},
		],
		name="missing_stage",
	)
	jobs = _read_jsonl(root / "initial_jobs.jsonl")[:1]
	_write_jsonl(root / "initial_jobs.jsonl", jobs)
	manifest = json.loads((root / "bootstrap_manifest.json").read_text(encoding="utf-8"))
	manifest["job_count"] = 1
	manifest["frozen_digests"]["initial_jobs"] = sha256_json(jobs)
	_write_json(root / "bootstrap_manifest.json", manifest)

	with pytest.raises(DataValidationError, match="same sample_key set"):
		plan_reflection_jobs(root, reflector_path)


def test_reflection_planning_rejects_an_escape_path_after_manifest_reseal(tmp_path: Path) -> None:
	root, reflector_path = _bootstrap_initials(
		tmp_path,
		[{"index": 0, "dataset": "escape", "target_prompt": "a blue cube"}],
		name="escape_path",
	)
	jobs = _read_jsonl(root / "initial_jobs.jsonl")
	jobs[0]["reflection_result_path"] = "../escaped/result.json"
	_write_jsonl(root / "initial_jobs.jsonl", jobs)
	manifest = json.loads((root / "bootstrap_manifest.json").read_text(encoding="utf-8"))
	manifest["frozen_digests"]["initial_jobs"] = sha256_json(jobs)
	_write_json(root / "bootstrap_manifest.json", manifest)

	with pytest.raises(DataValidationError, match="escapes bootstrap root"):
		plan_reflection_jobs(root, reflector_path)


def test_reflection_planning_rejects_a_modified_source_snapshot_hash(tmp_path: Path) -> None:
	root, reflector_path = _bootstrap_initials(
		tmp_path,
		[{"index": 0, "dataset": "hash", "target_prompt": "a blue cube"}],
		name="source_hash",
	)
	manifest = json.loads((root / "bootstrap_manifest.json").read_text(encoding="utf-8"))
	manifest["source_prompts_sha256"] = "f" * 64
	_write_json(root / "bootstrap_manifest.json", manifest)

	with pytest.raises(DataValidationError, match="snapshot hash mismatch"):
		plan_reflection_jobs(root, reflector_path)


def test_bootstrap_binds_initial_generation_and_reflection_before_planning_states(tmp_path: Path) -> None:
	config_path = tmp_path / "config.json"
	prompts_path = tmp_path / "prompts.jsonl"
	bootstrap_root = tmp_path / "bootstrap_run"
	_write_json(config_path, _config_payload(rollouts_per_action=1))
	_write_jsonl(
		prompts_path,
		[
			{
				"index": 17,
				"dataset": "unit_bootstrap",
				"tag": "counting",
				"target_prompt": "two blue cups on a table",
				"initial_generation_prompt": "one blue cup on a table",
				"corruption": {"type": "wrong_count"},
			}
		],
	)
	initial_summary = plan_initial_jobs(config_path, prompts_path, bootstrap_root)
	assert initial_summary["job_count"] == 1
	initial_job = _read_jsonl(bootstrap_root / "initial_jobs.jsonl")[0]
	initial_image = _make_image(
		bootstrap_root / initial_job["output_image"],
		(10, 20, 30),
		size=(32, 32),
	)
	_write_json(
		bootstrap_root / initial_job["result_path"],
		{
			"schema_version": SCHEMA_VERSION,
			"stage": INITIAL_STAGE,
			"job_id": initial_job["job_id"],
			"request_hash": initial_job["request_hash"],
			"policy_signature": initial_job["policy_signature"],
			"flux2_policy_digest": initial_job["flux2_policy_digest"],
			"executor_family": "flux2",
			"model_id": initial_job["policy"]["model_id"],
			"model_revision": initial_job["policy"]["model_revision"],
			"seed": initial_job["seed"],
			"status": "ok",
			"output_image": initial_job["output_image"],
			"output_sha256": sha256_file(initial_image),
			"call_count": 1,
		},
	)
	reflector_config = tmp_path / "reflector.json"
	_write_json(
		reflector_config,
		{
			"model": "/models/qwen/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"dtype": "bfloat16",
			"quantization": "none",
			"max_new_tokens": 512,
			"max_image_side": 512,
		},
	)
	reflection_summary = plan_reflection_jobs(bootstrap_root, reflector_config)
	assert reflection_summary["job_count"] == 1
	reflector_contract = json.loads((bootstrap_root / "reflector.lock.json").read_text(encoding="utf-8"))
	reflection_job = _read_jsonl(bootstrap_root / "reflection_jobs.jsonl")[0]
	reflection_signature = json.loads(reflection_prompt(reflection_job))
	assert reflection_signature[0] == {"role": "system", "text": TRIAGE_SYSTEM_PROMPT_V10}
	assert reflection_signature[1]["role"] == "user"
	assert reflection_job["target_prompt"] in reflection_signature[1]["text"]
	diagnosis = {
		"schema_version": DIAGNOSIS_SCHEMA,
		"fully_satisfied": False,
		"reason": "only one cup is visible",
		"constraint_audit": {
			"objects": "a blue cup and table are visible",
			"count": "only one cup is visible although two are requested",
			"attributes": "the visible cup is blue",
			"relations": "the visible cup is on the table",
			"action_state": "no action constraint applies",
			"scene": "the tabletop scene is otherwise correct",
		},
		"issues": [
			{
				"constraint_type": "count",
				"target_entity": "blue cups",
				"observed": "one blue cup",
				"expected": "two blue cups",
				"evidence": "only one cup silhouette is visible on the table",
				"confidence": 0.98,
			}
		],
		"preservation_constraints": ["preserve the existing cup and table"],
	}
	_write_json(
		bootstrap_root / reflection_job["result_path"],
		{
			"schema_version": SCHEMA_VERSION,
			"stage": REFLECTION_STAGE,
			"executor_family": "qwen",
			"model": "/models/qwen/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"model_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
			"seed": reflection_job["seed"],
			"decoding": dict(QWEN_DECODING_CONTRACT_V10),
			"prompt_version": reflection_job["prompt_version"],
			"job_id": reflection_job["job_id"],
			"request_hash": reflection_job["request_hash"],
			"reflector_signature": reflection_job["reflector_signature"],
			"distribution_signature": reflection_job["distribution_signature"],
			"diagnosis_schema": reflection_job["diagnosis_schema"],
			"source_image_sha256": reflection_job["source_image"]["sha256"],
			"prompt_sha256": reflection_job["prompt_sha256"],
			"dtype": reflector_contract["dtype"],
			"quantization": reflector_contract["quantization"],
			"allow_cpu_offload": reflector_contract["allow_cpu_offload"],
			"effective_quantization": reflector_contract["effective_quantization"],
			"max_image_side": reflector_contract["max_image_side"],
			"max_new_tokens": reflector_contract["max_new_tokens"],
			"status": "ok",
			"json_retry_count": 0,
			"diagnosis": diagnosis,
			"diagnosis_sha256": sha256_json(diagnosis),
		},
	)
	states_path = tmp_path / "clean_states.jsonl"
	state_summary = build_input_states(bootstrap_root, states_path, project_root=tmp_path)
	assert state_summary["eligible_state_count"] == 1
	clean_state = _read_jsonl(states_path)[0]
	assert clean_state["initial_record"]["job_id"] == initial_job["job_id"]
	assert clean_state["initial_record"]["output_sha256"] == sha256_file(initial_image)
	assert clean_state["initial_generator"]["family"] == "flux2"
	assert clean_state["provenance"]["initial_prompt_mode"] == INITIAL_PROMPT_MODE_COUNTERFACTUAL
	assert clean_state["provenance"]["prompt_corruption_applied"] is True
	assert clean_state["provenance"]["initial_state_origin"] == INITIAL_STATE_ORIGIN_COUNTERFACTUAL
	assert clean_state["provenance"]["initial_target_audit"] == {
		"protocol": "qwen_canonical_diagnosis_against_target_v1",
		"status": "failed",
		"fully_satisfied": False,
		"diagnosis_sha256": sha256_json(diagnosis),
	}
	assert clean_state.get("l1_action_plan") is None
	assert "edit" not in clean_state["diagnosis"]

	# The route-visible state is intentionally bbox-free.  The post-route L1
	# planner owns localization and must finish before counterfactual planning.
	with pytest.raises(DataValidationError, match="no L1 action plan"):
		plan_dataset(config_path, states_path, tmp_path / "missing_l1_plan", tmp_path)
	l1_root = tmp_path / "l1_plan_stage"
	plan_summary = plan_l1_action_jobs(
		states_path,
		reflector_config,
		l1_root,
		project_root=tmp_path,
	)
	assert plan_summary["job_count"] == 1
	l1_job = _read_jsonl(l1_root / "jobs.jsonl")[0]
	assert l1_job["diagnosis"] == diagnosis
	assert "quad_points" not in l1_job["diagnosis"]
	l1_signature = json.loads(l1_plan_prompt(l1_job))
	assert l1_signature[0] == {"role": "system", "text": L1_PLAN_SYSTEM_PROMPT_V10}
	assert l1_signature[1]["role"] == "user"
	assert "Canonical diagnosis JSON" in l1_signature[1]["text"]
	assert "schema_version" not in l1_signature[1]["text"]
	assert "vcot" not in l1_signature[1]["text"].casefold()
	l1_action_plan = build_level_action_plan(
		1,
		clean_state["target_prompt"],
		clean_state["diagnosis"],
		edits=[
			{
				"id": "edit_01",
				"priority": 1,
				"target_entity": "empty table area",
				"target_object": "empty table area",
				"issue": "missing second cup",
				"instruction": "add a second blue cup on the empty table area",
				"edit_prompt": "add a second blue cup on the empty table area",
				"whole_edit": "add a second blue cup on the empty table area",
				"quad_points": [[8, 2], [20, 2], [20, 14], [8, 14]],
				"box_xyxy": [8, 2, 21, 15],
			}
		],
		width=32,
		height=32,
	)
	_write_json(
		l1_root / l1_job["result_path"],
		{
			"schema_version": SCHEMA_VERSION,
			"stage": L1_PLAN_STAGE,
			"executor_family": "qwen",
			"job_id": l1_job["job_id"],
			"request_hash": l1_job["request_hash"],
			"planner_signature": l1_job["planner_signature"],
			"distribution_signature": l1_job["distribution_signature"],
			"diagnosis_schema": l1_job["diagnosis_schema"],
			"model": reflector_contract["model"],
			"model_revision": reflector_contract["model_revision"],
			"dtype": reflector_contract["dtype"],
			"quantization": reflector_contract["quantization"],
			"allow_cpu_offload": reflector_contract["allow_cpu_offload"],
			"effective_quantization": reflector_contract["effective_quantization"],
			"max_image_side": reflector_contract["max_image_side"],
			"max_new_tokens": reflector_contract["max_new_tokens"],
			"diagnosis_sha256": l1_job["diagnosis_sha256"],
			"source_image_sha256": l1_job["source_image"]["sha256"],
			"status": "ok",
			"json_retry_count": 0,
			"action_plan": l1_action_plan,
			"action_plan_sha256": sha256_json(l1_action_plan),
		},
	)
	l1_plans_path = tmp_path / "l1_action_plans.jsonl"
	assert collect_l1_action_plans(l1_root, l1_plans_path)["plan_count"] == 1

	counterfactual_root = tmp_path / "counterfactual"
	plan_dataset(
		config_path,
		states_path,
		counterfactual_root,
		tmp_path,
		l1_plans_path=l1_plans_path,
	)
	assert len(_read_jsonl(counterfactual_root / "jobs.jsonl")) == 3


def test_target_identical_initial_prompt_failure_becomes_a_traceable_repair_state(
	tmp_path: Path,
) -> None:
	root, reflector_path = _bootstrap_initials(
		tmp_path,
		[
			{
				"index": 23,
				"dataset": "natural_initial_failure",
				"target_prompt": "two blue cups on a table",
				# No initial_generation_prompt: bootstrap must use the target unchanged.
			}
		],
		name="natural_initial_failure",
	)
	prompt_row = _read_jsonl(root / "prompt_rows.jsonl")[0]
	initial_job = _read_jsonl(root / "initial_jobs.jsonl")[0]
	bootstrap_manifest = json.loads((root / "bootstrap_manifest.json").read_text(encoding="utf-8"))
	assert prompt_row["generation_prompt"] == prompt_row["target_prompt"]
	assert prompt_row["initial_prompt_mode"] == INITIAL_PROMPT_MODE_UNCHANGED
	assert prompt_row["prompt_corruption_applied"] is False
	assert initial_job["initial_prompt_mode"] == INITIAL_PROMPT_MODE_UNCHANGED
	assert initial_job["prompt_corruption_applied"] is False
	assert bootstrap_manifest["initial_prompt_modes"] == {INITIAL_PROMPT_MODE_UNCHANGED: 1}
	assert bootstrap_manifest["prompt_corruption_applied_count"] == 0

	plan_reflection_jobs(root, reflector_path)
	contract = json.loads((root / "reflector.lock.json").read_text(encoding="utf-8"))
	job = _read_jsonl(root / "reflection_jobs.jsonl")[0]
	diagnosis = {
		"schema_version": DIAGNOSIS_SCHEMA,
		"fully_satisfied": False,
		"reason": "the requested second cup is absent",
		"constraint_audit": {
			"objects": "one blue cup and a table are visible",
			"count": "one cup is visible instead of two",
			"attributes": "the visible cup is blue",
			"relations": "the visible cup is on the table",
			"action_state": "no action constraint applies",
			"scene": "the table scene is visible",
		},
		"issues": [
			{
				"constraint_type": "count",
				"target_entity": "blue cups",
				"observed": "one blue cup",
				"expected": "two blue cups",
				"evidence": "only one cup is visible",
				"confidence": 0.99,
			}
		],
		"preservation_constraints": ["preserve the visible blue cup and table"],
	}
	_write_json(
		root / job["result_path"],
		{
			"schema_version": SCHEMA_VERSION,
			"stage": REFLECTION_STAGE,
			"executor_family": "qwen",
			"model": contract["model"],
			"model_revision": contract["model_revision"],
			"seed": job["seed"],
			"decoding": contract["decoding"],
			"prompt_version": job["prompt_version"],
			"job_id": job["job_id"],
			"request_hash": job["request_hash"],
			"reflector_signature": job["reflector_signature"],
			"distribution_signature": job["distribution_signature"],
			"diagnosis_schema": job["diagnosis_schema"],
			"source_image_sha256": job["source_image"]["sha256"],
			"prompt_sha256": job["prompt_sha256"],
			"dtype": contract["dtype"],
			"quantization": contract["quantization"],
			"allow_cpu_offload": contract["allow_cpu_offload"],
			"effective_quantization": contract["effective_quantization"],
			"max_image_side": contract["max_image_side"],
			"max_new_tokens": contract["max_new_tokens"],
			"status": "ok",
			"json_retry_count": 0,
			"diagnosis": diagnosis,
			"diagnosis_sha256": sha256_json(diagnosis),
		},
	)
	states_path = tmp_path / "natural_initial_failure.states.jsonl"
	summary = build_input_states(root, states_path, project_root=tmp_path)
	assert summary["eligible_state_count"] == 1
	assert summary["eligible_initial_state_origins"] == {
		INITIAL_STATE_ORIGIN_NATURAL_FAILURE: 1
	}
	state = _read_jsonl(states_path)[0]
	assert state["target_prompt"] == prompt_row["generation_prompt"]
	assert state["provenance"]["initial_generation_prompt"] == state["target_prompt"]
	assert state["provenance"]["prompt_corruption_applied"] is False
	assert state["provenance"]["initial_state_origin"] == INITIAL_STATE_ORIGIN_NATURAL_FAILURE
	assert state["provenance"]["initial_target_audit"]["status"] == "failed"
