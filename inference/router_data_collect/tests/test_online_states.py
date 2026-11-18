"""Contract tests for exporting online V10 visited states into leakage-safe router data."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from inference.main_method.modules.contracts import (
	DIAGNOSIS_SCHEMA,
	build_qwen_analyzer_l1_distribution_contract,
	canonical_diagnosis_text,
)
from inference.main_method.modules.l3_policy import L3_PROMPT_VERSION, build_l3_policy_contract, l3_policy_digest
from inference.router_data_collect.canonical import iter_jsonl
from inference.router_data_collect.cli import build_parser
from inference.router_data_collect.online_states import (
	export_online_states,
	verify_online_states,
)
from inference.router_data_collect.schema import (
	DataValidationError,
	DatasetConfig,
	canonical_feature_diagnosis,
	canonicalize_state,
)


def _diagnosis(*, satisfied: bool = False, target: str = "red cup") -> dict:
	"""Construct the minimal canonical diagnosis fixture used by state tests."""
	return {
		"schema_version": DIAGNOSIS_SCHEMA,
		"fully_satisfied": satisfied,
		"reason": "all constraints are satisfied" if satisfied else f"the {target} is missing",
		"constraint_audit": {
			"objects": f"the {target} is absent" if not satisfied else f"the {target} is visible",
			"count": "one requested object was checked",
			"attributes": "the requested color was checked",
			"relations": "no additional relation is requested",
			"action_state": "no action is requested",
			"scene": "the neutral background is usable",
		},
		"issues": [] if satisfied else [
			{
				"constraint_type": "object_identity",
				"target_entity": target,
				"observed": "not visible",
				"expected": f"one visible {target}",
				"evidence": "the image contains only background",
				"confidence": 0.95,
			}
		],
		"preservation_constraints": ["preserve the neutral background"],
	}


def _write_sample(
	root: Path,
	name: str,
	prompt: str,
	*,
	family: str = "",
) -> Path:
	"""Materialize one synthetic V10 sample and return its sample directory."""
	run_config_path = root / "run_config.json"
	if not run_config_path.is_file():
		distribution = build_qwen_analyzer_l1_distribution_contract(
			model="/models/qwen/snapshots/" + "c" * 40,
			model_revision="c" * 40,
			dtype="bfloat16",
			quantization="8bit",
			allow_cpu_offload=False,
		)
		l3_policy = build_l3_policy_contract(
			family="flux2",
			model_id="black-forest-labs/FLUX.2-dev",
			model_revision="a" * 40,
			resolution=64,
			steps=1,
			guidance_scale=1.0,
			max_sequence_length=32,
			dtype="bfloat16",
			prompt_version=L3_PROMPT_VERSION,
		)
		run_config_path.parent.mkdir(parents=True, exist_ok=True)
		run_config_path.write_text(
			json.dumps({
				"qwen_analyzer_l1_distribution": distribution,
				"l3_policy_contract": l3_policy,
			}),
			encoding="utf-8",
		)
	sample = root / name
	sample.mkdir(parents=True)
	(sample / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
	status = {
		"state_schema_version": 10,
		"sample_id": name,
		"prompt": prompt,
		"initial_generator_policy_digest": l3_policy_digest(
			json.loads(run_config_path.read_text(encoding="utf-8"))["l3_policy_contract"]
		),
	}
	if family:
		status["prompt_family_id"] = family
	(sample / "status.json").write_text(json.dumps(status), encoding="utf-8")
	return sample


def _write_round(
	sample: Path,
	round_index: int,
	rgb: tuple[int, int, int],
	*,
	diagnosis: dict | None = None,
	with_image: bool = True,
) -> Path:
	round_dir = sample / f"round{round_index}"
	round_dir.mkdir()
	image_path = round_dir / "initial.png"
	if with_image:
		Image.new("RGB", (8, 6), rgb).save(image_path)
	payload = {
		"canonical_diagnosis": diagnosis or _diagnosis(),
		"raw_output": "audit-only model bytes that must never become a feature",
		"valid_json": True,
	}
	if with_image:
		payload["image"] = str(image_path.resolve())
	analysis_path = round_dir / "error_analysis.json"
	analysis_path.write_text(json.dumps(payload), encoding="utf-8")
	return analysis_path


def _external_state_config() -> DatasetConfig:
	return DatasetConfig.from_mapping(
		{
			"dataset_name": "online-state-test",
			"dataset_seed": 7,
			"rollouts_per_action": 1,
			"require_flux2_initial_state": False,
			"require_verified_initial_record": False,
			"flux2": {
				"model_id": "black-forest-labs/FLUX.2-dev",
				"model_revision": "a" * 40,
				"resolution": 64,
				"steps": 1,
				"guidance_scale": 1.0,
				"max_sequence_length": 32,
				"dtype": "bfloat16",
				"l1_roi_pad_ratio": 1.0,
				"l1_min_span": 2,
				"l1_max_edits": 3,
			},
		},
	)


def test_export_online_states_is_canonical_grouped_auditable_and_planner_ready(tmp_path: Path) -> None:
	"""Verify export online states is canonical grouped auditable and planner ready."""
	root = tmp_path / "online"
	prompt_a = "one red cup on a neutral table"
	prompt_b = "one blue bowl on a neutral table"
	sample_a = _write_sample(root, "sample_a", prompt_a)
	_write_round(sample_a, 0, (255, 0, 0), diagnosis=_diagnosis(target="red cup"))
	_write_round(sample_a, 1, (0, 0, 255), diagnosis=_diagnosis(target="red cup"))
	(sample_a / "timeline.jsonl").write_text(
		"\n".join(
			json.dumps(row)
			for row in (
				{"event": "qwen_diagnosis", "tag": "round_00_triage"},
				{"event": "route_head_decision", "tag": "round_00_triage", "decision": {"level": 2}},
				{"event": "global_flux2_edit", "round": 1, "instruction": "must remain audit-only"},
				{
					"event": "global_compare",
					"round": 1,
					"review": {"retained_repair": False, "winner_role": "original", "raw_output": "hidden"},
				},
				{"event": "qwen_diagnosis", "tag": "round_01_triage"},
			)
		) + "\n",
		encoding="utf-8",
	)

	# The explicit family links prompt A to prompt B transitively.  All their
	# states must therefore share the downstream split group.
	sample_b = _write_sample(root, "sample_b", prompt_a, family="declared-family")
	_write_round(sample_b, 0, (0, 255, 0), diagnosis=_diagnosis(target="red cup"))
	sample_c = _write_sample(root, "sample_c", prompt_b, family="declared-family")
	_write_round(sample_c, 0, (255, 255, 0), diagnosis=_diagnosis(target="blue bowl"))

	satisfied = _write_sample(root, "satisfied", "one green vase")
	_write_round(satisfied, 0, (0, 255, 255), diagnosis=_diagnosis(satisfied=True, target="green vase"))
	missing = _write_sample(root, "missing_image", "one purple lamp")
	_write_round(missing, 0, (0, 0, 0), diagnosis=_diagnosis(target="purple lamp"), with_image=False)

	output = tmp_path / "export"
	manifest = export_online_states([root], output, project_root=tmp_path)
	rows = list(iter_jsonl(output / "states.jsonl"))
	assert manifest["state_count"] == len(rows) == 4
	assert manifest["excluded_counts"] == {
		"fully_satisfied_l0": 1,
		"missing_round_initial_and_declared_image": 1,
	}
	assert len({row["prompt_family_id"] for row in rows}) == 1
	assert len({row["group_id"] for row in rows}) == 1
	assert len({row["lineage_id"] for row in rows if row["provenance"]["source_sample_id"] == "sample_a"}) == 1
	assert {row["provenance"]["qwen_distribution_signature"] for row in rows} == {
		manifest["qwen_distribution_signature"]
	}
	assert {row["provenance"]["l3_policy_digest"] for row in rows} == {
		manifest["l3_policy_digest"]
	}
	assert manifest["status_l3_policy_digest_verified_count"] == 4

	round_one = next(
		row for row in rows
		if row["provenance"]["source_sample_id"] == "sample_a" and row["source_round"] == 1
	)
	summary = round_one["provenance"]["prior_trajectory_summary"]
	assert summary["matched_current_diagnosis_event"] is True
	assert summary["counts"] == {
		"actions": 1,
		"l1": 0,
		"l2": 1,
		"l3": 0,
		"retained_candidates": 0,
		"rollbacks": 1,
	}
	assert summary["action_events"][0] == {
		"event": "global_flux2_edit",
		"source_round": 0,
		"reported_round": 1,
		"action_level": 2,
	}
	serialized = json.dumps(round_one["diagnosis"], ensure_ascii=False)
	assert "raw_output" not in serialized
	assert "instruction" not in serialized
	assert "rollback" not in canonical_diagnosis_text(round_one["diagnosis"])

	config = _external_state_config()
	for row in rows:
		canonical = canonicalize_state(row, config, tmp_path)
		assert canonical_feature_diagnosis(canonical) == canonical_diagnosis_text(row["diagnosis"])
	assert verify_online_states(output)["status"] == "verified"


def test_export_deduplicates_identical_state_content(tmp_path: Path) -> None:
	"""Verify export deduplicates identical state content."""
	root = tmp_path / "online"
	first = _write_sample(root, "first", "one red cup")
	second = _write_sample(root, "second", "one red cup")
	first_analysis = _write_round(first, 0, (12, 34, 56))
	second_analysis = _write_round(second, 0, (12, 34, 56))
	# Different source wrappers with the same prompt/image/diagnosis are one
	# route state, independent of where the online run was copied.
	assert first_analysis != second_analysis
	output = tmp_path / "export"
	manifest = export_online_states([root], output, project_root=tmp_path)
	rows = list(iter_jsonl(output / "states.jsonl"))
	assert manifest["eligible_before_dedup"] == 2
	assert manifest["state_count"] == 1
	assert manifest["duplicate_count"] == 1
	assert rows[0]["provenance"]["duplicate_source_count"] == 2


def test_export_rejects_mixed_l3_model_policies_before_materialization(tmp_path: Path) -> None:
	"""Verify export rejects mixed l3 model policies before materialization."""
	first_root = tmp_path / "first_run"
	second_root = tmp_path / "second_run"
	first = _write_sample(first_root, "sample", "one red cup")
	second = _write_sample(second_root, "sample", "one blue cup")
	_write_round(first, 0, (255, 0, 0), diagnosis=_diagnosis(target="red cup"))
	_write_round(second, 0, (0, 0, 255), diagnosis=_diagnosis(target="blue cup"))
	second_config_path = second_root / "run_config.json"
	second_config = json.loads(second_config_path.read_text(encoding="utf-8"))
	second_policy = dict(second_config["l3_policy_contract"])
	second_policy["executor"] = dict(second_policy["executor"], model_revision="b" * 40)
	second_config["l3_policy_contract"] = second_policy
	second_config_path.write_text(json.dumps(second_config), encoding="utf-8")
	second_status_path = second / "status.json"
	second_status = json.loads(second_status_path.read_text(encoding="utf-8"))
	second_status["initial_generator_policy_digest"] = l3_policy_digest(second_policy)
	second_status_path.write_text(json.dumps(second_status), encoding="utf-8")
	with pytest.raises(DataValidationError, match="mix initial-base L3 policies"):
		export_online_states([first_root, second_root], tmp_path / "export", project_root=tmp_path)


def test_export_rejects_status_and_run_config_l3_policy_mismatch(tmp_path: Path) -> None:
	"""Verify export rejects status and run config l3 policy mismatch."""
	root = tmp_path / "online"
	sample = _write_sample(root, "sample", "one red cup")
	_write_round(sample, 0, (255, 0, 0))
	status_path = sample / "status.json"
	status = json.loads(status_path.read_text(encoding="utf-8"))
	status["initial_generator_policy_digest"] = "f" * 64
	status_path.write_text(json.dumps(status), encoding="utf-8")
	with pytest.raises(DataValidationError, match="status/run_config L3 policy mismatch"):
		export_online_states([root], tmp_path / "export", project_root=tmp_path)


@pytest.mark.parametrize("leaked_key", ["raw_output", "action_plan", "route", "bbox", "level"])
def test_export_strictly_rejects_leakage_inside_canonical_diagnosis(
	tmp_path: Path,
	leaked_key: str,
) -> None:
	"""Verify export strictly rejects leakage inside canonical diagnosis."""
	root = tmp_path / "online"
	sample = _write_sample(root, "sample", "one red cup")
	diagnosis = _diagnosis()
	diagnosis[leaked_key] = "must be rejected"
	_write_round(sample, 0, (255, 0, 0), diagnosis=diagnosis)
	with pytest.raises(DataValidationError, match="non-canonical"):
		export_online_states([root], tmp_path / "export", project_root=tmp_path)


def test_cli_exposes_repeatable_online_state_export_inputs() -> None:
	"""Verify cli exposes repeatable online state export inputs."""
	args = build_parser().parse_args(
		[
			"export-online-states",
			"--input", "run_a",
			"--input", "run_b/sample",
			"--output-dir", "out",
		]
	)
	assert args.command == "export-online-states"
	assert args.input == [Path("run_a"), Path("run_b/sample")]
