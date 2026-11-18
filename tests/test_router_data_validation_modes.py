from __future__ import annotations

from pathlib import Path

import pytest

from inference.router_data_collect import evaluate_qwen, execute_flux2
from inference.router_data_collect.cli import build_parser as build_cli_parser
from inference.router_data_collect.validate import validate_dataset


def test_validate_cli_exposes_structure_only_and_parallel_artifact_scan():
	args = build_cli_parser().parse_args(
		[
			"validate",
			"--dataset-root",
			"/tmp/dataset",
			"--structure-only",
			"--artifact-workers",
			"16",
		]
	)
	assert args.structure_only is True
	assert args.artifact_workers == 16


def test_structure_only_rejects_artifact_requirements_before_io(tmp_path):
	with pytest.raises(ValueError, match="structure-only"):
		validate_dataset(tmp_path, scan_artifacts=False, require_results=True)
	with pytest.raises(ValueError, match="positive"):
		validate_dataset(tmp_path, artifact_workers=0)


def test_flux_shard_validation_still_runs_when_global_validation_is_skipped(monkeypatch, tmp_path, capsys):
	job = {"action_level": 2, "executor_family": "flux2", "job_id": "job_a"}
	completed = []
	monkeypatch.setattr(execute_flux2, "validate_dataset", lambda *_args, **_kwargs: pytest.fail("global scan ran"))
	monkeypatch.setattr(execute_flux2, "load_config", lambda *_args, **_kwargs: object())
	monkeypatch.setattr(execute_flux2, "iter_jsonl", lambda _path: iter([job]))
	monkeypatch.setattr(
		execute_flux2,
		"_completed_job",
		lambda _root, value: completed.append(value["job_id"]) or True,
	)
	execute_flux2.main(
		[
			"--dataset-root",
			str(tmp_path),
			"--action-level",
			"2",
			"--skip-global-validation",
			"--dry-run",
		]
	)
	assert completed == ["job_a"]
	assert "'pending': 0" in capsys.readouterr().out


def test_qwen_shard_validation_still_runs_when_global_validation_is_skipped(monkeypatch, tmp_path, capsys):
	source = tmp_path / "source.png"
	output = tmp_path / "output.png"
	result_path = tmp_path / "result.json"
	for path in (source, output, result_path):
		path.write_bytes(b"x")
	job = {
		"job_id": "job_a",
		"state_id": "state_a",
		"request_hash": "request",
		"policy_signature": "policy",
		"l3_policy_digest": "l3",
		"executor_family": "flux2",
		"source_image": {"path": str(source), "sha256": "source_sha"},
		"output_image": str(output),
		"result_path": str(result_path),
		"evaluation_path": str(tmp_path / "evaluation.json"),
	}
	result = {
		"status": "ok",
		"job_id": "job_a",
		"request_hash": "request",
		"policy_signature": "policy",
		"l3_policy_digest": "l3",
		"executor_family": "flux2",
		"output_sha256": "candidate_sha",
	}
	contract = {"evaluator_signature": "signature"}
	validated = []
	monkeypatch.setattr(evaluate_qwen, "validate_dataset", lambda *_args, **_kwargs: pytest.fail("global scan ran"))
	monkeypatch.setattr(
		evaluate_qwen,
		"iter_jsonl",
		lambda path: iter([{"state_id": "state_a"}]) if Path(path).name == "states.jsonl" else iter([job]),
	)
	monkeypatch.setattr(evaluate_qwen, "evaluator_contract", lambda **_kwargs: contract)
	monkeypatch.setattr(evaluate_qwen, "read_json", lambda path: result if Path(path) == result_path else {})
	monkeypatch.setattr(
		evaluate_qwen,
		"sha256_file",
		lambda path: "source_sha" if Path(path) == source else "candidate_sha",
	)
	monkeypatch.setattr(evaluate_qwen, "evaluation_prompt", lambda *_args: "prompt")
	monkeypatch.setattr(evaluate_qwen, "build_prompt_hash", lambda *_args: "prompt_hash")
	monkeypatch.setattr(
		evaluate_qwen,
		"_existing_is_valid",
		lambda *_args: validated.append("job_a") or True,
	)
	evaluate_qwen.main(
		[
			"--dataset-root",
			str(tmp_path),
			"--model",
			"/models/qwen",
			"--model-revision",
			"revision",
			"--skip-global-validation",
			"--dry-run",
		]
	)
	assert validated == ["job_a"]
	assert "'executable_results': 0" in capsys.readouterr().out
