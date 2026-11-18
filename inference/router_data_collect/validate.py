"""Cross-file invariant checks for planned and completed router data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any, Mapping

from .actions import action_contract_digest, assign_split, build_action_job, hashable_request_payload
from .aggregate import _load_and_validate_evaluation, _load_and_validate_result, aggregate_dataset
from .canonical import iter_jsonl, read_json, sha256_file, sha256_json, stable_id, stable_seed
from .evaluation_contract import validate_evaluator_contract
from .planner import load_config
from .schema import (
	ACTION_LEVELS,
	FORBIDDEN_DIAGNOSIS_KEYS,
	SCHEMA_VERSION,
	DataValidationError,
	canonical_feature_diagnosis,
	image_descriptor,
	normalize_prompt_for_group,
	validate_action_job,
)


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise DataValidationError(message)


def _contains_forbidden_key(value: Any) -> str | None:
	if isinstance(value, Mapping):
		for key, child in value.items():
			if str(key) in FORBIDDEN_DIAGNOSIS_KEYS:
				return str(key)
			found = _contains_forbidden_key(child)
			if found:
				return found
	elif isinstance(value, list):
		for child in value:
			found = _contains_forbidden_key(child)
			if found:
				return found
	return None


def _request_payload(job: Mapping[str, Any]) -> dict[str, Any]:
	excluded = {"job_id", "request_hash", "artifact_dir", "output_image", "result_path", "evaluation_path"}
	return hashable_request_payload({key: value for key, value in job.items() if key not in excluded})


def validate_dataset(
	dataset_root: Path,
	*,
	require_results: bool = False,
	require_evaluations: bool = False,
	require_targets: bool = False,
	scan_artifacts: bool = True,
	artifact_workers: int = 1,
) -> dict[str, Any]:
	"""Validate identities, balance, pairing, isolation and optional outputs.

	``scan_artifacts=False`` is a structure-only mode for supervisors which have
	frozen jobs and need to prove the global state/job/seed contract exactly once
	before launching disjoint workers.  Executors must still validate every
	result in their own shard before deciding whether it is safe to skip.

	Artifact validation is I/O and SHA-256 heavy, so strict full scans can use a
	bounded thread pool.  The default remains one worker for compatibility.
	"""

	if artifact_workers <= 0:
		raise ValueError("artifact_workers must be positive")
	if not scan_artifacts and (require_results or require_evaluations or require_targets):
		raise ValueError("structure-only validation cannot require result, evaluation, or target artifacts")

	dataset_root = dataset_root.resolve()
	config = load_config(dataset_root / "config.lock.json", strict_revision=True)
	manifest = read_json(dataset_root / "manifest.json")
	states = list(iter_jsonl(dataset_root / "states.jsonl"))
	jobs = list(iter_jsonl(dataset_root / "jobs.jsonl"))
	features = list(iter_jsonl(dataset_root / "features.jsonl"))
	evaluator_lock = None
	if (dataset_root / "evaluator.lock.json").is_file():
		try:
			evaluator_lock = validate_evaluator_contract(read_json(dataset_root / "evaluator.lock.json"))
		except Exception as exc:
			raise DataValidationError(f"invalid evaluator.lock.json: {exc}") from exc

	_require(manifest.get("schema_version") == SCHEMA_VERSION, "manifest schema_version mismatch")
	_require(manifest.get("policy_signature") == config.action_policy_signature, "manifest policy_signature mismatch")
	_require(manifest.get("l3_policy_contract") == config.l3_policy.as_dict(), "manifest L3 policy mismatch")
	_require(manifest.get("l3_policy_digest") == config.l3_policy.digest, "manifest L3 policy digest mismatch")
	_require(manifest.get("config_sha256") == sha256_json(config.as_dict()), "locked config hash mismatch")
	_require(manifest.get("state_count") == len(states), "manifest state_count mismatch")
	_require(manifest.get("job_count") == len(jobs), "manifest job_count mismatch")
	_require(manifest.get("action_contract_digest") == action_contract_digest(jobs), "action contract digest mismatch")
	_require(manifest.get("frozen_digests", {}).get("states") == sha256_json(states), "frozen states digest mismatch")
	_require(manifest.get("frozen_digests", {}).get("jobs") == sha256_json(jobs), "frozen jobs digest mismatch")
	_require(manifest.get("frozen_digests", {}).get("features") == sha256_json(features), "frozen features digest mismatch")

	state_ids = [row.get("state_id") for row in states]
	_require(len(state_ids) == len(set(state_ids)), "duplicate state_id in states.jsonl")
	state_by_id = {row["state_id"]: row for row in states}
	group_splits: dict[str, str] = {}
	for state in states:
		group_payload = (
			{"prompt_family_id": state["prompt_family_id"]}
			if state.get("prompt_family_id")
			else {"prompt_identity": normalize_prompt_for_group(state["target_prompt"])}
		)
		_require(state.get("schema_version") == SCHEMA_VERSION, f"state schema mismatch: {state['state_id']}")
		expected_group = stable_id(
			"group",
			group_payload,
			length=32,
		)
		_require(state.get("group_id") == expected_group, f"prompt group_id mismatch: {state['state_id']}")
		_require(state.get("split") == assign_split(state["group_id"], config.split_fractions), f"split mismatch: {state['state_id']}")
		prior = group_splits.setdefault(state["group_id"], state["split"])
		_require(prior == state["split"], f"group split leakage: {state['group_id']}")
		leak = _contains_forbidden_key(state.get("diagnosis"))
		_require(leak is None, f"route leakage key {leak!r} in state {state['state_id']}")
		image = state["current_image"]
		image_path = Path(image["path"])
		_require(image_descriptor(image_path) == image, f"state image changed or descriptor mismatch: {state['state_id']}")
		_require(sha256_json(state["diagnosis"]) == state.get("diagnosis_sha256"), f"diagnosis hash mismatch: {state['state_id']}")
		if config.require_verified_initial_record:
			record = state.get("initial_record") or {}
			generator = state.get("initial_generator") or {}
			expected_initial_family = generator.get("family")
			_require(
				expected_initial_family == config.l3_policy.family,
				f"initial generator/L3 family mismatch: {state['state_id']}",
			)
			_require(
				record.get("status") == "ok" and record.get("executor_family") == expected_initial_family,
				f"invalid initial record: {state['state_id']}",
			)
			_require(record.get("model_id") == generator.get("model_id"), f"initial record model mismatch: {state['state_id']}")
			_require(record.get("model_revision") == generator.get("model_revision"), f"initial record revision mismatch: {state['state_id']}")
			_require(record.get("seed") == generator.get("seed"), f"initial record seed mismatch: {state['state_id']}")
			_require(record.get("output_sha256") == image["sha256"], f"initial record image mismatch: {state['state_id']}")

	expected_job_count = len(states) * len(ACTION_LEVELS) * config.rollouts_per_action
	_require(len(jobs) == expected_job_count, f"expected {expected_job_count} jobs, found {len(jobs)}")
	job_ids = [row.get("job_id") for row in jobs]
	_require(len(job_ids) == len(set(job_ids)), "duplicate job_id")
	for field in ("result_path", "evaluation_path", "output_image"):
		values = [row.get(field) for row in jobs]
		_require(len(values) == len(set(values)), f"duplicate job {field}")

	jobs_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
	for job in jobs:
		state_id = job.get("state_id")
		_require(state_id in state_by_id, f"job references unknown state: {state_id}")
		validate_action_job(job, config)
		expected_job = build_action_job(
			state_by_id[state_id], config, int(job["action_level"]), int(job["rollout_index"])
		)
		_require(job == expected_job, f"job differs from deterministic action contract: {job.get('job_id')}")
		_request_hash = sha256_json(_request_payload(job))
		_require(job.get("request_hash") == _request_hash, f"request_hash mismatch: {job.get('job_id')}")
		expected_id = stable_id(
			"job",
			{
				"state_id": state_id,
				"level": int(job["action_level"]),
				"rollout_index": int(job["rollout_index"]),
				"request_hash": _request_hash,
			},
			length=32,
		)
		_require(job.get("job_id") == expected_id, f"job_id mismatch: {job.get('job_id')}")
		_require(job.get("split") == state_by_id[state_id]["split"], f"job split mismatch: {job['job_id']}")
		key = (state_id, int(job["rollout_index"]), int(job["action_level"]))
		_require(key not in jobs_by_key, f"duplicate state/rollout/level job: {key}")
		jobs_by_key[key] = job

	for state_id in state_by_id:
		for rollout_index in range(config.rollouts_per_action):
			expected_seed = stable_seed(config.dataset_seed, state_id, rollout_index)
			seeds = []
			for level in ACTION_LEVELS:
				key = (state_id, rollout_index, level)
				_require(key in jobs_by_key, f"missing job: {key}")
				seeds.append(int(jobs_by_key[key]["seed"]))
			_require(seeds == [expected_seed] * len(ACTION_LEVELS), f"unpaired seed: {state_id}/{rollout_index}")

	_require(len(features) == len(states), "features/state count mismatch")
	feature_by_id = {row.get("state_id"): row for row in features}
	_require(len(feature_by_id) == len(features), "duplicate feature state_id")
	for state_id, state in state_by_id.items():
		_require(state_id in feature_by_id, f"missing feature: {state_id}")
		feature = feature_by_id[state_id]
		expected_feature = {
			"schema_version": SCHEMA_VERSION,
			"state_id": state_id,
			"qwen_distribution_signature": (state.get("provenance") or {}).get("qwen_distribution_signature"),
			"group_id": state["group_id"],
			"split": state["split"],
			"image_path": state["current_image"]["path"],
			"image_sha256": state["current_image"]["sha256"],
			"target_prompt": state["target_prompt"],
			"diagnosis_text": canonical_feature_diagnosis(state),
		}
		_require(feature == expected_feature, f"feature differs from state contract: {state_id}")
		try:
			feature_diagnosis = json.loads(feature["diagnosis_text"])
		except Exception as exc:
			raise DataValidationError(f"feature diagnosis is invalid JSON: {state_id}: {exc}") from exc
		leak = _contains_forbidden_key(feature_diagnosis)
		_require(leak is None, f"feature route leakage key {leak!r}: {state_id}")

	result_counts = {"valid": 0, "invalid_or_missing": 0}
	evaluation_counts = {"valid": 0, "invalid_or_missing": 0}
	if scan_artifacts:
		def validate_artifacts(job: Mapping[str, Any]) -> tuple[str, bool, str | None, bool, str | None]:
			result, result_error = _load_and_validate_result(dataset_root, job)
			evaluation, evaluation_error = _load_and_validate_evaluation(
				dataset_root,
				job,
				state_by_id[job["state_id"]],
				evaluator_lock,
			)
			return (
				str(job["job_id"]),
				result is not None,
				result_error,
				evaluation is not None,
				evaluation_error,
			)

		def consume(outcomes: Any) -> None:
			for job_id, result_valid, result_error, evaluation_valid, evaluation_error in outcomes:
				result_counts["valid" if result_valid else "invalid_or_missing"] += 1
				if require_results and result_error:
					raise DataValidationError(f"job {job_id} result: {result_error}")
				evaluation_counts["valid" if evaluation_valid else "invalid_or_missing"] += 1
				if require_evaluations and evaluation_error:
					raise DataValidationError(f"job {job_id} evaluation: {evaluation_error}")

		if artifact_workers == 1:
			consume(map(validate_artifacts, jobs))
		else:
			# Bound submitted futures so a 59,100-job strict scan does not retain
			# tens of thousands of Future objects at once.
			batch_size = max(256, artifact_workers * 16)
			with ThreadPoolExecutor(max_workers=artifact_workers) as executor:
				for start in range(0, len(jobs), batch_size):
					consume(executor.map(validate_artifacts, jobs[start : start + batch_size]))
	else:
		result_counts["not_scanned"] = len(jobs)
		evaluation_counts["not_scanned"] = len(jobs)

	aggregation_summary = None
	if scan_artifacts and (require_targets or (dataset_root / "derived" / "summary.json").is_file()):
		aggregation_summary = aggregate_dataset(dataset_root, dry_run=True)
	if require_targets:
		derived_summary_path = dataset_root / "derived" / "summary.json"
		_require(derived_summary_path.is_file(), "derived/summary.json is missing")
		_require(read_json(derived_summary_path) == aggregation_summary, "derived summary is stale")
		targets_path = dataset_root / "derived" / "state_targets.jsonl"
		_require(targets_path.is_file(), "derived/state_targets.jsonl is missing")
		targets = list(iter_jsonl(targets_path))
		_require(len(targets) == aggregation_summary["trainable_state_count"], "derived target count mismatch")
		derived_rows = {
			"rollout_utilities": list(iter_jsonl(dataset_root / "derived" / "rollout_utilities.jsonl")),
			"state_targets": targets,
			"quarantine": list(iter_jsonl(dataset_root / "derived" / "quarantine.jsonl")),
			**{
				f"split_{name}": list(iter_jsonl(dataset_root / "derived" / f"{name}.jsonl"))
				for name in ("train", "validation", "test")
			},
		}
		for name, rows in derived_rows.items():
			_require(
				sha256_json(rows) == aggregation_summary["derived_digests"][name],
				f"derived {name} content is stale or modified",
			)
		for target in targets:
			probs = target.get("soft_targets") or []
			_require(len(probs) == len(ACTION_LEVELS), f"invalid soft target length: {target.get('state_id')}")
			_require(abs(sum(float(value) for value in probs) - 1.0) <= 1e-9, f"soft targets do not sum to one: {target.get('state_id')}")

	return {
		"schema_version": SCHEMA_VERSION,
		"status": "valid",
		"state_count": len(states),
		"job_count": len(jobs),
		"group_count": len(group_splits),
		"artifact_scan": {
			"mode": "full" if scan_artifacts else "skipped",
			"workers": artifact_workers if scan_artifacts else 0,
		},
		"result_counts": result_counts,
		"evaluation_counts": evaluation_counts,
		"aggregation": aggregation_summary,
	}
