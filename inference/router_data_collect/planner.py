"""Pure-CPU planning for a balanced Flux.2 counterfactual dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .actions import action_contract_digest, assign_split, build_action_job
from .canonical import canonical_json, iter_jsonl, read_json, sha256_file, sha256_json, write_if_identical_or_absent
from .schema import (
	ACTION_LEVELS,
	PROFILE_NAME,
	SCHEMA_VERSION,
	DataValidationError,
	DatasetConfig,
	canonical_feature_diagnosis,
	canonicalize_l1_action_plan,
	canonicalize_state,
	validate_action_job,
)


def load_config(path: Path, strict_revision: bool = True) -> DatasetConfig:
	"""Load and validate the dataset configuration used by all deterministic planners."""
	return DatasetConfig.from_mapping(read_json(path), strict_revision=strict_revision)


def load_state_rows(path: Path) -> list[dict[str, Any]]:
	"""Read either JSONL objects or a JSON list / {states:[...]} document."""

	if path.suffix.lower() == ".jsonl":
		return list(iter_jsonl(path))
	value = read_json(path)
	if isinstance(value, Mapping):
		value = value.get("states")
	if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
		raise DataValidationError(f"{path} must be JSONL objects, a JSON list, or {{states:[...]}}")
	return [dict(row) for row in value]


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
	return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def plan_dataset(
	config_path: Path,
	states_path: Path,
	output_dir: Path,
	project_root: Path,
	*,
	l1_plans_path: Path | None = None,
	dry_run: bool = False,
	strict_revision: bool = True,
) -> dict[str, Any]:
	"""Freeze states and exactly L1/L2/L3 x K requests before any inference."""

	config_path = config_path.resolve()
	states_path = states_path.resolve()
	output_dir = output_dir.resolve()
	project_root = project_root.resolve()
	config = load_config(config_path, strict_revision=strict_revision)
	raw_states = load_state_rows(states_path)
	if not raw_states:
		raise DataValidationError("state input is empty")

	plan_rows = load_state_rows(l1_plans_path.resolve()) if l1_plans_path else []
	plans_by_state: dict[str, dict[str, Any]] = {}
	for row in plan_rows:
		state_id = str(row.get("state_id") or "")
		if not state_id or state_id in plans_by_state:
			raise DataValidationError(f"invalid or duplicate external L1 plan state_id: {state_id!r}")
		plans_by_state[state_id] = row

	states: list[dict[str, Any]] = []
	seen_state_ids: set[str] = set()
	for row_index, raw in enumerate(raw_states):
		try:
			state = canonicalize_state(raw, config, project_root)
		except Exception as exc:
			raise DataValidationError(f"state row {row_index}: {exc}") from exc
		state_id = state["state_id"]
		if state_id in seen_state_ids:
			raise DataValidationError(f"duplicate state_id: {state_id}")
		seen_state_ids.add(state_id)
		plan_row = plans_by_state.get(state_id)
		if plan_row is not None:
			state_distribution = str(
				(state.get("provenance") or {}).get("qwen_distribution_signature") or ""
			).strip()
			plan_distribution = str(plan_row.get("distribution_signature") or "").strip()
			if state_distribution or plan_distribution:
				if not state_distribution or plan_distribution != state_distribution:
					raise DataValidationError(f"external L1 plan Qwen distribution mismatch: {state_id}")
			if plan_row.get("diagnosis_sha256") not in {None, state["diagnosis_sha256"]}:
				raise DataValidationError(f"external L1 plan diagnosis mismatch: {state_id}")
			if plan_row.get("source_image_sha256") not in {None, state["current_image"]["sha256"]}:
				raise DataValidationError(f"external L1 plan source mismatch: {state_id}")
			plan = canonicalize_l1_action_plan(
				plan_row.get("action_plan") or plan_row,
				target_prompt=state["target_prompt"],
				diagnosis=state["diagnosis"],
				width=state["current_image"]["width"],
				height=state["current_image"]["height"],
			)
			if plan is None:
				raise DataValidationError(f"external L1 plan is invalid: {state_id}")
			state["l1_action_plan"] = plan
			state["l1_action_plan_sha256"] = sha256_json(plan)
		if state.get("l1_action_plan") is None:
			raise DataValidationError(
				f"state {state_id} has no L1 action plan; diagnosis states may omit bboxes, "
				"but counterfactual planning requires --l1-plans before Flux execution"
			)
		state["split"] = assign_split(state["group_id"], config.split_fractions)
		states.append(state)
	states.sort(key=lambda row: row["state_id"])
	if plans_by_state and set(plans_by_state) != seen_state_ids:
		extra = sorted(set(plans_by_state) - seen_state_ids)
		raise DataValidationError(f"external L1 plans reference unknown states: {extra[:3]}")
	signed_distributions = {
		str((state.get("provenance") or {}).get("qwen_distribution_signature") or "").strip()
		for state in states
		if str((state.get("provenance") or {}).get("qwen_distribution_signature") or "").strip()
	}
	if len(signed_distributions) > 1:
		raise DataValidationError("states mix multiple Qwen analyzer/L1 distributions")

	jobs: list[dict[str, Any]] = []
	for state in states:
		for rollout_index in range(config.rollouts_per_action):
			# Deliberately nest rollout outside level: all three actions receive the
			# same paired seed for lower-variance counterfactual comparisons.
			for level in ACTION_LEVELS:
				job = build_action_job(state, config, level, rollout_index)
				validate_action_job(job, config)
				jobs.append(job)
	jobs.sort(key=lambda row: (row["state_id"], row["rollout_index"], row["action_level"]))

	expected_jobs = len(states) * len(ACTION_LEVELS) * config.rollouts_per_action
	if len(jobs) != expected_jobs or len({row["job_id"] for row in jobs}) != expected_jobs:
		raise DataValidationError("planner failed to produce unique, balanced action jobs")

	features = [
		{
			"schema_version": SCHEMA_VERSION,
			"state_id": state["state_id"],
			"qwen_distribution_signature": (state.get("provenance") or {}).get("qwen_distribution_signature"),
			"group_id": state["group_id"],
			"split": state["split"],
			"image_path": state["current_image"]["path"],
			"image_sha256": state["current_image"]["sha256"],
			"target_prompt": state["target_prompt"],
			"diagnosis_text": canonical_feature_diagnosis(state),
		}
		for state in states
	]
	config_snapshot = config.as_dict()
	manifest = {
		"schema_version": SCHEMA_VERSION,
		"dataset_name": config.dataset_name,
		"profile": PROFILE_NAME,
		"status": "planned",
		"state_count": len(states),
		"rollouts_per_action": config.rollouts_per_action,
		"action_levels": list(ACTION_LEVELS),
		"job_count": len(jobs),
		"policy_signature": config.action_policy_signature,
		"flux2_policy_contract": config.policy.collector_contract,
		"flux2_policy_digest": config.policy.collector_contract_digest,
		"l3_policy_contract": config.l3_policy.as_dict(),
		"l3_policy_digest": config.l3_policy.digest,
		"qwen_distribution_signature": next(iter(signed_distributions), None),
		"action_contract_digest": action_contract_digest(jobs),
		"transition_policy": {
			"kind": "atomic_joint_candidate_v2",
			"diagnosis": "canonical_v10_route_only_before_action_planning",
			"l1": "one_joint_flux2_call_with_full_image_plus_one_to_three_frozen_crops",
			"l2": "one_flux2_full_image_candidate",
			"l3": f"one_{config.l3_policy.family}_text_only_candidate_matching_initial_generator",
			"evaluation": "blind_pair_after_action; no deployment trajectory continuation",
		},
		"config_sha256": sha256_json(config_snapshot),
		"frozen_digests": {
			"states": sha256_json(states),
			"jobs": sha256_json(jobs),
			"features": sha256_json(features),
		},
		"source_manifest": {
			"config_path": str(config_path),
			"config_file_sha256": sha256_file(config_path),
			"states_path": str(states_path),
			"states_file_sha256": sha256_file(states_path),
			"l1_plans_path": str(l1_plans_path.resolve()) if l1_plans_path else None,
			"l1_plans_file_sha256": sha256_file(l1_plans_path.resolve()) if l1_plans_path else None,
		},
		"split_state_counts": {
			name: sum(state["split"] == name for state in states)
			for name in ("train", "validation", "test")
		},
		"files": {
			"config": "config.lock.json",
			"states": "states.jsonl",
			"jobs": "jobs.jsonl",
			"features": "features.jsonl",
		},
	}

	if not dry_run:
		artifacts = {
			output_dir / "config.lock.json": json.dumps(
				config_snapshot, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
			) + "\n",
			output_dir / "states.jsonl": _jsonl_text(states),
			output_dir / "jobs.jsonl": _jsonl_text(jobs),
			output_dir / "features.jsonl": _jsonl_text(features),
			output_dir / "manifest.json": json.dumps(
				manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
			) + "\n",
		}
		for path, text in artifacts.items():
			write_if_identical_or_absent(path, text)
	return manifest
