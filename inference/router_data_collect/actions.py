"""Deterministic action prompts and geometry for Flux.2 rollouts."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from inference.main_method.modules.contracts import ACTION_PLAN_SCHEMA, build_level_action_plan, validate_distribution_contract

from .canonical import canonical_json, sha256_json, stable_id, stable_seed
from .schema import ACTION_NAMES, ACTION_OPERATIONS, DataValidationError, DatasetConfig


def assign_split(group_id: str, fractions: Mapping[str, float]) -> str:
	"""Assign an entire lineage group to a split without using row order."""

	value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:16], 16) / float(16**16)
	cumulative = 0.0
	for name in ("train", "validation", "test"):
		cumulative += float(fractions[name])
		if value < cumulative or name == "test":
			return name
	raise AssertionError("unreachable split assignment")


def l2_prompt(target_prompt: str, diagnosis: Mapping[str, Any]) -> str:
	"""Compile a whole-image edit instruction from the canonical diagnosis."""
	return str(build_level_action_plan(2, target_prompt, diagnosis)["instruction"])


def l3_prompt(target_prompt: str, diagnosis: Mapping[str, Any], initial_generator_family: str) -> str:
	"""Compile a prompt-only reconstruction instruction for L3 regeneration."""
	return str(
		build_level_action_plan(
			3,
			target_prompt,
			diagnosis,
			initial_generator_family=initial_generator_family,
		)["instruction"]
	)


def hashable_request_payload(value: Mapping[str, Any]) -> dict[str, Any]:
	"""Remove location-only metadata while retaining source image content identity."""

	payload = dict(value)
	source = dict(payload.get("source_image") or {})
	source.pop("path", None)
	payload["source_image"] = source
	return payload


def build_action_job(state: Mapping[str, Any], config: DatasetConfig, level: int, rollout_index: int) -> dict[str, Any]:
	"""Create one immutable counterfactual job for a state/action/rollout."""

	if level not in ACTION_NAMES:
		raise ValueError(f"unsupported action level: {level}")
	seed = stable_seed(config.dataset_seed, state["state_id"], rollout_index)
	policy = config.policy.as_dict()
	diagnosis = state["diagnosis"]
	executor_family = "flux2" if level in {1, 2} else config.l3_policy.family
	common = {
		"schema_version": state["schema_version"],
		"state_id": state["state_id"],
		"group_id": state["group_id"],
		"split": state["split"],
		"action_level": level,
		"action_name": ACTION_NAMES[level],
		"operation": ACTION_OPERATIONS[level],
		"executor_family": executor_family,
		"rollout_index": rollout_index,
		"seed": seed,
		"round_index": int(state.get("source_round", 0)),
		"action_seed": seed,
		"policy_signature": config.action_policy_signature,
		"flux2_policy_digest": config.policy.collector_contract_digest,
		"l3_policy_contract": config.l3_policy.as_dict(),
		"l3_policy_digest": config.l3_policy.digest,
		"policy": policy,
		"initial_generator": dict(state["initial_generator"]),
		"source_image": state["current_image"],
		"target_prompt": state["target_prompt"],
		"diagnosis_sha256": state["diagnosis_sha256"],
	}
	if level == 1:
		plan = state.get("l1_action_plan")
		if not isinstance(plan, Mapping):
			raise DataValidationError(
				f"state {state['state_id']} has no post-route L1 action plan; run the L1 planning stage first"
			)
		validate_distribution_contract(plan)
		edits = [dict(edit) for edit in plan["edits"]]
		action = {
			"reference_roles": ["editable_full_image"]
			+ [f"localization_crop_{index + 1:02d}" for index in range(len(edits))],
			"prompt": str(plan["instruction"]),
			"edits": edits,
			"action_plan_schema": ACTION_PLAN_SCHEMA,
			"action_plan_sha256": state.get("l1_action_plan_sha256") or sha256_json(plan),
		}
	elif level == 2:
		plan = build_level_action_plan(2, state["target_prompt"], diagnosis)
		validate_distribution_contract(plan)
		action = {
			"reference_roles": ["full_image"],
			"prompt": str(plan["instruction"]),
			"edits": [],
			"action_plan_schema": ACTION_PLAN_SCHEMA,
			"action_plan_sha256": sha256_json(plan),
		}
	else:
		plan = build_level_action_plan(
			3,
			state["target_prompt"],
			diagnosis,
			initial_generator_family=state["initial_generator"]["family"],
		)
		validate_distribution_contract(plan)
		action = {
			"reference_roles": [],
			"prompt": str(plan["instruction"]),
			"edits": [],
			"action_plan_schema": ACTION_PLAN_SCHEMA,
			"action_plan_sha256": sha256_json(plan),
		}
	request_payload = {**common, **action}
	request_hash = sha256_json(hashable_request_payload(request_payload))
	job_id = stable_id(
		"job",
		{
			"state_id": state["state_id"],
			"level": level,
			"rollout_index": rollout_index,
			"request_hash": request_hash,
		},
		length=32,
	)
	artifact_dir = f"artifacts/{stable_id('state', state['state_id'], length=16)}/L{level}/rollout_{rollout_index:03d}"
	return {
		**request_payload,
		"job_id": job_id,
		"request_hash": request_hash,
		"artifact_dir": artifact_dir,
		"output_image": f"{artifact_dir}/output.png",
		"result_path": f"{artifact_dir}/result.json",
		"evaluation_path": f"{artifact_dir}/evaluation.json",
	}


def action_contract_digest(jobs: Sequence[Mapping[str, Any]]) -> str:
	"""Summarize the actual immutable requests represented by a manifest."""

	return sha256_json([job["request_hash"] for job in sorted(jobs, key=lambda row: row["job_id"])])


def canonical_job_line(job: Mapping[str, Any]) -> str:
	"""Serialize an action job to the stable JSONL representation used for hashing."""
	return canonical_json(dict(job))
