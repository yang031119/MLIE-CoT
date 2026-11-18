"""Aggregate balanced Flux.2 rollouts into utility-aware router targets."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from .canonical import canonical_json, iter_jsonl, read_json, sha256_file, sha256_json, stable_seed, write_json_atomic, write_jsonl_atomic
from .evaluation_contract import (
	BLIND_PROTOCOL,
	PROMPT_VERSION,
	SUCCESS_DEFINITION,
	prompt_hash as build_prompt_hash,
	validate_evaluator_contract,
)
from .planner import load_config
from .schema import (
	ACTION_LEVELS,
	SCHEMA_VERSION,
	DataValidationError,
	DatasetConfig,
	canonical_feature_diagnosis,
	validate_action_job,
)


EVALUATION_KEYS = (
	"original_target_satisfaction",
	"candidate_target_satisfaction",
	"targeted_fix",
	"preservation",
	"artifact_severity",
)


def _locked_qwen_distribution_signature(states: Sequence[Mapping[str, Any]]) -> str:
	signatures: list[str] = []
	for state in states:
		signature = str((state.get("provenance") or {}).get("qwen_distribution_signature") or "").strip().lower()
		if re.fullmatch(r"[0-9a-f]{64}", signature) is None:
			raise DataValidationError(
				f"state {state.get('state_id')} is missing a valid qwen_distribution_signature"
			)
		signatures.append(signature)
	unique = set(signatures)
	if len(unique) != 1:
		raise DataValidationError("states mix multiple Qwen analyzer/L1 distributions")
	return next(iter(unique))


def _relative_path(root: Path, value: str) -> Path:
	path = Path(value)
	return path if path.is_absolute() else root / path


def _score(value: Any, name: str) -> float:
	try:
		result = float(value)
	except (TypeError, ValueError) as exc:
		raise DataValidationError(f"evaluation {name} must be numeric") from exc
	if not math.isfinite(result) or not 0.0 <= result <= 1.0:
		raise DataValidationError(f"evaluation {name} must be in [0,1]")
	return result


def rollout_utility(evaluation: Mapping[str, Any], result: Mapping[str, Any], config: DatasetConfig) -> float:
	"""Compute one action utility; higher is better and cost is explicit."""

	scores = {name: _score(evaluation.get(name), name) for name in EVALUATION_KEYS}
	if not isinstance(evaluation.get("introduced_new_error"), bool):
		raise DataValidationError("evaluation introduced_new_error must be boolean")
	if not isinstance(evaluation.get("success"), bool):
		raise DataValidationError("evaluation success must be boolean")
	try:
		call_count = int(result.get("call_count", 1))
	except (TypeError, ValueError) as exc:
		raise DataValidationError("result call_count must be an integer") from exc
	if call_count <= 0:
		raise DataValidationError("result call_count must be positive")
	w = config.utility
	return (
		w["target_gain"] * (scores["candidate_target_satisfaction"] - scores["original_target_satisfaction"])
		+ w["targeted_fix"] * scores["targeted_fix"]
		+ w["preservation"] * scores["preservation"]
		- w["introduced_new_error"] * float(evaluation["introduced_new_error"])
		- w["artifact_severity"] * scores["artifact_severity"]
		+ w["success"] * float(evaluation["success"])
		- w["cost_per_call"] * call_count
	)


def softmax(values: Sequence[float], temperature: float) -> list[float]:
	"""Convert logits into normalized probabilities using a stable exponential shift."""
	if not values:
		raise DataValidationError("cannot softmax an empty utility vector")
	maximum = max(values)
	exponents = [math.exp((value - maximum) / temperature) for value in values]
	total = sum(exponents)
	return [value / total for value in exponents]


def _load_and_validate_result(root: Path, job: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
	"""Load one action result and verify hashes, candidate counts, seeds, and executor semantics."""
	path = _relative_path(root, str(job["result_path"]))
	if not path.is_file():
		return None, "missing_result"
	try:
		value = read_json(path)
	except Exception as exc:
		return None, f"invalid_result_json:{exc}"
	if not isinstance(value, Mapping):
		return None, "result_not_object"
	result = dict(value)
	for key in ("job_id", "request_hash", "policy_signature", "flux2_policy_digest", "l3_policy_digest"):
		if result.get(key) != job.get(key):
			return None, f"result_{key}_mismatch"
	if result.get("executor_family") != job.get("executor_family"):
		return None, "result_executor_family_mismatch"
	for key in ("state_id", "action_level", "rollout_index", "seed", "round_index", "action_seed"):
		if result.get(key) != job.get(key):
			return None, f"result_{key}_mismatch"
	if result.get("status") != "ok":
		return None, f"result_status:{result.get('status', 'missing')}"
	output_path = _relative_path(root, str(job["output_image"]))
	if not output_path.is_file():
		return None, "missing_output_image"
	if result.get("output_sha256") != sha256_file(output_path):
		return None, "output_sha256_mismatch"
	if result.get("output_image") != str(job["output_image"]):
		return None, "result_output_image_mismatch"
	if result.get("call_count") != 1:
		return None, "result_call_count_mismatch"
	if job.get("executor_family") == "flux2":
		if result.get("flux_call_count") != 1:
			return None, "result_flux_call_count_mismatch"
	elif result.get("executor_call_count") != 1:
		return None, "result_executor_call_count_mismatch"
	references = result.get("reference_records")
	if not isinstance(references, list):
		return None, "result_reference_records_missing"
	if int(job["action_level"]) == 1:
		if len(references) != len(job["edits"]):
			return None, "result_reference_count_mismatch"
		for reference_index, reference in enumerate(references, start=1):
			expected_role = f"localization_crop_{reference_index:02d}"
			if reference.get("role") != expected_role:
				return None, "result_reference_role_mismatch"
			reference_path = _relative_path(root, str(reference.get("image") or ""))
			if not reference_path.is_file() or reference.get("sha256") != sha256_file(reference_path):
				return None, "result_reference_image_mismatch"
		expected_reference_images = [str(job["source_image"]["path"]), *[str(item["image"]) for item in references]]
		if result.get("reference_images") != expected_reference_images:
			return None, "result_reference_images_mismatch"
		if result.get("rois_xyxy") != [list(item["roi_xyxy"]) for item in references]:
			return None, "result_rois_mismatch"
	elif references:
		return None, "non_l1_result_has_references"
	return result, None


def _load_and_validate_evaluation(
	root: Path,
	job: Mapping[str, Any],
	state: Mapping[str, Any] | None = None,
	evaluator_lock: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
	"""Load one blind evaluation and verify that it belongs to the exact action candidate pair."""
	path = _relative_path(root, str(job["evaluation_path"]))
	if not path.is_file():
		return None, "missing_evaluation"
	try:
		value = read_json(path)
	except Exception as exc:
		return None, f"invalid_evaluation_json:{exc}"
	if not isinstance(value, Mapping):
		return None, "evaluation_not_object"
	evaluation = dict(value)
	if evaluation.get("job_id") != job.get("job_id"):
		return None, "evaluation_job_id_mismatch"
	if evaluation.get("request_hash") != job.get("request_hash"):
		return None, "evaluation_request_hash_mismatch"
	if evaluation.get("status") != "ok":
		return None, f"evaluation_status:{evaluation.get('status', 'missing')}"
	if evaluation.get("blind_protocol") != BLIND_PROTOCOL:
		return None, "evaluation_not_blind"
	if not evaluation.get("evaluator_id") or not evaluation.get("prompt_version"):
		return None, "evaluation_contract_missing"
	if state is None or evaluator_lock is None:
		return None, "evaluation_lock_or_state_missing"
	if evaluation.get("evaluator_signature") != evaluator_lock.get("evaluator_signature"):
		return None, "evaluation_signature_mismatch"
	if evaluation.get("prompt_version") != PROMPT_VERSION:
		return None, "evaluation_prompt_version_mismatch"
	expected_order = ["original", "candidate"]
	if stable_seed(job["job_id"], "blind_order") % 2:
		expected_order.reverse()
	if evaluation.get("presentation_order") != expected_order:
		return None, "evaluation_presentation_order_mismatch"
	if evaluation.get("prompt_hash") != build_prompt_hash(state, expected_order):
		return None, "evaluation_prompt_hash_mismatch"
	output_path = _relative_path(root, str(job["output_image"]))
	source_path = Path(job["source_image"]["path"])
	if not source_path.is_file() or evaluation.get("source_image_sha256") != sha256_file(source_path):
		return None, "evaluation_source_image_hash_mismatch"
	if not output_path.is_file() or evaluation.get("candidate_image_sha256") != sha256_file(output_path):
		return None, "evaluation_candidate_image_hash_mismatch"
	if evaluation.get("success_definition") != SUCCESS_DEFINITION:
		return None, "evaluation_success_definition_mismatch"
	try:
		for name in EVALUATION_KEYS:
			_score(evaluation.get(name), name)
		if not isinstance(evaluation.get("introduced_new_error"), bool):
			raise DataValidationError("introduced_new_error must be boolean")
		if not isinstance(evaluation.get("success"), bool):
			raise DataValidationError("success must be boolean")
	except DataValidationError as exc:
		return None, f"invalid_evaluation:{exc}"
	return evaluation, None


def _pairwise_preferences(means: Sequence[float], epsilon: float) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for left in range(len(means)):
		for right in range(left + 1, len(means)):
			delta = means[left] - means[right]
			preference = 1 if delta > epsilon else (-1 if delta < -epsilon else 0)
			rows.append(
				{
					"left_level": ACTION_LEVELS[left],
					"right_level": ACTION_LEVELS[right],
					"utility_delta": delta,
					"preference": preference,
				}
			)
	return rows


def _training_balance_statistics(
	targets: list[dict[str, Any]],
	config: DatasetConfig,
) -> dict[str, Any]:
	"""Derive class weights from train soft mass; never reshape val/test."""

	train = [row for row in targets if row["split"] == "train"]
	if not train:
		return {
			"enabled": config.balance_train_soft_mass,
			"train_count": 0,
			"soft_target_mass": {f"L{level}": 0.0 for level in ACTION_LEVELS},
			"class_factors": {f"L{level}": 1.0 for level in ACTION_LEVELS},
			"effective_soft_target_mass": {f"L{level}": 0.0 for level in ACTION_LEVELS},
		}
	masses = [sum(float(row["soft_targets"][index]) for row in train) for index in range(len(ACTION_LEVELS))]
	total_mass = sum(masses)
	raw_factors = [
		total_mass / (len(ACTION_LEVELS) * mass) if mass > 0.0 else 1.0
		for mass in masses
	]
	maximum = config.max_train_class_weight
	minimum = 1.0 / maximum
	factors = (
		[min(maximum, max(minimum, factor)) for factor in raw_factors]
		if config.balance_train_soft_mass
		else [1.0] * len(ACTION_LEVELS)
	)
	effective_masses = [mass * factor for mass, factor in zip(masses, factors)]
	return {
		"enabled": config.balance_train_soft_mass,
		"train_count": len(train),
		"soft_target_mass": {
			f"L{level}": mass for level, mass in zip(ACTION_LEVELS, masses)
		},
		"class_factors": {
			f"L{level}": factor for level, factor in zip(ACTION_LEVELS, factors)
		},
		"effective_soft_target_mass": {
			f"L{level}": mass for level, mass in zip(ACTION_LEVELS, effective_masses)
		},
		"max_class_weight": config.max_train_class_weight,
	}


def aggregate_dataset(
	dataset_root: Path,
	*,
	dry_run: bool = False,
	artifact_workers: int = 1,
) -> dict[str, Any]:
	"""Create train/validation/test targets only from complete K-way evidence."""

	if artifact_workers <= 0:
		raise ValueError("artifact_workers must be positive")
	dataset_root = dataset_root.resolve()
	config = load_config(dataset_root / "config.lock.json", strict_revision=True)
	flux2_policy_contract = config.policy.collector_contract
	flux2_policy_digest = config.policy.collector_contract_digest
	l3_policy_contract = config.l3_policy.as_dict()
	l3_policy_digest = config.l3_policy.digest
	evaluator_lock_path = dataset_root / "evaluator.lock.json"
	if not evaluator_lock_path.is_file():
		raise DataValidationError("evaluator.lock.json is required before aggregation")
	try:
		evaluator_lock = validate_evaluator_contract(read_json(evaluator_lock_path))
	except Exception as exc:
		raise DataValidationError(f"invalid evaluator.lock.json: {exc}") from exc
	states = list(iter_jsonl(dataset_root / "states.jsonl"))
	qwen_distribution_signature = _locked_qwen_distribution_signature(states)
	planned_manifest = read_json(dataset_root / "manifest.json")
	if planned_manifest.get("qwen_distribution_signature") != qwen_distribution_signature:
		raise DataValidationError("planned manifest/state Qwen distribution signature mismatch")
	if planned_manifest.get("l3_policy_contract") != l3_policy_contract or planned_manifest.get("l3_policy_digest") != l3_policy_digest:
		raise DataValidationError("planned manifest/config initial-base L3 policy mismatch")
	jobs = list(iter_jsonl(dataset_root / "jobs.jsonl"))
	state_by_id = {row["state_id"]: row for row in states}
	if len(state_by_id) != len(states):
		raise DataValidationError("duplicate state_id in states manifest")
	if len({job.get("job_id") for job in jobs}) != len(jobs):
		raise DataValidationError("duplicate job_id in jobs manifest")
	expected_keys = {
		(state_id, rollout_index, level)
		for state_id in state_by_id
		for rollout_index in range(config.rollouts_per_action)
		for level in ACTION_LEVELS
	}
	actual_keys: set[tuple[str, int, int]] = set()
	for job in jobs:
		validate_action_job(job, config)
		key = (str(job.get("state_id")), int(job.get("rollout_index", -1)), int(job.get("action_level", -1)))
		if key in actual_keys:
			raise DataValidationError(f"duplicate state/rollout/action job: {key}")
		actual_keys.add(key)
		if key[0] in state_by_id:
			expected_seed = stable_seed(config.dataset_seed, key[0], key[1])
			if int(job.get("seed", 0)) != expected_seed:
				raise DataValidationError(f"job seed mismatch: {job.get('job_id')}")
	if actual_keys != expected_keys:
		missing = len(expected_keys - actual_keys)
		extra = len(actual_keys - expected_keys)
		raise DataValidationError(f"jobs are not exact L1/L2/L3 x K (missing={missing}, extra={extra})")
	jobs_by_state: dict[str, list[dict[str, Any]]] = {state_id: [] for state_id in state_by_id}
	for job in jobs:
		if job.get("state_id") not in jobs_by_state:
			raise DataValidationError(f"job references unknown state: {job.get('state_id')}")
		jobs_by_state[job["state_id"]].append(job)

	rollout_rows: list[dict[str, Any]] = []
	valid_by_state_level: dict[tuple[str, int], list[dict[str, Any]]] = {}
	def load_rollout(job: Mapping[str, Any]) -> dict[str, Any]:
		"""Validate one paired candidate and compute its scalar utility."""
		result, result_error = _load_and_validate_result(dataset_root, job)
		evaluation: dict[str, Any] | None = None
		evaluation_error: str | None = None
		utility: float | None = None
		if result is not None:
			evaluation, evaluation_error = _load_and_validate_evaluation(
				dataset_root, job, state_by_id[job["state_id"]], evaluator_lock
			)
		if result is not None and evaluation is not None:
			try:
				utility = rollout_utility(evaluation, result, config)
			except DataValidationError as exc:
				evaluation_error = f"invalid_utility:{exc}"
		reason = result_error or evaluation_error
		row = {
			"schema_version": SCHEMA_VERSION,
			"flux2_policy_digest": flux2_policy_digest,
			"l3_policy_digest": l3_policy_digest,
			"executor_family": job["executor_family"],
			"job_id": job["job_id"],
			"state_id": job["state_id"],
			"qwen_distribution_signature": qwen_distribution_signature,
			"group_id": job["group_id"],
			"split": job["split"],
			"action_level": job["action_level"],
			"rollout_index": job["rollout_index"],
			"seed": job["seed"],
			"valid": reason is None and utility is not None,
			"invalid_reason": reason,
			"utility": utility,
			"result_status": None if result is None else result.get("status"),
			"evaluation_status": None if evaluation is None else evaluation.get("status"),
		}
		if evaluation is not None:
			row["success"] = evaluation["success"]
			row["evaluation_scores"] = {name: evaluation[name] for name in EVALUATION_KEYS}
		if result is not None:
			row["call_count"] = int(result.get("call_count", 1))
		return row

	def consume(rows: Any) -> None:
		for row in rows:
			rollout_rows.append(row)
			if row["valid"]:
				valid_by_state_level.setdefault(
					(str(row["state_id"]), int(row["action_level"])),
					[],
				).append(row)

	if artifact_workers == 1:
		consume(map(load_rollout, jobs))
	else:
		# executor.map preserves the frozen job order. Submit bounded batches so
		# large K datasets do not retain one Future per action candidate.
		batch_size = max(256, artifact_workers * 16)
		with ThreadPoolExecutor(max_workers=artifact_workers) as executor:
			for start in range(0, len(jobs), batch_size):
				consume(executor.map(load_rollout, jobs[start : start + batch_size]))

	targets: list[dict[str, Any]] = []
	quarantine: list[dict[str, Any]] = []
	for state in states:
		state_id = state["state_id"]
		state_jobs = jobs_by_state.get(state_id, [])
		expected = len(ACTION_LEVELS) * config.rollouts_per_action
		level_counts = {
			str(level): len(valid_by_state_level.get((state_id, level), [])) for level in ACTION_LEVELS
		}
		job_level_counts = {
			str(level): sum(int(job["action_level"]) == level for job in state_jobs) for level in ACTION_LEVELS
		}
		reasons: list[str] = []
		if len(state_jobs) != expected:
			reasons.append(f"planned_job_count:{len(state_jobs)}!=expected:{expected}")
		for level in ACTION_LEVELS:
			if job_level_counts[str(level)] != config.rollouts_per_action:
				reasons.append(f"L{level}_planned:{job_level_counts[str(level)]}!=K")
			if level_counts[str(level)] != config.rollouts_per_action:
				reasons.append(f"L{level}_valid:{level_counts[str(level)]}!=K")
		if reasons:
			quarantine.append(
				{
					"schema_version": SCHEMA_VERSION,
					"flux2_policy_digest": flux2_policy_digest,
					"l3_policy_digest": l3_policy_digest,
					"state_id": state_id,
					"qwen_distribution_signature": qwen_distribution_signature,
					"group_id": state["group_id"],
					"split": state["split"],
					"reasons": reasons,
					"valid_rollout_counts": level_counts,
					"planned_job_counts": job_level_counts,
				}
			)
			continue

		level_summaries: dict[str, dict[str, Any]] = {}
		means: list[float] = []
		for level in ACTION_LEVELS:
			rows = sorted(valid_by_state_level[(state_id, level)], key=lambda value: value["rollout_index"])
			values = [float(row["utility"]) for row in rows]
			mean = fmean(values)
			std = pstdev(values) if len(values) > 1 else 0.0
			means.append(mean)
			level_summaries[str(level)] = {
				"mean_utility": mean,
				"std_utility": std,
				"standard_error": std / math.sqrt(len(values)),
				"success_rate": fmean(float(row.get("success", False)) for row in rows),
				"rollout_count": len(values),
				"job_ids": [row["job_id"] for row in rows],
			}
		probabilities = softmax(means, config.soft_target_temperature)
		best = max(means)
		best_levels = [level for level, value in zip(ACTION_LEVELS, means) if math.isclose(value, best, abs_tol=1e-12)]
		acceptable = [level for level, value in zip(ACTION_LEVELS, means) if best - value <= config.acceptable_epsilon]
		minimum = min(means)
		spread = best - minimum
		center = fmean(means)
		targets.append(
			{
				"schema_version": SCHEMA_VERSION,
				"flux2_policy_contract": flux2_policy_contract,
				"flux2_policy_digest": flux2_policy_digest,
				"l3_policy_contract": l3_policy_contract,
				"l3_policy_digest": l3_policy_digest,
				"initial_generator_family": config.l3_policy.family,
				"l3_initial_identity_binding": state["initial_generator"]["l3_identity_binding"],
				"state_id": state_id,
				"qwen_distribution_signature": qwen_distribution_signature,
				"group_id": state["group_id"],
				"split": state["split"],
				"dataset": state["dataset"],
				"error_family": state["error_family"],
				"source_round": state["source_round"],
				"image_path": state["current_image"]["path"],
				"image_sha256": state["current_image"]["sha256"],
				"target_prompt": state["target_prompt"],
				"diagnosis_text": canonical_feature_diagnosis(state),
				"action_mask": [1, 1, 1],
				"utility_vector": means,
				"centered_utility_vector": [value - center for value in means],
				"regret_vector": [best - value for value in means],
				"utility_spread": spread,
				"soft_targets": probabilities,
				"best_levels": best_levels,
				"hard_target": best_levels[0] if len(best_levels) == 1 else None,
				"acceptable_levels": acceptable,
				"pairwise_preferences": _pairwise_preferences(means, config.acceptable_epsilon),
				"action_statistics": level_summaries,
			}
		)

	training_balance = _training_balance_statistics(targets, config)
	rollout_rows.sort(key=lambda row: (row["state_id"], row["rollout_index"], row["action_level"]))
	targets.sort(key=lambda row: row["state_id"])
	quarantine.sort(key=lambda row: row["state_id"])
	summary = {
		"schema_version": SCHEMA_VERSION,
		"status": "aggregated",
		"flux2_policy_contract": flux2_policy_contract,
		"flux2_policy_digest": flux2_policy_digest,
		"l3_policy_contract": l3_policy_contract,
		"l3_policy_digest": l3_policy_digest,
		"initial_generator_family": config.l3_policy.family,
		"l3_initial_identity_binding_counts": dict(
			sorted(Counter(row["l3_initial_identity_binding"] for row in targets).items())
		),
		"state_count": len(states),
		"job_count": len(jobs),
		"valid_rollout_count": sum(row["valid"] for row in rollout_rows),
		"invalid_rollout_count": sum(not row["valid"] for row in rollout_rows),
		"trainable_state_count": len(targets),
		"quarantined_state_count": len(quarantine),
		"qwen_distribution_signature": qwen_distribution_signature,
		"split_target_counts": {
			name: sum(row["split"] == name for row in targets) for name in ("train", "validation", "test")
		},
		"target_distribution": {
			"unique_hard_target": dict(
				sorted(Counter(f"L{row['hard_target']}" for row in targets if row["hard_target"] is not None).items())
			),
			"ties": sum(row["hard_target"] is None for row in targets),
			"by_error_family": {
				family: dict(
					sorted(
						Counter(
							"tie" if row["hard_target"] is None else f"L{row['hard_target']}"
							for row in targets
							if row["error_family"] == family
						).items()
					)
				)
				for family in sorted({row["error_family"] for row in targets})
			},
		},
		"training_balance": training_balance,
		"aggregation": {
			"rollout_reducer": "arithmetic_mean",
			"requires_exact_valid_k_per_action": True,
			"soft_target_temperature": config.soft_target_temperature,
			"acceptable_epsilon": config.acceptable_epsilon,
		},
		"evaluator_signature": evaluator_lock["evaluator_signature"],
		"derived_digests": {
			"rollout_utilities": sha256_json(rollout_rows),
			"state_targets": sha256_json(targets),
			"quarantine": sha256_json(quarantine),
			**{
				f"split_{name}": sha256_json([row for row in targets if row["split"] == name])
				for name in ("train", "validation", "test")
			},
		},
	}
	if not dry_run:
		derived = dataset_root / "derived"
		write_jsonl_atomic(derived / "rollout_utilities.jsonl", rollout_rows)
		write_jsonl_atomic(derived / "state_targets.jsonl", targets)
		write_jsonl_atomic(derived / "quarantine.jsonl", quarantine)
		for split in ("train", "validation", "test"):
			write_jsonl_atomic(derived / f"{split}.jsonl", [row for row in targets if row["split"] == split])
		write_json_atomic(derived / "summary.json", summary)
	return summary
