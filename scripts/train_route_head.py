#!/usr/bin/env python3
"""Extract frozen V10 Qwen route features and train the three-class MLP head.

The input is the exact ``derived/{train,validation,test}.jsonl`` bundle emitted
by ``inference.router_data_collect.aggregate``.  Feature extraction deliberately calls
the online V10 ``QwenPlanner.encode_route_features`` implementation with the
raw, unannotated current image.  The Qwen backbone is never optimized; only
``RouteMLP`` parameters are passed to AdamW.

GPU extraction is a separate, cache-producing stage.  ``train`` accepts that
cache directly, which keeps data/loss/checkpoint tests CPU-only and lets MLP
experiments run without repeatedly loading Qwen.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from inference.main_method.modules.contracts import (  # noqa: E402
	DIAGNOSIS_SCHEMA,
	ROUTER_ACTION_POLICY_VERSION,
	ROUTER_REPAIR_EXECUTOR_FAMILY,
	build_qwen_analyzer_l1_distribution_contract,
	canonical_diagnosis_text,
	diagnosis_payload_is_valid,
	model_visible_diagnosis_text,
	qwen_effective_quantization,
)
from inference.main_method.modules.route_head import (  # noqa: E402
	DIAGNOSIS_AUX_FEATURE_NAMES,
	ROUTE_CANDIDATE_QUALITY_SCHEMA,
	ROUTE_CHECKPOINT_VERSION,
	ROUTE_DIAGNOSIS_GROUP_EB_SCHEMA,
	ROUTE_NAMES,
	RouteMLP,
	build_diagnosis_aux_contract,
	build_route_encoder_messages,
	canonical_qwen_dtype_name,
	canonical_qwen_quantization_name,
	diagnosis_aux_vector,
)
from inference.main_method.modules.flux_policy import (  # noqa: E402
	flux2_policy_digest,
	validate_flux2_policy_contract,
)
from inference.main_method.modules.l3_policy import (  # noqa: E402
	l3_policy_digest,
	validate_l3_policy_contract,
)
from inference.router_data_collect.canonical import sha256_json  # noqa: E402
from inference.router_data_collect.router_loss import utility_aware_router_loss  # noqa: E402
from inference.router_data_collect.schema import SCHEMA_VERSION as DATA_SCHEMA_VERSION  # noqa: E402


SPLITS = ("train", "validation", "test")
LABEL_LEVELS = (1, 2, 3)
FEATURE_SCHEMA = "vcot_v10_qwen_route_feature"
FEATURE_SCHEMA_VERSION = 2
FEATURE_CACHE_SCHEMA = "vcot_v10_qwen_route_feature_cache_v2"
CHECKPOINT_VERSION = ROUTE_CHECKPOINT_VERSION
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$", flags=re.IGNORECASE)
TARGET_MODES = ("aggregate_utility", "candidate_quality")
TRAIN_TARGET_SHRINKAGE_MODES = ("none", "empirical_bayes_l2", "diagnosis_group_eb_l2")
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))


class TrainingDataError(ValueError):
	"""Raised when an aggregate split or feature cache violates the V10 contract."""


def _canonical_json(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
	return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _json_safe(value: Any) -> Any:
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, Mapping):
		return {str(key): _json_safe(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
	if isinstance(value, (list, tuple)):
		return [_json_safe(child) for child in value]
	return str(value)


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
	try:
		value = json.loads(path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise TrainingDataError(f"Cannot read JSON {path}: {exc}") from exc
	if not isinstance(value, dict):
		raise TrainingDataError(f"Expected a JSON object: {path}")
	return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	try:
		with path.open("r", encoding="utf-8") as handle:
			for line_number, line in enumerate(handle, start=1):
				line = line.strip()
				if not line:
					continue
				value = json.loads(line)
				if not isinstance(value, dict):
					raise TrainingDataError(f"{path}:{line_number} must be a JSON object")
				rows.append(value)
	except TrainingDataError:
		raise
	except Exception as exc:
		raise TrainingDataError(f"Cannot read JSONL {path}: {exc}") from exc
	return rows


def _finite_vector(value: Any, name: str, *, length: int = 3) -> list[float]:
	if not isinstance(value, list) or len(value) != length:
		raise TrainingDataError(f"{name} must contain exactly {length} values")
	try:
		result = [float(item) for item in value]
	except (TypeError, ValueError) as exc:
		raise TrainingDataError(f"{name} must be numeric") from exc
	if not all(math.isfinite(item) for item in result):
		raise TrainingDataError(f"{name} must be finite")
	return result


def _resolve_image_path(dataset_root: Path, value: Any) -> Path:
	path = Path(str(value or ""))
	if not path.is_absolute():
		path = dataset_root / path
	return path.resolve()


def _feature_input_digest(row: Mapping[str, Any]) -> str:
	return _sha256_text(
		_canonical_json(
			{
				"state_id": row["state_id"],
				"image_sha256": row["image_sha256"],
				"target_prompt": row["target_prompt"],
				"diagnosis_text": row["diagnosis_text"],
			}
		)
	)


def _validate_pairwise(row: Mapping[str, Any], utility: Sequence[float], epsilon: float, prefix: str) -> None:
	pairs = row.get("pairwise_preferences", row.get("pairwise"))
	if not isinstance(pairs, list) or len(pairs) != 3:
		raise TrainingDataError(f"{prefix}.pairwise_preferences must contain all three level pairs")
	seen: set[tuple[int, int]] = set()
	for pair in pairs:
		if not isinstance(pair, Mapping):
			raise TrainingDataError(f"{prefix}.pairwise_preferences entries must be objects")
		try:
			left = int(pair.get("left_level"))
			right = int(pair.get("right_level"))
		except (TypeError, ValueError) as exc:
			raise TrainingDataError(f"{prefix}.pairwise_preferences levels must be integers") from exc
		if (left, right) not in {(1, 2), (1, 3), (2, 3)} or (left, right) in seen:
			raise TrainingDataError(f"{prefix}.pairwise_preferences has an invalid or duplicate pair")
		seen.add((left, right))
		delta = utility[left - 1] - utility[right - 1]
		try:
			recorded_delta = float(pair.get("utility_delta"))
			preference = int(pair.get("preference"))
		except (TypeError, ValueError) as exc:
			raise TrainingDataError(f"{prefix}.pairwise_preferences values are invalid") from exc
		if not math.isclose(recorded_delta, delta, abs_tol=1e-7):
			raise TrainingDataError(f"{prefix}.pairwise_preferences utility_delta disagrees with utility_vector")
		expected = 1 if delta > epsilon else (-1 if delta < -epsilon else 0)
		if preference != expected:
			raise TrainingDataError(f"{prefix}.pairwise_preferences preference disagrees with aggregate epsilon")
	if seen != {(1, 2), (1, 3), (2, 3)}:
		raise TrainingDataError(f"{prefix}.pairwise_preferences is incomplete")


def _validate_row(
	row: dict[str, Any],
	*,
	split: str,
	dataset_root: Path,
	epsilon: float,
	verify_image_hashes: bool,
	qwen_distribution_signature: str,
	flux2_policy_contract: Mapping[str, Any],
	flux2_policy_contract_digest: str,
	l3_policy_contract: Mapping[str, Any],
	l3_policy_contract_digest: str,
	allow_legacy_l3_identity_ablation: bool,
) -> dict[str, Any]:
	prefix = f"{split}:{row.get('state_id', '<missing>')}"
	if row.get("schema_version") != DATA_SCHEMA_VERSION:
		raise TrainingDataError(f"{prefix}.schema_version must be {DATA_SCHEMA_VERSION!r}")
	if row.get("split") != split:
		raise TrainingDataError(f"{prefix}.split does not match {split!r}")
	for key in ("state_id", "group_id", "target_prompt", "diagnosis_text", "image_path", "image_sha256"):
		if not str(row.get(key) or "").strip():
			raise TrainingDataError(f"{prefix}.{key} is required")
	if row.get("qwen_distribution_signature") != qwen_distribution_signature:
		raise TrainingDataError(f"{prefix}.qwen_distribution_signature differs from aggregate summary")
	if row.get("flux2_policy_digest") != flux2_policy_contract_digest:
		raise TrainingDataError(f"{prefix}.flux2_policy_digest differs from aggregate summary")
	try:
		row_flux2_policy = validate_flux2_policy_contract(row.get("flux2_policy_contract"))
	except (TypeError, ValueError) as exc:
		raise TrainingDataError(f"{prefix}.flux2_policy_contract is invalid: {exc}") from exc
	if row_flux2_policy != dict(flux2_policy_contract):
		raise TrainingDataError(f"{prefix}.flux2_policy_contract differs from aggregate summary")
	if row.get("l3_policy_digest") != l3_policy_contract_digest:
		raise TrainingDataError(f"{prefix}.l3_policy_digest differs from aggregate summary")
	try:
		row_l3_policy = validate_l3_policy_contract(row.get("l3_policy_contract"))
	except (TypeError, ValueError) as exc:
		raise TrainingDataError(f"{prefix}.l3_policy_contract is invalid: {exc}") from exc
	if row_l3_policy != dict(l3_policy_contract):
		raise TrainingDataError(f"{prefix}.l3_policy_contract differs from aggregate summary")
	if row.get("initial_generator_family") != l3_policy_contract["executor"]["family"]:
		raise TrainingDataError(f"{prefix}.initial_generator_family differs from its L3 executor")
	l3_identity_binding = row.get("l3_initial_identity_binding")
	if l3_identity_binding not in {
		"exact_model_identity",
		"legacy_bagel_family_only_unverified_revision",
	}:
		raise TrainingDataError(f"{prefix}.l3_initial_identity_binding is missing or invalid")
	if l3_identity_binding != "exact_model_identity" and not allow_legacy_l3_identity_ablation:
		raise TrainingDataError(
			f"{prefix}.l3_initial_identity_binding is legacy/unverified; main training requires exact model identity. "
			"Use --allow-legacy-l3-identity-ablation only for an explicitly reported ablation."
		)
	if (
		l3_identity_binding == "legacy_bagel_family_only_unverified_revision"
		and l3_policy_contract["executor"]["family"] != "bagel"
	):
		raise TrainingDataError(f"{prefix}.legacy L3 identity binding is only valid for BAGEL")
	if row.get("action_mask") != [1, 1, 1]:
		raise TrainingDataError(f"{prefix}.action_mask must be [1,1,1]")
	soft_targets = _finite_vector(row.get("soft_targets"), f"{prefix}.soft_targets")
	if any(value < 0.0 or value > 1.0 for value in soft_targets) or not math.isclose(sum(soft_targets), 1.0, abs_tol=1e-6):
		raise TrainingDataError(f"{prefix}.soft_targets must be probabilities summing to one")
	utility = _finite_vector(row.get("utility_vector"), f"{prefix}.utility_vector")
	centered = _finite_vector(row.get("centered_utility_vector"), f"{prefix}.centered_utility_vector")
	center = sum(utility) / 3.0
	if any(not math.isclose(value, source - center, abs_tol=1e-7) for value, source in zip(centered, utility)):
		raise TrainingDataError(f"{prefix}.centered_utility_vector disagrees with utility_vector")
	_validate_pairwise(row, utility, epsilon, prefix)
	try:
		diagnosis = json.loads(str(row["diagnosis_text"]))
	except json.JSONDecodeError as exc:
		raise TrainingDataError(f"{prefix}.diagnosis_text is not JSON") from exc
	if not diagnosis_payload_is_valid(diagnosis):
		raise TrainingDataError(f"{prefix}.diagnosis_text violates the V10 diagnosis schema")
	if diagnosis.get("schema_version") != DIAGNOSIS_SCHEMA:
		raise TrainingDataError(f"{prefix}.diagnosis_text has the wrong schema version")
	canonical = canonical_diagnosis_text(diagnosis)
	if str(row["diagnosis_text"]) != canonical:
		raise TrainingDataError(f"{prefix}.diagnosis_text must use the shared canonical serialization")
	# Exercising the shared message builder here catches future route-field or
	# role drift before an expensive feature-extraction run.
	messages = build_route_encoder_messages(str(row["target_prompt"]), canonical, "<raw-current-image>")
	visible_diagnosis = model_visible_diagnosis_text(diagnosis)
	if messages[-1]["content"][0]["text"] != visible_diagnosis:
		raise TrainingDataError(f"{prefix}.diagnosis facts changed inside the shared route encoder")
	image_path = _resolve_image_path(dataset_root, row["image_path"])
	if not image_path.is_file():
		raise TrainingDataError(f"{prefix}.image_path does not exist: {image_path}")
	image_sha256 = str(row["image_sha256"]).lower()
	if not re.fullmatch(r"[0-9a-f]{64}", image_sha256):
		raise TrainingDataError(f"{prefix}.image_sha256 is invalid")
	if verify_image_hashes and _sha256_file(image_path) != image_sha256:
		raise TrainingDataError(f"{prefix}.image_sha256 does not match the raw current image")
	confidence = float(row.get("confidence", 1.0))
	sample_weight = float(row.get("sample_weight", 1.0))
	if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
		raise TrainingDataError(f"{prefix}.confidence must be in [0,1]")
	if not math.isfinite(sample_weight) or sample_weight < 0.0:
		raise TrainingDataError(f"{prefix}.sample_weight must be finite and non-negative")
	result = dict(row)
	result["image_path"] = str(image_path)
	result["image_sha256"] = image_sha256
	result["confidence"] = confidence
	result["sample_weight"] = sample_weight
	result["feature_input_digest"] = _feature_input_digest(result)
	return result


@dataclass(frozen=True)
class DatasetBundle:
	root: Path
	rows: dict[str, list[dict[str, Any]]]
	summary: dict[str, Any]
	class_factors: tuple[float, float, float]
	acceptable_epsilon: float
	qwen_distribution_signature: str
	flux2_policy_contract: dict[str, Any]
	flux2_policy_digest: str
	l3_policy_contract: dict[str, Any]
	l3_policy_digest: str
	l3_initial_identity_binding_counts: dict[str, int]
	allow_legacy_l3_identity_ablation: bool
	target_mode: str
	train_target_shrinkage: str
	target_processing: dict[str, Any]
	target_digest: str
	digest: str


def _softmax_vector(values: Sequence[float], temperature: float) -> list[float]:
	if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
		raise TrainingDataError("soft-target temperature must be finite and positive")
	maximum = max(values)
	exponents = [math.exp((float(value) - maximum) / float(temperature)) for value in values]
	total = sum(exponents)
	return [value / total for value in exponents]


def _reliable_paired_mask(
	per_level: Sequence[Sequence[float]],
	*,
	acceptable_epsilon: float,
	sign_epsilon: float = 1e-8,
) -> list[bool]:
	"""Keep only mean preferences supported by a strict paired-seed majority."""

	rollout_counts = {len(values) for values in per_level}
	if len(per_level) != len(LABEL_LEVELS) or len(rollout_counts) != 1:
		raise TrainingDataError("candidate-quality levels must have one equal paired rollout count")
	rollout_count = rollout_counts.pop()
	if rollout_count <= 0:
		raise TrainingDataError("candidate-quality paired rollout count must be positive")
	required_votes = rollout_count // 2 + 1
	masks: list[bool] = []
	for left, right in PAIR_INDICES:
		differences = [float(a) - float(b) for a, b in zip(per_level[left], per_level[right])]
		mean_delta = sum(differences) / rollout_count
		if abs(mean_delta) <= acceptable_epsilon:
			masks.append(False)
			continue
		direction = 1.0 if mean_delta > 0.0 else -1.0
		consistent_votes = sum(direction * value > sign_epsilon for value in differences)
		masks.append(consistent_votes >= required_votes)
	return masks


def _apply_candidate_quality_targets(
	dataset_root: Path,
	rows: dict[str, list[dict[str, Any]]],
	*,
	acceptable_epsilon: float,
	soft_target_temperature: float,
) -> dict[str, Any]:
	"""Remove repeated-original judge noise from cross-level quality targets.

	The aggregate utility contains ``target_gain * (candidate - original)`` and
	a per-call cost.  Within one state/paired seed, the original image and the
	one-call action budget are identical for L1/L2/L3, so both terms must cancel
	when learning the route.  Reconstructing candidate-only quality prevents
	three independent re-scores of the same original image from changing labels.
	"""

	config = _read_json(dataset_root / "config.lock.json")
	utility_weights = config.get("utility")
	if not isinstance(utility_weights, Mapping):
		raise TrainingDataError("config.lock.json is missing utility weights")
	try:
		target_gain = float(utility_weights["target_gain"])
		cost_per_call = float(utility_weights["cost_per_call"])
	except (KeyError, TypeError, ValueError) as exc:
		raise TrainingDataError("config.lock.json candidate-quality weights are invalid") from exc
	if not all(math.isfinite(value) for value in (target_gain, cost_per_call)):
		raise TrainingDataError("candidate-quality weights must be finite")

	rollout_path = dataset_root / "derived" / "rollout_utilities.jsonl"
	if not rollout_path.is_file():
		raise TrainingDataError("candidate_quality target mode requires derived/rollout_utilities.jsonl")
	rollout_rows = _read_jsonl(rollout_path)
	summary = _read_json(dataset_root / "derived" / "summary.json")
	expected_rollout_digest = str(summary.get("derived_digests", {}).get("rollout_utilities") or "")
	actual_rollout_digest = sha256_json(rollout_rows)
	if expected_rollout_digest != actual_rollout_digest:
		raise TrainingDataError("derived/rollout_utilities.jsonl digest mismatch")
	expected_states = {
		str(row["state_id"]): row
		for split_rows in rows.values()
		for row in split_rows
	}
	by_key: dict[tuple[str, int, int], float] = {}
	seeds_by_pair: dict[tuple[str, int], dict[int, int]] = {}
	for rollout in rollout_rows:
		if rollout.get("valid") is not True:
			continue
		try:
			state_id = str(rollout["state_id"])
			level = int(rollout["action_level"])
			rollout_index = int(rollout["rollout_index"])
			seed = int(rollout["seed"])
			utility = float(rollout["utility"])
			call_count = int(rollout["call_count"])
			original_score = float(rollout["evaluation_scores"]["original_target_satisfaction"])
		except (KeyError, TypeError, ValueError) as exc:
			raise TrainingDataError("invalid candidate-quality rollout row") from exc
		if state_id not in expected_states:
			continue
		expected = expected_states[state_id]
		if level not in LABEL_LEVELS or rollout_index < 0 or seed <= 0 or call_count <= 0:
			raise TrainingDataError(f"invalid candidate-quality rollout identity: {state_id}/L{level}/{rollout_index}")
		if not all(math.isfinite(value) for value in (utility, original_score)) or not 0.0 <= original_score <= 1.0:
			raise TrainingDataError(f"invalid candidate-quality rollout score: {state_id}/L{level}/{rollout_index}")
		for field, expected_value in (
			("split", expected["split"]),
			("group_id", expected["group_id"]),
			("qwen_distribution_signature", expected["qwen_distribution_signature"]),
			("flux2_policy_digest", expected["flux2_policy_digest"]),
			("l3_policy_digest", expected["l3_policy_digest"]),
		):
			if rollout.get(field) != expected_value:
				raise TrainingDataError(
					f"candidate-quality rollout {field} mismatch: {state_id}/L{level}/{rollout_index}"
				)
		key = (state_id, level, rollout_index)
		if key in by_key:
			raise TrainingDataError(f"duplicate candidate-quality rollout: {key}")
		quality = utility + target_gain * original_score + cost_per_call * call_count
		if not math.isfinite(quality):
			raise TrainingDataError(f"non-finite candidate quality: {key}")
		by_key[key] = quality
		seeds_by_pair.setdefault((state_id, rollout_index), {})[level] = seed

	for split, split_rows in rows.items():
		for index, source in enumerate(split_rows):
			state_id = str(source["state_id"])
			rollout_count = int(source.get("action_statistics", {}).get("1", {}).get("rollout_count", 0))
			if rollout_count <= 0:
				rollout_count = len([
					key for key in by_key if key[0] == state_id and key[1] == 1
				])
			if rollout_count <= 0:
				raise TrainingDataError(f"candidate-quality state has no paired rollouts: {state_id}")
			per_level: list[list[float]] = []
			for level in LABEL_LEVELS:
				values = []
				for rollout_index in range(rollout_count):
					key = (state_id, level, rollout_index)
					if key not in by_key:
						raise TrainingDataError(f"missing candidate-quality rollout: {key}")
					values.append(by_key[key])
				per_level.append(values)
			means = [sum(values) / len(values) for values in per_level]
			center = sum(means) / len(means)
			best = max(means)
			for rollout_index in range(rollout_count):
				paired_seeds = seeds_by_pair.get((state_id, rollout_index), {})
				if set(paired_seeds) != set(LABEL_LEVELS) or len(set(paired_seeds.values())) != 1:
					raise TrainingDataError(f"unpaired candidate-quality seed: {state_id}/{rollout_index}")
			pairwise_mask = _reliable_paired_mask(
				per_level,
				acceptable_epsilon=acceptable_epsilon,
			)
			row = dict(source)
			row["aggregate_utility_vector"] = list(source["utility_vector"])
			row["utility_vector"] = means
			row["centered_utility_vector"] = [value - center for value in means]
			row["regret_vector"] = [best - value for value in means]
			row["utility_spread"] = best - min(means)
			row["soft_targets"] = _softmax_vector(means, soft_target_temperature)
			row["quality_pairwise_mask"] = pairwise_mask
			row["quality_rollout_vectors"] = [
				[per_level[level][rollout_index] for level in range(len(LABEL_LEVELS))]
				for rollout_index in range(rollout_count)
			]
			row["target_mode"] = "candidate_quality"
			split_rows[index] = row
	return {
		"schema": ROUTE_CANDIDATE_QUALITY_SCHEMA,
		"rollout_utilities_digest": actual_rollout_digest,
		"pairwise_mask_policy": "mean_gap_and_strict_same_direction_paired_seed_majority_v2",
	}


def _apply_empirical_bayes_l2_shrinkage(
	rows: dict[str, list[dict[str, Any]]],
	*,
	acceptable_epsilon: float,
	soft_target_temperature: float,
) -> dict[str, Any]:
	"""Shrink noisy train action deltas toward the train-only L2 prior."""

	train = rows["train"]
	if len(train) < 2:
		raise TrainingDataError("empirical_bayes_l2 requires at least two train states")
	rollout_counts: set[int] = set()
	for row in train:
		vectors = row.get("quality_rollout_vectors")
		if not isinstance(vectors, list) or len(vectors) < 2:
			raise TrainingDataError("empirical_bayes_l2 requires at least two paired candidate-quality rollouts")
		rollout_counts.add(len(vectors))
		for vector in vectors:
			_finite_vector(vector, "quality_rollout_vectors row")
	if len(rollout_counts) != 1:
		raise TrainingDataError("empirical_bayes_l2 requires one shared rollout count")
	parameters: dict[str, Any] = {
		"schema": "vcot_v10_train_only_empirical_bayes_l2_delta_v2",
		"reference_level": 2,
		"pairwise_mask_policy": "intersection_with_raw_paired_seed_reliability",
		"actions": {},
	}
	for action_index in (0, 2):
		means: list[float] = []
		within_variances: list[float] = []
		for row in train:
			vectors = row.get("quality_rollout_vectors")
			if not isinstance(vectors, list) or len(vectors) < 2:
				raise TrainingDataError("empirical_bayes_l2 requires paired candidate-quality rollouts")
			deltas = [float(vector[action_index]) - float(vector[1]) for vector in vectors]
			means.append(sum(deltas) / len(deltas))
			center = means[-1]
			within_variances.append(
				sum((value - center) ** 2 for value in deltas) / (len(deltas) - 1)
			)
		global_mean = sum(means) / len(means)
		mean_center = global_mean
		mean_variance = sum((value - mean_center) ** 2 for value in means) / (len(means) - 1)
		within_variance = sum(within_variances) / len(within_variances)
		rollout_count = next(iter(rollout_counts))
		between_variance = max(0.0, mean_variance - within_variance / rollout_count)
		denominator = between_variance + within_variance / rollout_count
		shrinkage = between_variance / denominator if denominator > 0 else 0.0
		parameters["actions"][f"L{action_index + 1}_minus_L2"] = {
			"global_mean_delta": global_mean,
			"mean_delta_variance": mean_variance,
			"pooled_within_variance": within_variance,
			"between_state_variance": between_variance,
			"rollout_count": rollout_count,
			"reliability": shrinkage,
		}

	for index, source in enumerate(train):
		vectors = source["quality_rollout_vectors"]
		posterior_deltas: dict[int, float] = {}
		for action_index in (0, 2):
			key = f"L{action_index + 1}_minus_L2"
			parameter = parameters["actions"][key]
			delta = sum(float(vector[action_index]) - float(vector[1]) for vector in vectors) / len(vectors)
			posterior_deltas[action_index] = (
				float(parameter["global_mean_delta"]) * (1.0 - float(parameter["reliability"]))
				+ delta * float(parameter["reliability"])
			)
		means = [posterior_deltas[0], 0.0, posterior_deltas[2]]
		center = sum(means) / len(means)
		best = max(means)
		row = dict(source)
		row["unshrunk_candidate_quality_vector"] = list(source["utility_vector"])
		row["unshrunk_quality_pairwise_mask"] = list(source["quality_pairwise_mask"])
		row["utility_vector"] = means
		row["centered_utility_vector"] = [value - center for value in means]
		row["regret_vector"] = [best - value for value in means]
		row["utility_spread"] = best - min(means)
		row["soft_targets"] = _softmax_vector(means, soft_target_temperature)
		raw_pairwise_mask = source.get("quality_pairwise_mask")
		if not isinstance(raw_pairwise_mask, list) or len(raw_pairwise_mask) != len(PAIR_INDICES):
			raise TrainingDataError("empirical_bayes_l2 requires candidate-quality pairwise masks")
		row["quality_pairwise_mask"] = [
			bool(raw_pairwise_mask[pair_index]) and abs(means[left] - means[right]) > acceptable_epsilon
			for pair_index, (left, right) in enumerate(PAIR_INDICES)
		]
		row["train_target_shrinkage"] = "empirical_bayes_l2"
		train[index] = row
	return parameters


def _diagnosis_group_signature(row: Mapping[str, Any]) -> tuple[str, ...]:
	try:
		diagnosis = json.loads(str(row["diagnosis_text"]))
	except (KeyError, TypeError, json.JSONDecodeError) as exc:
		raise TrainingDataError("diagnosis_group_eb_l2 requires canonical diagnosis_text") from exc
	issues = diagnosis.get("issues") if isinstance(diagnosis, Mapping) else None
	if not isinstance(issues, list) or not issues:
		raise TrainingDataError("diagnosis_group_eb_l2 requires at least one diagnosis issue")
	types: list[str] = []
	for issue in issues:
		constraint_type = str(issue.get("constraint_type") or "").strip().lower() if isinstance(issue, Mapping) else ""
		if not constraint_type:
			raise TrainingDataError("diagnosis_group_eb_l2 found an issue without constraint_type")
		types.append(constraint_type)
	return tuple(sorted(types))


def _apply_diagnosis_group_eb_l2(
	rows: dict[str, list[dict[str, Any]]],
	*,
	acceptable_epsilon: float,
	soft_target_temperature: float,
	prior_strength: float = 5.0,
) -> dict[str, Any]:
	"""Denoise train targets by diagnosis structure, backed off to global L2."""

	if not math.isfinite(float(prior_strength)) or float(prior_strength) <= 0.0:
		raise TrainingDataError("diagnosis_group_eb_l2 prior_strength must be finite and positive")
	train = rows["train"]
	if len(train) < 2:
		raise TrainingDataError("diagnosis_group_eb_l2 requires at least two train states")
	group_rows: dict[tuple[str, ...], list[dict[str, Any]]] = {}
	for row in train:
		group_rows.setdefault(_diagnosis_group_signature(row), []).append(row)
	global_deltas = {
		action_index: sum(float(row["utility_vector"][action_index]) - float(row["utility_vector"][1]) for row in train)
		/ len(train)
		for action_index in (0, 2)
	}
	group_parameters: dict[tuple[str, ...], dict[str, Any]] = {}
	for signature, members in group_rows.items():
		posterior = [0.0, 0.0, 0.0]
		for action_index in (0, 2):
			mean_delta = sum(
				float(row["utility_vector"][action_index]) - float(row["utility_vector"][1])
				for row in members
			) / len(members)
			posterior[action_index] = (
				len(members) * mean_delta + float(prior_strength) * global_deltas[action_index]
			) / (len(members) + float(prior_strength))
		pairwise_mask: list[bool] = []
		for left, right in PAIR_INDICES:
			posterior_delta = posterior[left] - posterior[right]
			direction = 1.0 if posterior_delta > 0.0 else -1.0
			deltas = [float(row["utility_vector"][left]) - float(row["utility_vector"][right]) for row in members]
			consistent = sum(direction * delta > 1e-8 for delta in deltas)
			opposed = sum(direction * delta < -1e-8 for delta in deltas)
			pairwise_mask.append(
				abs(posterior_delta) > acceptable_epsilon
				and consistent >= 3
				and consistent > opposed
			)
		group_parameters[signature] = {
			"member_count": len(members),
			"posterior_delta_vector": posterior,
			"quality_pairwise_mask": pairwise_mask,
		}
	for index, source in enumerate(train):
		signature = _diagnosis_group_signature(source)
		parameter = group_parameters[signature]
		means = [float(value) for value in parameter["posterior_delta_vector"]]
		center = sum(means) / len(means)
		best = max(means)
		row = dict(source)
		row["unshrunk_candidate_quality_vector"] = list(source["utility_vector"])
		row["unshrunk_quality_pairwise_mask"] = list(source["quality_pairwise_mask"])
		row["utility_vector"] = means
		row["centered_utility_vector"] = [value - center for value in means]
		row["regret_vector"] = [best - value for value in means]
		row["utility_spread"] = best - min(means)
		row["soft_targets"] = _softmax_vector(means, soft_target_temperature)
		row["quality_pairwise_mask"] = list(parameter["quality_pairwise_mask"])
		row["train_target_shrinkage"] = "diagnosis_group_eb_l2"
		row["diagnosis_group_signature"] = list(signature)
		train[index] = row
	return {
		"schema": ROUTE_DIAGNOSIS_GROUP_EB_SCHEMA,
		"reference_level": 2,
		"prior_strength": float(prior_strength),
		"global_delta_vector": [global_deltas[0], 0.0, global_deltas[2]],
		"group_count": len(group_parameters),
		"groups": {
			"|".join(signature): parameter
			for signature, parameter in sorted(group_parameters.items())
		},
	}


def load_dataset_bundle(
	dataset_root: Path,
	*,
	verify_image_hashes: bool = True,
	allow_legacy_l3_identity_ablation: bool = False,
	target_mode: str = "aggregate_utility",
	train_target_shrinkage: str = "none",
) -> DatasetBundle:
	dataset_root = dataset_root.resolve()
	target_mode = str(target_mode or "aggregate_utility").strip().lower()
	if target_mode not in TARGET_MODES:
		raise TrainingDataError(f"unsupported target_mode: {target_mode!r}")
	train_target_shrinkage = str(train_target_shrinkage or "none").strip().lower()
	if train_target_shrinkage not in TRAIN_TARGET_SHRINKAGE_MODES:
		raise TrainingDataError(f"unsupported train_target_shrinkage: {train_target_shrinkage!r}")
	if train_target_shrinkage != "none" and target_mode != "candidate_quality":
		raise TrainingDataError("train target shrinkage requires target_mode=candidate_quality")
	derived = dataset_root / "derived"
	summary = _read_json(derived / "summary.json")
	if summary.get("schema_version") != DATA_SCHEMA_VERSION:
		raise TrainingDataError("derived/summary.json schema_version mismatch")
	qwen_distribution_signature = str(summary.get("qwen_distribution_signature") or "").strip().lower()
	if re.fullmatch(r"[0-9a-f]{64}", qwen_distribution_signature) is None:
		raise TrainingDataError("derived/summary.json is missing a valid qwen_distribution_signature")
	try:
		flux2_policy_contract = validate_flux2_policy_contract(summary.get("flux2_policy_contract"))
	except (TypeError, ValueError) as exc:
		raise TrainingDataError(f"derived/summary.json Flux.2 collector policy is invalid: {exc}") from exc
	flux2_policy_contract_digest = flux2_policy_digest(flux2_policy_contract)
	if summary.get("flux2_policy_digest") != flux2_policy_contract_digest:
		raise TrainingDataError("derived/summary.json Flux.2 collector-policy digest mismatch")
	try:
		l3_policy_contract = validate_l3_policy_contract(summary.get("l3_policy_contract"))
	except (TypeError, ValueError) as exc:
		raise TrainingDataError(f"derived/summary.json initial-base L3 policy is invalid: {exc}") from exc
	l3_policy_contract_digest = l3_policy_digest(l3_policy_contract)
	if summary.get("l3_policy_digest") != l3_policy_contract_digest:
		raise TrainingDataError("derived/summary.json initial-base L3 policy digest mismatch")
	if summary.get("initial_generator_family") != l3_policy_contract["executor"]["family"]:
		raise TrainingDataError("derived/summary.json initial generator differs from its L3 policy")
	try:
		epsilon = float(summary["aggregation"]["acceptable_epsilon"])
	except (KeyError, TypeError, ValueError) as exc:
		raise TrainingDataError("derived/summary.json is missing aggregation.acceptable_epsilon") from exc
	if not math.isfinite(epsilon) or epsilon < 0.0:
		raise TrainingDataError("aggregation.acceptable_epsilon must be finite and non-negative")
	try:
		factor_map = summary["training_balance"]["class_factors"]
		class_factors = tuple(float(factor_map[f"L{level}"]) for level in LABEL_LEVELS)
	except (KeyError, TypeError, ValueError) as exc:
		raise TrainingDataError("summary training_balance.class_factors must contain L1/L2/L3") from exc
	if any(not math.isfinite(value) or value <= 0.0 for value in class_factors):
		raise TrainingDataError("class_factors must be finite and positive")
	rows: dict[str, list[dict[str, Any]]] = {}
	seen_states: set[str] = set()
	group_split: dict[str, str] = {}
	for split in SPLITS:
		path = derived / f"{split}.jsonl"
		if not path.is_file():
			raise TrainingDataError(f"Missing aggregate split: {path}")
		validated: list[dict[str, Any]] = []
		for raw in _read_jsonl(path):
			row = _validate_row(
				raw,
				split=split,
				dataset_root=dataset_root,
				epsilon=epsilon,
				verify_image_hashes=verify_image_hashes,
				qwen_distribution_signature=qwen_distribution_signature,
				flux2_policy_contract=flux2_policy_contract,
				flux2_policy_contract_digest=flux2_policy_contract_digest,
				l3_policy_contract=l3_policy_contract,
				l3_policy_contract_digest=l3_policy_contract_digest,
				allow_legacy_l3_identity_ablation=allow_legacy_l3_identity_ablation,
			)
			state_id = str(row["state_id"])
			if state_id in seen_states:
				raise TrainingDataError(f"Duplicate state_id across derived splits: {state_id}")
			seen_states.add(state_id)
			group_id = str(row["group_id"])
			previous_split = group_split.setdefault(group_id, split)
			if previous_split != split:
				raise TrainingDataError(f"group_id leaks across splits: {group_id}")
			validated.append(row)
		rows[split] = validated
	if not rows["train"]:
		raise TrainingDataError("derived/train.jsonl is empty")
	target_processing: dict[str, Any] = {"target_mode": target_mode, "train_target_shrinkage": train_target_shrinkage}
	if target_mode == "candidate_quality":
		try:
			soft_target_temperature = float(summary.get("aggregation", {}).get("soft_target_temperature", 0.5))
		except (TypeError, ValueError) as exc:
			raise TrainingDataError("aggregate soft-target temperature is invalid") from exc
		target_processing["candidate_quality"] = _apply_candidate_quality_targets(
			dataset_root,
			rows,
			acceptable_epsilon=epsilon,
			soft_target_temperature=soft_target_temperature,
		)
		if train_target_shrinkage == "empirical_bayes_l2":
			target_processing["empirical_bayes"] = _apply_empirical_bayes_l2_shrinkage(
				rows,
				acceptable_epsilon=epsilon,
				soft_target_temperature=soft_target_temperature,
			)
		elif train_target_shrinkage == "diagnosis_group_eb_l2":
			target_processing["diagnosis_group_eb"] = _apply_diagnosis_group_eb_l2(
				rows,
				acceptable_epsilon=epsilon,
				soft_target_temperature=soft_target_temperature,
			)
	binding_counts = dict(sorted(Counter(
		str(row["l3_initial_identity_binding"])
		for split_rows in rows.values()
		for row in split_rows
	).items()))
	declared_binding_counts = {
		str(key): int(value)
		for key, value in dict(summary.get("l3_initial_identity_binding_counts") or {}).items()
	}
	if binding_counts != declared_binding_counts:
		raise TrainingDataError("derived/summary.json L3 initial identity binding counts mismatch")
	digest_payload = {
		"qwen_distribution_signature": qwen_distribution_signature,
		"flux2_policy_digest": flux2_policy_contract_digest,
		"l3_policy_digest": l3_policy_contract_digest,
		"allow_legacy_l3_identity_ablation": bool(allow_legacy_l3_identity_ablation),
		"l3_initial_identity_binding_counts": binding_counts,
		"splits": {
			split: [row["feature_input_digest"] for row in split_rows]
			for split, split_rows in rows.items()
		},
	}
	target_digest = _sha256_text(
		_canonical_json(
			{
				"target_mode": target_mode,
				"train_target_shrinkage": train_target_shrinkage,
				"target_processing": target_processing,
				"splits": {
					split: [
						{
							"state_id": row["state_id"],
							"utility_vector": row["utility_vector"],
							"soft_targets": row["soft_targets"],
							"pairwise_mask": row.get("quality_pairwise_mask", [True, True, True]),
							"observed_quality_vector": row.get(
								"unshrunk_candidate_quality_vector", row["utility_vector"]
							),
							"safe_pairwise_mask": row.get(
								"unshrunk_quality_pairwise_mask",
								row.get("quality_pairwise_mask", [True, True, True]),
							),
							"confidence": row["confidence"],
							"sample_weight": row["sample_weight"],
						}
						for row in split_rows
					]
					for split, split_rows in rows.items()
				},
			}
		)
	)
	return DatasetBundle(
		root=dataset_root,
		rows=rows,
		summary=summary,
		class_factors=class_factors,
		acceptable_epsilon=epsilon,
		qwen_distribution_signature=qwen_distribution_signature,
		flux2_policy_contract=flux2_policy_contract,
		flux2_policy_digest=flux2_policy_contract_digest,
		l3_policy_contract=l3_policy_contract,
		l3_policy_digest=l3_policy_contract_digest,
		l3_initial_identity_binding_counts=binding_counts,
		allow_legacy_l3_identity_ablation=bool(allow_legacy_l3_identity_ablation),
		target_mode=target_mode,
		train_target_shrinkage=train_target_shrinkage,
		target_processing=target_processing,
		target_digest=target_digest,
		digest=_sha256_text(_canonical_json(digest_payload)),
	)


def load_raw_route_image(path: Path):
	"""Load the exact online route input: current RGB image, no ruler or resize."""

	from PIL import Image

	with Image.open(path) as loaded:
		return loaded.convert("RGB")


def _resolved_chat_template(processor: Any) -> Any:
	value = getattr(processor, "chat_template", None)
	if value is None:
		value = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
	return _json_safe(value)


def _safe_to_dict(value: Any) -> dict[str, Any]:
	method = getattr(value, "to_dict", None)
	if not callable(method):
		return {}
	try:
		result = method()
	except Exception as exc:
		raise TrainingDataError(f"Cannot freeze processor configuration: {exc}") from exc
	if not isinstance(result, Mapping):
		raise TrainingDataError("Processor to_dict() must return a mapping")
	return dict(result)


def build_feature_contract(
	*,
	qwen_model: str,
	qwen_revision: str,
	processor_metadata: Mapping[str, Any],
	encoder_source_digest: str,
	qwen_dtype: str = "bfloat16",
	qwen_quantization: str = "8bit",
	qwen_allow_cpu_offload: bool = False,
	qwen_effective_quantization_name: str | None = None,
	qwen_distribution_signature: str,
) -> dict[str, Any]:
	"""Build the immutable contract saved in caches and deployment checkpoints."""

	if IMMUTABLE_REVISION_RE.fullmatch(str(qwen_revision).strip()) is None:
		raise TrainingDataError("qwen_revision must be an immutable 40-64 character hexadecimal commit")
	try:
		qwen_dtype = canonical_qwen_dtype_name(qwen_dtype)
		qwen_quantization = canonical_qwen_quantization_name(qwen_quantization)
	except ValueError as exc:
		raise TrainingDataError(str(exc)) from exc
	if not isinstance(qwen_allow_cpu_offload, bool):
		raise TrainingDataError("qwen_allow_cpu_offload must be a boolean")
	if re.fullmatch(r"[0-9a-f]{64}", str(qwen_distribution_signature).strip().lower()) is None:
		raise TrainingDataError("qwen_distribution_signature must be a SHA-256 digest")
	expected_effective = qwen_effective_quantization(qwen_quantization, qwen_allow_cpu_offload)
	effective_quantization = str(qwen_effective_quantization_name or expected_effective).strip().lower()
	if effective_quantization != expected_effective:
		raise TrainingDataError(
			"Qwen effective quantization differs from requested quantization/CPU-offload policy"
		)
	processor_component = {
		"loader": "transformers.AutoProcessor.from_pretrained(local_files_only=True,trust_remote_code=True)",
		"processor_class": str(processor_metadata.get("processor_class") or ""),
		"tokenizer_class": str(processor_metadata.get("tokenizer_class") or ""),
		"tokenizer_padding_side": str(processor_metadata.get("tokenizer_padding_side") or ""),
		"model_input_names": list(processor_metadata.get("model_input_names") or []),
		"resolved_processor_config": _json_safe(processor_metadata.get("processor_config") or {}),
		"resolved_image_processor_config": _json_safe(processor_metadata.get("image_processor_config") or {}),
		"resolved_tokenizer_init_kwargs": _json_safe(processor_metadata.get("tokenizer_init_kwargs") or {}),
	}
	chat_template_component = {
		"resolved_template": _json_safe(processor_metadata.get("chat_template")),
		"apply_chat_template": {
			"tokenize": False,
			"add_generation_prompt": False,
			"enable_thinking": False,
			"type_error_fallback_omits": ["enable_thinking"],
		},
		"messages_builder_source_sha256": _sha256_text(inspect.getsource(build_route_encoder_messages)),
	}
	preprocess_component = {
		"source": "aggregate.image_path/raw current image",
		"open": "PIL.Image.open",
		"color_mode": "RGB",
		"annotation": "none",
		"manual_resize_before_processor": "none",
		"processor_call": {"text": "batch chat strings", "images": "one raw image per row", "padding": True, "return_tensors": "pt"},
	}
	pooling_component = {
		"backbone_runtime": {
			"dtype": str(qwen_dtype),
			"quantization": str(qwen_quantization),
			"allow_cpu_offload": qwen_allow_cpu_offload,
			"effective_quantization": effective_quantization,
		},
		"backbone_kwargs": {"output_hidden_states": True, "use_cache": False, "return_dict": True},
		"hidden_layer": -1,
		"token_position": -1,
		"padding_precondition": "tokenizer.padding_side=left",
		"expression": "hidden_states[-1][:, -1, :].detach().float().cpu()",
		"output_dtype": "float32",
		"encoder_source_sha256": encoder_source_digest,
	}
	components = {
		"processor": processor_component,
		"chat_template": chat_template_component,
		"preprocess": preprocess_component,
		"pooling": pooling_component,
	}
	component_digests = {
		f"{name}_contract_digest": _sha256_text(_canonical_json(value))
		for name, value in components.items()
	}
	contract = {
		"feature_schema": FEATURE_SCHEMA,
		"feature_schema_version": FEATURE_SCHEMA_VERSION,
		"diagnosis_schema": DIAGNOSIS_SCHEMA,
		"label_levels": list(LABEL_LEVELS),
		"qwen_model": str(qwen_model),
		"qwen_revision": str(qwen_revision).lower(),
		"qwen_dtype": qwen_dtype,
		"qwen_quantization": qwen_quantization,
		"qwen_allow_cpu_offload": qwen_allow_cpu_offload,
		"qwen_effective_quantization": effective_quantization,
		"qwen_distribution_signature": str(qwen_distribution_signature).strip().lower(),
		"components": components,
		**component_digests,
	}
	contract["processor_chat_template_preprocess_pooling_contract_digest"] = _sha256_text(
		_canonical_json(components)
	)
	contract["feature_contract_digest"] = _sha256_text(
		_canonical_json({key: value for key, value in contract.items() if key != "feature_contract_digest"})
	)
	return contract


def runtime_feature_contract(
	planner: Any,
	qwen_model: str,
	qwen_revision: str,
	*,
	qwen_dtype: str,
	qwen_quantization: str,
	qwen_allow_cpu_offload: bool,
	qwen_distribution_signature: str,
) -> dict[str, Any]:
	processor = planner.processor
	tokenizer = getattr(processor, "tokenizer", None)
	image_processor = getattr(processor, "image_processor", None)
	if str(getattr(tokenizer, "padding_side", "")) != "left":
		raise TrainingDataError("Online-compatible route pooling requires tokenizer.padding_side='left'")
	processor_config = _safe_to_dict(processor)
	image_processor_config = _safe_to_dict(image_processor) if image_processor is not None else {}
	metadata = {
		"processor_class": f"{processor.__class__.__module__}.{processor.__class__.__qualname__}",
		"tokenizer_class": (
			f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}" if tokenizer is not None else ""
		),
		"tokenizer_padding_side": getattr(tokenizer, "padding_side", ""),
		"model_input_names": list(getattr(processor, "model_input_names", None) or []),
		"chat_template": _resolved_chat_template(processor),
		"processor_config": processor_config,
		"image_processor_config": image_processor_config,
		"tokenizer_init_kwargs": getattr(tokenizer, "init_kwargs", None) or {},
	}
	return build_feature_contract(
		qwen_model=qwen_model,
		qwen_revision=qwen_revision,
		processor_metadata=metadata,
		encoder_source_digest=_sha256_text(inspect.getsource(planner.__class__.encode_route_features)),
		qwen_dtype=qwen_dtype,
		qwen_quantization=qwen_quantization,
		qwen_allow_cpu_offload=qwen_allow_cpu_offload,
		qwen_effective_quantization_name=str(getattr(planner, "effective_quantization", "")),
		qwen_distribution_signature=qwen_distribution_signature,
	)


def _static_test_feature_contract(
	qwen_model: str = "/test/qwen",
	qwen_revision: str = "a" * 40,
	qwen_distribution_signature: str | None = None,
) -> dict[str, Any]:
	"""Small public helper for CPU tests and externally produced feature caches."""

	if qwen_distribution_signature is None:
		qwen_distribution_signature = build_qwen_analyzer_l1_distribution_contract(
			model=qwen_model,
			model_revision=qwen_revision,
			dtype="bfloat16",
			quantization="8bit",
			allow_cpu_offload=False,
		)["distribution_signature"]
	return build_feature_contract(
		qwen_model=qwen_model,
		qwen_revision=qwen_revision,
		processor_metadata={
			"processor_class": "test.FrozenProcessor",
			"tokenizer_class": "test.FrozenTokenizer",
			"tokenizer_padding_side": "left",
			"model_input_names": ["input_ids", "attention_mask", "pixel_values"],
			"chat_template": "test-template-v1",
		},
		encoder_source_digest="0" * 64,
		qwen_dtype="bfloat16",
		qwen_quantization="8bit",
		qwen_distribution_signature=qwen_distribution_signature,
	)


def save_feature_cache(
	path: Path,
	bundle: DatasetBundle,
	features_by_split: Mapping[str, torch.Tensor],
	feature_contract: Mapping[str, Any],
) -> None:
	"""Write a complete cache bound to the aggregate rows (also used by tests)."""

	splits: dict[str, Any] = {}
	input_dim: int | None = None
	for split in SPLITS:
		features = features_by_split.get(split)
		if features is None:
			raise TrainingDataError(f"Missing {split} features")
		features = features.detach().float().cpu()
		if features.ndim != 2 or features.shape[0] != len(bundle.rows[split]):
			raise TrainingDataError(f"{split} feature tensor shape does not match aggregate rows")
		if features.shape[0] > 0:
			if input_dim is None:
				input_dim = int(features.shape[1])
			elif input_dim != int(features.shape[1]):
				raise TrainingDataError("Feature dimensions differ across splits")
		splits[split] = {
			"features": features,
			"state_ids": [row["state_id"] for row in bundle.rows[split]],
			"input_digests": [row["feature_input_digest"] for row in bundle.rows[split]],
		}
	if input_dim is None or input_dim <= 0:
		raise TrainingDataError("Feature cache contains no non-empty feature tensor")
	contract = dict(feature_contract)
	payload = {
		"cache_schema": FEATURE_CACHE_SCHEMA,
		"dataset_digest": bundle.digest,
		"qwen_distribution_signature": bundle.qwen_distribution_signature,
		"feature_contract": contract,
		"feature_contract_digest": contract.get("feature_contract_digest"),
		"input_dim": input_dim,
		"splits": splits,
	}
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(payload, path)


def load_feature_cache(path: Path, bundle: DatasetBundle) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
	payload = torch.load(path, map_location="cpu", weights_only=False)
	if not isinstance(payload, Mapping) or payload.get("cache_schema") != FEATURE_CACHE_SCHEMA:
		raise TrainingDataError("Feature cache schema mismatch")
	if payload.get("dataset_digest") != bundle.digest:
		raise TrainingDataError("Feature cache belongs to a different aggregate dataset")
	if payload.get("qwen_distribution_signature") != bundle.qwen_distribution_signature:
		raise TrainingDataError("Feature cache Qwen distribution differs from aggregate data")
	contract = payload.get("feature_contract")
	if not isinstance(contract, Mapping):
		raise TrainingDataError("Feature cache is missing feature_contract")
	contract = dict(contract)
	if contract.get("feature_schema") != FEATURE_SCHEMA or contract.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
		raise TrainingDataError("Feature cache feature schema mismatch")
	if contract.get("diagnosis_schema") != DIAGNOSIS_SCHEMA or contract.get("label_levels") != list(LABEL_LEVELS):
		raise TrainingDataError("Feature cache diagnosis or label contract mismatch")
	if contract.get("qwen_distribution_signature") != bundle.qwen_distribution_signature:
		raise TrainingDataError("Feature contract Qwen distribution differs from aggregate data")
	expected_digest = _sha256_text(
		_canonical_json({key: value for key, value in contract.items() if key != "feature_contract_digest"})
	)
	if contract.get("feature_contract_digest") != expected_digest or payload.get("feature_contract_digest") != expected_digest:
		raise TrainingDataError("Feature cache contract digest mismatch")
	split_payloads = payload.get("splits")
	if not isinstance(split_payloads, Mapping):
		raise TrainingDataError("Feature cache has no split payloads")
	features_by_split: dict[str, torch.Tensor] = {}
	input_dim = int(payload.get("input_dim", 0))
	for split in SPLITS:
		split_payload = split_payloads.get(split)
		if not isinstance(split_payload, Mapping):
			raise TrainingDataError(f"Feature cache is missing split {split}")
		expected_ids = [row["state_id"] for row in bundle.rows[split]]
		expected_inputs = [row["feature_input_digest"] for row in bundle.rows[split]]
		if split_payload.get("state_ids") != expected_ids or split_payload.get("input_digests") != expected_inputs:
			raise TrainingDataError(f"Feature cache {split} row identities do not match aggregate data")
		features = split_payload.get("features")
		if not isinstance(features, torch.Tensor) or features.ndim != 2:
			raise TrainingDataError(f"Feature cache {split} features must be a rank-2 tensor")
		if tuple(features.shape) != (len(expected_ids), input_dim):
			raise TrainingDataError(f"Feature cache {split} tensor shape mismatch")
		if not torch.isfinite(features.float()).all():
			raise TrainingDataError(f"Feature cache {split} contains non-finite values")
		features_by_split[split] = features.float().cpu()
	return features_by_split, contract


def _extract_selected_rows(
	planner: Any,
	rows: Sequence[dict[str, Any]],
	indices: Sequence[int],
	batch_size: int,
) -> torch.Tensor:
	feature_chunks: list[torch.Tensor] = []
	for start in range(0, len(indices), max(1, batch_size)):
		batch_indices = indices[start : start + max(1, batch_size)]
		batch_rows = [rows[index] for index in batch_indices]
		images = [load_raw_route_image(Path(row["image_path"])) for row in batch_rows]
		try:
			try:
				features = planner.encode_route_features(
					[str(row["target_prompt"]) for row in batch_rows],
					[str(row["diagnosis_text"]) for row in batch_rows],
					images,
				)
			except Exception:
				if len(batch_rows) == 1:
					raise
				features = torch.stack(
					[
						planner.encode_route_features(
							[str(row["target_prompt"])], [str(row["diagnosis_text"])], [image]
						)[0]
						for row, image in zip(batch_rows, images)
					]
				)
			feature_chunks.append(features.detach().float().cpu())
		finally:
			for image in images:
				image.close()
		print(
			json.dumps({"event": "route_feature_batch", "done": min(start + len(batch_indices), len(indices)), "total": len(indices)}),
			flush=True,
		)
	if not feature_chunks:
		return torch.empty((0, 0), dtype=torch.float32)
	return torch.cat(feature_chunks, dim=0)


def extract_worker(args: argparse.Namespace) -> int:
	from inference.self_route_version.modules.vcot_branch import QwenPlanner, parse_dtype

	validate_pinned_qwen_identity(str(args.qwen_model), str(args.qwen_revision))
	if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
		raise TrainingDataError("extract-worker shard index/count are invalid")
	if args.extract_batch_size <= 0:
		raise TrainingDataError("--extract-batch-size must be positive")
	bundle = load_dataset_bundle(
		args.dataset_root,
		verify_image_hashes=not args.skip_image_hash_check,
		allow_legacy_l3_identity_ablation=bool(getattr(args, "allow_legacy_l3_identity_ablation", False)),
		target_mode=str(getattr(args, "target_mode", "aggregate_utility")),
		train_target_shrinkage=str(getattr(args, "train_target_shrinkage", "none")),
	)
	planner = QwenPlanner(
		model_path=str(args.qwen_model),
		dtype=parse_dtype(args.qwen_dtype),
		max_memory_per_gpu=args.qwen_max_memory_per_gpu,
		min_free_gib=args.qwen_min_free_gib,
		per_gpu_extra_reserve={},
		quantization=args.qwen_quantization,
		allow_cpu_offload=bool(args.qwen_allow_cpu_offload),
		route_head_path=None,
	)
	for parameter in planner.model.parameters():
		parameter.requires_grad_(False)
	planner.model.eval()
	contract = runtime_feature_contract(
		planner,
		str(args.qwen_model),
		str(args.qwen_revision),
		qwen_dtype=args.qwen_dtype,
		qwen_quantization=args.qwen_quantization,
		qwen_allow_cpu_offload=bool(args.qwen_allow_cpu_offload),
		qwen_distribution_signature=bundle.qwen_distribution_signature,
	)
	split_payloads: dict[str, Any] = {}
	input_dim: int | None = None
	for split in SPLITS:
		rows = bundle.rows[split]
		indices = [index for index in range(len(rows)) if index % args.shard_count == args.shard_index]
		features = _extract_selected_rows(planner, rows, indices, args.extract_batch_size)
		if features.shape[0] > 0:
			input_dim = int(features.shape[1]) if input_dim is None else input_dim
			if int(features.shape[1]) != input_dim:
				raise TrainingDataError("Qwen emitted inconsistent feature dimensions")
		split_payloads[split] = {
			"indices": indices,
			"features": features,
			"input_digests": [rows[index]["feature_input_digest"] for index in indices],
		}
	payload = {
		"cache_schema": FEATURE_CACHE_SCHEMA,
		"partial": True,
		"shard_index": args.shard_index,
		"shard_count": args.shard_count,
		"dataset_digest": bundle.digest,
		"qwen_distribution_signature": bundle.qwen_distribution_signature,
		"feature_contract": contract,
		"feature_contract_digest": contract["feature_contract_digest"],
		"input_dim": input_dim,
		"splits": split_payloads,
	}
	args.shard_output.parent.mkdir(parents=True, exist_ok=True)
	torch.save(payload, args.shard_output)
	return 0


def merge_feature_shards(shard_paths: Sequence[Path], feature_cache: Path, bundle: DatasetBundle) -> None:
	if not shard_paths:
		raise TrainingDataError("No feature shards were provided")
	payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]
	first = payloads[0]
	contract = first.get("feature_contract")
	digest = first.get("feature_contract_digest")
	input_dims = {int(payload["input_dim"]) for payload in payloads if payload.get("input_dim") is not None}
	if len(input_dims) != 1:
		raise TrainingDataError("Feature shards have inconsistent or empty feature dimensions")
	input_dim = input_dims.pop()
	features_by_split: dict[str, torch.Tensor] = {}
	for payload in payloads:
		if payload.get("dataset_digest") != bundle.digest:
			raise TrainingDataError("Feature shard dataset digest mismatch")
		if payload.get("qwen_distribution_signature") != bundle.qwen_distribution_signature:
			raise TrainingDataError("Feature shard Qwen distribution mismatch")
		if payload.get("feature_contract_digest") != digest or payload.get("feature_contract") != contract:
			raise TrainingDataError("Feature shard runtime contracts differ")
	for split in SPLITS:
		rows = bundle.rows[split]
		ordered: list[torch.Tensor | None] = [None] * len(rows)
		for payload in payloads:
			split_payload = payload["splits"][split]
			indices = split_payload["indices"]
			features = split_payload["features"].float()
			inputs = split_payload["input_digests"]
			if len(indices) != len(features) or len(indices) != len(inputs):
				raise TrainingDataError(f"Feature shard {split} lengths differ")
			for local_index, row_index in enumerate(indices):
				if row_index < 0 or row_index >= len(rows) or ordered[row_index] is not None:
					raise TrainingDataError(f"Feature shard {split} index is duplicate or out of range")
				if inputs[local_index] != rows[row_index]["feature_input_digest"]:
					raise TrainingDataError(f"Feature shard {split} input digest mismatch")
				ordered[row_index] = features[local_index]
		if any(value is None for value in ordered):
			raise TrainingDataError(f"Feature shards do not completely cover {split}")
		features_by_split[split] = (
			torch.stack([value for value in ordered if value is not None])
			if ordered
			else torch.empty((0, input_dim), dtype=torch.float32)
		)
	save_feature_cache(feature_cache, bundle, features_by_split, contract)


def launch_extractors(args: argparse.Namespace) -> Path:
	validate_pinned_qwen_identity(str(args.qwen_model), str(args.qwen_revision))
	if args.extract_batch_size <= 0:
		raise TrainingDataError("--extract-batch-size must be positive")
	bundle = load_dataset_bundle(
		args.dataset_root,
		verify_image_hashes=not args.skip_image_hash_check,
		allow_legacy_l3_identity_ablation=bool(getattr(args, "allow_legacy_l3_identity_ablation", False)),
		target_mode=str(getattr(args, "target_mode", "aggregate_utility")),
		train_target_shrinkage=str(getattr(args, "train_target_shrinkage", "none")),
	)
	gpus = [value.strip() for value in args.gpu_indices.split(",") if value.strip()]
	if not gpus:
		raise TrainingDataError("--gpu-indices is empty")
	shard_dir = args.output_dir / "feature_shards"
	shard_dir.mkdir(parents=True, exist_ok=True)
	processes: list[tuple[subprocess.Popen[str], Any, Path, Path]] = []
	for shard_index, gpu in enumerate(gpus):
		shard_output = shard_dir / f"shard_{shard_index:02d}.pt"
		log_path = shard_dir / f"shard_{shard_index:02d}.log"
		command = [
			sys.executable,
			str(Path(__file__).resolve()),
			"extract-worker",
			"--dataset-root", str(args.dataset_root),
			"--shard-index", str(shard_index),
			"--shard-count", str(len(gpus)),
			"--shard-output", str(shard_output),
			"--qwen-model", str(args.qwen_model),
			"--qwen-revision", str(args.qwen_revision),
			"--qwen-dtype", args.qwen_dtype,
			"--qwen-quantization", args.qwen_quantization,
			"--qwen-max-memory-per-gpu", args.qwen_max_memory_per_gpu,
			"--qwen-min-free-gib", str(args.qwen_min_free_gib),
			"--extract-batch-size", str(args.extract_batch_size),
			"--target-mode", str(getattr(args, "target_mode", "aggregate_utility")),
			"--train-target-shrinkage", str(getattr(args, "train_target_shrinkage", "none")),
		]
		if args.skip_image_hash_check:
			command.append("--skip-image-hash-check")
		if args.qwen_allow_cpu_offload:
			command.append("--qwen-allow-cpu-offload")
		if bool(getattr(args, "allow_legacy_l3_identity_ablation", False)):
			command.append("--allow-legacy-l3-identity-ablation")
		environment = os.environ.copy()
		environment["CUDA_VISIBLE_DEVICES"] = gpu
		log_handle = log_path.open("w", encoding="utf-8")
		process = subprocess.Popen(command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
		processes.append((process, log_handle, log_path, shard_output))
	shard_paths: list[Path] = []
	for process, log_handle, log_path, shard_output in processes:
		return_code = process.wait()
		log_handle.close()
		if return_code != 0 or not shard_output.is_file():
			raise RuntimeError(f"Qwen route feature extractor failed with code={return_code}; see {log_path}")
		shard_paths.append(shard_output)
	feature_cache = args.feature_cache or (args.output_dir / "features.pt")
	merge_feature_shards(shard_paths, feature_cache, bundle)
	return feature_cache


def _row_tensors(
	rows: Sequence[Mapping[str, Any]],
	*,
	exact_tie_quality_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
	if not math.isfinite(float(exact_tie_quality_weight)) or not 0.0 <= float(exact_tie_quality_weight) <= 1.0:
		raise TrainingDataError("exact_tie_quality_weight must be in [0,1]")
	regrets = []
	efficiency_regrets = []
	pairwise_masks = []
	safe_pairwise_masks = []
	sample_weights = []
	quality_row_weights = []
	for row in rows:
		utility = [float(value) for value in row["utility_vector"]]
		best = max(utility)
		regrets.append([best - value for value in utility])
		observed_utility = [
			float(value)
			for value in row.get("unshrunk_candidate_quality_vector", row["utility_vector"])
		]
		if len(observed_utility) != 3 or any(not math.isfinite(value) for value in observed_utility):
			raise TrainingDataError("observed candidate quality vector must contain three finite values")
		observed_best = max(observed_utility)
		efficiency_regrets.append([observed_best - value for value in observed_utility])
		mask = row.get("quality_pairwise_mask")
		pairwise_masks.append(list(mask) if isinstance(mask, list) and len(mask) == 3 else [True, True, True])
		safe_mask = row.get("unshrunk_quality_pairwise_mask", mask)
		safe_pairwise_masks.append(
			list(safe_mask) if isinstance(safe_mask, list) and len(safe_mask) == 3 else [True, True, True]
		)
		sample_weights.append(float(row["sample_weight"]))
		quality_row_weights.append(
			float(exact_tie_quality_weight)
			if max(observed_utility) - min(observed_utility) <= 1e-8
			else 1.0
		)
	return {
		"soft_targets": torch.tensor([row["soft_targets"] for row in rows], dtype=torch.float32),
		"utility_targets": torch.tensor([row["centered_utility_vector"] for row in rows], dtype=torch.float32),
		"regret_targets": torch.tensor(regrets, dtype=torch.float32),
		"efficiency_regret_targets": torch.tensor(efficiency_regrets, dtype=torch.float32),
		"pairwise_mask": torch.tensor(pairwise_masks, dtype=torch.float32),
		"safe_pairwise_mask": torch.tensor(safe_pairwise_masks, dtype=torch.float32),
		"confidence": torch.tensor([row["confidence"] for row in rows], dtype=torch.float32),
		"sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
		"quality_row_weight": torch.tensor(quality_row_weights, dtype=torch.float32),
	}


def _loss_kwargs(args: argparse.Namespace) -> dict[str, float]:
	return {
		"utility_temperature": float(getattr(args, "utility_temperature", 0.5)),
		"pairwise_temperature": float(getattr(args, "pairwise_temperature", 0.5)),
		"policy_temperature": float(getattr(args, "policy_temperature", 1.0)),
		"quality_tie_epsilon": float(getattr(args, "quality_tie_epsilon", 0.0)),
		"soft_ce_weight": float(getattr(args, "soft_ce_weight", 1.0)),
		"utility_alignment_weight": float(getattr(args, "utility_alignment_weight", 0.15)),
		"pairwise_weight": float(getattr(args, "pairwise_weight", 0.20)),
		"expected_regret_weight": float(getattr(args, "expected_regret_weight", 0.0)),
		"efficiency_weight": float(getattr(args, "efficiency_weight", 0.0)),
		"uncertain_safe_prior_weight": float(getattr(args, "uncertain_safe_prior_weight", 0.0)),
		"safe_action_index": int(getattr(args, "safe_action_index", 1)),
	}


def _efficiency_cost_tensor(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
	values = getattr(args, "efficiency_costs", (204.5771, 154.4044, 151.5316))
	if len(values) != 3 or any(not math.isfinite(float(value)) for value in values):
		raise TrainingDataError("--efficiency-costs must contain three finite values")
	return torch.tensor([float(value) for value in values], dtype=torch.float32, device=device)


def _append_training_diagnosis_aux_features(
	features: Mapping[str, torch.Tensor],
	rows: Mapping[str, Sequence[Mapping[str, Any]]],
	*,
	std_floor: float = 0.1,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
	if not math.isfinite(float(std_floor)) or float(std_floor) <= 0.0:
		raise TrainingDataError("diagnosis auxiliary std_floor must be finite and positive")
	raw = {
		split: torch.tensor(
			[diagnosis_aux_vector(row["diagnosis_text"]) for row in rows[split]],
			dtype=torch.float32,
		)
		for split in SPLITS
	}
	mean = raw["train"].mean(dim=0)
	std = raw["train"].std(dim=0).clamp_min(float(std_floor))
	contract = build_diagnosis_aux_contract(mean.tolist(), std.tolist(), std_floor=float(std_floor))
	augmented = {
		split: torch.cat([features[split].float(), (raw[split] - mean) / std], dim=-1)
		for split in SPLITS
	}
	return augmented, contract


@torch.no_grad()
def evaluate_head(
	model: RouteMLP,
	features: torch.Tensor,
	rows: Sequence[Mapping[str, Any]],
	class_weight: torch.Tensor,
	args: argparse.Namespace,
	device: torch.device,
) -> dict[str, Any]:
	if not rows:
		return {"rows": 0, "loss": None, "soft_ce": None, "utility_alignment": None, "pairwise": None}
	model.eval()
	targets = _row_tensors(
		rows,
		exact_tie_quality_weight=float(getattr(args, "exact_tie_quality_weight", 0.0)),
	)
	component_sums = {
		"loss": 0.0,
		"soft_ce": 0.0,
		"utility_alignment": 0.0,
		"pairwise": 0.0,
		"expected_regret": 0.0,
		"efficiency": 0.0,
		"safe_prior": 0.0,
	}
	predictions: list[int] = []
	soft_labels: list[int] = []
	regrets: list[float] = []
	selected_utilities: list[float] = []
	acceptable_hits: list[bool] = []
	efficiency_costs = _efficiency_cost_tensor(args, device)
	for start in range(0, len(rows), args.mlp_batch_size):
		stop = min(start + args.mlp_batch_size, len(rows))
		indices = slice(start, stop)
		logits = model(features[indices].to(device))
		loss, components = utility_aware_router_loss(
			logits,
			targets["soft_targets"][indices].to(device),
			targets["utility_targets"][indices].to(device),
			regret_targets=targets["regret_targets"][indices].to(device),
			efficiency_regret_targets=targets["efficiency_regret_targets"][indices].to(device),
			pairwise_mask=targets["pairwise_mask"][indices].to(device),
			safe_pairwise_mask=targets["safe_pairwise_mask"][indices].to(device),
			efficiency_costs=efficiency_costs,
			confidence=targets["confidence"][indices].to(device),
			sample_weight=targets["sample_weight"][indices].to(device),
			quality_row_weight=targets["quality_row_weight"][indices].to(device),
			class_weight=class_weight,
			**_loss_kwargs(args),
		)
		batch_rows = stop - start
		for key in component_sums:
			value = loss if key == "loss" else components[key]
			component_sums[key] += float(value.detach().cpu().item()) * batch_rows
		batch_predictions = logits.argmax(dim=-1).detach().cpu().tolist()
		predictions.extend(int(value) for value in batch_predictions)
		soft_labels.extend(targets["soft_targets"][indices].argmax(dim=-1).tolist())
		for offset, prediction in enumerate(batch_predictions):
			utility = rows[start + offset]["utility_vector"]
			regrets.append(max(utility) - utility[int(prediction)])
			selected_utilities.append(float(utility[int(prediction)]))
			acceptable_hits.append(
				max(utility) - utility[int(prediction)] <= float(getattr(args, "acceptable_epsilon", 0.25)) + 1e-12
			)
	confusion = [[0, 0, 0] for _ in range(3)]
	for label, prediction in zip(soft_labels, predictions):
		confusion[label][prediction] += 1
	return {
		"rows": len(rows),
		**{key: value / len(rows) for key, value in component_sums.items()},
		"soft_argmax_accuracy": sum(left == right for left, right in zip(soft_labels, predictions)) / len(rows),
		"mean_observed_regret": sum(regrets) / len(regrets),
		"mean_selected_utility": sum(selected_utilities) / len(selected_utilities),
		"acceptable_quality_rate": sum(acceptable_hits) / len(acceptable_hits),
		"mean_selected_efficiency_cost": sum(
			float(efficiency_costs[prediction].detach().cpu().item()) for prediction in predictions
		) / len(predictions),
		"prediction_distribution": dict(Counter(f"L{value + 1}" for value in predictions)),
		"confusion_matrix_rows_soft_argmax_cols_prediction": confusion,
	}


def _training_device(value: str) -> torch.device:
	if value == "auto":
		return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	device = torch.device(value)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise TrainingDataError(f"Requested unavailable training device: {value}")
	return device


def validate_pinned_qwen_identity(qwen_model: str, qwen_revision: str) -> None:
	"""Fail before GPU allocation when a local snapshot is not revision-pinned."""

	revision = str(qwen_revision).strip().lower()
	if IMMUTABLE_REVISION_RE.fullmatch(revision) is None:
		raise TrainingDataError("qwen_revision must be an immutable 40-64 character hexadecimal commit")
	model_path = Path(str(qwen_model)).expanduser()
	if not model_path.is_dir():
		raise TrainingDataError(f"qwen_model must be an existing local snapshot directory: {model_path}")
	if model_path.name.lower() != revision:
		raise TrainingDataError(
			"qwen_model must point directly to snapshots/<qwen_revision>; aliases such as snapshots/main are forbidden"
		)


def train_head(args: argparse.Namespace) -> dict[str, Any]:
	if args.hidden_dim <= 0 or args.mlp_batch_size <= 0 or args.epochs <= 0:
		raise TrainingDataError("hidden_dim, mlp_batch_size, and epochs must be positive")
	if not 0.0 <= args.dropout < 1.0:
		raise TrainingDataError("dropout must be in [0,1)")
	if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
		raise TrainingDataError("learning_rate must be positive and weight_decay non-negative")
	temperatures = (
		float(getattr(args, "utility_temperature", 0.5)),
		float(getattr(args, "pairwise_temperature", 0.5)),
		float(getattr(args, "policy_temperature", 1.0)),
	)
	if any(not math.isfinite(value) or value <= 0.0 for value in temperatures):
		raise TrainingDataError("loss temperatures must be positive")
	if any(
		not math.isfinite(float(value)) or float(value) < 0.0
		for value in (
			getattr(args, "soft_ce_weight", 1.0),
			getattr(args, "utility_alignment_weight", 0.15),
			getattr(args, "pairwise_weight", 0.20),
			getattr(args, "expected_regret_weight", 0.0),
			getattr(args, "efficiency_weight", 0.0),
			getattr(args, "uncertain_safe_prior_weight", 0.0),
		)
	):
		raise TrainingDataError("loss weights must be non-negative")
	bundle = load_dataset_bundle(
		args.dataset_root,
		verify_image_hashes=not args.skip_image_hash_check,
		allow_legacy_l3_identity_ablation=bool(getattr(args, "allow_legacy_l3_identity_ablation", False)),
		target_mode=str(getattr(args, "target_mode", "aggregate_utility")),
		train_target_shrinkage=str(getattr(args, "train_target_shrinkage", "none")),
	)
	feature_cache = args.feature_cache or (args.output_dir / "features.pt")
	features, feature_contract = load_feature_cache(feature_cache, bundle)
	args.acceptable_epsilon = bundle.acceptable_epsilon
	qwen_feature_dim = int(features["train"].shape[1])
	diagnosis_aux_enabled = bool(getattr(args, "diagnosis_aux_features", False))
	diagnosis_aux_contract: dict[str, Any] | None = None
	if diagnosis_aux_enabled:
		features, diagnosis_aux_contract = _append_training_diagnosis_aux_features(features, bundle.rows)
	input_dim = int(features["train"].shape[1])
	class_factors = (
		tuple(args.class_factors)
		if args.class_factors is not None
		else ((1.0, 1.0, 1.0) if bundle.target_mode == "candidate_quality" else bundle.class_factors)
	)
	if len(class_factors) != 3 or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in class_factors):
		raise TrainingDataError("--class-factors must contain three finite positive values")
	device = _training_device(args.train_device)
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	if device.type == "cuda":
		torch.cuda.manual_seed_all(args.seed)
	model = RouteMLP(input_dim=input_dim, hidden_dim=args.hidden_dim, num_labels=3, dropout=args.dropout).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	class_weight = torch.tensor(class_factors, dtype=torch.float32, device=device)
	efficiency_costs = _efficiency_cost_tensor(args, device)
	train_targets = _row_tensors(
		bundle.rows["train"],
		exact_tie_quality_weight=float(getattr(args, "exact_tie_quality_weight", 0.0)),
	)
	best_state: dict[str, torch.Tensor] | None = None
	best_epoch = 0
	best_validation_loss = math.inf
	best_validation_regret = math.inf
	best_validation_efficiency = math.inf
	best_selection_key: tuple[float, ...] | None = None
	epochs_without_improvement = 0
	history: list[dict[str, Any]] = []
	for epoch in range(1, args.epochs + 1):
		model.train()
		generator = torch.Generator(device="cpu").manual_seed(args.seed + epoch)
		permutation = torch.randperm(len(bundle.rows["train"]), generator=generator)
		train_loss_sum = 0.0
		train_rows = 0
		for start in range(0, len(permutation), args.mlp_batch_size):
			batch = permutation[start : start + args.mlp_batch_size]
			optimizer.zero_grad(set_to_none=True)
			logits = model(features["train"][batch].to(device))
			loss, _components = utility_aware_router_loss(
				logits,
				train_targets["soft_targets"][batch].to(device),
				train_targets["utility_targets"][batch].to(device),
				regret_targets=train_targets["regret_targets"][batch].to(device),
				efficiency_regret_targets=train_targets["efficiency_regret_targets"][batch].to(device),
				pairwise_mask=train_targets["pairwise_mask"][batch].to(device),
				safe_pairwise_mask=train_targets["safe_pairwise_mask"][batch].to(device),
				efficiency_costs=efficiency_costs,
				confidence=train_targets["confidence"][batch].to(device),
				sample_weight=train_targets["sample_weight"][batch].to(device),
				quality_row_weight=train_targets["quality_row_weight"][batch].to(device),
				class_weight=class_weight,
				**_loss_kwargs(args),
			)
			loss.backward()
			if args.max_grad_norm > 0.0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
			optimizer.step()
			train_loss_sum += float(loss.detach().cpu().item()) * len(batch)
			train_rows += len(batch)
		validation = evaluate_head(
			model,
			features["validation"] if bundle.rows["validation"] else features["train"],
			bundle.rows["validation"] if bundle.rows["validation"] else bundle.rows["train"],
			class_weight,
			args,
			device,
		)
		validation_loss = float(validation["loss"])
		validation_regret = float(validation["mean_observed_regret"])
		validation_efficiency = float(validation["mean_selected_efficiency_cost"])
		history.append(
			{
				"epoch": epoch,
				"train_loss": train_loss_sum / max(1, train_rows),
				"validation_loss": validation_loss,
				"validation_quality_regret": validation_regret,
				"validation_efficiency_cost": validation_efficiency,
				"validation_prediction_distribution": validation["prediction_distribution"],
			}
		)
		selection_metric = str(getattr(args, "model_selection_metric", "validation_loss"))
		selection_key = (
			(validation_regret, validation_efficiency, validation_loss)
			if selection_metric == "quality_regret"
			else (validation_loss, validation_regret, validation_efficiency)
		)
		improved = best_selection_key is None or selection_key[0] < best_selection_key[0] - args.min_delta
		if (
			not improved
			and best_selection_key is not None
			and math.isclose(selection_key[0], best_selection_key[0], abs_tol=args.min_delta)
			and selection_key[1:] < best_selection_key[1:]
		):
			improved = True
		if improved:
			best_validation_loss = validation_loss
			best_validation_regret = validation_regret
			best_validation_efficiency = validation_efficiency
			best_selection_key = selection_key
			best_epoch = epoch
			best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1
		if args.patience > 0 and epochs_without_improvement >= args.patience:
			break
	if best_state is None:
		raise RuntimeError("MLP training did not produce a checkpoint")
	model.load_state_dict(best_state)
	evaluated_splits = ["train", "validation"]
	if bool(getattr(args, "evaluate_test", False)):
		evaluated_splits.append("test")
	metrics = {
		split: evaluate_head(model, features[split], bundle.rows[split], class_weight, args, device)
		for split in evaluated_splits
	}
	config = {
		"version": CHECKPOINT_VERSION,
		"input_dim": input_dim,
		"hidden_dim": args.hidden_dim,
		"num_labels": 3,
		"dropout": args.dropout,
		"label_levels": list(LABEL_LEVELS),
		"label_names": {str(level): ROUTE_NAMES[level] for level in LABEL_LEVELS},
		"l0_policy": "native_qwen_only",
		"diagnosis_schema": DIAGNOSIS_SCHEMA,
		"action_policy_version": ROUTER_ACTION_POLICY_VERSION,
		"repair_executor_family": ROUTER_REPAIR_EXECUTOR_FAMILY,
		"flux2_policy_contract": bundle.flux2_policy_contract,
		"flux2_policy_digest": bundle.flux2_policy_digest,
		"l3_policy_contract": bundle.l3_policy_contract,
		"l3_policy_digest": bundle.l3_policy_digest,
		"allow_legacy_l3_identity_ablation": bundle.allow_legacy_l3_identity_ablation,
		"l3_training_initial_identity_binding_counts": bundle.l3_initial_identity_binding_counts,
		"feature_schema": FEATURE_SCHEMA,
		"feature_schema_version": FEATURE_SCHEMA_VERSION,
		"feature_contract_digest": feature_contract["feature_contract_digest"],
		"processor_chat_template_preprocess_pooling_contract_digest": feature_contract[
			"processor_chat_template_preprocess_pooling_contract_digest"
		],
		"processor_contract_digest": feature_contract["processor_contract_digest"],
		"chat_template_contract_digest": feature_contract["chat_template_contract_digest"],
		"preprocess_contract_digest": feature_contract["preprocess_contract_digest"],
		"pooling_contract_digest": feature_contract["pooling_contract_digest"],
		"qwen_model": feature_contract["qwen_model"],
		"qwen_revision": feature_contract["qwen_revision"],
		"qwen_dtype": feature_contract["qwen_dtype"],
		"qwen_quantization": feature_contract["qwen_quantization"],
		"qwen_allow_cpu_offload": feature_contract["qwen_allow_cpu_offload"],
		"qwen_effective_quantization": feature_contract["qwen_effective_quantization"],
		"qwen_distribution_signature": feature_contract["qwen_distribution_signature"],
		"feature_contract": feature_contract,
		"qwen_feature_dim": qwen_feature_dim,
		"diagnosis_aux_features_enabled": diagnosis_aux_enabled,
		"diagnosis_aux_contract": diagnosis_aux_contract,
		"diagnosis_aux_contract_digest": (
			_sha256_text(_canonical_json(diagnosis_aux_contract)) if diagnosis_aux_contract is not None else None
		),
		"dataset_digest": bundle.digest,
		"target_mode": bundle.target_mode,
		"train_target_shrinkage": bundle.train_target_shrinkage,
		"target_processing": bundle.target_processing,
		"target_digest": bundle.target_digest,
		"class_factors": [float(value) for value in class_factors],
		"loss": _loss_kwargs(args),
		"model_selection_metric": str(getattr(args, "model_selection_metric", "validation_loss")),
		"evaluated_splits": evaluated_splits,
		"exact_tie_quality_weight": float(getattr(args, "exact_tie_quality_weight", 0.0)),
		"efficiency_costs": [float(value) for value in getattr(args, "efficiency_costs", (204.5771, 154.4044, 151.5316))],
		"seed": args.seed,
		"best_epoch": best_epoch,
		"optimizer": {
			"name": "AdamW",
			"learning_rate": float(args.learning_rate),
			"weight_decay": float(args.weight_decay),
			"max_grad_norm": float(args.max_grad_norm),
			"batch_size": int(args.mlp_batch_size),
			"epochs": int(args.epochs),
			"patience": int(args.patience),
			"min_delta": float(args.min_delta),
		},
	}
	model_dir = args.output_dir / "model"
	model_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = model_dir / "route_head.pt"
	torch.save({"state_dict": best_state, "config": config}, checkpoint_path)
	if args.export_checkpoint is not None:
		args.export_checkpoint.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(checkpoint_path, args.export_checkpoint)
	_write_json(model_dir / "config.json", config)
	summary = {
		"checkpoint": str(checkpoint_path),
		"feature_cache": str(feature_cache),
		"dataset_digest": bundle.digest,
		"target_mode": bundle.target_mode,
		"train_target_shrinkage": bundle.train_target_shrinkage,
		"target_processing": bundle.target_processing,
		"target_digest": bundle.target_digest,
		"qwen_distribution_signature": bundle.qwen_distribution_signature,
		"flux2_policy_digest": bundle.flux2_policy_digest,
		"l3_policy_digest": bundle.l3_policy_digest,
		"allow_legacy_l3_identity_ablation": bundle.allow_legacy_l3_identity_ablation,
		"l3_initial_identity_binding_counts": bundle.l3_initial_identity_binding_counts,
		"initial_generator_family": bundle.l3_policy_contract["executor"]["family"],
		"rows": {split: len(bundle.rows[split]) for split in SPLITS},
		"class_factors": list(class_factors),
		"best_epoch": best_epoch,
		"best_validation_loss": best_validation_loss,
		"best_validation_quality_regret": best_validation_regret,
		"best_validation_efficiency_cost": best_validation_efficiency,
		"epochs_run": len(history),
		"metrics": metrics,
		"evaluated_splits": evaluated_splits,
		"gpu_feature_extraction_validated": getattr(args, "command", "train") == "all",
	}
	_write_json(model_dir / "train_summary.json", summary)
	_write_json(model_dir / "history.json", history)
	return summary


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("--dataset-root", type=Path, required=True, help="Root containing aggregate derived/*.jsonl")
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--feature-cache", type=Path, default=None)
	parser.add_argument(
		"--skip-image-hash-check",
		action="store_true",
		help="Skip expensive image bytes verification (path and declared SHA-256 format are still checked).",
	)
	parser.add_argument(
		"--allow-legacy-l3-identity-ablation",
		action="store_true",
		help=(
			"Explicit ablation only: admit legacy BAGEL rows whose original model revision is unknown. "
			"Main training rejects these rows by default."
		),
	)
	parser.add_argument(
		"--target-mode",
		choices=TARGET_MODES,
		default="candidate_quality",
		help=(
			"candidate_quality removes repeated-original judge noise and equal one-call cost before routing; "
			"aggregate_utility reproduces the frozen aggregate labels"
		),
	)
	parser.add_argument(
		"--train-target-shrinkage",
		choices=TRAIN_TARGET_SHRINKAGE_MODES,
		default="none",
		help="Train-only paired-delta denoising; validation and test labels remain untouched.",
	)


def _add_qwen_arguments(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("--qwen-model", required=True, help="Pinned local Qwen snapshot path")
	parser.add_argument("--qwen-revision", required=True, help="Immutable 40-64 hex snapshot commit")
	parser.add_argument("--qwen-dtype", default="bfloat16")
	parser.add_argument("--qwen-quantization", default="8bit", choices=("none", "8bit", "int8", "4bit", "int4", "nf4"))
	parser.add_argument("--qwen-allow-cpu-offload", action="store_true")
	parser.add_argument("--qwen-max-memory-per-gpu", default="78GiB")
	parser.add_argument("--qwen-min-free-gib", type=float, default=10.0)
	parser.add_argument("--extract-batch-size", type=int, default=8)


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
	parser.add_argument("--export-checkpoint", type=Path, default=None)
	parser.add_argument("--train-device", default="auto")
	parser.add_argument("--hidden-dim", type=int, default=512)
	parser.add_argument("--dropout", type=float, default=0.15)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-3)
	parser.add_argument("--epochs", type=int, default=300)
	parser.add_argument("--patience", type=int, default=40)
	parser.add_argument("--min-delta", type=float, default=1e-6)
	parser.add_argument("--mlp-batch-size", type=int, default=256)
	parser.add_argument("--max-grad-norm", type=float, default=5.0)
	parser.add_argument("--seed", type=int, default=20260715)
	parser.add_argument("--class-factors", type=float, nargs=3, default=None, metavar=("L1", "L2", "L3"))
	parser.add_argument(
		"--diagnosis-aux-features",
		action="store_true",
		help="Append train-normalized canonical-diagnosis structure counts to the frozen Qwen feature.",
	)
	parser.add_argument("--utility-temperature", type=float, default=0.5)
	parser.add_argument("--pairwise-temperature", type=float, default=0.5)
	parser.add_argument("--policy-temperature", type=float, default=1.0)
	parser.add_argument("--quality-tie-epsilon", type=float, default=0.0)
	parser.add_argument("--soft-ce-weight", type=float, default=0.5)
	parser.add_argument("--utility-alignment-weight", type=float, default=0.1)
	parser.add_argument("--pairwise-weight", type=float, default=0.20)
	parser.add_argument("--expected-regret-weight", type=float, default=0.0)
	parser.add_argument("--efficiency-weight", type=float, default=0.001)
	parser.add_argument(
		"--uncertain-safe-prior-weight",
		type=float,
		default=0.0,
		help="L2 prior used only when paired seeds do not reliably support an L1/L3 quality escape.",
	)
	parser.add_argument("--safe-action-index", type=int, choices=(0, 1, 2), default=1, help=argparse.SUPPRESS)
	parser.add_argument(
		"--exact-tie-quality-weight",
		"--exact-tie-row-weight",
		dest="exact_tie_quality_weight",
		type=float,
		default=0.0,
		help=(
			"Quality-loss weight for rows whose three observed candidate qualities are exactly equal; "
			"base reliability and efficiency tie-breaking remain active."
		),
	)
	parser.add_argument(
		"--efficiency-costs",
		type=float,
		nargs=3,
		default=(204.5771, 154.4044, 151.5316),
		metavar=("L1_SECONDS", "L2_SECONDS", "L3_SECONDS"),
		help="Train-split median generation seconds; only breaks exact quality ties.",
	)
	parser.add_argument(
		"--model-selection-metric",
		choices=("quality_regret", "validation_loss"),
		default="quality_regret",
	)
	parser.add_argument(
		"--evaluate-test",
		action="store_true",
		help="Evaluate the test split only after the complete training configuration has been locked.",
	)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	subparsers = parser.add_subparsers(dest="command", required=True)
	validate_parser = subparsers.add_parser("validate-data", help="CPU-only aggregate contract validation")
	_add_dataset_arguments(validate_parser)
	extract_parser = subparsers.add_parser("extract", help="Launch one frozen-Qwen feature worker per GPU")
	_add_dataset_arguments(extract_parser)
	_add_qwen_arguments(extract_parser)
	extract_parser.add_argument("--gpu-indices", default="0,1,2,3")
	worker = subparsers.add_parser("extract-worker", help=argparse.SUPPRESS)
	worker.add_argument("--dataset-root", type=Path, required=True)
	worker.add_argument("--shard-index", type=int, required=True)
	worker.add_argument("--shard-count", type=int, required=True)
	worker.add_argument("--shard-output", type=Path, required=True)
	worker.add_argument("--skip-image-hash-check", action="store_true")
	worker.add_argument("--allow-legacy-l3-identity-ablation", action="store_true")
	worker.add_argument("--target-mode", choices=TARGET_MODES, default="candidate_quality")
	worker.add_argument(
		"--train-target-shrinkage",
		choices=TRAIN_TARGET_SHRINKAGE_MODES,
		default="none",
	)
	_add_qwen_arguments(worker)
	train_parser = subparsers.add_parser("train", help="CPU/GPU MLP-only training from a feature cache")
	_add_dataset_arguments(train_parser)
	_add_train_arguments(train_parser)
	all_parser = subparsers.add_parser("all", help="Validate, extract frozen features, then train")
	_add_dataset_arguments(all_parser)
	_add_qwen_arguments(all_parser)
	_add_train_arguments(all_parser)
	all_parser.add_argument("--gpu-indices", default="0,1,2,3")
	return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	started = time.time()
	if args.command == "validate-data":
		bundle = load_dataset_bundle(
			args.dataset_root,
			verify_image_hashes=not args.skip_image_hash_check,
			allow_legacy_l3_identity_ablation=bool(args.allow_legacy_l3_identity_ablation),
			target_mode=str(getattr(args, "target_mode", "aggregate_utility")),
			train_target_shrinkage=str(getattr(args, "train_target_shrinkage", "none")),
		)
		print(
			json.dumps(
				{
					"status": "ok",
		"dataset_digest": bundle.digest,
		"qwen_distribution_signature": bundle.qwen_distribution_signature,
		"rows": {split: len(bundle.rows[split]) for split in SPLITS},
					"class_factors": bundle.class_factors,
					"target_mode": bundle.target_mode,
					"train_target_shrinkage": bundle.train_target_shrinkage,
					"target_digest": bundle.target_digest,
					"allow_legacy_l3_identity_ablation": bundle.allow_legacy_l3_identity_ablation,
				},
				ensure_ascii=False,
				indent=2,
			),
			flush=True,
		)
		return 0
	if args.command == "extract-worker":
		return extract_worker(args)
	if args.command in {"extract", "all"}:
		feature_cache = launch_extractors(args)
		print(json.dumps({"event": "feature_cache_ready", "path": str(feature_cache)}), flush=True)
	if args.command in {"train", "all"}:
		summary = train_head(args)
		summary["duration_sec"] = round(time.time() - started, 3)
		print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
