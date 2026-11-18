"""Stable, row-order-independent sharding for independent GPU workers."""

from __future__ import annotations

from collections import Counter
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .actions import action_contract_digest
from .canonical import (
	canonical_json,
	iter_jsonl,
	read_json,
	sha256_file,
	sha256_json,
	write_json_atomic,
	write_jsonl_atomic,
)
from .planner import load_config, load_state_rows
from .schema import DataValidationError, SCHEMA_VERSION, normalize_prompt_for_group, normalize_text


SHARD_POLICY = "vcot_prompt_family_sha256_mod_v1"


def shard_key(row: Mapping[str, Any]) -> str:
	"""Keep prompt variants/trajectories together when family metadata exists."""

	for key in ("prompt_family_id", "base_prompt_id", "lineage_id", "group_id", "sample_key"):
		value = normalize_text(row.get(key))
		if value:
			return f"{key}:{value}"
	target = normalize_prompt_for_group(row.get("target_prompt") or row.get("prompt"))
	if target:
		return f"target_prompt:{target}"
	value = normalize_text(row.get("state_id") or row.get("sample_id") or row.get("id"))
	if value:
		return f"identity:{value}"
	raise DataValidationError("cannot derive stable shard key from row")


def stable_shard_id(key: str, num_shards: int) -> int:
	"""Map a stable row identity to a deterministic shard via hashing."""
	if int(num_shards) <= 0:
		raise ValueError("num_shards must be positive")
	digest = hashlib.sha256((SHARD_POLICY + "\0" + str(key)).encode("utf-8")).digest()
	return int.from_bytes(digest[:8], "big") % int(num_shards)


def partition_rows(rows: Sequence[Mapping[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
	"""Partition rows by stable identity while preserving deterministic ordering."""
	num_shards = int(num_shards)
	if num_shards <= 0:
		raise ValueError("num_shards must be positive")
	shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
	for row in rows:
		key = shard_key(row)
		owner = stable_shard_id(key, num_shards)
		shards[owner].append({**dict(row), "shard_key": key, "shard_id": owner, "num_shards": num_shards})
	for shard in shards:
		shard.sort(key=lambda row: (row["shard_key"], canonical_json(row)))
	return shards


def partition_file(input_path: Path, output_root: Path, num_shards: int, *, dry_run: bool = False) -> dict[str, Any]:
	"""Write deterministic hash-based shards and a manifest proving full disjoint coverage."""
	input_path = input_path.resolve()
	output_root = output_root.resolve()
	rows = load_state_rows(input_path)
	if not rows:
		raise DataValidationError("cannot shard an empty input")
	shards = partition_rows(rows, num_shards)
	manifest = {
		"schema_version": SCHEMA_VERSION,
		"shard_policy": SHARD_POLICY,
		"num_shards": int(num_shards),
		"source_path": str(input_path),
		"source_sha256": sha256_file(input_path),
		"row_count": len(rows),
		"shard_counts": [len(shard) for shard in shards],
		"shard_digests": [sha256_json(shard) for shard in shards],
	}
	if not dry_run:
		for shard_id, shard in enumerate(shards):
			shard_root = output_root / "shards" / f"shard_{shard_id:02d}"
			write_jsonl_atomic(shard_root / "input_prompts.jsonl", shard)
			write_json_atomic(
				shard_root / "owner.json",
				{
					"schema_version": SCHEMA_VERSION,
					"shard_policy": SHARD_POLICY,
					"shard_id": shard_id,
					"num_shards": int(num_shards),
					"row_count": len(shard),
					"rows_digest": sha256_json(shard),
				},
			)
		write_json_atomic(output_root / "shard_manifest.json", manifest)
	return manifest


def validate_partition(output_root: Path) -> dict[str, Any]:
	"""Verify shard hashes, ownership, row counts, and reconstruction of the original file."""
	root = output_root.resolve()
	manifest = read_json(root / "shard_manifest.json")
	if manifest.get("schema_version") != SCHEMA_VERSION:
		raise DataValidationError("shard manifest schema mismatch")
	if manifest.get("shard_policy") != SHARD_POLICY:
		raise DataValidationError("unsupported shard policy")
	num_shards = int(manifest.get("num_shards", 0))
	if num_shards <= 0:
		raise DataValidationError("invalid shard count")
	manifest_counts = manifest.get("shard_counts")
	manifest_digests = manifest.get("shard_digests")
	if not isinstance(manifest_counts, list) or len(manifest_counts) != num_shards:
		raise DataValidationError("shard manifest counts are incomplete")
	if not isinstance(manifest_digests, list) or len(manifest_digests) != num_shards:
		raise DataValidationError("shard manifest digests are incomplete")
	seen: set[str] = set()
	count = 0
	for shard_id in range(num_shards):
		shard_root = root / "shards" / f"shard_{shard_id:02d}"
		owner = read_json(shard_root / "owner.json")
		rows = list(iter_jsonl(shard_root / "input_prompts.jsonl"))
		rows_digest = sha256_json(rows)
		if (
			owner.get("schema_version") != SCHEMA_VERSION
			or owner.get("shard_policy") != SHARD_POLICY
			or owner.get("shard_id") != shard_id
			or owner.get("num_shards") != num_shards
			or owner.get("row_count") != len(rows)
			or owner.get("rows_digest") != rows_digest
			or manifest_counts[shard_id] != len(rows)
			or manifest_digests[shard_id] != rows_digest
		):
			raise DataValidationError(f"shard {shard_id} owner/digest mismatch")
		for row in rows:
			key = shard_key(row)
			if (
				row.get("shard_key") != key
				or int(row.get("shard_id", -1)) != shard_id
				or int(row.get("num_shards", -1)) != num_shards
				or stable_shard_id(key, num_shards) != shard_id
			):
				raise DataValidationError(f"row assigned to wrong shard: {key}")
			identity = canonical_json(dict(row))
			if identity in seen:
				raise DataValidationError("duplicate row across shards")
			seen.add(identity)
		count += len(rows)
	if count != int(manifest.get("row_count", -1)):
		raise DataValidationError("shard coverage count mismatch")
	return {"status": "valid", "row_count": count, "num_shards": num_shards, "shard_policy": SHARD_POLICY}


def merge_completed_shards(
	output_root: Path,
	merged_root: Path,
	*,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Merge validated per-card derived targets into one training-only bundle.

	Action images/results remain in their independently owned shard directories;
	the merged rows already contain absolute source-image paths.  Global class
	weights are recomputed after concatenation instead of averaging per-shard
	weights.
	"""

	root = output_root.resolve()
	merged_root = merged_root.resolve()
	partition = validate_partition(root)
	from .aggregate import _training_balance_statistics
	from .validate import validate_dataset

	config_snapshot: dict[str, Any] | None = None
	evaluator_snapshot: dict[str, Any] | None = None
	aggregation_snapshot: dict[str, Any] | None = None
	qwen_distribution_signature: str | None = None
	flux2_policy_digest_snapshot: str | None = None
	flux2_policy_contract_snapshot: dict[str, Any] | None = None
	l3_policy_digest_snapshot: str | None = None
	l3_policy_contract_snapshot: dict[str, Any] | None = None
	initial_generator_family_snapshot: str | None = None
	targets: list[dict[str, Any]] = []
	rollouts: list[dict[str, Any]] = []
	quarantine: list[dict[str, Any]] = []
	shard_records: list[dict[str, Any]] = []
	state_owner: dict[str, int] = {}
	group_owner: dict[str, int] = {}
	job_ids: set[str] = set()
	for shard_id in range(int(partition["num_shards"])):
		dataset = root / "shards" / f"shard_{shard_id:02d}" / "dataset"
		validation = validate_dataset(
			dataset,
			require_results=True,
			require_evaluations=True,
			require_targets=True,
		)
		current_config = read_json(dataset / "config.lock.json")
		current_evaluator = read_json(dataset / "evaluator.lock.json")
		current_summary = read_json(dataset / "derived" / "summary.json")
		current_signature = str(current_summary.get("qwen_distribution_signature") or "").strip().lower()
		current_flux2_digest = str(current_summary.get("flux2_policy_digest") or "").strip().lower()
		current_flux2_contract = current_summary.get("flux2_policy_contract")
		current_l3_digest = str(current_summary.get("l3_policy_digest") or "").strip().lower()
		current_l3_contract = current_summary.get("l3_policy_contract")
		current_initial_family = str(current_summary.get("initial_generator_family") or "")
		if re.fullmatch(r"[0-9a-f]{64}", current_signature) is None:
			raise DataValidationError(f"shard {shard_id} has no valid Qwen distribution signature")
		if re.fullmatch(r"[0-9a-f]{64}", current_flux2_digest) is None or not isinstance(current_flux2_contract, Mapping):
			raise DataValidationError(f"shard {shard_id} has no valid Flux.2 collector-policy lock")
		if re.fullmatch(r"[0-9a-f]{64}", current_l3_digest) is None or not isinstance(current_l3_contract, Mapping):
			raise DataValidationError(f"shard {shard_id} has no valid initial-base L3 policy lock")
		if config_snapshot is None:
			config_snapshot = dict(current_config)
			evaluator_snapshot = dict(current_evaluator)
			aggregation_snapshot = dict(current_summary.get("aggregation") or {})
			qwen_distribution_signature = current_signature
			flux2_policy_digest_snapshot = current_flux2_digest
			flux2_policy_contract_snapshot = dict(current_flux2_contract)
			l3_policy_digest_snapshot = current_l3_digest
			l3_policy_contract_snapshot = dict(current_l3_contract)
			initial_generator_family_snapshot = current_initial_family
		if current_config != config_snapshot:
			raise DataValidationError(f"shard {shard_id} uses a different dataset config")
		if current_evaluator != evaluator_snapshot:
			raise DataValidationError(f"shard {shard_id} uses a different evaluator contract")
		if dict(current_summary.get("aggregation") or {}) != aggregation_snapshot:
			raise DataValidationError(f"shard {shard_id} uses different aggregation settings")
		if current_signature != qwen_distribution_signature:
			raise DataValidationError("completed shards mix Qwen analyzer/L1 distributions")
		if current_flux2_digest != flux2_policy_digest_snapshot or dict(current_flux2_contract) != flux2_policy_contract_snapshot:
			raise DataValidationError("completed shards mix Flux.2 collector policies")
		if (
			current_l3_digest != l3_policy_digest_snapshot
			or dict(current_l3_contract) != l3_policy_contract_snapshot
			or current_initial_family != initial_generator_family_snapshot
		):
			raise DataValidationError("completed shards mix initial-image bases or L3 executor policies")
		current_targets = list(iter_jsonl(dataset / "derived" / "state_targets.jsonl"))
		current_rollouts = list(iter_jsonl(dataset / "derived" / "rollout_utilities.jsonl"))
		current_quarantine = list(iter_jsonl(dataset / "derived" / "quarantine.jsonl"))
		for row in [*current_targets, *current_rollouts, *current_quarantine]:
			if row.get("qwen_distribution_signature") != current_signature:
				raise DataValidationError(f"shard {shard_id} mixes Qwen distributions in derived rows")
			if row.get("flux2_policy_digest") != current_flux2_digest:
				raise DataValidationError(f"shard {shard_id} mixes Flux.2 policy signatures in derived rows")
			if row.get("l3_policy_digest") != current_l3_digest:
				raise DataValidationError(f"shard {shard_id} mixes L3 policy signatures in derived rows")
		for row in [*current_targets, *current_quarantine]:
			state_id = str(row.get("state_id") or "")
			if not state_id or state_id in state_owner:
				raise DataValidationError(f"duplicate/invalid state across shards: {state_id!r}")
			state_owner[state_id] = shard_id
			group_id = str(row.get("group_id") or "")
			if group_id:
				previous = group_owner.setdefault(group_id, shard_id)
				if previous != shard_id:
					raise DataValidationError(f"group crosses shard ownership: {group_id}")
		for row in current_rollouts:
			job_id = str(row.get("job_id") or "")
			if not job_id or job_id in job_ids:
				raise DataValidationError(f"duplicate/invalid rollout job across shards: {job_id!r}")
			job_ids.add(job_id)
		targets.extend(current_targets)
		rollouts.extend(current_rollouts)
		quarantine.extend(current_quarantine)
		shard_records.append(
			{
				"shard_id": shard_id,
				"dataset_root": str(dataset),
				"state_count": validation["state_count"],
				"job_count": validation["job_count"],
				"target_count": len(current_targets),
				"quarantine_count": len(current_quarantine),
				"target_digest": sha256_json(current_targets),
			}
		)
	if (
		config_snapshot is None or evaluator_snapshot is None or aggregation_snapshot is None
		or qwen_distribution_signature is None or flux2_policy_digest_snapshot is None
		or flux2_policy_contract_snapshot is None
		or l3_policy_digest_snapshot is None or l3_policy_contract_snapshot is None
		or initial_generator_family_snapshot is None
	):
		raise DataValidationError("no completed shard datasets found")
	targets.sort(key=lambda row: row["state_id"])
	rollouts.sort(key=lambda row: (row["state_id"], row["rollout_index"], row["action_level"]))
	quarantine.sort(key=lambda row: row["state_id"])
	config = load_config(root / "shards" / "shard_00" / "dataset" / "config.lock.json", strict_revision=True)
	training_balance = _training_balance_statistics(targets, config)
	split_rows = {
		name: [row for row in targets if row["split"] == name]
		for name in ("train", "validation", "test")
	}
	manifest = {
		"schema_version": SCHEMA_VERSION,
		"status": "merged_training_bundle",
		"shard_policy": SHARD_POLICY,
		"num_shards": int(partition["num_shards"]),
		"source_root": str(root),
		"state_count": len(state_owner),
		"valid_rollout_count": sum(bool(row.get("valid")) for row in rollouts),
		"invalid_rollout_count": sum(not bool(row.get("valid")) for row in rollouts),
		"trainable_state_count": len(targets),
		"quarantined_state_count": len(quarantine),
		"qwen_distribution_signature": qwen_distribution_signature,
		"flux2_policy_contract": flux2_policy_contract_snapshot,
		"flux2_policy_digest": flux2_policy_digest_snapshot,
		"l3_policy_contract": l3_policy_contract_snapshot,
		"l3_policy_digest": l3_policy_digest_snapshot,
		"initial_generator_family": initial_generator_family_snapshot,
		"l3_initial_identity_binding_counts": dict(sorted(Counter(
			str(row.get("l3_initial_identity_binding") or "") for row in targets
		).items())),
		"split_target_counts": {name: len(rows) for name, rows in split_rows.items()},
		"target_distribution": {
			"unique_hard_target": dict(
				sorted(Counter(f"L{row['hard_target']}" for row in targets if row.get("hard_target") is not None).items())
			),
			"ties": sum(row.get("hard_target") is None for row in targets),
		},
		"training_balance": training_balance,
		"aggregation": aggregation_snapshot,
		"evaluator_signature": evaluator_snapshot["evaluator_signature"],
		"shards": shard_records,
		"derived_digests": {
			"rollout_utilities": sha256_json(rollouts),
			"state_targets": sha256_json(targets),
			"quarantine": sha256_json(quarantine),
			**{f"split_{name}": sha256_json(rows) for name, rows in split_rows.items()},
		},
	}
	if not dry_run:
		write_json_atomic(merged_root / "config.lock.json", config_snapshot)
		write_json_atomic(merged_root / "evaluator.lock.json", evaluator_snapshot)
		derived = merged_root / "derived"
		write_jsonl_atomic(derived / "rollout_utilities.jsonl", rollouts)
		write_jsonl_atomic(derived / "state_targets.jsonl", targets)
		write_jsonl_atomic(derived / "quarantine.jsonl", quarantine)
		for name, rows in split_rows.items():
			write_jsonl_atomic(derived / f"{name}.jsonl", rows)
		write_json_atomic(derived / "summary.json", manifest)
	return manifest


def merge_training_bundles(
	bundle_roots: Sequence[Path],
	merged_root: Path,
	*,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Merge disjoint, already validated training bundles.

	This is the second-stage merge used when independently collected phases
	(such as a pilot and its remainder) share the exact same frozen contracts.
	The source bundles are verified by their recorded row digests before global
	class weights and split files are recomputed.
	"""

	if len(bundle_roots) < 2:
		raise DataValidationError("merge-training-bundles requires at least two inputs")
	from .aggregate import _training_balance_statistics

	roots = [Path(root).resolve() for root in bundle_roots]
	if len(set(roots)) != len(roots):
		raise DataValidationError("training bundle inputs must be distinct")
	merged_root = merged_root.resolve()

	config_snapshot: dict[str, Any] | None = None
	evaluator_snapshot: dict[str, Any] | None = None
	contract_snapshot: dict[str, Any] | None = None
	targets: list[dict[str, Any]] = []
	rollouts: list[dict[str, Any]] = []
	quarantine: list[dict[str, Any]] = []
	source_records: list[dict[str, Any]] = []
	state_ids: set[str] = set()
	group_ids: set[str] = set()
	job_ids: set[str] = set()

	contract_keys = (
		"schema_version",
		"qwen_distribution_signature",
		"flux2_policy_digest",
		"flux2_policy_contract",
		"l3_policy_digest",
		"l3_policy_contract",
		"initial_generator_family",
		"aggregation",
	)
	for root in roots:
		summary_path = root / "derived" / "summary.json"
		if not summary_path.is_file():
			raise DataValidationError(f"training bundle has no summary: {root}")
		summary = read_json(summary_path)
		if summary.get("status") != "merged_training_bundle":
			raise DataValidationError(f"input is not a merged training bundle: {root}")
		current_config = read_json(root / "config.lock.json")
		current_evaluator = read_json(root / "evaluator.lock.json")
		current_contract = {key: summary.get(key) for key in contract_keys}
		if config_snapshot is None:
			config_snapshot = dict(current_config)
			evaluator_snapshot = dict(current_evaluator)
			contract_snapshot = current_contract
		elif current_config != config_snapshot:
			raise DataValidationError("training bundles use different dataset configs")
		elif current_evaluator != evaluator_snapshot:
			raise DataValidationError("training bundles use different evaluator contracts")
		elif current_contract != contract_snapshot:
			raise DataValidationError("training bundles use different collection contracts")

		current_targets = list(iter_jsonl(root / "derived" / "state_targets.jsonl"))
		current_rollouts = list(iter_jsonl(root / "derived" / "rollout_utilities.jsonl"))
		current_quarantine = list(iter_jsonl(root / "derived" / "quarantine.jsonl"))
		digests = summary.get("derived_digests") or {}
		for name, rows in (
			("state_targets", current_targets),
			("rollout_utilities", current_rollouts),
			("quarantine", current_quarantine),
		):
			if digests.get(name) != sha256_json(rows):
				raise DataValidationError(f"training bundle {name} digest mismatch: {root}")
		if int(summary.get("state_count", -1)) != len(current_targets) + len(current_quarantine):
			raise DataValidationError(f"training bundle state count mismatch: {root}")
		if int(summary.get("valid_rollout_count", -1)) != sum(bool(row.get("valid")) for row in current_rollouts):
			raise DataValidationError(f"training bundle valid rollout count mismatch: {root}")
		if int(summary.get("invalid_rollout_count", -1)) != sum(not bool(row.get("valid")) for row in current_rollouts):
			raise DataValidationError(f"training bundle invalid rollout count mismatch: {root}")

		local_state_ids = {str(row.get("state_id") or "") for row in [*current_targets, *current_quarantine]}
		if "" in local_state_ids or state_ids.intersection(local_state_ids):
			raise DataValidationError("duplicate/invalid state across training bundles")
		state_ids.update(local_state_ids)
		for row in [*current_targets, *current_quarantine]:
			group_id = str(row.get("group_id") or "")
			if group_id and group_id in group_ids:
				raise DataValidationError(f"group crosses training bundles: {group_id}")
			if group_id:
				group_ids.add(group_id)
		for row in current_rollouts:
			job_id = str(row.get("job_id") or "")
			if not job_id or job_id in job_ids:
				raise DataValidationError(f"duplicate/invalid rollout job across training bundles: {job_id!r}")
			if str(row.get("state_id") or "") not in local_state_ids:
				raise DataValidationError(f"rollout references an unknown state: {job_id}")
			job_ids.add(job_id)

		targets.extend(current_targets)
		rollouts.extend(current_rollouts)
		quarantine.extend(current_quarantine)
		source_records.append(
			{
				"bundle_root": str(root),
				"state_count": len(local_state_ids),
				"target_count": len(current_targets),
				"valid_rollout_count": sum(bool(row.get("valid")) for row in current_rollouts),
				"summary_digest": sha256_file(summary_path),
			}
		)

	if config_snapshot is None or evaluator_snapshot is None or contract_snapshot is None:
		raise DataValidationError("no completed training bundles found")
	targets.sort(key=lambda row: row["state_id"])
	rollouts.sort(key=lambda row: (row["state_id"], row["rollout_index"], row["action_level"]))
	quarantine.sort(key=lambda row: row["state_id"])
	config = load_config(roots[0] / "config.lock.json", strict_revision=True)
	training_balance = _training_balance_statistics(targets, config)
	split_rows = {name: [row for row in targets if row["split"] == name] for name in ("train", "validation", "test")}
	manifest = {
		**contract_snapshot,
		"status": "merged_training_bundle",
		"merge_policy": "disjoint_completed_training_bundles_v1",
		"source_bundles": source_records,
		"state_count": len(state_ids),
		"valid_rollout_count": sum(bool(row.get("valid")) for row in rollouts),
		"invalid_rollout_count": sum(not bool(row.get("valid")) for row in rollouts),
		"trainable_state_count": len(targets),
		"quarantined_state_count": len(quarantine),
		"l3_initial_identity_binding_counts": dict(sorted(Counter(
			str(row.get("l3_initial_identity_binding") or "") for row in targets
		).items())),
		"split_target_counts": {name: len(rows) for name, rows in split_rows.items()},
		"target_distribution": {
			"unique_hard_target": dict(sorted(Counter(f"L{row['hard_target']}" for row in targets if row.get("hard_target") is not None).items())),
			"ties": sum(row.get("hard_target") is None for row in targets),
		},
		"training_balance": training_balance,
		"evaluator_signature": evaluator_snapshot["evaluator_signature"],
		"derived_digests": {
			"rollout_utilities": sha256_json(rollouts),
			"state_targets": sha256_json(targets),
			"quarantine": sha256_json(quarantine),
			**{f"split_{name}": sha256_json(rows) for name, rows in split_rows.items()},
		},
	}
	if not dry_run:
		write_json_atomic(merged_root / "config.lock.json", config_snapshot)
		write_json_atomic(merged_root / "evaluator.lock.json", evaluator_snapshot)
		derived = merged_root / "derived"
		write_jsonl_atomic(derived / "rollout_utilities.jsonl", rollouts)
		write_jsonl_atomic(derived / "state_targets.jsonl", targets)
		write_jsonl_atomic(derived / "quarantine.jsonl", quarantine)
		for name, rows in split_rows.items():
			write_jsonl_atomic(derived / f"{name}.jsonl", rows)
		write_json_atomic(derived / "summary.json", manifest)
	return manifest


def merge_completed_datasets(
	dataset_roots: Sequence[Path],
	merged_root: Path,
	*,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Merge disjoint completed datasets into one directly usable dataset root.

	Unlike :func:`merge_training_bundles`, this preserves the complete dataset
	interface (states, jobs, features, results and evaluations).  Large per-state
	artifact trees are exposed through relative symlinks instead of copied.
	"""

	if len(dataset_roots) < 2:
		raise DataValidationError("merge-completed-datasets requires at least two inputs")
	from .aggregate import aggregate_dataset
	from .validate import validate_dataset

	roots = [Path(root).resolve() for root in dataset_roots]
	if len(set(roots)) != len(roots):
		raise DataValidationError("completed dataset inputs must be distinct")
	merged_root = merged_root.resolve()

	config_snapshot: dict[str, Any] | None = None
	evaluator_snapshot: dict[str, Any] | None = None
	manifest_snapshot: dict[str, Any] | None = None
	states: list[dict[str, Any]] = []
	jobs: list[dict[str, Any]] = []
	features: list[dict[str, Any]] = []
	state_ids: set[str] = set()
	group_ids: set[str] = set()
	job_ids: set[str] = set()
	artifact_owners: dict[str, Path] = {}
	source_records: list[dict[str, Any]] = []

	contract_keys = (
		"schema_version",
		"dataset_name",
		"profile",
		"rollouts_per_action",
		"action_levels",
		"policy_signature",
		"flux2_policy_contract",
		"flux2_policy_digest",
		"l3_policy_contract",
		"l3_policy_digest",
		"qwen_distribution_signature",
		"transition_policy",
	)
	for root in roots:
		validation = validate_dataset(
			root,
			require_results=True,
			require_evaluations=True,
			require_targets=True,
		)
		current_config = read_json(root / "config.lock.json")
		current_evaluator = read_json(root / "evaluator.lock.json")
		current_manifest = read_json(root / "manifest.json")
		current_contract = {key: current_manifest.get(key) for key in contract_keys}
		if config_snapshot is None:
			config_snapshot = dict(current_config)
			evaluator_snapshot = dict(current_evaluator)
			manifest_snapshot = current_contract
		elif current_config != config_snapshot:
			raise DataValidationError("completed datasets use different dataset configs")
		elif current_evaluator != evaluator_snapshot:
			raise DataValidationError("completed datasets use different evaluator contracts")
		elif current_contract != manifest_snapshot:
			raise DataValidationError("completed datasets use different collection contracts")

		current_states = list(iter_jsonl(root / "states.jsonl"))
		current_jobs = list(iter_jsonl(root / "jobs.jsonl"))
		current_features = list(iter_jsonl(root / "features.jsonl"))
		local_state_ids = {str(row.get("state_id") or "") for row in current_states}
		local_job_ids = {str(row.get("job_id") or "") for row in current_jobs}
		if "" in local_state_ids or state_ids.intersection(local_state_ids):
			raise DataValidationError("duplicate/invalid state across completed datasets")
		if "" in local_job_ids or job_ids.intersection(local_job_ids):
			raise DataValidationError("duplicate/invalid job across completed datasets")
		state_ids.update(local_state_ids)
		job_ids.update(local_job_ids)
		for state in current_states:
			group_id = str(state.get("group_id") or "")
			if group_id and group_id in group_ids:
				raise DataValidationError(f"group crosses completed datasets: {group_id}")
			if group_id:
				group_ids.add(group_id)
		for job in current_jobs:
			artifact_dir = Path(str(job.get("artifact_dir") or ""))
			if len(artifact_dir.parts) < 2 or artifact_dir.parts[0] != "artifacts":
				raise DataValidationError(f"invalid relative artifact directory: {artifact_dir}")
			state_artifact_name = artifact_dir.parts[1]
			source_artifact = root / "artifacts" / state_artifact_name
			if not source_artifact.is_dir():
				raise DataValidationError(f"missing state artifact directory: {source_artifact}")
			previous = artifact_owners.setdefault(state_artifact_name, source_artifact)
			if previous != source_artifact:
				raise DataValidationError(f"artifact directory collision: {state_artifact_name}")

		states.extend(current_states)
		jobs.extend(current_jobs)
		features.extend(current_features)
		source_records.append(
			{
				"dataset_root": str(root),
				"state_count": validation["state_count"],
				"job_count": validation["job_count"],
				"manifest_sha256": sha256_file(root / "manifest.json"),
				"derived_summary_sha256": sha256_file(root / "derived" / "summary.json"),
			}
		)

	if config_snapshot is None or evaluator_snapshot is None or manifest_snapshot is None:
		raise DataValidationError("no completed datasets found")
	states.sort(key=lambda row: row["state_id"])
	jobs.sort(key=lambda row: (row["state_id"], row["rollout_index"], row["action_level"]))
	features.sort(key=lambda row: row["state_id"])
	manifest = {
		**manifest_snapshot,
		"status": "completed_merged_dataset",
		"merge_policy": "disjoint_completed_datasets_symlink_artifacts_v1",
		"state_count": len(states),
		"job_count": len(jobs),
		"action_contract_digest": action_contract_digest(jobs),
		"config_sha256": sha256_json(config_snapshot),
		"frozen_digests": {
			"states": sha256_json(states),
			"jobs": sha256_json(jobs),
			"features": sha256_json(features),
		},
		"source_manifest": {
			"merge_policy": "disjoint_completed_datasets_symlink_artifacts_v1",
			"datasets": source_records,
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
	if dry_run:
		return manifest

	write_json_atomic(merged_root / "config.lock.json", config_snapshot)
	write_json_atomic(merged_root / "evaluator.lock.json", evaluator_snapshot)
	write_jsonl_atomic(merged_root / "states.jsonl", states)
	write_jsonl_atomic(merged_root / "jobs.jsonl", jobs)
	write_jsonl_atomic(merged_root / "features.jsonl", features)
	write_json_atomic(merged_root / "manifest.json", manifest)
	artifacts_root = merged_root / "artifacts"
	artifacts_root.mkdir(parents=True, exist_ok=True)
	for name, source in sorted(artifact_owners.items()):
		destination = artifacts_root / name
		relative_source = Path(os.path.relpath(source, start=artifacts_root))
		if destination.is_symlink():
			if destination.readlink() != relative_source:
				raise DataValidationError(f"existing artifact symlink differs: {destination}")
		elif destination.exists():
			raise DataValidationError(f"artifact destination already exists: {destination}")
		else:
			destination.symlink_to(relative_source, target_is_directory=True)
	aggregation = aggregate_dataset(merged_root, dry_run=False)
	validation = validate_dataset(
		merged_root,
		require_results=True,
		require_evaluations=True,
		require_targets=True,
	)
	return {
		**manifest,
		"aggregation": aggregation,
		"validation_status": validation["status"],
		"artifact_symlink_count": len(artifact_owners),
	}


__all__ = [
	"SHARD_POLICY",
	"merge_completed_shards",
	"merge_completed_datasets",
	"merge_training_bundles",
	"partition_file",
	"partition_rows",
	"shard_key",
	"stable_shard_id",
	"validate_partition",
]
