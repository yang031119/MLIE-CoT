"""Plan and collect V10 reflections over already-materialized initial images.

This is the bridge between existing immutable initial assets and the router-data
pipeline.  It copies only the small metadata snapshots into a work directory;
the multi-gigabyte image corpus remains in its asset root and is referenced by
an absolute, hash-locked path.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from inference.main_method.modules.contracts import (
	DEFAULT_MAX_REPAIR_ACTIONS_V10,
	DIAGNOSIS_SCHEMA,
	diagnosis_payload_is_valid,
)

from .bootstrap import (
	REFLECTION_STAGE,
	REFLECTION_PROMPT_VERSION,
	_reflection_request_payload,
	_jsonl_text,
	_write_artifacts,
	load_reflector_config,
	reflection_prompt,
	validate_reflection_job_contract,
	validate_reflection_result,
)
from .canonical import (
	iter_jsonl,
	read_json,
	sha256_file,
	sha256_json,
	sha256_text,
	stable_id,
	stable_seed,
)
from .planner import load_config, load_state_rows
from .schema import (
	SCHEMA_VERSION,
	DataValidationError,
	canonicalize_state,
	image_descriptor,
	normalize_text,
)
from .sharding import SHARD_POLICY, shard_key, stable_shard_id


EXISTING_SOURCE_MODE = "verified_external_initial_assets_v1"
EXISTING_LOCK_SCHEMA = "vcot_v10_existing_initial_assets_lock_v1"
EXISTING_LOCK_FILE = "existing_assets.lock.json"
SOURCE_MANIFEST_SNAPSHOT = "source_manifest.snapshot.json"
SOURCE_SAMPLES_SNAPSHOT = "source_samples.snapshot.jsonl"
LEGACY_ASSET_SCHEMA = "v10_legacy_initial_assets_v1"
_SHARD_LOCK_KEYS = {
	"shard_sample_count",
	"shard_policy",
	"shard_id",
	"num_shards",
	"full_assignment_digest",
	"shard_assignment_digest",
}


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise DataValidationError(message)


def _safe_relative_asset_path(value: Any) -> str:
	text = normalize_text(value)
	_require(bool(text), "existing initial-image path is empty")
	path = PurePosixPath(text)
	_require(not path.is_absolute(), f"asset image path must be relative: {text!r}")
	_require(
		all(part not in {"", ".", ".."} for part in path.parts),
		f"asset image path contains an unsafe component: {text!r}",
	)
	return path.as_posix()


def _path_inside(root: Path, relative: str) -> Path:
	root = root.resolve()
	path = (root / _safe_relative_asset_path(relative)).resolve()
	try:
		path.relative_to(root)
	except ValueError as exc:
		raise DataValidationError(f"asset path escapes the locked asset root: {relative!r}") from exc
	return path


def _source_lock_payload(lock: Mapping[str, Any]) -> dict[str, Any]:
	# ``source_signature`` identifies the immutable full corpus, independent of
	# how it is distributed across workers.  This keeps per-sample seeds and job
	# identities stable between one-card and four-card planning.
	return {
		key: value
		for key, value in lock.items()
		if key != "source_signature" and key not in _SHARD_LOCK_KEYS
	}


def _read_legacy_source(asset_root: Path) -> tuple[dict[str, Any], str, list[dict[str, Any]], str]:
	"""Read the legacy asset manifest and return its rows with source hashes."""
	asset_root = asset_root.resolve()
	manifest_path = asset_root / "manifest.json"
	_require(manifest_path.is_file(), f"legacy asset manifest is missing: {manifest_path}")
	manifest_text = manifest_path.read_text(encoding="utf-8")
	manifest = json.loads(manifest_text)
	_require(isinstance(manifest, Mapping), "legacy asset manifest must be an object")
	_require(manifest.get("schema_version") == LEGACY_ASSET_SCHEMA, "unsupported existing-asset schema")
	_require(manifest.get("self_contained") is True, "existing asset corpus must be self-contained")
	_require(
		manifest.get("migration_contract") == "prompt_and_initial_image_only",
		"existing asset corpus has an incompatible migration contract",
	)
	_require(
		manifest.get("old_analyses_or_candidates_copied") is False,
		"existing asset corpus must not contain migrated action labels",
	)
	samples_relpath = _safe_relative_asset_path(manifest.get("samples_path"))
	samples_path = _path_inside(asset_root, samples_relpath)
	_require(samples_path.is_file(), f"legacy asset samples are missing: {samples_path}")
	samples_text = samples_path.read_text(encoding="utf-8")
	_require(
		manifest.get("samples_sha256") == sha256_text(samples_text),
		"legacy samples hash differs from its manifest",
	)
	rows = load_state_rows(samples_path)
	_require(bool(rows), "legacy asset sample table is empty")
	_require(
		manifest.get("counts", {}).get("samples") == len(rows),
		"legacy asset manifest sample count mismatch",
	)
	return dict(manifest), manifest_text, rows, samples_text


def _validate_asset_row(
	row: Mapping[str, Any],
	*,
	asset_root: Path,
	seen_sample_ids: set[str],
	seen_asset_ids: set[str],
) -> tuple[dict[str, Any], Path]:
	"""Validate asset row and reject contract violations."""
	_require(row.get("schema_version") == LEGACY_ASSET_SCHEMA, "legacy sample schema mismatch")
	sample_id = normalize_text(row.get("sample_id"))
	asset_id = normalize_text(row.get("asset_id"))
	_require(bool(sample_id) and sample_id not in seen_sample_ids, f"invalid or duplicate sample_id: {sample_id!r}")
	_require(bool(asset_id) and asset_id not in seen_asset_ids, f"invalid or duplicate asset_id: {asset_id!r}")
	seen_sample_ids.add(sample_id)
	seen_asset_ids.add(asset_id)
	_require(bool(normalize_text(row.get("target_prompt"))), f"legacy target prompt is empty: {sample_id}")
	_require(bool(normalize_text(row.get("lineage_id"))), f"legacy lineage_id is empty: {sample_id}")
	_require(bool(normalize_text(row.get("variant_id"))), f"legacy variant_id is empty: {sample_id}")
	_require(row.get("migration_contract") == "prompt_and_initial_image_only", f"legacy migration contract mismatch: {sample_id}")
	generator = row.get("initial_generator")
	_require(isinstance(generator, Mapping) and normalize_text(generator.get("family")).lower() == "bagel", f"legacy initial generator must be BAGEL: {sample_id}")
	image = row.get("initial_image")
	_require(isinstance(image, Mapping), f"legacy initial image descriptor is missing: {sample_id}")
	image_relpath = _safe_relative_asset_path(image.get("path"))
	image_path = _path_inside(asset_root, image_relpath)
	actual = image_descriptor(image_path)
	_require(actual["sha256"] == image.get("sha256"), f"legacy image hash mismatch: {sample_id}")
	_require(actual["width"] == image.get("width"), f"legacy image width mismatch: {sample_id}")
	_require(actual["height"] == image.get("height"), f"legacy image height mismatch: {sample_id}")
	_require(image_path.stat().st_size == image.get("bytes"), f"legacy image byte count mismatch: {sample_id}")
	return actual, image_path


def _source_assignments(
	rows: list[dict[str, Any]],
	num_shards: int,
) -> list[dict[str, Any]]:
	"""Return the exact shared prompt-family assignment table."""

	_require(isinstance(num_shards, int) and not isinstance(num_shards, bool) and num_shards > 0, "num_shards must be positive")
	assignments: list[dict[str, Any]] = []
	seen: set[str] = set()
	seen_assets: set[str] = set()
	for index, row in enumerate(rows):
		_require(row.get("schema_version") == LEGACY_ASSET_SCHEMA, f"legacy sample schema mismatch at row {index}")
		sample_id = normalize_text(row.get("sample_id"))
		_require(bool(sample_id) and sample_id not in seen, f"invalid or duplicate source sample_id at row {index}: {sample_id!r}")
		seen.add(sample_id)
		asset_id = normalize_text(row.get("asset_id"))
		_require(bool(asset_id) and asset_id not in seen_assets, f"invalid or duplicate source asset_id at row {index}: {asset_id!r}")
		seen_assets.add(asset_id)
		key = shard_key(row)
		assignments.append(
			{
				"source_sample_id": sample_id,
				"shard_key": key,
				"shard_id": stable_shard_id(key, num_shards),
			}
		)
	assignments.sort(key=lambda row: row["source_sample_id"])
	return assignments


def plan_existing_reflections(
	assets_root: Path,
	work_root: Path,
	reflector_config_path: Path,
	*,
	shard_id: int = 0,
	num_shards: int = 1,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Freeze route-free Qwen jobs over verified external initial images."""

	asset_root = assets_root.resolve()
	root = work_root.resolve()
	contract = load_reflector_config(reflector_config_path.resolve())
	manifest, manifest_text, raw_rows, samples_text = _read_legacy_source(asset_root)
	_require(isinstance(shard_id, int) and not isinstance(shard_id, bool), "shard_id must be an integer")
	_require(isinstance(num_shards, int) and not isinstance(num_shards, bool) and num_shards > 0, "num_shards must be positive")
	_require(0 <= shard_id < num_shards, "shard_id must satisfy 0 <= shard_id < num_shards")
	assignments = _source_assignments(raw_rows, num_shards)
	selected_sample_ids = {
		row["source_sample_id"] for row in assignments if int(row["shard_id"]) == shard_id
	}
	selected_rows = [row for row in raw_rows if normalize_text(row.get("sample_id")) in selected_sample_ids]
	shard_assignments = [row for row in assignments if int(row["shard_id"]) == shard_id]
	source_manifest_sha = sha256_text(manifest_text)
	source_samples_sha = sha256_text(samples_text)
	lock: dict[str, Any] = {
		"schema_version": EXISTING_LOCK_SCHEMA,
		"source_mode": EXISTING_SOURCE_MODE,
		"asset_root": str(asset_root),
		"source_manifest_path": "manifest.json",
		"source_manifest_sha256": source_manifest_sha,
		"source_manifest_snapshot": SOURCE_MANIFEST_SNAPSHOT,
		"source_samples_path": _safe_relative_asset_path(manifest["samples_path"]),
		"source_samples_sha256": source_samples_sha,
		"source_samples_snapshot": SOURCE_SAMPLES_SNAPSHOT,
		"source_content_digest": manifest.get("content_digest"),
		"sample_count": len(raw_rows),
		"full_sample_count": len(raw_rows),
		"shard_sample_count": len(selected_rows),
		"shard_policy": SHARD_POLICY,
		"shard_id": shard_id,
		"num_shards": num_shards,
		"full_assignment_digest": sha256_json(assignments),
		"shard_assignment_digest": sha256_json(shard_assignments),
		"image_path_policy": "absolute_path_under_locked_asset_root_plus_sha256",
		"image_copy_policy": "metadata_snapshots_only_no_image_copy",
	}
	lock["source_signature"] = stable_id("existing_source", _source_lock_payload(lock), length=32)

	seen_samples: set[str] = set()
	seen_assets: set[str] = set()
	jobs: list[dict[str, Any]] = []
	for row in selected_rows:
		descriptor, _image_path = _validate_asset_row(
			row,
			asset_root=asset_root,
			seen_sample_ids=seen_samples,
			seen_asset_ids=seen_assets,
		)
		sample_id = normalize_text(row["sample_id"])
		lineage_id = normalize_text(row["lineage_id"])
		prompt_family_id = normalize_text(row.get("prompt_family_id")) or lineage_id
		source_prompt_id = normalize_text(row.get("source_index_group")) or sample_id
		row_hash = sha256_json(row)
		initial_job_id = stable_id(
			"verified_external_initial",
			{
				"source_signature": lock["source_signature"],
				"sample_id": sample_id,
				"image_sha256": descriptor["sha256"],
			},
			length=32,
		)
		initial_request_hash = sha256_json(
			{
				"source_signature": lock["source_signature"],
				"source_row_sha256": row_hash,
				"source_image": {key: value for key, value in descriptor.items() if key != "path"},
				"initial_generator": "bagel/verified_external",
			}
		)
		base: dict[str, Any] = {
			"schema_version": SCHEMA_VERSION,
			"stage": REFLECTION_STAGE,
			"source_mode": EXISTING_SOURCE_MODE,
			"source_signature": lock["source_signature"],
			"source_sample_id": sample_id,
			"source_asset_id": normalize_text(row["asset_id"]),
			"source_row_sha256": row_hash,
			"sample_id": sample_id,
			"sample_key": stable_id(
				"prompt",
				{
					"dataset": "vcot_v7_legacy_initial_assets",
					"source_prompt_id": source_prompt_id,
					"sample_id": sample_id,
					"variant_id": normalize_text(row["variant_id"]),
				},
				length=32,
			),
			"dataset": "vcot_v7_legacy_initial_assets",
			"source_prompt_id": source_prompt_id,
			"lineage_id": lineage_id,
			"base_prompt_id": source_prompt_id,
			"prompt_family_id": prompt_family_id,
			"variant_id": normalize_text(row["variant_id"]),
			"target_prompt": normalize_text(row["target_prompt"]),
			"initial_generation_prompt": normalize_text(row.get("initial_generation_prompt")),
			"error_family": normalize_text(row.get("error_family")) or "unknown",
			"initial_job_id": initial_job_id,
			"initial_request_hash": initial_request_hash,
			"initial_generator": "bagel/verified_external",
			"source_image": descriptor,
			"analysis_reason": "verified_external_initial_generation",
			"source_round": 0,
			"max_global_rounds": DEFAULT_MAX_REPAIR_ACTIONS_V10,
			"reflector_signature": contract["reflector_signature"],
			"prompt_version": REFLECTION_PROMPT_VERSION,
			"diagnosis_schema": DIAGNOSIS_SCHEMA,
			"distribution_signature": contract["distribution_signature"],
			"seed": stable_seed(contract["seed"], "reflection", initial_job_id),
		}
		base["prompt_sha256"] = sha256_text(reflection_prompt(base))
		request_hash = sha256_json(_reflection_request_payload(base))
		job_id = stable_id("reflection", {"request_hash": request_hash}, length=32)
		jobs.append(
			{
				**base,
				"job_id": job_id,
				"request_hash": request_hash,
				"result_path": f"results/{stable_id('sample', sample_id, length=20)}.json",
			}
		)
	jobs.sort(key=lambda row: row["job_id"])
	for job in jobs:
		validate_reflection_job_contract(job, contract)
	reflection_manifest = {
		"schema_version": SCHEMA_VERSION,
		"stage": REFLECTION_STAGE,
		"status": "planned",
		"source_mode": EXISTING_SOURCE_MODE,
		"source_signature": lock["source_signature"],
		"source_lock_sha256": sha256_json(lock),
		"job_count": len(jobs),
		"reflector_signature": contract["reflector_signature"],
		"distribution_signature": contract["distribution_signature"],
		"diagnosis_schema": DIAGNOSIS_SCHEMA,
		"frozen_digest": sha256_json(jobs),
		"source_manifest_sha256": source_manifest_sha,
		"source_samples_sha256": source_samples_sha,
		"full_source_sample_count": len(raw_rows),
		"shard_sample_count": len(selected_rows),
		"shard_policy": SHARD_POLICY,
		"shard_id": shard_id,
		"num_shards": num_shards,
		"full_assignment_digest": sha256_json(assignments),
		"shard_assignment_digest": sha256_json(shard_assignments),
		"image_copy_count": 0,
	}
	artifacts = {
		root / EXISTING_LOCK_FILE: json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
		root / SOURCE_MANIFEST_SNAPSHOT: manifest_text,
		root / SOURCE_SAMPLES_SNAPSHOT: samples_text,
		root / "reflector.lock.json": json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
		root / "reflection_jobs.jsonl": _jsonl_text(jobs),
		root / "reflection_manifest.json": json.dumps(
			reflection_manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
		) + "\n",
	}
	_write_artifacts(artifacts, dry_run)
	return reflection_manifest


def _load_and_validate_source_lock(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	"""Verify copied source manifests, row assignments, and image hashes before planning any work."""
	root = root.resolve()
	lock_path = root / EXISTING_LOCK_FILE
	_require(lock_path.is_file(), f"existing asset lock is missing: {lock_path}")
	lock = read_json(lock_path)
	_require(isinstance(lock, Mapping), "existing asset lock must be an object")
	_require(lock.get("schema_version") == EXISTING_LOCK_SCHEMA, "existing asset lock schema mismatch")
	_require(lock.get("source_mode") == EXISTING_SOURCE_MODE, "existing asset lock mode mismatch")
	_require(
		lock.get("source_signature") == stable_id("existing_source", _source_lock_payload(lock), length=32),
		"existing asset lock signature mismatch",
	)
	asset_root = Path(str(lock.get("asset_root")))
	_require(asset_root.is_absolute(), "locked asset root must be absolute")
	asset_root = asset_root.resolve()
	manifest_path = _path_inside(asset_root, str(lock.get("source_manifest_path")))
	samples_path = _path_inside(asset_root, str(lock.get("source_samples_path")))
	manifest_snapshot = _path_inside(root, str(lock.get("source_manifest_snapshot")))
	samples_snapshot = _path_inside(root, str(lock.get("source_samples_snapshot")))
	for path, expected, label in (
		(manifest_path, lock.get("source_manifest_sha256"), "source manifest"),
		(samples_path, lock.get("source_samples_sha256"), "source samples"),
		(manifest_snapshot, lock.get("source_manifest_sha256"), "manifest snapshot"),
		(samples_snapshot, lock.get("source_samples_sha256"), "samples snapshot"),
	):
		_require(path.is_file(), f"{label} is missing: {path}")
		_require(sha256_file(path) == expected, f"{label} hash mismatch: {path}")
	manifest = read_json(manifest_snapshot)
	_require(manifest.get("schema_version") == LEGACY_ASSET_SCHEMA, "source manifest snapshot schema mismatch")
	_require(manifest.get("samples_sha256") == lock.get("source_samples_sha256"), "source manifest/sample lock mismatch")
	_require(manifest.get("content_digest") == lock.get("source_content_digest"), "source content digest mismatch")
	rows = list(iter_jsonl(samples_snapshot))
	_require(len(rows) == lock.get("sample_count"), "source sample snapshot count mismatch")
	_require(len(rows) == lock.get("full_sample_count"), "full source sample count mismatch")
	return dict(lock), rows


def validate_existing_reflection_work(
	work_root: Path,
	*,
	verify_images: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path], dict[str, dict[str, Any]]]:
	"""Validate a complete external-source reflection plan and resolve its images."""

	root = work_root.resolve()
	contract = load_reflector_config(root / "reflector.lock.json")
	jobs = list(iter_jsonl(root / "reflection_jobs.jsonl"))
	manifest = read_json(root / "reflection_manifest.json")
	lock, source_rows = _load_and_validate_source_lock(root)
	_require(manifest.get("schema_version") == SCHEMA_VERSION, "reflection manifest schema mismatch")
	_require(manifest.get("stage") == REFLECTION_STAGE, "invalid reflection manifest stage")
	_require(manifest.get("status") == "planned", "invalid reflection manifest status")
	_require(manifest.get("source_mode") == EXISTING_SOURCE_MODE, "reflection manifest source mode mismatch")
	_require(manifest.get("source_signature") == lock["source_signature"], "reflection/source signature mismatch")
	_require(manifest.get("source_lock_sha256") == sha256_json(lock), "reflection source lock hash mismatch")
	_require(manifest.get("reflector_signature") == contract["reflector_signature"], "reflection manifest/reflector mismatch")
	_require(manifest.get("distribution_signature") == contract["distribution_signature"], "reflection manifest/Qwen distribution mismatch")
	_require(manifest.get("diagnosis_schema") == DIAGNOSIS_SCHEMA, "reflection manifest diagnosis schema mismatch")
	_require(manifest.get("job_count") == len(jobs), "reflection manifest job_count mismatch")
	_require(manifest.get("frozen_digest") == sha256_json(jobs), "reflection jobs differ from manifest")
	shard_id = lock.get("shard_id")
	num_shards = lock.get("num_shards")
	_require(isinstance(shard_id, int) and not isinstance(shard_id, bool), "locked shard_id must be an integer")
	_require(isinstance(num_shards, int) and not isinstance(num_shards, bool) and num_shards > 0, "locked num_shards must be positive")
	_require(0 <= shard_id < num_shards, "locked shard_id is outside num_shards")
	assignments = _source_assignments(source_rows, num_shards)
	shard_assignments = [row for row in assignments if int(row["shard_id"]) == shard_id]
	expected_sample_ids = {row["source_sample_id"] for row in shard_assignments}
	_require(lock.get("shard_policy") == SHARD_POLICY, "existing source shard policy mismatch")
	_require(lock.get("full_assignment_digest") == sha256_json(assignments), "full source assignment digest mismatch")
	_require(lock.get("shard_assignment_digest") == sha256_json(shard_assignments), "source shard assignment digest mismatch")
	_require(lock.get("shard_sample_count") == len(shard_assignments), "source shard sample count mismatch")
	for key, expected in (
		("full_source_sample_count", len(source_rows)),
		("shard_sample_count", len(shard_assignments)),
		("shard_policy", SHARD_POLICY),
		("shard_id", shard_id),
		("num_shards", num_shards),
		("full_assignment_digest", sha256_json(assignments)),
		("shard_assignment_digest", sha256_json(shard_assignments)),
	):
		_require(manifest.get(key) == expected, f"reflection manifest {key} mismatch")
	_require(len({job.get("job_id") for job in jobs}) == len(jobs), "duplicate reflection job_id")
	_require(len({job.get("source_sample_id") for job in jobs}) == len(jobs), "duplicate reflection source sample")
	rows_by_sample = {normalize_text(row.get("sample_id")): row for row in source_rows}
	_require(len(rows_by_sample) == len(source_rows), "duplicate sample_id in source snapshot")
	_require(
		expected_sample_ids == {normalize_text(job.get("source_sample_id")) for job in jobs},
		"reflection jobs do not cover their assigned source shard exactly",
	)
	assignment_by_sample = {row["source_sample_id"]: row for row in shard_assignments}
	asset_root = Path(str(lock["asset_root"])).resolve()
	paths: dict[str, Path] = {}
	for job in jobs:
		validate_reflection_job_contract(job, contract)
		_require(job.get("source_mode") == EXISTING_SOURCE_MODE, f"reflection source mode mismatch: {job.get('job_id')}")
		_require(job.get("source_signature") == lock["source_signature"], f"reflection source signature mismatch: {job.get('job_id')}")
		row = rows_by_sample[normalize_text(job["source_sample_id"])]
		assignment = assignment_by_sample[normalize_text(job["source_sample_id"])]
		_require(
			stable_shard_id(assignment["shard_key"], num_shards) == shard_id,
			f"reflection source row belongs to another shard: {job.get('job_id')}",
		)
		row_hash = sha256_json(row)
		_require(job.get("source_row_sha256") == row_hash, f"reflection source row mismatch: {job.get('job_id')}")
		_require(job.get("source_asset_id") == row.get("asset_id"), f"reflection asset id mismatch: {job.get('job_id')}")
		_require(job.get("sample_id") == row.get("sample_id"), f"reflection sample id mismatch: {job.get('job_id')}")
		_require(job.get("lineage_id") == row.get("lineage_id"), f"reflection lineage mismatch: {job.get('job_id')}")
		_require(job.get("target_prompt") == normalize_text(row.get("target_prompt")), f"reflection prompt mismatch: {job.get('job_id')}")
		sample_id = normalize_text(row.get("sample_id"))
		lineage_id = normalize_text(row.get("lineage_id"))
		source_prompt_id = normalize_text(row.get("source_index_group")) or sample_id
		variant_id = normalize_text(row.get("variant_id"))
		expected_sample_key = stable_id(
			"prompt",
			{
				"dataset": "vcot_v7_legacy_initial_assets",
				"source_prompt_id": source_prompt_id,
				"sample_id": sample_id,
				"variant_id": variant_id,
			},
			length=32,
		)
		for key, expected in (
			("dataset", "vcot_v7_legacy_initial_assets"),
			("sample_key", expected_sample_key),
			("source_prompt_id", source_prompt_id),
			("base_prompt_id", source_prompt_id),
			("prompt_family_id", normalize_text(row.get("prompt_family_id")) or lineage_id),
			("variant_id", variant_id),
			("initial_generation_prompt", normalize_text(row.get("initial_generation_prompt"))),
			("error_family", normalize_text(row.get("error_family")) or "unknown"),
			("initial_generator", "bagel/verified_external"),
			("analysis_reason", "verified_external_initial_generation"),
			("source_round", 0),
			("max_global_rounds", DEFAULT_MAX_REPAIR_ACTIONS_V10),
		):
			_require(job.get(key) == expected, f"reflection {key} binding mismatch: {job.get('job_id')}")
		image = row.get("initial_image")
		_require(isinstance(image, Mapping), f"source image descriptor missing: {job.get('job_id')}")
		path = _path_inside(asset_root, str(image.get("path")))
		_require(Path(str(job["source_image"]["path"])).is_absolute(), f"external reflection path must be absolute: {job.get('job_id')}")
		_require(Path(str(job["source_image"]["path"])).resolve() == path, f"reflection image path is outside its locked asset: {job.get('job_id')}")
		if verify_images:
			actual = image_descriptor(path)
			_require(actual == job.get("source_image"), f"reflection source image changed: {job.get('job_id')}")
			_require(actual["sha256"] == image.get("sha256"), f"reflection/source image hash mismatch: {job.get('job_id')}")
		else:
			actual = dict(job["source_image"])
		expected_initial_job_id = stable_id(
			"verified_external_initial",
			{
				"source_signature": lock["source_signature"],
				"sample_id": sample_id,
				"image_sha256": actual["sha256"],
			},
			length=32,
		)
		expected_initial_request_hash = sha256_json(
			{
				"source_signature": lock["source_signature"],
				"source_row_sha256": row_hash,
				"source_image": {key: value for key, value in actual.items() if key != "path"},
				"initial_generator": "bagel/verified_external",
			}
		)
		_require(job.get("initial_job_id") == expected_initial_job_id, f"external initial job identity mismatch: {job.get('job_id')}")
		_require(job.get("initial_request_hash") == expected_initial_request_hash, f"external initial request mismatch: {job.get('job_id')}")
		paths[str(job["job_id"])] = path
	return contract, jobs, paths, rows_by_sample


def collect_existing_states(
	work_root: Path,
	output_path: Path,
	config_path: Path,
	*,
	project_root: Path | None = None,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Validate Qwen results, exclude L0, and emit planner-ready state rows."""

	root = work_root.resolve()
	output_path = output_path.resolve()
	project_root = (project_root or Path.cwd()).resolve()
	config = load_config(config_path.resolve(), strict_revision=True)
	_require(
		config.require_flux2_initial_state is False,
		"legacy BAGEL states require config.require_flux2_initial_state=false",
	)
	_require(
		config.require_verified_initial_record is False,
		"legacy BAGEL provenance is not a reproducible Flux.2 initial record; set require_verified_initial_record=false",
	)
	_require(config.l3_policy.family == "bagel", "legacy BAGEL states require config.l3.family=bagel")
	_require(
		config.allow_legacy_external_l3_identity,
		"legacy BAGEL states have no immutable initial revision; explicitly set "
		"allow_legacy_external_l3_identity=true to use family-only L3 binding",
	)
	contract, jobs, paths, source_rows = validate_existing_reflection_work(root, verify_images=True)
	source_lock = read_json(root / EXISTING_LOCK_FILE)
	states: list[dict[str, Any]] = []
	excluded: list[dict[str, Any]] = []
	for job in jobs:
		result, diagnosis = validate_reflection_result(root, job, contract)
		_require(
			diagnosis_payload_is_valid(result.get("diagnosis")) and result.get("diagnosis") == diagnosis,
			f"reflection diagnosis is not native strict V10 JSON: {job['job_id']}",
		)
		_require(
			result.get("eligible_for_router") is (not diagnosis["fully_satisfied"]),
			f"reflection eligibility flag mismatch: {job['job_id']}",
		)
		row = source_rows[normalize_text(job["source_sample_id"])]
		if diagnosis["fully_satisfied"]:
			excluded.append(
				{
					"sample_id": job["sample_id"],
					"lineage_id": job["lineage_id"],
					"prompt_family_id": job["prompt_family_id"],
					"reflection_job_id": job["job_id"],
					"reason": "fully_satisfied_l0",
				}
			)
			continue
		state_id = stable_id(
			"state",
			{
				"reflection_job_id": job["job_id"],
				"source_sha256": job["source_image"]["sha256"],
				"diagnosis_sha256": sha256_json(diagnosis),
			},
			length=32,
		)
		state: dict[str, Any] = {
			"state_id": state_id,
			"sample_id": job["sample_id"],
			"lineage_id": job["lineage_id"],
			"base_prompt_id": job["base_prompt_id"],
			"prompt_family_id": job["prompt_family_id"],
			"variant_id": job["variant_id"],
			"source_prompt_id": job["source_prompt_id"],
			"dataset": job["dataset"],
			"error_family": job["error_family"],
			"source_round": 0,
			"target_prompt": job["target_prompt"],
			"current_image": str(paths[job["job_id"]]),
			"initial_generator": {
				"family": "bagel",
				"verification": "verified_external",
				"provenance_quality": "legacy_partial_model_identity",
				"model_id": "",
				"model_revision": "",
				"seed": None,
			},
			"initial_record": {
				"status": "ok",
				"executor_family": "verified_external",
				"job_id": job["initial_job_id"],
				"request_hash": job["initial_request_hash"],
				"output_sha256": job["source_image"]["sha256"],
				"reproducible_generation": False,
			},
			"diagnosis": diagnosis,
			"provenance": {
				"input_mode": "bagel/verified_external",
				"work_root": str(root),
				"source_signature": job["source_signature"],
				"shard_policy": source_lock["shard_policy"],
				"shard_id": source_lock["shard_id"],
				"num_shards": source_lock["num_shards"],
				"source_asset_id": job["source_asset_id"],
				"source_row_sha256": job["source_row_sha256"],
				"source_partition": row.get("partition"),
				"source_split": row.get("split"),
				"source_sampling": dict(row.get("sampling") or {}),
				"source_corpus": dict(row.get("source") or {}),
				"initial_generation_prompt": row.get("initial_generation_prompt"),
				"corruption": dict(row.get("corruption") or {}),
				"reflection_job_id": job["job_id"],
				"reflector_model": contract["model"],
				"reflector_model_revision": contract["model_revision"],
				"qwen_distribution_signature": contract["distribution_signature"],
				"diagnosis_schema": DIAGNOSIS_SCHEMA,
				"json_retry_count": result["json_retry_count"],
			},
		}
		# Exercise the exact downstream state validator now.  The raw state is
		# retained because the planner owns the final canonical serialization.
		canonicalize_state(state, config, project_root)
		states.append(state)
	states.sort(key=lambda row: row["state_id"])
	excluded.sort(key=lambda row: row["sample_id"])
	summary = {
		"schema_version": SCHEMA_VERSION,
		"status": "existing_states_collected",
		"source_mode": EXISTING_SOURCE_MODE,
		"shard_policy": source_lock["shard_policy"],
		"shard_id": source_lock["shard_id"],
		"num_shards": source_lock["num_shards"],
		"full_source_sample_count": source_lock["full_sample_count"],
		"reflection_job_count": len(jobs),
		"eligible_state_count": len(states),
		"excluded_count": len(excluded),
		"excluded_reasons": {"fully_satisfied_l0": len(excluded)},
		"state_digest": sha256_json(states),
		"excluded_digest": sha256_json(excluded),
		"config_sha256": sha256_file(config_path.resolve()),
		"reflector_signature": contract["reflector_signature"],
		"qwen_distribution_signature": contract["distribution_signature"],
		"diagnosis_schema": DIAGNOSIS_SCHEMA,
	}
	artifacts = {
		output_path: _jsonl_text(states),
		output_path.with_suffix(output_path.suffix + ".excluded.jsonl"): _jsonl_text(excluded),
		output_path.with_suffix(output_path.suffix + ".summary.json"): json.dumps(
			summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
		) + "\n",
	}
	_write_artifacts(artifacts, dry_run)
	return summary
