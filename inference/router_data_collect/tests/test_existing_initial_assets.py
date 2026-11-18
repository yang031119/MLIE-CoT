"""End-to-end contract tests for importing, sharding, and consuming existing initial-image assets."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from inference.main_method.modules.contracts import (
	DIAGNOSIS_SCHEMA,
	TRIAGE_PROMPT_VERSION_V10,
	build_level_action_plan,
)
from inference.router_data_collect.bootstrap import REFLECTION_STAGE, plan_l1_action_jobs
from inference.router_data_collect.canonical import (
	canonical_json,
	iter_jsonl,
	sha256_file,
	sha256_json,
	sha256_text,
)
from inference.router_data_collect.cli import build_parser
from inference.router_data_collect.execute_reflections_qwen import (
	DIAGNOSIS_RETRY_INSTRUCTION,
	_normalize_diagnosis_response,
	_validate_plan,
)
from inference.router_data_collect.existing_initial_assets import (
	_source_assignments,
	collect_existing_states,
	plan_existing_reflections,
	validate_existing_reflection_work,
)
from inference.router_data_collect.planner import plan_dataset
from inference.router_data_collect.qwen_batching import generate_with_one_json_retry
from inference.router_data_collect.schema import DataValidationError


def _diagnosis(*, satisfied: bool) -> dict:
	"""Construct the minimal canonical diagnosis fixture used by asset tests."""
	return {
		"schema_version": DIAGNOSIS_SCHEMA,
		"fully_satisfied": satisfied,
		"reason": "all requested content is present" if satisfied else "only one cup is visible",
		"constraint_audit": {
			"objects": "a cup is visible",
			"count": "one cup is visible" if not satisfied else "two cups are visible",
			"attributes": "the visible cups are red",
			"relations": "the cups are on the table",
			"action_state": "no action constraint was requested",
			"scene": "the tabletop scene is correct",
		},
		"issues": [] if satisfied else [
			{
				"constraint_type": "count",
				"target_entity": "red cups",
				"observed": "one red cup",
				"expected": "two red cups",
				"evidence": "only one cup silhouette is visible",
				"confidence": 0.98,
			}
		],
		"preservation_constraints": ["preserve the table and existing cup"],
	}


def _write_asset_corpus(
	root: Path,
	count: int = 2,
	*,
	family_size: int | None = None,
) -> list[dict]:
	"""Materialize a synthetic asset corpus with deterministic image hashes."""
	rows: list[dict] = []
	for index in range(count):
		image_rel = f"images/{index:02d}/image.png"
		image_path = root / image_rel
		image_path.parent.mkdir(parents=True, exist_ok=True)
		Image.new("RGB", (16, 12), (200, index * 30, 20)).save(image_path)
		lineage = "lineage_shared" if family_size is None else f"lineage_{index // family_size:04d}"
		rows.append(
			{
				"schema_version": "v10_legacy_initial_assets_v1",
				"sample_id": f"legacy_{index}",
				"asset_id": f"asset_{index}",
				"lineage_id": lineage,
				"split": "train",
				"partition": "core",
				"source_index_group": f"v7_index_{index:05d}",
				"source_index": index,
				"variant_id": f"variant_{index}",
				"target_prompt": "a photo of two red cups",
				"initial_generation_prompt": "a photo of one red cup",
				"error_family": "counting",
				"corruption": {"type": "wrong_count", "target_count": 2, "corrupt_count": 1},
				"initial_image": {
					"path": image_rel,
					"sha256": sha256_file(image_path),
					"bytes": image_path.stat().st_size,
					"width": 16,
					"height": 12,
					"format": "PNG",
				},
				"initial_generator": {
					"family": "bagel",
					"provenance_quality": "legacy_partial",
					"model_revision": None,
					"per_sample_seed": None,
				},
				"sampling": {"source_index_weight": 1.0},
				"source": {"corpus": "vcot_v7_router_data_qwen9b", "sample_id": f"old_{index}"},
				"migration_contract": "prompt_and_initial_image_only",
			}
		)
	samples_text = "".join(canonical_json(row) + "\n" for row in rows)
	(root / "samples.jsonl").write_text(samples_text, encoding="utf-8", newline="\n")
	manifest = {
		"schema_version": "v10_legacy_initial_assets_v1",
		"self_contained": True,
		"migration_contract": "prompt_and_initial_image_only",
		"old_analyses_or_candidates_copied": False,
		"samples_path": "samples.jsonl",
		"samples_sha256": sha256_text(samples_text),
		"content_digest": sha256_json(
			[{"sample_id": row["sample_id"], "image_sha256": row["initial_image"]["sha256"]} for row in rows]
		),
		"counts": {"samples": len(rows)},
	}
	(root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
	return rows


def _write_reflector_config(path: Path) -> None:
	path.write_text(
		json.dumps(
			{
				"model": "Qwen/Qwen3-VL-9B-Instruct",
				"model_revision": "b" * 40,
				"dtype": "bfloat16",
				"quantization": "none",
				"max_new_tokens": 256,
				"max_image_side": 128,
				"allow_cpu_offload": False,
				"seed": 19,
			}
		),
		encoding="utf-8",
	)


def _write_action_config(path: Path, *, require_flux2: bool = False) -> None:
	"""Write the smallest valid action config, optionally requiring FLUX.2."""
	path.write_text(
		json.dumps(
			{
				"dataset_name": "legacy-test",
				"dataset_seed": 23,
				"rollouts_per_action": 1,
				"require_flux2_initial_state": require_flux2,
				"require_verified_initial_record": False,
				"allow_legacy_external_l3_identity": True,
				"flux2": {
					"model_id": "black-forest-labs/FLUX.2-dev",
					"model_revision": "a" * 40,
					"resolution": 16,
					"steps": 1,
					"guidance_scale": 1.0,
					"max_sequence_length": 32,
					"dtype": "bfloat16",
					"l1_roi_pad_ratio": 1.0,
					"l1_min_span": 2,
					"l1_max_edits": 3,
				},
				"l3": {
					"family": "bagel",
					"model_id": "ByteDance-Seed/BAGEL-7B-MoT",
					"model_revision": "b" * 40,
					"auxiliary_model_id": "vcot/BAGEL-7B-MoT-finetuned",
					"auxiliary_model_revision": "c" * 40,
					"resolution": 16,
					"steps": 1,
					"guidance_scale": 4.0,
					"max_sequence_length": None,
					"max_latent_size": 64,
					"dtype": "bfloat16",
				},
			}
		),
		encoding="utf-8",
	)


def _write_results(work: Path, *, first_satisfied: bool = False) -> None:
	"""Write synthetic evaluation results with a controllable satisfied first row."""
	contract = json.loads((work / "reflector.lock.json").read_text(encoding="utf-8"))
	for index, job in enumerate(iter_jsonl(work / "reflection_jobs.jsonl")):
		diagnosis = _diagnosis(satisfied=first_satisfied and index == 0)
		result = {
			"schema_version": job["schema_version"],
			"stage": REFLECTION_STAGE,
			"status": "ok",
			"executor_family": "qwen",
			"model": contract["model"],
			"model_revision": contract["model_revision"],
			"seed": job["seed"],
			"decoding": contract["decoding"],
			"job_id": job["job_id"],
			"request_hash": job["request_hash"],
			"reflector_signature": contract["reflector_signature"],
			"distribution_signature": contract["distribution_signature"],
			"diagnosis_schema": contract["diagnosis_schema"],
			"prompt_version": job["prompt_version"],
			"prompt_sha256": job["prompt_sha256"],
			"source_image_sha256": job["source_image"]["sha256"],
			"dtype": contract["dtype"],
			"quantization": contract["quantization"],
			"allow_cpu_offload": contract["allow_cpu_offload"],
			"effective_quantization": contract["effective_quantization"],
			"max_image_side": contract["max_image_side"],
			"max_new_tokens": contract["max_new_tokens"],
			"json_retry_count": index % 2,
			"eligible_for_router": not diagnosis["fully_satisfied"],
			"diagnosis": diagnosis,
			"diagnosis_sha256": sha256_json(diagnosis),
			"raw_response": canonical_json(diagnosis),
		}
		path = work / job["result_path"]
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(result), encoding="utf-8")


def test_existing_assets_plan_collect_and_feed_unified_planners(tmp_path: Path) -> None:
	"""Verify existing assets plan collect and feed unified planners."""
	assets = tmp_path / "assets"
	assets.mkdir()
	_write_asset_corpus(assets)
	reflector = tmp_path / "reflector.json"
	_write_reflector_config(reflector)
	work = tmp_path / "reflection_work"
	manifest = plan_existing_reflections(assets, work, reflector)
	assert manifest["job_count"] == 2
	assert manifest["image_copy_count"] == 0
	assert not list(work.rglob("*.png"))

	contract = json.loads((work / "reflector.lock.json").read_text(encoding="utf-8"))
	jobs = list(iter_jsonl(work / "reflection_jobs.jsonl"))
	paths = _validate_plan(work, jobs, contract)
	assert set(paths) == {job["job_id"] for job in jobs}
	assert all(path.is_relative_to(assets.resolve()) for path in paths.values())
	assert all(Path(job["source_image"]["path"]).is_absolute() for job in jobs)
	assert {job["lineage_id"] for job in jobs} == {"lineage_shared"}
	assert {job["prompt_family_id"] for job in jobs} == {"lineage_shared"}

	_write_results(work, first_satisfied=True)
	config = tmp_path / "config.json"
	_write_action_config(config)
	states_path = tmp_path / "states.jsonl"
	summary = collect_existing_states(work, states_path, config, project_root=tmp_path)
	assert summary["eligible_state_count"] == 1
	assert summary["excluded_count"] == 1
	states = list(iter_jsonl(states_path))
	assert states[0]["initial_generator"] == {
		"family": "bagel",
		"model_id": "",
		"model_revision": "",
		"provenance_quality": "legacy_partial_model_identity",
		"seed": None,
		"verification": "verified_external",
	}
	assert states[0]["provenance"]["input_mode"] == "bagel/verified_external"
	assert states[0]["prompt_family_id"] == "lineage_shared"

	plan_root = tmp_path / "l1_work"
	l1_manifest = plan_l1_action_jobs(states_path, reflector, plan_root, project_root=tmp_path)
	assert l1_manifest["job_count"] == 1
	state = states[0]
	action_plan = build_level_action_plan(
		1,
		state["target_prompt"],
		state["diagnosis"],
		edits=[
			{
				"id": "edit_01",
				"target_entity": "red cups",
				"instruction": "add the missing red cup",
				"quad_points": [[1, 1], [5, 1], [5, 5], [1, 5]],
			}
		],
		width=16,
		height=12,
	)
	l1_plans = tmp_path / "l1_plans.jsonl"
	l1_plans.write_text(
		canonical_json(
			{
				"state_id": state["state_id"],
				"distribution_signature": state["provenance"]["qwen_distribution_signature"],
				"diagnosis_sha256": sha256_json(state["diagnosis"]),
				"source_image_sha256": sha256_file(Path(state["current_image"])),
				"action_plan": action_plan,
			}
		) + "\n",
		encoding="utf-8",
	)
	dataset_root = tmp_path / "dataset"
	planned = plan_dataset(
		config,
		states_path,
		dataset_root,
		tmp_path,
		l1_plans_path=l1_plans,
		dry_run=False,
	)
	assert planned["state_count"] == 1
	assert planned["job_count"] == 3
	assert planned["qwen_distribution_signature"] == state["provenance"]["qwen_distribution_signature"]
	action_jobs = list(iter_jsonl(dataset_root / "jobs.jsonl"))
	assert {job["action_level"]: job["executor_family"] for job in action_jobs} == {
		1: "flux2",
		2: "flux2",
		3: "bagel",
	}
	assert next(job for job in action_jobs if job["action_level"] == 3)["initial_generator"][
		"l3_identity_binding"
	] == "legacy_bagel_family_only_unverified_revision"
	tampered_plans = tmp_path / "tampered_l1_plans.jsonl"
	tampered = next(iter_jsonl(l1_plans))
	tampered["distribution_signature"] = "0" * 64
	tampered_plans.write_text(canonical_json(tampered) + "\n", encoding="utf-8")
	with pytest.raises(DataValidationError, match="Qwen distribution mismatch"):
		plan_dataset(
			config,
			states_path,
			tmp_path / "tampered_dataset",
			tmp_path,
			l1_plans_path=tampered_plans,
			dry_run=True,
		)


def test_existing_collection_rejects_qwen_runtime_distribution_drift(tmp_path: Path) -> None:
	"""Verify existing collection rejects qwen runtime distribution drift."""
	assets = tmp_path / "assets"
	assets.mkdir()
	_write_asset_corpus(assets, count=1)
	reflector = tmp_path / "reflector.json"
	_write_reflector_config(reflector)
	work = tmp_path / "reflection_work"
	plan_existing_reflections(assets, work, reflector)
	_write_results(work)
	job = next(iter_jsonl(work / "reflection_jobs.jsonl"))
	result_path = work / job["result_path"]
	result = json.loads(result_path.read_text(encoding="utf-8"))
	result["max_image_side"] = int(result["max_image_side"]) + 32
	result_path.write_text(json.dumps(result), encoding="utf-8")
	config = tmp_path / "config.json"
	_write_action_config(config)

	with pytest.raises(DataValidationError, match="input max side mismatch"):
		collect_existing_states(work, tmp_path / "states.jsonl", config, project_root=tmp_path)


def test_external_image_tamper_and_flux2_only_config_are_rejected(tmp_path: Path) -> None:
	"""Verify external image tamper and flux2 only config are rejected."""
	assets = tmp_path / "assets"
	assets.mkdir()
	rows = _write_asset_corpus(assets, count=1)
	reflector = tmp_path / "reflector.json"
	_write_reflector_config(reflector)
	work = tmp_path / "work"
	plan_existing_reflections(assets, work, reflector)
	Image.new("RGB", (16, 12), (1, 2, 3)).save(assets / rows[0]["initial_image"]["path"])
	contract = json.loads((work / "reflector.lock.json").read_text(encoding="utf-8"))
	jobs = list(iter_jsonl(work / "reflection_jobs.jsonl"))
	with pytest.raises(DataValidationError, match="changed|hash mismatch"):
		_validate_plan(work, jobs, contract)

	# Re-plan in a fresh corpus so collection reaches the action-config guard.
	assets2 = tmp_path / "assets2"
	assets2.mkdir()
	_write_asset_corpus(assets2, count=1)
	work2 = tmp_path / "work2"
	plan_existing_reflections(assets2, work2, reflector)
	_write_results(work2)
	strict_config = tmp_path / "strict.json"
	_write_action_config(strict_config, require_flux2=True)
	with pytest.raises(DataValidationError, match="require_flux2_initial_state=false"):
		collect_existing_states(work2, tmp_path / "states.jsonl", strict_config)


def test_reflection_json_retry_is_strict_and_happens_exactly_once() -> None:
	"""Verify reflection json retry is strict and happens exactly once."""
	valid = canonical_json(_diagnosis(satisfied=False))

	class FakePlanner:
		"""Return one invalid response followed by a deterministic valid diagnosis."""
		def __init__(self) -> None:
			self.calls = 0

		def _generate(self, _messages, _images, _max_new_tokens):
			self.calls += 1
			return '{"fully_satisfied": false, "route": "L1"}' if self.calls == 1 else valid

	planner = FakePlanner()
	result = generate_with_one_json_retry(
		planner,
		[[{"role": "user", "content": [{"type": "text", "text": "diagnose"}]}]],
		[[object()]],
		128,
		_normalize_diagnosis_response,
		DIAGNOSIS_RETRY_INSTRUCTION,
	)
	assert planner.calls == 2
	assert result[0]["valid_json"] is True
	assert result[0]["json_retry_count"] == 1
	assert result[0]["diagnosis"] == _diagnosis(satisfied=False)


def test_four_shards_are_disjoint_complete_and_keep_families_together(tmp_path: Path) -> None:
	"""Verify four shards are disjoint complete and keep families together."""
	assets = tmp_path / "assets"
	assets.mkdir()
	rows = _write_asset_corpus(assets, count=20, family_size=3)
	reflector = tmp_path / "reflector.json"
	_write_reflector_config(reflector)
	owners: dict[str, int] = {}
	union: set[str] = set()
	job_ids: set[str] = set()
	source_signatures: set[str] = set()
	full_assignment_digests: set[str] = set()
	for shard_id in range(4):
		work = tmp_path / f"work_{shard_id}"
		manifest = plan_existing_reflections(
			assets,
			work,
			reflector,
			shard_id=shard_id,
			num_shards=4,
		)
		contract, jobs, _paths, _source_rows = validate_existing_reflection_work(work)
		assert contract["prompt_version"] == TRIAGE_PROMPT_VERSION_V10
		ids = {job["source_sample_id"] for job in jobs}
		assert union.isdisjoint(ids)
		union.update(ids)
		job_ids.update(job["job_id"] for job in jobs)
		lock = json.loads((work / "existing_assets.lock.json").read_text(encoding="utf-8"))
		assert lock["full_sample_count"] == len(rows)
		assert lock["shard_sample_count"] == len(jobs) == manifest["shard_sample_count"]
		assert lock["shard_id"] == shard_id and lock["num_shards"] == 4
		assert sha256_file(work / "source_samples.snapshot.jsonl") == sha256_file(assets / "samples.jsonl")
		assert not list(work.rglob("*.png"))
		source_signatures.add(lock["source_signature"])
		full_assignment_digests.add(lock["full_assignment_digest"])
		for job in jobs:
			family = job["prompt_family_id"]
			assert owners.setdefault(family, shard_id) == shard_id
	assert union == {row["sample_id"] for row in rows}
	assert len(job_ids) == len(rows)
	assert len(source_signatures) == 1
	assert len(full_assignment_digests) == 1

	# Worker ownership is operational metadata: it must not perturb the request
	# identity or Qwen seed relative to single-worker planning.
	one_work = tmp_path / "work_single"
	plan_existing_reflections(assets, one_work, reflector)
	one_jobs = list(iter_jsonl(one_work / "reflection_jobs.jsonl"))
	assert {job["job_id"] for job in one_jobs} == job_ids


def test_assignment_policy_covers_all_11440_synthetic_rows_without_family_split() -> None:
	"""Verify assignment policy covers all 11440 synthetic rows without family split."""
	rows = [
		{
			"schema_version": "v10_legacy_initial_assets_v1",
			"sample_id": f"sample_{index:05d}",
			"asset_id": f"asset_{index:05d}",
			"lineage_id": f"family_{index // 4:05d}",
		}
		for index in range(11_440)
	]
	assignments = _source_assignments(rows, 4)
	by_shard = [
		{row["source_sample_id"] for row in assignments if row["shard_id"] == shard_id}
		for shard_id in range(4)
	]
	assert sum(len(shard) for shard in by_shard) == 11_440
	assert len(set().union(*by_shard)) == 11_440
	for left in range(4):
		for right in range(left + 1, 4):
			assert by_shard[left].isdisjoint(by_shard[right])
	family_owners: dict[str, set[int]] = {}
	for assignment in assignments:
		family = assignment["shard_key"]
		family_owners.setdefault(family, set()).add(int(assignment["shard_id"]))
	assert all(len(owners) == 1 for owners in family_owners.values())


def test_existing_reflection_cli_exposes_explicit_shard_ownership() -> None:
	"""Verify existing reflection cli exposes explicit shard ownership."""
	args = build_parser().parse_args(
		[
			"plan-existing-reflections",
			"--assets-root", "assets",
			"--work-root", "work/shard_03",
			"--reflector-config", "reflector.json",
			"--shard-id", "3",
			"--num-shards", "4",
		]
	)
	assert args.shard_id == 3
	assert args.num_shards == 4
