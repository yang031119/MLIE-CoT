"""Pure-CPU front half of the clean Flux.2 router-data rebuild.

The stages are intentionally materialized rather than hidden in one GPU loop:
prompt rows -> frozen initial Flux.2 jobs -> verified images -> frozen reflection
jobs -> verified route-free diagnoses -> planner input states.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from inference.main_method.modules.contracts import (
	DIAGNOSIS_SCHEMA,
	L1_PLAN_PROMPT_VERSION_V10,
	L1_PLAN_SYSTEM_PROMPT_V10,
	L1_PLAN_USER_TEMPLATE_V10,
	QWEN_ANALYZER_L1_DTYPE_V10,
	QWEN_ANALYZER_L1_INPUT_MAX_SIDE_V10,
	QWEN_ANALYZER_L1_MAX_NEW_TOKENS_V10,
	QWEN_ANALYZER_L1_QUANTIZATION_V10,
	TRIAGE_PROMPT_VERSION_V10,
	TRIAGE_SYSTEM_PROMPT_V10,
	TRIAGE_USER_TEMPLATE_V10,
	build_qwen_analyzer_l1_distribution_contract,
	model_visible_diagnosis_text,
	validate_qwen_analyzer_l1_distribution_contract,
)

from .canonical import (
	canonical_json,
	iter_jsonl,
	read_json,
	sha256_file,
	sha256_json,
	sha256_text,
	stable_id,
	stable_seed,
	write_if_identical_or_absent,
)
from .planner import load_config, load_state_rows
from .schema import (
	SCHEMA_VERSION,
	DataValidationError,
	DatasetConfig,
	canonicalize_diagnosis,
	canonicalize_l1_action_plan,
	canonicalize_state,
	image_descriptor,
	immutable_revision,
	normalize_text,
	strict_bool,
)


INITIAL_STAGE = "flux2_initial_generation_v2"
REFLECTION_STAGE = "qwen_route_free_reflection_v2"
REFLECTION_PROMPT_VERSION = TRIAGE_PROMPT_VERSION_V10
L1_PLAN_STAGE = "qwen_post_route_l1_plan_v1"
L1_PLAN_PROMPT_VERSION = L1_PLAN_PROMPT_VERSION_V10
SOURCE_PROMPTS_SNAPSHOT = "source_prompts.snapshot"
INITIAL_PROMPT_MODE_UNCHANGED = "target_prompt_unchanged"
INITIAL_PROMPT_MODE_COUNTERFACTUAL = "designed_prompt_counterfactual"
INITIAL_STATE_ORIGIN_NATURAL_FAILURE = "natural_initial_generation_failure"
INITIAL_STATE_ORIGIN_COUNTERFACTUAL = "designed_prompt_counterfactual"


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise DataValidationError(message)


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
	return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def _write_artifacts(artifacts: Mapping[Path, str], dry_run: bool) -> None:
	if dry_run:
		return
	conflicts = [
		path
		for path, text in artifacts.items()
		if path.exists() and path.read_text(encoding="utf-8") != text
	]
	if conflicts:
		raise DataValidationError(f"refusing partial write; incompatible artifacts already exist: {conflicts}")
	for path, text in artifacts.items():
		write_if_identical_or_absent(path, text)


def _relative(root: Path, value: str) -> Path:
	root = root.resolve()
	path = Path(value)
	resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
	try:
		resolved.relative_to(root)
	except ValueError as exc:
		raise DataValidationError(f"artifact path escapes bootstrap root: {value!r}") from exc
	return resolved


def classify_initial_prompt_pair(target_prompt: Any, generation_prompt: Any) -> tuple[str, bool]:
	"""Classify whether the initial request changed the target text.

	A target-identical initial prompt is intentional: if the generated image
	fails the later canonical target audit, its image is a natural generator
	failure and is a valid repair state.  Prompt equality itself never decides
	whether the image is fully satisfied.
	"""

	target = normalize_text(target_prompt)
	generation = normalize_text(generation_prompt)
	_require(bool(target) and bool(generation), "initial prompt classification requires non-empty text")
	changed = generation != target
	return (
		INITIAL_PROMPT_MODE_COUNTERFACTUAL if changed else INITIAL_PROMPT_MODE_UNCHANGED,
		changed,
	)


def _canonical_prompt_row(raw: Mapping[str, Any], config: DatasetConfig, index: int) -> dict[str, Any]:
	"""Normalize one prompt row while preserving stable identity, family, and source provenance."""
	target_prompt = normalize_text(raw.get("target_prompt") or raw.get("prompt"))
	generation_prompt = normalize_text(
		raw.get("initial_generation_prompt") or raw.get("generation_prompt") or target_prompt
	)
	_require(bool(target_prompt), f"prompt row {index} has no target_prompt/prompt")
	_require(bool(generation_prompt), f"prompt row {index} has no generation prompt")
	initial_prompt_mode, prompt_corruption_applied = classify_initial_prompt_pair(
		target_prompt, generation_prompt
	)
	sample_value = next(
		(raw[key] for key in ("sample_id", "id", "index") if key in raw and raw[key] is not None),
		None,
	)
	sample_id = normalize_text(sample_value)
	_require(bool(sample_id), f"prompt row {index} needs sample_id/id/index")
	dataset = normalize_text(raw.get("dataset") or "router_bootstrap")
	source_prompt_value = next(
		(raw[key] for key in ("source_prompt_id", "index") if key in raw and raw[key] is not None),
		sample_id,
	)
	source_prompt_id = normalize_text(source_prompt_value)
	lineage_id = normalize_text(raw.get("lineage_id") or f"{dataset}:{source_prompt_id}")
	base_prompt_id = normalize_text(raw.get("base_prompt_id") or source_prompt_id)
	prompt_family_id = normalize_text(raw.get("prompt_family_id") or f"{dataset}:{base_prompt_id}")
	variant_id = normalize_text(raw.get("variant_id") or sample_id)
	seed_value = raw.get("initial_seed")
	seed = (
		int(seed_value)
		if isinstance(seed_value, int) and not isinstance(seed_value, bool)
		else stable_seed(config.dataset_seed, "initial_flux2", dataset, source_prompt_id, sample_id)
	)
	_require(seed > 0, f"prompt row {index} initial seed must be positive")
	corruption = raw.get("corruption") if isinstance(raw.get("corruption"), Mapping) else {}
	error_family = normalize_text(
		raw.get("error_family") or raw.get("tag") or corruption.get("type") or "unknown"
	)
	identity = {
		"dataset": dataset,
		"source_prompt_id": source_prompt_id,
		"sample_id": sample_id,
		"variant_id": variant_id,
	}
	return {
		"schema_version": SCHEMA_VERSION,
		"sample_id": sample_id,
		"sample_key": stable_id("prompt", identity, length=32),
		"dataset": dataset,
		"source_prompt_id": source_prompt_id,
		"lineage_id": lineage_id,
		"base_prompt_id": base_prompt_id,
		"prompt_family_id": prompt_family_id,
		"variant_id": variant_id,
		"variant_strategy": normalize_text(raw.get("variant_strategy") or "original"),
		"transform_version": normalize_text(raw.get("transform_version") or "none"),
		"shard_key": normalize_text(raw.get("shard_key")),
		"shard_id": raw.get("shard_id"),
		"num_shards": raw.get("num_shards"),
		"target_prompt": target_prompt,
		"generation_prompt": generation_prompt,
		"initial_prompt_mode": initial_prompt_mode,
		"prompt_corruption_applied": prompt_corruption_applied,
		"initial_seed": seed,
		"error_family": error_family,
		"corruption": dict(corruption),
		"source_metadata": {
			key: raw[key]
			for key in ("index", "tag", "include", "exclude")
			if key in raw
		},
	}


def validate_prompt_row_contract(row: Mapping[str, Any]) -> None:
	"""Validate prompt row contract and reject contract violations."""
	_require(row.get("schema_version") == SCHEMA_VERSION, "prompt row schema mismatch")
	identity = {
		"dataset": row.get("dataset"),
		"source_prompt_id": row.get("source_prompt_id"),
		"sample_id": row.get("sample_id"),
		"variant_id": row.get("variant_id"),
	}
	_require(all(normalize_text(value) for value in identity.values()), "prompt row identity is incomplete")
	_require(
		row.get("sample_key") == stable_id("prompt", identity, length=32),
		f"prompt sample_key mismatch: {row.get('sample_key')}",
	)
	_require(bool(normalize_text(row.get("target_prompt"))), f"prompt target is empty: {row.get('sample_key')}")
	_require(bool(normalize_text(row.get("generation_prompt"))), f"generation prompt is empty: {row.get('sample_key')}")
	expected_mode, expected_changed = classify_initial_prompt_pair(
		row.get("target_prompt"), row.get("generation_prompt")
	)
	_require(row.get("initial_prompt_mode") == expected_mode, f"initial prompt mode mismatch: {row.get('sample_key')}")
	_require(
		row.get("prompt_corruption_applied") is expected_changed,
		f"prompt corruption flag mismatch: {row.get('sample_key')}",
	)
	_require(
		isinstance(row.get("initial_seed"), int)
		and not isinstance(row.get("initial_seed"), bool)
		and int(row["initial_seed"]) > 0,
		f"prompt initial_seed is invalid: {row.get('sample_key')}",
	)


def validate_source_prompts_snapshot(root: Path, manifest: Mapping[str, Any]) -> None:
	"""Validate source prompts snapshot and reject contract violations."""
	_require(
		manifest.get("source_prompts_snapshot") == SOURCE_PROMPTS_SNAPSHOT,
		"bootstrap source prompt snapshot path mismatch",
	)
	path = _relative(root, str(manifest["source_prompts_snapshot"]))
	_require(path.is_file(), "bootstrap source prompt snapshot is missing")
	_require(
		manifest.get("source_prompts_sha256") == sha256_file(path),
		"bootstrap source prompt snapshot hash mismatch",
	)


def validate_initial_prompt_mode_manifest(
	manifest: Mapping[str, Any], prompt_rows: Iterable[Mapping[str, Any]]
) -> None:
	"""Validate initial prompt mode manifest and reject contract violations."""
	rows = list(prompt_rows)
	expected_modes = dict(sorted(Counter(row["initial_prompt_mode"] for row in rows).items()))
	expected_changed = sum(bool(row["prompt_corruption_applied"]) for row in rows)
	_require(manifest.get("initial_prompt_modes") == expected_modes, "bootstrap initial prompt-mode counts mismatch")
	_require(
		manifest.get("prompt_corruption_applied_count") == expected_changed,
		"bootstrap prompt-corruption count mismatch",
	)


def _initial_request_payload(job: Mapping[str, Any]) -> dict[str, Any]:
	return {
		key: value
		for key, value in job.items()
		if key not in {"job_id", "request_hash", "output_image", "result_path", "reflection_result_path"}
	}


def validate_initial_job_contract(job: Mapping[str, Any], config: DatasetConfig) -> None:
	"""Validate initial job contract and reject contract violations."""
	_require(job.get("schema_version") == SCHEMA_VERSION, "initial job schema mismatch")
	_require(job.get("stage") == INITIAL_STAGE, f"invalid initial stage: {job.get('job_id')}")
	_require(job.get("executor_family") == "flux2", f"invalid initial executor: {job.get('job_id')}")
	_require(
		job.get("operation") == "generate_text_only_initial",
		f"invalid initial operation: {job.get('job_id')}",
	)
	_require(job.get("policy_signature") == config.policy.signature, f"initial policy signature mismatch: {job.get('job_id')}")
	_require(
		job.get("flux2_policy_digest") == config.policy.collector_contract_digest,
		f"initial Flux.2 collector-policy digest mismatch: {job.get('job_id')}",
	)
	_require(job.get("policy") == config.policy.as_dict(), f"initial policy mismatch: {job.get('job_id')}")
	expected_mode, expected_changed = classify_initial_prompt_pair(
		job.get("target_prompt"), job.get("generation_prompt")
	)
	_require(job.get("initial_prompt_mode") == expected_mode, f"initial prompt mode mismatch: {job.get('job_id')}")
	_require(
		job.get("prompt_corruption_applied") is expected_changed,
		f"initial prompt corruption flag mismatch: {job.get('job_id')}",
	)
	request_hash = sha256_json(_initial_request_payload(job))
	_require(job.get("request_hash") == request_hash, f"initial request hash mismatch: {job.get('job_id')}")
	_require(
		job.get("job_id") == stable_id("initial", {"request_hash": request_hash}, length=32),
		f"initial job_id mismatch: {job.get('job_id')}",
	)


def plan_initial_jobs(
	config_path: Path,
	prompts_path: Path,
	bootstrap_root: Path,
	*,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Freeze one reproducible Flux.2 initial image request per prompt row."""

	config_path = config_path.resolve()
	prompts_path = prompts_path.resolve()
	root = bootstrap_root.resolve()
	config = load_config(config_path, strict_revision=True)
	raw_rows = load_state_rows(prompts_path)
	rows = [_canonical_prompt_row(raw, config, index) for index, raw in enumerate(raw_rows)]
	_require(bool(rows), "prompt manifest is empty")
	rows.sort(key=lambda row: (row["dataset"], row["source_prompt_id"], row["sample_id"], row["variant_id"]))
	identities = [(row["dataset"], row["source_prompt_id"], row["sample_id"], row["variant_id"]) for row in rows]
	_require(len(identities) == len(set(identities)), "duplicate dataset/source_prompt_id/sample_id/variant_id")
	source_prompts_text = prompts_path.read_text(encoding="utf-8")

	jobs: list[dict[str, Any]] = []
	for row in rows:
		base = {
			"schema_version": SCHEMA_VERSION,
			"stage": INITIAL_STAGE,
			"sample_id": row["sample_id"],
			"sample_key": row["sample_key"],
			"dataset": row["dataset"],
			"source_prompt_id": row["source_prompt_id"],
			"lineage_id": row["lineage_id"],
			"base_prompt_id": row["base_prompt_id"],
			"prompt_family_id": row["prompt_family_id"],
			"variant_id": row["variant_id"],
			"variant_strategy": row["variant_strategy"],
			"transform_version": row["transform_version"],
			"target_prompt": row["target_prompt"],
			"generation_prompt": row["generation_prompt"],
			"initial_prompt_mode": row["initial_prompt_mode"],
			"prompt_corruption_applied": row["prompt_corruption_applied"],
			"error_family": row["error_family"],
			"seed": row["initial_seed"],
			"executor_family": "flux2",
			"operation": "generate_text_only_initial",
			"policy_signature": config.policy.signature,
			"flux2_policy_digest": config.policy.collector_contract_digest,
			"policy": config.policy.as_dict(),
		}
		request_hash = sha256_json(base)
		job_id = stable_id("initial", {"request_hash": request_hash}, length=32)
		artifact_dir = f"initial/{stable_id('sample', identities[len(jobs)], length=16)}"
		jobs.append(
			{
				**base,
				"job_id": job_id,
				"request_hash": request_hash,
				"output_image": f"{artifact_dir}/image.png",
				"result_path": f"{artifact_dir}/result.json",
				"reflection_result_path": f"reflections/{stable_id('sample', identities[len(jobs)], length=16)}/result.json",
			}
		)
	manifest = {
		"schema_version": SCHEMA_VERSION,
		"stage": INITIAL_STAGE,
		"status": "planned",
		"prompt_count": len(rows),
		"job_count": len(jobs),
		"initial_prompt_modes": dict(
			sorted(Counter(row["initial_prompt_mode"] for row in rows).items())
		),
		"prompt_corruption_applied_count": sum(
			bool(row["prompt_corruption_applied"]) for row in rows
		),
		"policy_signature": config.policy.signature,
		"flux2_policy_contract": config.policy.collector_contract,
		"flux2_policy_digest": config.policy.collector_contract_digest,
		"config_sha256": sha256_json(config.as_dict()),
		"source_prompts_input_sha256": sha256_file(prompts_path),
		"source_prompts_sha256": sha256_text(source_prompts_text),
		"source_prompts_snapshot": SOURCE_PROMPTS_SNAPSHOT,
		"frozen_digests": {"prompts": sha256_json(rows), "initial_jobs": sha256_json(jobs)},
	}
	artifacts = {
		root / "bootstrap_config.lock.json": json.dumps(
			config.as_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
		) + "\n",
		root / "prompt_rows.jsonl": _jsonl_text(rows),
		root / SOURCE_PROMPTS_SNAPSHOT: source_prompts_text,
		root / "initial_jobs.jsonl": _jsonl_text(jobs),
		root / "bootstrap_manifest.json": json.dumps(
			manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
		) + "\n",
	}
	_write_artifacts(artifacts, dry_run)
	return manifest


def validate_initial_result(root: Path, job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
	"""Validate initial result and reject contract violations."""
	result_path = _relative(root, str(job["result_path"]))
	output_path = _relative(root, str(job["output_image"]))
	_require(result_path.is_file(), f"missing initial result: {result_path}")
	result = read_json(result_path)
	_require(isinstance(result, Mapping), f"initial result is not an object: {result_path}")
	for key in ("job_id", "request_hash", "policy_signature", "flux2_policy_digest", "seed"):
		_require(result.get(key) == job.get(key), f"initial result {key} mismatch: {job['job_id']}")
	_require(result.get("status") == "ok", f"initial result is not ok: {job['job_id']}")
	_require(result.get("executor_family") == "flux2", f"initial result is not Flux.2: {job['job_id']}")
	_require(result.get("model_id") == job["policy"]["model_id"], f"initial model_id mismatch: {job['job_id']}")
	_require(result.get("model_revision") == job["policy"]["model_revision"], f"initial revision mismatch: {job['job_id']}")
	_require(output_path.is_file(), f"initial output image missing: {output_path}")
	descriptor = image_descriptor(output_path)
	_require(
		descriptor["width"] == int(job["policy"]["resolution"])
		and descriptor["height"] == int(job["policy"]["resolution"]),
		f"initial output resolution mismatch: {job['job_id']}",
	)
	_require(result.get("output_sha256") == descriptor["sha256"], f"initial output hash mismatch: {job['job_id']}")
	_require(result.get("output_image") == job["output_image"], f"initial output path mismatch: {job['job_id']}")
	return dict(result), descriptor


def load_reflector_config(path: Path) -> dict[str, Any]:
	"""Load the reflection configuration and bind it to the frozen Qwen distribution contract."""
	value = read_json(path)
	_require(isinstance(value, Mapping), "reflector config must be an object")
	seed = value.get("seed", 20260715)
	_require(isinstance(seed, int) and not isinstance(seed, bool) and seed > 0, "reflector seed must be a positive integer")
	allow_cpu_offload = strict_bool(value.get("allow_cpu_offload", False), "reflector allow_cpu_offload")
	provided_distribution = value.get("qwen_distribution")
	if provided_distribution is not None:
		_require(isinstance(provided_distribution, Mapping), "qwen_distribution must be an object")
	try:
		distribution = build_qwen_analyzer_l1_distribution_contract(
			model=value.get("model"),
			model_revision=value.get("model_revision"),
			dtype=value.get("dtype", QWEN_ANALYZER_L1_DTYPE_V10),
			quantization=value.get("quantization", QWEN_ANALYZER_L1_QUANTIZATION_V10),
			allow_cpu_offload=allow_cpu_offload,
			input_max_side=value.get("max_image_side", QWEN_ANALYZER_L1_INPUT_MAX_SIDE_V10),
			max_new_tokens=value.get("max_new_tokens", QWEN_ANALYZER_L1_MAX_NEW_TOKENS_V10),
			require_immutable_revision=True,
		)
		if provided_distribution is not None:
			locked_distribution = validate_qwen_analyzer_l1_distribution_contract(
				provided_distribution,
				require_immutable_revision=True,
			)
			_require(locked_distribution == distribution, "flat Qwen settings differ from qwen_distribution lock")
	except ValueError as exc:
		raise DataValidationError(str(exc)) from exc
	for key, expected in (
		("distribution_signature", distribution["distribution_signature"]),
		("diagnosis_schema", DIAGNOSIS_SCHEMA),
		("analyzer_prompt_version", REFLECTION_PROMPT_VERSION),
		("l1_planner_prompt_version", L1_PLAN_PROMPT_VERSION),
	):
		if key in value:
			_require(value.get(key) == expected, f"reflector {key} mismatch")
	contract = {
		"stage": REFLECTION_STAGE,
		"prompt_version": REFLECTION_PROMPT_VERSION,
		"model": distribution["model"],
		"model_revision": immutable_revision(distribution["model_revision"], "reflector model_revision"),
		"dtype": distribution["dtype"],
		"quantization": distribution["quantization"],
		"effective_quantization": distribution["effective_quantization"],
		"max_new_tokens": distribution["max_new_tokens"],
		"max_image_side": distribution["input_max_side"],
		"allow_cpu_offload": distribution["allow_cpu_offload"],
		"seed": seed,
		"decoding": distribution["decoding"],
		"diagnosis_schema": DIAGNOSIS_SCHEMA,
		"analyzer_prompt_version": REFLECTION_PROMPT_VERSION,
		"l1_planner_prompt_version": L1_PLAN_PROMPT_VERSION,
		"distribution_signature": distribution["distribution_signature"],
		"qwen_distribution": distribution,
	}
	_require(bool(contract["model"]), "reflector model is required")
	_require(contract["dtype"] in {"bfloat16", "float16", "float32"}, "invalid reflector dtype")
	_require(contract["quantization"] in {"none", "8bit", "4bit", "nf4"}, "invalid reflector quantization")
	_require(contract["max_new_tokens"] > 0 and contract["max_image_side"] > 0, "reflector limits must be positive")
	contract["reflector_signature"] = stable_id("reflector", contract, length=32)
	return contract


def reflection_user_prompt(job: Mapping[str, Any]) -> str:
	"""Render the model-facing reflection prompt from an immutable action job."""
	return TRIAGE_USER_TEMPLATE_V10.format(
		prompt=job["target_prompt"],
		width=job["source_image"]["width"],
		height=job["source_image"]["height"],
	)


def reflection_prompt(job: Mapping[str, Any]) -> str:
	"""Stable hash material for the exact online system/user text pair."""

	return canonical_json(
		[
			{"role": "system", "text": TRIAGE_SYSTEM_PROMPT_V10},
			{"role": "user", "text": reflection_user_prompt(job)},
		]
	)


def l1_plan_user_prompt(job: Mapping[str, Any]) -> str:
	"""Render the local-repair planning prompt from the canonical diagnosis."""
	return L1_PLAN_USER_TEMPLATE_V10.format(
		prompt=job["target_prompt"],
		diagnosis_json=model_visible_diagnosis_text(job["diagnosis"], indent=2),
		width=job["source_image"]["width"],
		height=job["source_image"]["height"],
	)


def l1_plan_prompt(job: Mapping[str, Any]) -> str:
	"""Stable hash material for the exact online system/user text pair."""

	return canonical_json(
		[
			{"role": "system", "text": L1_PLAN_SYSTEM_PROMPT_V10},
			{"role": "user", "text": l1_plan_user_prompt(job)},
		]
	)


def _reflection_request_payload(job: Mapping[str, Any]) -> dict[str, Any]:
	payload = {
		key: value
		for key, value in job.items()
		if key not in {"job_id", "request_hash", "result_path"}
	}
	source = dict(payload["source_image"])
	source.pop("path", None)
	payload["source_image"] = source
	return payload


def validate_reflection_job_contract(job: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
	"""Validate reflection job contract and reject contract violations."""
	_require(job.get("schema_version") == SCHEMA_VERSION, "reflection job schema mismatch")
	_require(job.get("stage") == REFLECTION_STAGE, f"invalid reflection stage: {job.get('job_id')}")
	_require(
		job.get("reflector_signature") == contract.get("reflector_signature"),
		f"reflection signature mismatch: {job.get('job_id')}",
	)
	_require(job.get("prompt_version") == REFLECTION_PROMPT_VERSION, f"reflection prompt version mismatch: {job.get('job_id')}")
	_require(job.get("diagnosis_schema") == DIAGNOSIS_SCHEMA, f"reflection diagnosis schema mismatch: {job.get('job_id')}")
	_require(
		job.get("distribution_signature") == contract.get("distribution_signature"),
		f"reflection Qwen distribution mismatch: {job.get('job_id')}",
	)
	_require(
		job.get("prompt_sha256") == sha256_text(reflection_prompt(job)),
		f"reflection prompt hash mismatch: {job.get('job_id')}",
	)
	_require(
		job.get("seed") == stable_seed(contract["seed"], "reflection", job.get("initial_job_id")),
		f"reflection seed derivation mismatch: {job.get('job_id')}",
	)
	request_hash = sha256_json(_reflection_request_payload(job))
	_require(job.get("request_hash") == request_hash, f"reflection request hash mismatch: {job.get('job_id')}")
	_require(
		job.get("job_id") == stable_id("reflection", {"request_hash": request_hash}, length=32),
		f"reflection job_id mismatch: {job.get('job_id')}",
	)


def plan_reflection_jobs(
	bootstrap_root: Path,
	reflector_config_path: Path,
	*,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Bind reflection requests to verified initial images and one Qwen contract."""

	root = bootstrap_root.resolve()
	contract = load_reflector_config(reflector_config_path.resolve())
	prompt_rows = list(iter_jsonl(root / "prompt_rows.jsonl"))
	initial_jobs = list(iter_jsonl(root / "initial_jobs.jsonl"))
	bootstrap_manifest = read_json(root / "bootstrap_manifest.json")
	locked_config = load_config(root / "bootstrap_config.lock.json", strict_revision=True)
	_require(bootstrap_manifest.get("stage") == INITIAL_STAGE, "invalid bootstrap manifest stage")
	_require(bootstrap_manifest.get("status") == "planned", "invalid bootstrap manifest status")
	_require(
		bootstrap_manifest.get("config_sha256") == sha256_json(locked_config.as_dict()),
		"bootstrap config differs from manifest",
	)
	_require(
		bootstrap_manifest.get("policy_signature") == locked_config.policy.signature,
		"bootstrap policy signature mismatch",
	)
	validate_source_prompts_snapshot(root, bootstrap_manifest)
	validate_initial_prompt_mode_manifest(bootstrap_manifest, prompt_rows)
	_require(
		bootstrap_manifest.get("frozen_digests", {}).get("prompts") == sha256_json(prompt_rows),
		"prompt rows differ from bootstrap manifest",
	)
	_require(
		bootstrap_manifest.get("frozen_digests", {}).get("initial_jobs") == sha256_json(initial_jobs),
		"initial jobs differ from bootstrap manifest",
	)
	_require(bootstrap_manifest.get("prompt_count") == len(prompt_rows), "bootstrap prompt_count mismatch")
	_require(bootstrap_manifest.get("job_count") == len(initial_jobs), "bootstrap job_count mismatch")
	for row in prompt_rows:
		validate_prompt_row_contract(row)
	_require(
		len({job.get("job_id") for job in initial_jobs}) == len(initial_jobs),
		"duplicate initial job_id",
	)
	_require(
		len({job.get("sample_key") for job in initial_jobs}) == len(initial_jobs),
		"duplicate initial sample_key",
	)
	for initial_job in initial_jobs:
		validate_initial_job_contract(initial_job, locked_config)
		_relative(root, str(initial_job["output_image"]))
		_relative(root, str(initial_job["result_path"]))
		_relative(root, str(initial_job["reflection_result_path"]))
	prompts = {row["sample_key"]: row for row in prompt_rows}
	_require(len(prompts) == len(prompt_rows), "duplicate prompt sample_key")
	_require(
		set(prompts) == {job.get("sample_key") for job in initial_jobs},
		"prompt rows and initial jobs do not contain the same sample_key set",
	)
	jobs: list[dict[str, Any]] = []
	for initial_job in initial_jobs:
		_require(
			initial_job.get("sample_key") in prompts,
			f"unknown initial sample_key: {initial_job.get('sample_key')}",
		)
		_initial_result, descriptor = validate_initial_result(root, initial_job)
		row = prompts[initial_job["sample_key"]]
		for key in (
			"sample_id",
			"sample_key",
			"dataset",
			"source_prompt_id",
			"lineage_id",
			"base_prompt_id",
			"prompt_family_id",
			"variant_id",
			"target_prompt",
			"error_family",
			"initial_prompt_mode",
			"prompt_corruption_applied",
		):
			_require(initial_job.get(key) == row.get(key), f"initial/prompt {key} mismatch: {row['sample_key']}")
		_require(
			initial_job.get("generation_prompt") == row.get("generation_prompt"),
			f"initial/prompt generation_prompt mismatch: {row['sample_key']}",
		)
		_require(
			initial_job.get("seed") == row.get("initial_seed"),
			f"initial/prompt seed mismatch: {row['sample_key']}",
		)
		_require(
			initial_job.get("policy") == locked_config.policy.as_dict(),
			f"initial policy mismatch: {row['sample_key']}",
		)
		base = {
			"schema_version": SCHEMA_VERSION,
			"stage": REFLECTION_STAGE,
			"sample_id": row["sample_id"],
			"sample_key": row["sample_key"],
			"dataset": row["dataset"],
			"source_prompt_id": row["source_prompt_id"],
			"lineage_id": row["lineage_id"],
			"base_prompt_id": row["base_prompt_id"],
			"prompt_family_id": row["prompt_family_id"],
			"variant_id": row["variant_id"],
			"target_prompt": row["target_prompt"],
			"error_family": row["error_family"],
			"initial_prompt_mode": row["initial_prompt_mode"],
			"prompt_corruption_applied": row["prompt_corruption_applied"],
			"initial_job_id": initial_job["job_id"],
			"initial_request_hash": initial_job["request_hash"],
			"source_image": descriptor,
			"reflector_signature": contract["reflector_signature"],
			"prompt_version": REFLECTION_PROMPT_VERSION,
			"diagnosis_schema": DIAGNOSIS_SCHEMA,
			"distribution_signature": contract["distribution_signature"],
			"seed": stable_seed(contract["seed"], "reflection", initial_job["job_id"]),
		}
		base["prompt_sha256"] = sha256_text(reflection_prompt(base))
		request_hash = sha256_json(_reflection_request_payload(base))
		job_id = stable_id("reflection", {"request_hash": request_hash}, length=32)
		jobs.append(
			{
				**base,
				"job_id": job_id,
				"request_hash": request_hash,
				"result_path": initial_job["reflection_result_path"],
			}
		)
	jobs.sort(key=lambda row: row["job_id"])
	manifest = {
		"schema_version": SCHEMA_VERSION,
		"stage": REFLECTION_STAGE,
		"status": "planned",
		"job_count": len(jobs),
		"reflector_signature": contract["reflector_signature"],
		"distribution_signature": contract["distribution_signature"],
		"diagnosis_schema": DIAGNOSIS_SCHEMA,
		"frozen_digest": sha256_json(jobs),
	}
	artifacts = {
		root / "reflector.lock.json": json.dumps(
			contract, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
		) + "\n",
		root / "reflection_jobs.jsonl": _jsonl_text(jobs),
		root / "reflection_manifest.json": json.dumps(
			manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
		) + "\n",
	}
	_write_artifacts(artifacts, dry_run)
	return manifest


def validate_reflection_result(
	root: Path,
	job: Mapping[str, Any],
	contract: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
	"""Validate reflection output against the source job while excluding action and trajectory leakage."""
	path = _relative(root, str(job["result_path"]))
	_require(path.is_file(), f"missing reflection result: {path}")
	result = read_json(path)
	_require(isinstance(result, Mapping), f"reflection result is not an object: {path}")
	contract = dict(contract or load_reflector_config(root / "reflector.lock.json"))
	_require(result.get("schema_version") == SCHEMA_VERSION, f"reflection result schema mismatch: {job['job_id']}")
	for key in (
		"job_id",
		"request_hash",
		"reflector_signature",
		"distribution_signature",
		"diagnosis_schema",
		"source_image_sha256",
		"prompt_sha256",
	):
		expected = job["source_image"]["sha256"] if key == "source_image_sha256" else job[key]
		_require(result.get(key) == expected, f"reflection result {key} mismatch: {job['job_id']}")
	_require(result.get("status") == "ok", f"reflection result is not ok: {job['job_id']}")
	_require(result.get("stage") == REFLECTION_STAGE, f"reflection result stage mismatch: {job['job_id']}")
	_require(result.get("executor_family") == "qwen", f"reflection executor mismatch: {job['job_id']}")
	_require(result.get("prompt_version") == job.get("prompt_version"), f"reflection prompt version mismatch: {job['job_id']}")
	_require(result.get("model") == contract.get("model"), f"reflection model mismatch: {job['job_id']}")
	_require(
		result.get("model_revision") == contract.get("model_revision"),
		f"reflection revision mismatch: {job['job_id']}",
	)
	_require(result.get("seed") == job.get("seed"), f"reflection seed mismatch: {job['job_id']}")
	_require(result.get("decoding") == contract.get("decoding"), f"reflection decoding mismatch: {job['job_id']}")
	_require(result.get("dtype") == contract.get("dtype"), f"reflection dtype mismatch: {job['job_id']}")
	_require(result.get("quantization") == contract.get("quantization"), f"reflection quantization mismatch: {job['job_id']}")
	_require(result.get("allow_cpu_offload") == contract.get("allow_cpu_offload"), f"reflection CPU-offload mismatch: {job['job_id']}")
	_require(result.get("effective_quantization") == contract.get("effective_quantization"), f"reflection effective quantization mismatch: {job['job_id']}")
	_require(result.get("max_image_side") == contract.get("max_image_side"), f"reflection input max side mismatch: {job['job_id']}")
	_require(result.get("max_new_tokens") == contract.get("max_new_tokens"), f"reflection token budget mismatch: {job['job_id']}")
	_require(result.get("json_retry_count") in {0, 1}, f"reflection JSON retry count mismatch: {job['job_id']}")
	diagnosis = canonicalize_diagnosis(result.get("diagnosis"))
	_require(result.get("diagnosis_sha256") == sha256_json(diagnosis), f"reflection diagnosis hash mismatch: {job['job_id']}")
	return dict(result), diagnosis


def build_input_states(
	bootstrap_root: Path,
	output_path: Path,
	*,
	project_root: Path | None = None,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Create only verified problematic states for the 3 x K action planner."""

	root = bootstrap_root.resolve()
	output_path = output_path.resolve()
	project_root = (project_root or Path.cwd()).resolve()
	config = DatasetConfig.from_mapping(read_json(root / "bootstrap_config.lock.json"), strict_revision=True)
	prompt_rows = list(iter_jsonl(root / "prompt_rows.jsonl"))
	initial_job_rows = list(iter_jsonl(root / "initial_jobs.jsonl"))
	reflection_jobs = list(iter_jsonl(root / "reflection_jobs.jsonl"))
	bootstrap_manifest = read_json(root / "bootstrap_manifest.json")
	reflection_manifest = read_json(root / "reflection_manifest.json")
	reflector_contract = load_reflector_config(root / "reflector.lock.json")
	_require(bootstrap_manifest.get("stage") == INITIAL_STAGE, "invalid bootstrap manifest stage")
	_require(reflection_manifest.get("stage") == REFLECTION_STAGE, "invalid reflection manifest stage")
	_require(bootstrap_manifest.get("status") == "planned", "invalid bootstrap manifest status")
	_require(reflection_manifest.get("status") == "planned", "invalid reflection manifest status")
	_require(
		bootstrap_manifest.get("config_sha256") == sha256_json(config.as_dict()),
		"bootstrap config differs from manifest",
	)
	_require(
		bootstrap_manifest.get("policy_signature") == config.policy.signature,
		"bootstrap policy signature mismatch",
	)
	validate_source_prompts_snapshot(root, bootstrap_manifest)
	validate_initial_prompt_mode_manifest(bootstrap_manifest, prompt_rows)
	_require(
		reflection_manifest.get("reflector_signature") == reflector_contract["reflector_signature"],
		"reflection manifest/lock signature mismatch",
	)
	_require(
		reflection_manifest.get("distribution_signature") == reflector_contract["distribution_signature"],
		"reflection manifest/Qwen distribution mismatch",
	)
	_require(
		reflection_manifest.get("diagnosis_schema") == DIAGNOSIS_SCHEMA,
		"reflection manifest diagnosis schema mismatch",
	)
	_require(
		bootstrap_manifest.get("frozen_digests", {}).get("prompts") == sha256_json(prompt_rows),
		"prompt rows differ from bootstrap manifest",
	)
	_require(
		bootstrap_manifest.get("frozen_digests", {}).get("initial_jobs") == sha256_json(initial_job_rows),
		"initial jobs differ from bootstrap manifest",
	)
	_require(
		reflection_manifest.get("frozen_digest") == sha256_json(reflection_jobs),
		"reflection jobs differ from reflection manifest",
	)
	_require(bootstrap_manifest.get("prompt_count") == len(prompt_rows), "bootstrap prompt_count mismatch")
	_require(bootstrap_manifest.get("job_count") == len(initial_job_rows), "bootstrap job_count mismatch")
	_require(reflection_manifest.get("job_count") == len(reflection_jobs), "reflection job_count mismatch")
	for row in prompt_rows:
		validate_prompt_row_contract(row)
	_require(
		len({job.get("job_id") for job in initial_job_rows}) == len(initial_job_rows),
		"duplicate initial job_id",
	)
	_require(
		len({job.get("job_id") for job in reflection_jobs}) == len(reflection_jobs),
		"duplicate reflection job_id",
	)
	_require(
		len({job.get("sample_key") for job in initial_job_rows}) == len(initial_job_rows),
		"duplicate initial sample_key",
	)
	_require(
		len({job.get("sample_key") for job in reflection_jobs}) == len(reflection_jobs),
		"duplicate reflection sample_key",
	)
	for initial_job in initial_job_rows:
		validate_initial_job_contract(initial_job, config)
	prompts = {row["sample_key"]: row for row in prompt_rows}
	initial_jobs = {row["sample_key"]: row for row in initial_job_rows}
	_require(
		len(prompts) == len(prompt_rows) and len(initial_jobs) == len(initial_job_rows),
		"duplicate bootstrap sample_key",
	)
	_require(
		set(prompts) == set(initial_jobs) == {job.get("sample_key") for job in reflection_jobs},
		"prompt, initial, and reflection stages do not contain the same sample_key set",
	)
	states: list[dict[str, Any]] = []
	excluded: list[dict[str, Any]] = []
	for reflection_job in reflection_jobs:
		sample_id = reflection_job["sample_id"]
		sample_key = reflection_job["sample_key"]
		_require(sample_key in prompts and sample_key in initial_jobs, f"unknown reflection sample_key: {sample_key}")
		validate_reflection_job_contract(reflection_job, reflector_contract)
		row = prompts[sample_key]
		initial_job = initial_jobs[sample_key]
		for key in (
			"sample_id",
			"sample_key",
			"dataset",
			"source_prompt_id",
			"lineage_id",
			"base_prompt_id",
			"prompt_family_id",
			"variant_id",
			"target_prompt",
			"error_family",
			"initial_prompt_mode",
			"prompt_corruption_applied",
		):
			_require(reflection_job.get(key) == row.get(key), f"reflection/prompt {key} mismatch: {sample_key}")
		for key in (
			"sample_id",
			"sample_key",
			"dataset",
			"source_prompt_id",
			"lineage_id",
			"base_prompt_id",
			"prompt_family_id",
			"variant_id",
			"target_prompt",
			"error_family",
			"initial_prompt_mode",
			"prompt_corruption_applied",
		):
			_require(initial_job.get(key) == row.get(key), f"initial/prompt {key} mismatch: {sample_key}")
		_require(
			initial_job.get("generation_prompt") == row.get("generation_prompt"),
			f"initial/prompt generation_prompt mismatch: {sample_key}",
		)
		_require(initial_job.get("seed") == row.get("initial_seed"), f"initial/prompt seed mismatch: {sample_key}")
		initial_result, descriptor = validate_initial_result(root, initial_job)
		_require(
			reflection_job["initial_job_id"] == initial_job["job_id"],
			f"reflection/initial job mismatch: {sample_key}",
		)
		_require(
			reflection_job["initial_request_hash"] == initial_job["request_hash"],
			f"reflection/initial request mismatch: {sample_key}",
		)
		_require(
			reflection_job["source_image"] == descriptor,
			f"reflection source descriptor mismatch: {sample_key}",
		)
		_reflection_result, diagnosis = validate_reflection_result(root, reflection_job, reflector_contract)
		prompt_corruption_applied = row["prompt_corruption_applied"]
		initial_state_origin = (
			INITIAL_STATE_ORIGIN_COUNTERFACTUAL
			if prompt_corruption_applied
			else INITIAL_STATE_ORIGIN_NATURAL_FAILURE
		)
		target_audit = {
			"protocol": "qwen_canonical_diagnosis_against_target_v1",
			"status": "passed" if diagnosis["fully_satisfied"] else "failed",
			"fully_satisfied": diagnosis["fully_satisfied"],
			"diagnosis_sha256": sha256_json(diagnosis),
		}
		if diagnosis["fully_satisfied"]:
			excluded.append(
				{
					"sample_id": sample_id,
					"sample_key": sample_key,
					"dataset": row["dataset"],
					"source_prompt_id": row["source_prompt_id"],
					"reason": "fully_satisfied_l0",
					"initial_prompt_mode": row["initial_prompt_mode"],
					"prompt_corruption_applied": prompt_corruption_applied,
					"initial_generation_prompt": row["generation_prompt"],
					"initial_target_audit": target_audit,
				}
			)
			continue
		state = {
			"state_id": stable_id(
				"state",
				{
					"reflection_job_id": reflection_job["job_id"],
					"source_sha256": descriptor["sha256"],
					"diagnosis_sha256": sha256_json(diagnosis),
				},
				length=32,
			),
			"sample_id": sample_id,
			"lineage_id": row["lineage_id"],
			"base_prompt_id": row["base_prompt_id"],
			"prompt_family_id": row["prompt_family_id"],
			"variant_id": row["variant_id"],
			"source_prompt_id": row["source_prompt_id"],
			"dataset": row["dataset"],
			"error_family": row["error_family"],
			"target_prompt": row["target_prompt"],
			"current_image": descriptor["path"],
			"initial_generator": {
				"family": "flux2",
				"model_id": initial_job["policy"]["model_id"],
				"model_revision": initial_job["policy"]["model_revision"],
				"seed": initial_job["seed"],
			},
			"initial_record": initial_result,
			"diagnosis": diagnosis,
			"provenance": {
				"initial_state_origin": initial_state_origin,
				"initial_prompt_mode": row["initial_prompt_mode"],
				"prompt_corruption_applied": prompt_corruption_applied,
				"initial_generation_prompt": row["generation_prompt"],
				"initial_target_audit": target_audit,
				"bootstrap_root": str(root),
				"initial_job_id": initial_job["job_id"],
				"reflection_job_id": reflection_job["job_id"],
				"qwen_distribution_signature": reflector_contract["distribution_signature"],
				"diagnosis_schema": DIAGNOSIS_SCHEMA,
				"corruption": row["corruption"],
			},
		}
		# Run the final planner's exact validation now, not after expensive
		# counterfactual execution.
		canonicalize_state(state, config, project_root)
		states.append(state)
	states.sort(key=lambda row: row["state_id"])
	excluded.sort(key=lambda row: row["sample_key"])
	summary = {
		"schema_version": SCHEMA_VERSION,
		"status": "states_built",
		"reflection_job_count": len(reflection_jobs),
		"eligible_state_count": len(states),
		"excluded_count": len(excluded),
		"eligible_initial_state_origins": dict(
			sorted(Counter(state["provenance"]["initial_state_origin"] for state in states).items())
		),
		"excluded_initial_prompt_modes": dict(
			sorted(Counter(row["initial_prompt_mode"] for row in excluded).items())
		),
		"qwen_distribution_signature": reflector_contract["distribution_signature"],
		"diagnosis_schema": DIAGNOSIS_SCHEMA,
		"state_digest": sha256_json(states),
		"excluded_digest": sha256_json(excluded),
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


def load_l1_planner_config(path: Path) -> dict[str, Any]:
	"""Reuse the pinned Qwen runtime while giving L1 planning its own signature."""

	base = load_reflector_config(path)
	contract = {
		**{key: value for key, value in base.items() if key not in {"stage", "prompt_version", "reflector_signature"}},
		"stage": L1_PLAN_STAGE,
		"prompt_version": L1_PLAN_PROMPT_VERSION,
	}
	contract["planner_signature"] = stable_id("l1_planner", contract, length=32)
	return contract


def _l1_plan_request_payload(job: Mapping[str, Any]) -> dict[str, Any]:
	payload = {key: value for key, value in job.items() if key not in {"job_id", "request_hash", "result_path"}}
	source = dict(payload["source_image"])
	source.pop("path", None)
	payload["source_image"] = source
	return payload


def validate_l1_plan_job_contract(job: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
	"""Validate l1 plan job contract and reject contract violations."""
	_require(job.get("schema_version") == SCHEMA_VERSION, "L1 planner job schema mismatch")
	_require(job.get("stage") == L1_PLAN_STAGE, f"invalid L1 planner stage: {job.get('job_id')}")
	_require(job.get("planner_signature") == contract.get("planner_signature"), "L1 planner signature mismatch")
	_require(job.get("prompt_version") == L1_PLAN_PROMPT_VERSION, "L1 planner prompt version mismatch")
	_require(job.get("diagnosis_schema") == DIAGNOSIS_SCHEMA, "L1 planner diagnosis schema mismatch")
	_require(
		job.get("distribution_signature") == contract.get("distribution_signature"),
		"L1 planner Qwen distribution mismatch",
	)
	_require(job.get("prompt_sha256") == sha256_text(l1_plan_prompt(job)), "L1 planner prompt hash mismatch")
	request_hash = sha256_json(_l1_plan_request_payload(job))
	_require(job.get("request_hash") == request_hash, "L1 planner request hash mismatch")
	_require(job.get("job_id") == stable_id("l1_plan", {"request_hash": request_hash}, length=32), "L1 planner job_id mismatch")


def plan_l1_action_jobs(
	states_path: Path,
	planner_config_path: Path,
	plan_root: Path,
	*,
	project_root: Path | None = None,
	dry_run: bool = False,
) -> dict[str, Any]:
	"""Freeze post-route L1 localization jobs without mutating diagnosis rows."""

	states_path = states_path.resolve()
	root = plan_root.resolve()
	project_root = (project_root or Path.cwd()).resolve()
	contract = load_l1_planner_config(planner_config_path.resolve())
	rows = load_state_rows(states_path)
	_require(bool(rows), "L1 planner state input is empty")
	jobs: list[dict[str, Any]] = []
	seen: set[str] = set()
	for index, row in enumerate(rows):
		state_id = normalize_text(row.get("state_id") or row.get("sample_id"))
		_require(bool(state_id) and state_id not in seen, f"invalid or duplicate state_id at row {index}")
		seen.add(state_id)
		target_prompt = normalize_text(row.get("target_prompt") or row.get("prompt"))
		diagnosis = canonicalize_diagnosis(row.get("diagnosis") or row.get("analysis") or row)
		_require(not diagnosis["fully_satisfied"] and bool(diagnosis["issues"]), f"L1 planning requires problematic state: {state_id}")
		image_value = row.get("current_image") or row.get("source_image")
		if isinstance(image_value, Mapping):
			image_value = image_value.get("path")
		image_path = Path(str(image_value))
		if not image_path.is_absolute():
			image_path = project_root / image_path
		descriptor = image_descriptor(image_path)
		base = {
			"schema_version": SCHEMA_VERSION,
			"stage": L1_PLAN_STAGE,
			"state_id": state_id,
			"target_prompt": target_prompt,
			"diagnosis": diagnosis,
			"diagnosis_sha256": sha256_json(diagnosis),
			"source_image": descriptor,
			"planner_signature": contract["planner_signature"],
			"prompt_version": L1_PLAN_PROMPT_VERSION,
			"diagnosis_schema": DIAGNOSIS_SCHEMA,
			"distribution_signature": contract["distribution_signature"],
			"seed": stable_seed(contract["seed"], "l1_plan", state_id),
		}
		base["prompt_sha256"] = sha256_text(l1_plan_prompt(base))
		request_hash = sha256_json(_l1_plan_request_payload(base))
		job_id = stable_id("l1_plan", {"request_hash": request_hash}, length=32)
		jobs.append(
			{
				**base,
				"job_id": job_id,
				"request_hash": request_hash,
				"result_path": f"results/{stable_id('state', state_id, length=20)}.json",
			}
		)
	jobs.sort(key=lambda row: row["state_id"])
	for job in jobs:
		validate_l1_plan_job_contract(job, contract)
	manifest = {
		"schema_version": SCHEMA_VERSION,
		"stage": L1_PLAN_STAGE,
		"status": "planned",
		"job_count": len(jobs),
		"planner_signature": contract["planner_signature"],
		"distribution_signature": contract["distribution_signature"],
		"diagnosis_schema": DIAGNOSIS_SCHEMA,
		"jobs_digest": sha256_json(jobs),
		"source_states_sha256": sha256_file(states_path),
	}
	_write_artifacts(
		{
			root / "planner.lock.json": json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
			root / "jobs.jsonl": _jsonl_text(jobs),
			root / "manifest.json": json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
		},
		dry_run,
	)
	return manifest


def collect_l1_action_plans(plan_root: Path, output_path: Path, *, dry_run: bool = False) -> dict[str, Any]:
	"""Validate per-job Qwen results and emit a deterministic state-keyed plan table."""

	root = plan_root.resolve()
	contract = load_l1_planner_config(root / "planner.lock.json")
	jobs = list(iter_jsonl(root / "jobs.jsonl"))
	manifest = read_json(root / "manifest.json")
	_require(manifest.get("schema_version") == SCHEMA_VERSION, "L1 planner manifest schema mismatch")
	_require(manifest.get("stage") == L1_PLAN_STAGE, "L1 planner manifest stage mismatch")
	_require(manifest.get("status") == "planned", "L1 planner manifest status mismatch")
	_require(manifest.get("job_count") == len(jobs), "L1 planner manifest job_count mismatch")
	_require(manifest.get("planner_signature") == contract.get("planner_signature"), "L1 planner manifest signature mismatch")
	_require(manifest.get("distribution_signature") == contract.get("distribution_signature"), "L1 planner manifest/Qwen distribution mismatch")
	_require(manifest.get("diagnosis_schema") == DIAGNOSIS_SCHEMA, "L1 planner manifest diagnosis schema mismatch")
	_require(manifest.get("jobs_digest") == sha256_json(jobs), "L1 planner jobs differ from manifest")
	plans: list[dict[str, Any]] = []
	for job in jobs:
		validate_l1_plan_job_contract(job, contract)
		result_path = _relative(root, str(job["result_path"]))
		_require(result_path.is_file(), f"missing L1 plan result: {result_path}")
		result = read_json(result_path)
		_require(isinstance(result, Mapping), f"L1 plan result is not an object: {job['state_id']}")
		_require(result.get("schema_version") == SCHEMA_VERSION, f"L1 plan result schema mismatch: {job['state_id']}")
		_require(result.get("stage") == L1_PLAN_STAGE, f"L1 plan result stage mismatch: {job['state_id']}")
		_require(result.get("executor_family") == "qwen", f"L1 plan executor mismatch: {job['state_id']}")
		_require(result.get("json_retry_count") in {0, 1}, f"L1 plan JSON retry count mismatch: {job['state_id']}")
		for key in (
			"job_id",
			"request_hash",
			"planner_signature",
			"distribution_signature",
			"diagnosis_schema",
			"diagnosis_sha256",
			"source_image_sha256",
		):
			expected = job["source_image"]["sha256"] if key == "source_image_sha256" else job[key]
			_require(result.get(key) == expected, f"L1 plan result {key} mismatch: {job['state_id']}")
		_require(result.get("status") == "ok", f"L1 plan result is not ok: {job['state_id']}")
		_require(result.get("model") == contract.get("model"), f"L1 plan model mismatch: {job['state_id']}")
		_require(result.get("model_revision") == contract.get("model_revision"), f"L1 plan revision mismatch: {job['state_id']}")
		_require(result.get("dtype") == contract.get("dtype"), f"L1 plan dtype mismatch: {job['state_id']}")
		_require(result.get("quantization") == contract.get("quantization"), f"L1 plan quantization mismatch: {job['state_id']}")
		_require(result.get("allow_cpu_offload") == contract.get("allow_cpu_offload"), f"L1 plan CPU-offload mismatch: {job['state_id']}")
		_require(result.get("effective_quantization") == contract.get("effective_quantization"), f"L1 plan effective quantization mismatch: {job['state_id']}")
		_require(result.get("max_image_side") == contract.get("max_image_side"), f"L1 plan input max side mismatch: {job['state_id']}")
		_require(result.get("max_new_tokens") == contract.get("max_new_tokens"), f"L1 plan token budget mismatch: {job['state_id']}")
		plan = result.get("action_plan")
		_require(isinstance(plan, Mapping), f"L1 action plan missing: {job['state_id']}")
		canonical_plan = canonicalize_l1_action_plan(
			plan,
			target_prompt=job["target_prompt"],
			diagnosis=job["diagnosis"],
			width=int(job["source_image"]["width"]),
			height=int(job["source_image"]["height"]),
		)
		_require(canonical_plan is not None, f"L1 action plan is invalid: {job['state_id']}")
		for key in (
			"schema_version",
			"level",
			"target_prompt",
			"diagnosis",
			"action",
			"instruction",
			"candidate_count",
			"editor_call_count",
		):
			_require(plan.get(key) == canonical_plan.get(key), f"L1 action plan {key} mismatch: {job['state_id']}")
		_require(result.get("action_plan_sha256") == sha256_json(plan), f"L1 action plan hash mismatch: {job['state_id']}")
		plans.append(
			{
				"schema_version": SCHEMA_VERSION,
				"state_id": job["state_id"],
				"distribution_signature": contract["distribution_signature"],
				"diagnosis_schema": DIAGNOSIS_SCHEMA,
				"diagnosis_sha256": job["diagnosis_sha256"],
				"source_image_sha256": job["source_image"]["sha256"],
				"action_plan": canonical_plan,
				"action_plan_sha256": sha256_json(canonical_plan),
				"planner_job_id": job["job_id"],
			}
		)
	plans.sort(key=lambda row: row["state_id"])
	summary = {
		"schema_version": SCHEMA_VERSION,
		"status": "collected",
		"plan_count": len(plans),
		"qwen_distribution_signature": contract["distribution_signature"],
		"diagnosis_schema": DIAGNOSIS_SCHEMA,
		"plans_digest": sha256_json(plans),
	}
	if not dry_run:
		write_if_identical_or_absent(output_path.resolve(), _jsonl_text(plans))
		write_if_identical_or_absent(
			output_path.resolve().with_suffix(output_path.suffix + ".summary.json"),
			json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
		)
	return summary
