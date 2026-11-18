"""GPU executor for route-free Qwen reflection jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

from inference.main_method.modules.contracts import (
	DIAGNOSIS_RETRY_INSTRUCTION_V10,
	TRIAGE_SYSTEM_PROMPT_V10,
	diagnosis_payload_is_valid,
	validate_loaded_qwen_quantization_runtime,
)

from .bootstrap import (
	REFLECTION_STAGE,
	_relative,
	load_reflector_config,
	reflection_user_prompt,
	validate_reflection_job_contract,
)
from .canonical import archive_existing, iter_jsonl, read_json, sha256_json, write_json_atomic
from .execute_flux2 import _resolve_pinned_model
from .existing_initial_assets import EXISTING_SOURCE_MODE, validate_existing_reflection_work
from .qwen_batching import generate_with_one_json_retry
from .schema import DataValidationError, canonicalize_diagnosis, image_descriptor


# Backward-compatible import name; the bytes now live in the shared online/offline contract.
DIAGNOSIS_RETRY_INSTRUCTION = DIAGNOSIS_RETRY_INSTRUCTION_V10


def _extract_object(text: str) -> dict[str, Any]:
	clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
	decoder = json.JSONDecoder()
	for index, character in enumerate(clean):
		if character != "{":
			continue
		try:
			value, _ = decoder.raw_decode(clean[index:])
		except json.JSONDecodeError:
			continue
		if isinstance(value, dict):
			return value
	raise DataValidationError("Qwen reflection did not return a JSON object")


def _validate_diagnosis_contract(diagnosis: Mapping[str, Any]) -> None:
	if not diagnosis_payload_is_valid(diagnosis):
		raise DataValidationError("reflection does not satisfy the canonical V10 diagnosis contract")


def _normalize_diagnosis_response(raw: str, _index: int = 0) -> dict[str, Any]:
	raw_diagnosis = _extract_object(raw)
	if not diagnosis_payload_is_valid(raw_diagnosis):
		raise DataValidationError("Qwen reflection returned non-canonical V10 diagnosis JSON")
	diagnosis = canonicalize_diagnosis(raw_diagnosis)
	_validate_diagnosis_contract(diagnosis)
	return {"valid_json": True, "diagnosis": diagnosis}


def _valid_completed(root: Path, job: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
	"""Accept a cached reflection only when its job and prompt hashes still match."""
	path = _relative(root, str(job["result_path"]))
	if not path.is_file():
		return False
	try:
		result = read_json(path)
		diagnosis = canonicalize_diagnosis(result.get("diagnosis"))
		_validate_diagnosis_contract(diagnosis)
	except Exception:
		return False
	return (
		isinstance(result, Mapping)
		and result.get("status") == "ok"
		and result.get("stage") == REFLECTION_STAGE
		and result.get("job_id") == job["job_id"]
		and result.get("request_hash") == job["request_hash"]
		and result.get("reflector_signature") == contract["reflector_signature"]
		and result.get("distribution_signature") == contract["distribution_signature"]
		and result.get("diagnosis_schema") == contract["diagnosis_schema"]
		and result.get("prompt_sha256") == job["prompt_sha256"]
		and result.get("executor_family") == "qwen"
		and result.get("model") == contract["model"]
		and result.get("model_revision") == contract["model_revision"]
		and result.get("seed") == job["seed"]
		and result.get("decoding") == contract["decoding"]
		and result.get("dtype") == contract["dtype"]
		and result.get("quantization") == contract["quantization"]
		and result.get("allow_cpu_offload") == contract["allow_cpu_offload"]
		and result.get("effective_quantization") == contract["effective_quantization"]
		and result.get("max_image_side") == contract["max_image_side"]
		and result.get("max_new_tokens") == contract["max_new_tokens"]
		and result.get("source_image_sha256") == job["source_image"]["sha256"]
		and result.get("diagnosis_sha256") == sha256_json(diagnosis)
		and result.get("json_retry_count") in {0, 1}
	)


def _validate_plan(
	root: Path,
	jobs: list[dict[str, Any]],
	contract: Mapping[str, Any],
) -> dict[str, Path]:
	"""Validate plan and reject contract violations."""
	manifest = read_json(root / "reflection_manifest.json")
	if manifest.get("source_mode") == EXISTING_SOURCE_MODE:
		external_contract, external_jobs, source_paths, _source_rows = validate_existing_reflection_work(
			root,
			verify_images=True,
		)
		if external_contract != contract or external_jobs != jobs:
			raise DataValidationError("existing-source reflection plan changed during validation")
		return source_paths

	source_paths: dict[str, Path] = {}
	if len({job.get("job_id") for job in jobs}) != len(jobs):
		raise DataValidationError("duplicate reflection job_id")
	for job in jobs:
		validate_reflection_job_contract(job, contract)
		path = _relative(root, str(job["source_image"]["path"]))
		if image_descriptor(path) != job["source_image"]:
			raise DataValidationError(f"reflection source changed: {job.get('job_id')}")
		source_paths[str(job["job_id"])] = path
	if manifest.get("stage") != REFLECTION_STAGE:
		raise DataValidationError("invalid reflection manifest stage")
	if manifest.get("status") != "planned":
		raise DataValidationError("invalid reflection manifest status")
	if manifest.get("job_count") != len(jobs):
		raise DataValidationError("reflection manifest job_count mismatch")
	if manifest.get("reflector_signature") != contract["reflector_signature"]:
		raise DataValidationError("reflection manifest/lock signature mismatch")
	if manifest.get("distribution_signature") != contract["distribution_signature"]:
		raise DataValidationError("reflection manifest/Qwen distribution mismatch")
	if manifest.get("diagnosis_schema") != contract["diagnosis_schema"]:
		raise DataValidationError("reflection manifest diagnosis schema mismatch")
	if manifest.get("frozen_digest") != sha256_json(jobs):
		raise DataValidationError("reflection jobs differ from reflection manifest")
	return source_paths


def build_parser() -> argparse.ArgumentParser:
	"""Construct the CLI parser for resumable batched reflection generation."""
	parser = argparse.ArgumentParser(description="Execute route-free Qwen reflection jobs")
	parser.add_argument("--bootstrap-root", type=Path, required=True)
	parser.add_argument("--max-memory-per-gpu")
	parser.add_argument("--min-free-gib", type=float, default=16.0)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--limit", type=int)
	parser.add_argument("--retry-errors", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	return parser


def main(argv: list[str] | None = None) -> None:
	"""Run this module's command-line workflow and return its process status."""
	args = build_parser().parse_args(argv)
	root = args.bootstrap_root.resolve()
	contract = load_reflector_config(root / "reflector.lock.json")
	jobs = list(iter_jsonl(root / "reflection_jobs.jsonl"))
	source_paths = _validate_plan(root, jobs, contract)
	selected = jobs[max(0, args.start) :]
	if args.limit is not None:
		selected = selected[: max(0, args.limit)]
	pending: list[dict[str, Any]] = []
	for job in selected:
		if _valid_completed(root, job, contract):
			continue
		path = _relative(root, str(job["result_path"]))
		if path.is_file():
			try:
				existing = read_json(path)
			except Exception as exc:
				raise DataValidationError(f"corrupt reflection result {path}: {exc}") from exc
			identity_ok = (
				isinstance(existing, Mapping)
				and existing.get("job_id") == job["job_id"]
				and existing.get("request_hash") == job["request_hash"]
				and existing.get("reflector_signature") == contract["reflector_signature"]
			)
			if not identity_ok:
				raise DataValidationError(f"stale reflection result: {path}")
			if existing.get("status") != "error":
				raise DataValidationError(f"invalid reflection result: {path}")
			if not args.retry_errors:
				continue
		pending.append(job)
	print({"selected": len(selected), "pending": len(pending), "dry_run": args.dry_run})
	if args.dry_run or not pending:
		return

	from PIL import Image
	from inference.main_method.modules.image_utils import pad_image_to_multiple, prepare_qwen_image
	from inference.main_method.modules.vcot_branch import QwenPlanner, parse_dtype

	model_path = _resolve_pinned_model(contract["model"], contract["model_revision"])
	planner = QwenPlanner(
		model_path=str(model_path),
		dtype=parse_dtype(contract["dtype"]),
		max_memory_per_gpu=args.max_memory_per_gpu,
		min_free_gib=args.min_free_gib,
		per_gpu_extra_reserve={},
		quantization=contract["quantization"],
		allow_cpu_offload=contract["allow_cpu_offload"],
		route_head_path=None,
	)
	try:
		validate_loaded_qwen_quantization_runtime(planner, contract["qwen_distribution"])
	except ValueError as exc:
		raise DataValidationError(str(exc)) from exc
	counts = {"ok": 0, "error": 0}
	step = max(1, int(args.batch_size))
	for start in range(0, len(pending), step):
		chunk = pending[start : start + step]
		prepared_images = []
		messages_batch = []
		try:
			for job in chunk:
				source_path = source_paths[str(job["job_id"])]
				with Image.open(source_path) as loaded:
					# Online V10 triage/route encoding is deliberately unannotated.
					# Coordinate rulers are introduced only in the post-route L1 planner.
					prepared = prepare_qwen_image(loaded.convert("RGB"), max_side=contract["max_image_side"])
					prepared = pad_image_to_multiple(prepared, multiple=32)
				prepared_images.append(prepared)
				messages_batch.append(
					[
						{"role": "system", "content": [{"type": "text", "text": TRIAGE_SYSTEM_PROMPT_V10}]},
						{
							"role": "user",
							"content": [
								{"type": "image", "image": prepared},
								{"type": "text", "text": reflection_user_prompt(job)},
							],
						},
					]
				)
			for job in chunk:
				archive_existing(_relative(root, str(job["result_path"])))
			started = time.monotonic()
			responses = generate_with_one_json_retry(
				planner,
				messages_batch,
				[[image] for image in prepared_images],
				contract["max_new_tokens"],
				_normalize_diagnosis_response,
				DIAGNOSIS_RETRY_INSTRUCTION_V10,
			)
			for job, response in zip(chunk, responses):
				path = _relative(root, str(job["result_path"]))
				base = {
					"schema_version": job["schema_version"], "stage": REFLECTION_STAGE, "executor_family": "qwen",
					"model": contract["model"], "model_revision": contract["model_revision"], "seed": job["seed"],
					"decoding": contract["decoding"], "job_id": job["job_id"], "request_hash": job["request_hash"],
					"reflector_signature": contract["reflector_signature"], "prompt_version": job["prompt_version"],
					"distribution_signature": contract["distribution_signature"],
					"diagnosis_schema": contract["diagnosis_schema"],
					"prompt_sha256": job["prompt_sha256"], "source_image_sha256": job["source_image"]["sha256"],
					"dtype": contract["dtype"], "quantization": contract["quantization"],
					"allow_cpu_offload": contract["allow_cpu_offload"],
					"effective_quantization": contract["effective_quantization"],
					"max_image_side": contract["max_image_side"], "max_new_tokens": contract["max_new_tokens"],
					"json_retry_count": int(response.get("json_retry_count", 0)),
				}
				if response.get("valid_json"):
					diagnosis = response["diagnosis"]
					payload = {**base, "status": "ok", "eligible_for_router": not diagnosis["fully_satisfied"], "diagnosis": diagnosis, "diagnosis_sha256": sha256_json(diagnosis), "raw_response": response["raw_response"], "duration_seconds": time.monotonic() - started}
				else:
					exc = response.get("generation_exception") or response.get("validation_exception") or DataValidationError("invalid diagnosis after one JSON retry")
					payload = {**base, "status": "error", "error_type": type(exc).__name__, "error_message": str(exc), "raw_response": response.get("raw_response", ""), "duration_seconds": time.monotonic() - started}
				if "initial_raw_response" in response:
					payload["initial_raw_response"] = response["initial_raw_response"]
				write_json_atomic(path, payload)
				counts[payload["status"]] += 1
		finally:
			for image in prepared_images:
				try:
					image.close()
				except Exception:
					pass
		print({"progress": f"{min(start + len(chunk), len(pending))}/{len(pending)}", **counts}, flush=True)
	print(counts)


if __name__ == "__main__":
	main()
