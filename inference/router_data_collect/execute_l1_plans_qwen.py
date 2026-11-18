"""Batch executor for frozen post-route V10 L1 localization plans."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Mapping

from inference.main_method.modules.contracts import (
	L1_PLAN_RETRY_INSTRUCTION_V10,
	L1_PLAN_SYSTEM_PROMPT_V10,
	validate_loaded_qwen_quantization_runtime,
)

from .bootstrap import (
	L1_PLAN_STAGE,
	_relative,
	l1_plan_user_prompt,
	load_l1_planner_config,
	validate_l1_plan_job_contract,
)
from .canonical import archive_existing, iter_jsonl, read_json, sha256_json, write_json_atomic
from .execute_flux2 import _resolve_pinned_model
from .qwen_batching import generate_with_one_json_retry
from .schema import DataValidationError


def build_parser() -> argparse.ArgumentParser:
	"""Construct the CLI parser for resumable batched L1-plan generation."""
	parser = argparse.ArgumentParser(description="Execute V10 post-route L1 planning jobs in stage batches")
	parser.add_argument("--plan-root", type=Path, required=True)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--max-memory-per-gpu")
	parser.add_argument("--min-free-gib", type=float, default=16.0)
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--limit", type=int)
	parser.add_argument("--retry-errors", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	return parser


def _completed(root: Path, job: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
	"""Reuse an L1 plan only when its durable result matches the current job."""
	path = _relative(root, str(job["result_path"]))
	if not path.is_file():
		return False
	try:
		value = read_json(path)
	except Exception:
		return False
	return (
		isinstance(value, Mapping)
		and value.get("status") == "ok"
		and value.get("job_id") == job["job_id"]
		and value.get("request_hash") == job["request_hash"]
		and value.get("planner_signature") == contract["planner_signature"]
		and value.get("distribution_signature") == contract["distribution_signature"]
		and value.get("diagnosis_schema") == contract["diagnosis_schema"]
		and value.get("model") == contract["model"]
		and value.get("model_revision") == contract["model_revision"]
		and value.get("dtype") == contract["dtype"]
		and value.get("quantization") == contract["quantization"]
		and value.get("allow_cpu_offload") == contract["allow_cpu_offload"]
		and value.get("effective_quantization") == contract["effective_quantization"]
		and value.get("max_image_side") == contract["max_image_side"]
		and value.get("max_new_tokens") == contract["max_new_tokens"]
		and value.get("diagnosis_sha256") == job["diagnosis_sha256"]
		and value.get("source_image_sha256") == job["source_image"]["sha256"]
		and value.get("action_plan_sha256") == sha256_json(value.get("action_plan"))
		and value.get("json_retry_count") in {0, 1}
	)


def main(argv: list[str] | None = None) -> None:
	"""Run this module's command-line workflow and return its process status."""
	args = build_parser().parse_args(argv)
	root = args.plan_root.resolve()
	contract = load_l1_planner_config(root / "planner.lock.json")
	jobs = list(iter_jsonl(root / "jobs.jsonl"))
	manifest = read_json(root / "manifest.json")
	if (
		manifest.get("stage") != L1_PLAN_STAGE
		or manifest.get("jobs_digest") != sha256_json(jobs)
		or manifest.get("planner_signature") != contract["planner_signature"]
		or manifest.get("distribution_signature") != contract["distribution_signature"]
		or manifest.get("diagnosis_schema") != contract["diagnosis_schema"]
	):
		raise DataValidationError("invalid L1 planner manifest")
	for job in jobs:
		validate_l1_plan_job_contract(job, contract)
	selected = jobs[max(0, args.start) :]
	if args.limit is not None:
		selected = selected[: max(0, args.limit)]
	pending: list[dict[str, Any]] = []
	for job in selected:
		if _completed(root, job, contract):
			continue
		path = _relative(root, str(job["result_path"]))
		if path.is_file():
			existing = read_json(path)
			identity_ok = (
				isinstance(existing, Mapping)
				and existing.get("job_id") == job["job_id"]
				and existing.get("request_hash") == job["request_hash"]
			)
			if not identity_ok:
				raise DataValidationError(f"stale L1 plan result: {path}")
			if existing.get("status") != "error" or not args.retry_errors:
				continue
		pending.append(job)
	print({"selected": len(selected), "pending": len(pending), "batch_size": args.batch_size, "dry_run": args.dry_run})
	if args.dry_run or not pending:
		return

	from PIL import Image
	from inference.main_method.modules.image_utils import annotate_image_with_ticks_32, pad_image_to_multiple, prepare_qwen_image
	from inference.main_method.modules.parsing import extract_json, normalize_l1_plan
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
		prepared: list[Any] = []
		messages_batch = []
		try:
			for job in chunk:
				source_path = Path(job["source_image"]["path"])
				with Image.open(source_path) as loaded:
					image = annotate_image_with_ticks_32(loaded.convert("RGB"))
					image = prepare_qwen_image(image, max_side=contract["max_image_side"])
					image = pad_image_to_multiple(image, multiple=32)
				prepared.append(image)
				messages_batch.append(
					[
						{"role": "system", "content": [{"type": "text", "text": L1_PLAN_SYSTEM_PROMPT_V10}]},
						{
							"role": "user",
							"content": [
								{"type": "image", "image": image},
								{"type": "text", "text": l1_plan_user_prompt(job)},
							],
						},
					]
				)
			for job in chunk:
				archive_existing(_relative(root, str(job["result_path"])))
			started = time.monotonic()

			def validate_response(raw: str, index: int) -> Mapping[str, Any]:
				"""Normalize one response against the corresponding chunk job."""
				job = chunk[index]
				return normalize_l1_plan(
					extract_json(raw),
					raw,
					job["diagnosis"],
					job["target_prompt"],
					int(job["source_image"]["width"]),
					int(job["source_image"]["height"]),
				)

			responses = generate_with_one_json_retry(
				planner,
				messages_batch,
				[[image] for image in prepared],
				contract["max_new_tokens"],
				validate_response,
				L1_PLAN_RETRY_INSTRUCTION_V10,
			)
			for job, response in zip(chunk, responses):
				path = _relative(root, str(job["result_path"]))
				base = {
					"schema_version": job["schema_version"],
					"stage": L1_PLAN_STAGE,
					"executor_family": "qwen",
					"job_id": job["job_id"],
					"request_hash": job["request_hash"],
					"planner_signature": job["planner_signature"],
					"distribution_signature": contract["distribution_signature"],
					"diagnosis_schema": contract["diagnosis_schema"],
					"model": contract["model"],
					"model_revision": contract["model_revision"],
					"dtype": contract["dtype"],
					"quantization": contract["quantization"],
					"allow_cpu_offload": contract["allow_cpu_offload"],
					"effective_quantization": contract["effective_quantization"],
					"max_image_side": contract["max_image_side"],
					"max_new_tokens": contract["max_new_tokens"],
					"diagnosis_sha256": job["diagnosis_sha256"],
					"source_image_sha256": job["source_image"]["sha256"],
					"json_retry_count": int(response.get("json_retry_count", 0)),
				}
				if response.get("valid_json"):
					action_plan = {
						key: value
						for key, value in response.items()
						if key not in {
							"raw_output", "raw_response", "initial_raw_response", "valid_json",
							"reason", "json_retry_count", "generation_exception", "validation_exception",
						}
					}
					payload = {
						**base,
						"status": "ok",
						"action_plan": action_plan,
						"action_plan_sha256": sha256_json(action_plan),
						"raw_response": response["raw_response"],
						"duration_seconds": time.monotonic() - started,
					}
				else:
					exc = response.get("generation_exception") or response.get("validation_exception") or DataValidationError(str(response.get("reason") or "invalid L1 plan after one JSON retry"))
					payload = {
						**base,
						"status": "error",
						"error_type": type(exc).__name__,
						"error_message": str(exc),
						"raw_response": response.get("raw_response", ""),
						"duration_seconds": time.monotonic() - started,
					}
				if "initial_raw_response" in response:
					payload["initial_raw_response"] = response["initial_raw_response"]
				write_json_atomic(path, payload)
				counts[payload["status"]] += 1
		finally:
			for image in prepared:
				try:
					image.close()
				except Exception:
					pass
		print({"progress": f"{min(start + len(chunk), len(pending))}/{len(pending)}", **counts}, flush=True)
	print(counts)


if __name__ == "__main__":
	main()
