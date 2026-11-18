"""Blind paired Qwen evaluator for frozen router counterfactuals.

The evaluator sees the target, the diagnosis, and a randomized presentation of
the original/candidate images.  It never receives the action level, executor,
seed, model name, or rollout index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

from .canonical import archive_existing, iter_jsonl, read_json, sha256_file, stable_seed, write_if_identical_or_absent, write_json_atomic
from .evaluation_contract import (
	BLIND_PROTOCOL,
	PROMPT_VERSION,
	SUCCESS_DEFINITION,
	evaluation_prompt,
	evaluator_contract,
	prompt_hash as build_prompt_hash,
)
from .execute_flux2 import _resolve_pinned_model
from .qwen_batching import generate_with_one_json_retry
from .schema import DataValidationError
from .validate import validate_dataset


EVALUATOR_JSON_RETRY_INSTRUCTION = """Your previous response failed JSON schema validation. Return one corrected JSON object only. It must contain exactly the top-level keys image_a, image_b, and rationale. Both image objects must contain exactly target_satisfaction, diagnosed_issue_resolution, preservation, artifact_severity, and preservation_violation, with numeric scores in [0,1] and a boolean preservation_violation."""


def _path(root: Path, value: str) -> Path:
	path = Path(value)
	return path if path.is_absolute() else root / path


def _extract_json_object(text: str) -> dict[str, Any]:
	clean = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
	decoder = json.JSONDecoder()
	for index, char in enumerate(clean):
		if char != "{":
			continue
		try:
			value, _ = decoder.raw_decode(clean[index:])
		except json.JSONDecodeError:
			continue
		if isinstance(value, dict):
			return value
	raise DataValidationError("evaluator did not return a JSON object")


def _normalize_evaluation(value: Mapping[str, Any], order: list[str]) -> dict[str, Any]:
	"""Normalize evaluator JSON while failing safely to the original image on invalid output."""
	if sorted(order) != ["candidate", "original"] or len(order) != 2:
		raise DataValidationError(f"invalid blind presentation order: {order}")
	required_top = {"image_a", "image_b", "rationale"}
	if set(value) != required_top:
		raise DataValidationError(
			f"evaluator keys must be exactly {sorted(required_top)}, got {sorted(value)}"
		)
	required_metrics = {
		"target_satisfaction",
		"diagnosed_issue_resolution",
		"preservation",
		"artifact_severity",
		"preservation_violation",
	}
	by_label: dict[str, dict[str, Any]] = {}
	for label in ("image_a", "image_b"):
		raw_metrics = value.get(label)
		if not isinstance(raw_metrics, Mapping) or set(raw_metrics) != required_metrics:
			actual = sorted(raw_metrics) if isinstance(raw_metrics, Mapping) else type(raw_metrics).__name__
			raise DataValidationError(
				f"{label} keys must be exactly {sorted(required_metrics)}, got {actual}"
			)
		metrics: dict[str, Any] = {}
		for name in (
			"target_satisfaction",
			"diagnosed_issue_resolution",
			"preservation",
			"artifact_severity",
		):
			raw_score = raw_metrics.get(name)
			if isinstance(raw_score, bool):
				raise DataValidationError(f"{label}.{name} must be numeric")
			try:
				score = float(raw_score)
			except (TypeError, ValueError) as exc:
				raise DataValidationError(f"{label}.{name} must be numeric") from exc
			if not 0.0 <= score <= 1.0:
				raise DataValidationError(f"{label}.{name} must be in [0,1]")
			metrics[name] = score
		if not isinstance(raw_metrics.get("preservation_violation"), bool):
			raise DataValidationError(f"{label}.preservation_violation must be boolean")
		metrics["preservation_violation"] = raw_metrics["preservation_violation"]
		by_label[label] = metrics

	# Source roles are mapped only after Qwen has returned symmetric A/B scores.
	role_to_label = {
		order[0]: "image_a",
		order[1]: "image_b",
	}
	original = by_label[role_to_label["original"]]
	candidate = by_label[role_to_label["candidate"]]
	result: dict[str, Any] = {
		"original_target_satisfaction": original["target_satisfaction"],
		"candidate_target_satisfaction": candidate["target_satisfaction"],
		"targeted_fix": candidate["diagnosed_issue_resolution"],
		"preservation": candidate["preservation"],
		"artifact_severity": candidate["artifact_severity"],
		"introduced_new_error": candidate["preservation_violation"],
	}
	result["rationale"] = " ".join(str(value.get("rationale") or "").split())[:500]
	# A deterministic success definition makes this diagnostic comparable across
	# evaluator prompts and avoids asking the VLM for a redundant free-form label.
	result["success"] = bool(
		result["candidate_target_satisfaction"] >= 0.80
		and result["targeted_fix"] >= 0.70
		and result["artifact_severity"] <= 0.30
		and not result["introduced_new_error"]
	)
	return result


def build_evaluation_messages(
	prompt: str,
	order: list[str],
	images_by_role: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[Any]]:
	"""Return the exact two-image order used by both batch and scalar paths."""

	if sorted(order) != ["candidate", "original"]:
		raise DataValidationError(f"invalid blind presentation order: {order}")
	ordered_images = [images_by_role[order[0]], images_by_role[order[1]]]
	messages = [
		{
			"role": "user",
			"content": [
				{"type": "text", "text": "A:"},
				{"type": "image", "image": ordered_images[0]},
				{"type": "text", "text": "B:"},
				{"type": "image", "image": ordered_images[1]},
				{"type": "text", "text": prompt},
			],
		}
	]
	return messages, ordered_images


def _existing_is_valid(
	path: Path,
	job: Mapping[str, Any],
	evaluator_signature: str,
	prompt_hash: str,
	source_sha256: str,
	candidate_sha256: str,
) -> bool:
	"""Reuse an evaluation only when prompt, image, and model hashes all match."""
	if not path.is_file():
		return False
	try:
		value = read_json(path)
	except Exception:
		return False
	return (
		isinstance(value, Mapping)
		and value.get("status") == "ok"
		and value.get("job_id") == job.get("job_id")
		and value.get("request_hash") == job.get("request_hash")
		and value.get("evaluator_signature") == evaluator_signature
		and value.get("prompt_hash") == prompt_hash
		and value.get("source_image_sha256") == source_sha256
		and value.get("candidate_image_sha256") == candidate_sha256
		and value.get("blind_protocol") == BLIND_PROTOCOL
	)


def build_parser() -> argparse.ArgumentParser:
	"""Construct the CLI parser for hash-validated Qwen evaluation."""
	parser = argparse.ArgumentParser(description="Blindly evaluate Flux.2 counterfactual image pairs with Qwen")
	parser.add_argument("--dataset-root", type=Path, required=True)
	parser.add_argument("--model", required=True, help="Pinned local Qwen snapshot or repository ID")
	parser.add_argument("--model-revision", required=True, help="Exact checkpoint revision used for evaluator identity")
	parser.add_argument("--dtype", default="bfloat16")
	parser.add_argument("--quantization", default="8bit", choices=("none", "8bit", "4bit", "nf4"))
	parser.add_argument("--max-memory-per-gpu")
	parser.add_argument("--min-free-gib", type=float, default=16.0)
	parser.add_argument("--allow-cpu-offload", action="store_true")
	parser.add_argument("--max-new-tokens", type=int, default=384)
	parser.add_argument("--max-image-side", type=int, default=1280)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=1,
		help="Two-image evaluation micro-batch target; failures are recursively bisected to isolate bad rows.",
	)
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--limit", type=int)
	parser.add_argument("--retry-errors", action="store_true")
	parser.add_argument(
		"--skip-global-validation",
		action="store_true",
		help=(
			"supervisor-only: skip repeated full-dataset validation after a persisted round "
			"structure marker; selected results/evaluations are still identity/hash-validated"
		),
	)
	parser.add_argument("--dry-run", action="store_true", help="Validate and count without importing GPU libraries")
	return parser


def main(argv: list[str] | None = None) -> None:
	"""Run this module's command-line workflow and return its process status."""
	args = build_parser().parse_args(argv)
	root = args.dataset_root.resolve()
	if not args.skip_global_validation:
		validate_dataset(root)
	states = {row["state_id"]: row for row in iter_jsonl(root / "states.jsonl")}
	jobs = list(iter_jsonl(root / "jobs.jsonl"))[max(0, args.start) :]
	if args.limit is not None:
		jobs = jobs[: max(0, args.limit)]
	evaluator_id = f"qwen:{args.model}@{args.model_revision}"
	contract = evaluator_contract(
		model=args.model,
		model_revision=args.model_revision,
		dtype=args.dtype,
		quantization=args.quantization,
		max_new_tokens=args.max_new_tokens,
		max_image_side=args.max_image_side,
		allow_cpu_offload=args.allow_cpu_offload,
	)
	lock_path = root / "evaluator.lock.json"
	if lock_path.is_file() and read_json(lock_path) != contract:
		raise DataValidationError("evaluator arguments differ from the immutable evaluator.lock.json")
	if not args.dry_run:
		write_if_identical_or_absent(
			lock_path,
			json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
		)
	selected: list[tuple[dict[str, Any], list[str], str, str, str, str]] = []
	for job in jobs:
		result_path = _path(root, str(job["result_path"]))
		if not result_path.is_file():
			continue
		try:
			result = read_json(result_path)
		except Exception:
			continue
		output_path = _path(root, str(job["output_image"]))
		source_path = Path(job["source_image"]["path"])
		if (
			result.get("status") != "ok"
			or result.get("job_id") != job["job_id"]
			or result.get("request_hash") != job["request_hash"]
			or result.get("policy_signature") != job["policy_signature"]
			or result.get("l3_policy_digest") != job["l3_policy_digest"]
			or result.get("executor_family") != job["executor_family"]
			or not output_path.is_file()
			or not source_path.is_file()
		):
			continue
		candidate_sha256 = sha256_file(output_path)
		source_sha256 = sha256_file(source_path)
		if result.get("output_sha256") != candidate_sha256 or source_sha256 != job["source_image"]["sha256"]:
			raise DataValidationError(f"result/source image hash mismatch for job {job['job_id']}")
		order = ["original", "candidate"]
		if stable_seed(job["job_id"], "blind_order") % 2:
			order.reverse()
		prompt = evaluation_prompt(states[job["state_id"]], order)
		prompt_hash = build_prompt_hash(states[job["state_id"]], order)
		evaluation_path = _path(root, str(job["evaluation_path"]))
		if _existing_is_valid(
			evaluation_path,
			job,
			contract["evaluator_signature"],
			prompt_hash,
			source_sha256,
			candidate_sha256,
		):
			continue
		if evaluation_path.is_file() and not args.retry_errors:
			raise DataValidationError(f"stale, invalid, or failed evaluation exists: {evaluation_path}")
		selected.append((job, order, prompt, prompt_hash, source_sha256, candidate_sha256))
	print(
		{
			"executable_results": len(selected),
			"dry_run": args.dry_run,
			"evaluator_id": evaluator_id,
			"evaluator_signature": contract["evaluator_signature"],
			"requested_batch_size": max(1, int(args.batch_size)),
			"images_per_sample": 2,
			"batch_policy": "two_image_batch_with_recursive_bisection_and_scalar_fallback",
		}
	)
	if args.dry_run or not selected:
		return

	# Heavy model dependencies intentionally begin here.
	from PIL import Image
	from inference.main_method.modules.vcot_branch import QwenPlanner, parse_dtype

	model_path = _resolve_pinned_model(args.model, args.model_revision)
	planner = QwenPlanner(
		model_path=str(model_path),
		dtype=parse_dtype(args.dtype),
		max_memory_per_gpu=args.max_memory_per_gpu,
		min_free_gib=args.min_free_gib,
		per_gpu_extra_reserve={},
		quantization=args.quantization,
		allow_cpu_offload=args.allow_cpu_offload,
		route_head_path=None,
	)
	counts = {"ok": 0, "error": 0}
	completed = 0
	step = max(1, int(args.batch_size))
	for chunk_start in range(0, len(selected), step):
		chunk = selected[chunk_start : chunk_start + step]
		prepared: list[dict[str, Any]] = []
		try:
			for job, order, prompt, prompt_hash, source_sha256, candidate_sha256 in chunk:
				started = time.monotonic()
				evaluation_path = _path(root, str(job["evaluation_path"]))
				archive_existing(evaluation_path)
				base = {
					"schema_version": job["schema_version"],
					"job_id": job["job_id"],
					"state_id": job["state_id"],
					"request_hash": job["request_hash"],
					"blind_protocol": BLIND_PROTOCOL,
					"evaluator_id": evaluator_id,
					"evaluator_signature": contract["evaluator_signature"],
					"prompt_version": PROMPT_VERSION,
					"prompt_hash": prompt_hash,
					"presentation_order": order,
					"source_image_sha256": source_sha256,
					"candidate_image_sha256": candidate_sha256,
				}
				images_by_role: dict[str, Any] = {}
				try:
					output_path = _path(root, str(job["output_image"]))
					if not output_path.is_file():
						raise DataValidationError("candidate output image is missing")
					for role, path in (
						("original", Path(job["source_image"]["path"])),
						("candidate", output_path),
					):
						with Image.open(path) as loaded:
							image = loaded.convert("RGB")
						long_side = max(image.size)
						if args.max_image_side > 0 and long_side > args.max_image_side:
							scale = args.max_image_side / float(long_side)
							resized = image.resize(
								(max(1, round(image.width * scale)), max(1, round(image.height * scale))),
								Image.Resampling.LANCZOS,
							)
							image.close()
							image = resized
						images_by_role[role] = image
					messages, ordered_images = build_evaluation_messages(prompt, order, images_by_role)
					prepared.append(
						{
							"job": job,
							"order": order,
							"evaluation_path": evaluation_path,
							"base": base,
							"started": started,
							"messages": messages,
							"images": ordered_images,
							"owned_images": list(images_by_role.values()),
							"prepare_error": None,
						}
					)
				except Exception as exc:
					for image in images_by_role.values():
						image.close()
					prepared.append(
						{
							"job": job,
							"evaluation_path": evaluation_path,
							"base": base,
							"started": started,
							"messages": [],
							"images": [],
							"owned_images": [],
							"prepare_error": exc,
						}
					)
			valid_indices = [index for index, item in enumerate(prepared) if item["prepare_error"] is None]
			responses_by_index: dict[int, dict[str, Any]] = {}
			if valid_indices:
				def validate_response(raw: str, local_index: int) -> dict[str, Any]:
					"""Normalize one response using its original prepared-item order."""
					prepared_index = valid_indices[local_index]
					return {
						"valid_json": True,
						"normalized": _normalize_evaluation(
							_extract_json_object(raw),
							prepared[prepared_index]["order"],
						),
					}

				responses = generate_with_one_json_retry(
					planner,
					[prepared[index]["messages"] for index in valid_indices],
					[prepared[index]["images"] for index in valid_indices],
					args.max_new_tokens,
					validate_response,
					EVALUATOR_JSON_RETRY_INSTRUCTION,
				)
				responses_by_index.update(zip(valid_indices, responses))
			for index, item in enumerate(prepared):
				job = item["job"]
				base = item["base"]
				response = responses_by_index.get(index)
				try:
					if item["prepare_error"] is not None:
						raise item["prepare_error"]
					if not isinstance(response, Mapping):
						raise RuntimeError("Qwen evaluator produced no response record")
					if not response.get("valid_json"):
						raise (
							response.get("generation_exception")
							or response.get("validation_exception")
							or DataValidationError("invalid evaluator JSON after one retry")
						)
					normalized = response.get("normalized")
					if not isinstance(normalized, Mapping):
						raise DataValidationError("validated evaluator response is missing normalized metrics")
					payload = {
						**base,
						"status": "ok",
						**normalized,
						"success_definition": SUCCESS_DEFINITION,
						"json_retry_count": int(response.get("json_retry_count", 0)),
						"raw_response": response.get("raw_response", ""),
						"duration_seconds": time.monotonic() - item["started"],
					}
					if "initial_raw_response" in response:
						payload["initial_raw_response"] = response["initial_raw_response"]
				except Exception as exc:
					payload = {
						**base,
						"status": "error",
						"error_type": type(exc).__name__,
						"error_message": str(exc),
						"json_retry_count": int(response.get("json_retry_count", 0)) if isinstance(response, Mapping) else 0,
						"raw_response": response.get("raw_response", "") if isinstance(response, Mapping) else "",
						"duration_seconds": time.monotonic() - item["started"],
					}
					if isinstance(response, Mapping) and "initial_raw_response" in response:
						payload["initial_raw_response"] = response["initial_raw_response"]
				write_json_atomic(item["evaluation_path"], payload)
				counts[payload["status"]] += 1
				completed += 1
				print({"progress": f"{completed}/{len(selected)}", "job_id": job["job_id"], "status": payload["status"]}, flush=True)
		finally:
			for item in prepared:
				for image in item["owned_images"]:
					image.close()
	print(counts)


if __name__ == "__main__":
	main()
