"""GPU executor for the frozen Flux.2 jobs.

Heavy dependencies are imported only after argument parsing and the dry-run
exit, so planning, validation and ``--help`` remain CPU-only.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any, Mapping

from inference.main_method.modules.contracts import l1_crop_rois

from .aggregate import _load_and_validate_result
from .canonical import archive_existing, iter_jsonl, read_json, sha256_file, write_json_atomic
from .planner import load_config
from .schema import DataValidationError
from .validate import validate_dataset


def _artifact_path(root: Path, value: str) -> Path:
	path = Path(value)
	return path if path.is_absolute() else root / path


def _resolve_pinned_model(model_id: str, revision: str) -> Path:
	"""Resolve a model only at an immutable revision and return its local snapshot."""
	local = Path(model_id).expanduser()
	if local.exists():
		# Validate the caller-visible snapshot name before resolving symlinks.
		# Hugging Face caches may represent snapshots/<commit> as a symlink to a
		# shared materialized directory (for example ``main``). Comparing the
		# resolved target basename incorrectly rejects that still-explicit commit
		# lock, while accepting ``.../snapshots/main`` would weaken the contract.
		if local.name != revision:
			raise DataValidationError(
				f"local model directory basename {local.name!r} does not match locked revision {revision!r}; "
				"point model_id at the exact snapshots/<revision> directory"
			)
		return local.resolve()
	roots = [
		Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub",
	]
	cache_name = "models--" + model_id.replace("/", "--")
	for root in roots:
		candidate = root / cache_name / "snapshots" / revision
		if candidate.is_dir():
			return candidate.resolve()
	raise FileNotFoundError(
		f"Pinned Flux.2 snapshot not found for {model_id}@{revision}; "
		"download that exact revision or set model_id to a local snapshot directory"
	)


def _completed_job(root: Path, job: Mapping[str, Any]) -> bool:
	result, _error = _load_and_validate_result(root, job)
	return result is not None


def _base_result(job: Mapping[str, Any]) -> dict[str, Any]:
	return {
		"schema_version": job["schema_version"],
		"job_id": job["job_id"],
		"state_id": job["state_id"],
		"request_hash": job["request_hash"],
		"policy_signature": job["policy_signature"],
		"flux2_policy_digest": job["flux2_policy_digest"],
		"l3_policy_digest": job["l3_policy_digest"],
		"executor_family": job["executor_family"],
		"action_level": job["action_level"],
		"rollout_index": job["rollout_index"],
		"seed": job["seed"],
		"round_index": job["round_index"],
		"action_seed": job["action_seed"],
	}


def _run_l1(
	service: Any,
	torch: Any,
	image_module: Any,
	root: Path,
	job: Mapping[str, Any],
	references: list[dict[str, Any]],
) -> Path:
	"""Execute one joint L1 call with one editable full image and 1--3 frozen crops."""

	policy = job["policy"]
	resolution = int(policy["resolution"])
	source_path = Path(job["source_image"]["path"])
	with image_module.open(source_path) as loaded:
		full_image = loaded.convert("RGB").resize((resolution, resolution), image_module.Resampling.LANCZOS)
	original_width = int(job["source_image"]["width"])
	original_height = int(job["source_image"]["height"])
	artifact_dir = _artifact_path(root, str(job["artifact_dir"]))
	artifact_dir.mkdir(parents=True, exist_ok=True)
	scaled_quads = [
		[
			[
				int(round(point[0] * resolution / original_width)),
				int(round(point[1] * resolution / original_height)),
			]
			for point in edit["quad_points"]
		]
		for edit in job["edits"]
	]
	rois = l1_crop_rois(
		scaled_quads,
		resolution,
		resolution,
		pad_ratio=float(policy["l1_roi_pad_ratio"]),
		min_span=int(policy["l1_min_span"]),
	)
	crops = []
	for edit_index, (edit, roi) in enumerate(zip(job["edits"], rois)):
		crop = full_image.crop(tuple(roi)).resize((resolution, resolution), image_module.Resampling.LANCZOS)
		crop_path = artifact_dir / f"localization_crop_{edit_index + 1:02d}.png"
		crop.save(crop_path)
		crops.append(crop)
		references.append(
			{
				"role": f"localization_crop_{edit_index + 1:02d}",
				"edit_id": edit.get("id"),
				"roi_xyxy": roi,
				"image": str(crop_path.relative_to(root)),
				"sha256": sha256_file(crop_path),
			}
		)
	try:
		with torch.inference_mode():
			result = service.pipe(
				image=[full_image, *crops],
				prompt=str(job["prompt"]),
				height=resolution,
				width=resolution,
				num_inference_steps=int(policy["steps"]),
				guidance_scale=float(policy["guidance_scale"]),
				generator=torch.Generator(device=service.device).manual_seed(int(job["seed"])),
				max_sequence_length=int(policy["max_sequence_length"]),
			)
		current = result.images[0].convert("RGB")
	finally:
		full_image.close()
		for crop in crops:
			crop.close()
	if current.size != (resolution, resolution):
		resized = current.resize((resolution, resolution), image_module.Resampling.LANCZOS)
		current.close()
		current = resized
	output_path = _artifact_path(root, str(job["output_image"]))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	try:
		current.save(output_path)
	finally:
		current.close()
	return output_path


def _run_single_job(service: Any, torch: Any, image_module: Any, root: Path, job: Mapping[str, Any]) -> dict[str, Any]:
	"""Execute one validated FLUX.2 action and persist the exact candidate and request audit."""
	started = time.monotonic()
	base = _base_result(job)
	result_path = _artifact_path(root, str(job["result_path"]))
	references: list[dict[str, Any]] = []
	try:
		source_path = Path(job["source_image"]["path"])
		if not source_path.is_file() or sha256_file(source_path) != job["source_image"]["sha256"]:
			raise DataValidationError("source image is missing or changed after planning")
		level = int(job["action_level"])
		policy = job["policy"]
		if level == 1:
			output_path = _run_l1(service, torch, image_module, root, job, references)
			call_count = 1
		elif level == 2:
			output_path = service.edit(
				{
					"input_image": str(source_path),
					"output_image": str(_artifact_path(root, str(job["output_image"]))),
					"prompt": job["prompt"],
					"seed": int(job["seed"]),
					"resolution": int(policy["resolution"]),
					"steps": int(policy["steps"]),
					"guidance_scale": float(policy["guidance_scale"]),
					"max_sequence_length": int(policy["max_sequence_length"]),
				}
			)
			call_count = 1
		elif level == 3:
			output_path = service.generate(
				{
					"output_image": str(_artifact_path(root, str(job["output_image"]))),
					"prompt": job["prompt"],
					"seed": int(job["seed"]),
					"resolution": int(policy["resolution"]),
					"steps": int(policy["steps"]),
					"guidance_scale": float(policy["guidance_scale"]),
					"max_sequence_length": int(policy["max_sequence_length"]),
				}
			)
			call_count = 1
		else:
			raise DataValidationError(f"unsupported action level: {level}")
		output_path = Path(output_path)
		payload = {
			**base,
			"status": "ok",
			"output_image": str(output_path.relative_to(root)),
			"output_sha256": sha256_file(output_path),
			"call_count": call_count,
			"flux_call_count": call_count,
			"reference_images": (
				[str(job["source_image"]["path"]), *[str(item["image"]) for item in references]]
				if level == 1
				else ([str(job["source_image"]["path"])] if level == 2 else [])
			),
			"rois_xyxy": [list(item["roi_xyxy"]) for item in references],
			"reference_records": references,
			"duration_seconds": time.monotonic() - started,
		}
	except Exception as exc:
		payload = {
			**base,
			"status": "error",
			"error_type": type(exc).__name__,
			"error_message": str(exc),
			"completed_reference_records": references,
			"duration_seconds": time.monotonic() - started,
		}
	write_json_atomic(result_path, payload)
	return payload


def build_parser() -> argparse.ArgumentParser:
	"""Construct the CLI parser for hash-validated FLUX.2 action execution."""
	parser = argparse.ArgumentParser(description="Execute frozen Flux.2 L1/L2 and Flux-initial L3 jobs")
	parser.add_argument("--dataset-root", type=Path, required=True)
	parser.add_argument("--device", default="cuda:0")
	parser.add_argument(
		"--batch-size",
		type=int,
		default=2,
		help="Text-only L3 batch size. Conditional L1/L2 remain scalar unless a verified nested-reference adapter is added.",
	)
	parser.add_argument("--action-level", type=int, choices=(1, 2, 3), action="append")
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--limit", type=int)
	parser.add_argument("--retry-errors", action="store_true")
	parser.add_argument(
		"--skip-global-validation",
		action="store_true",
		help=(
			"supervisor-only: skip the repeated full-dataset validation after a persisted "
			"round structure-validation marker; selected shard results are still hash-validated"
		),
	)
	parser.add_argument("--dry-run", action="store_true", help="Validate and print selection without importing GPU libraries")
	return parser


def main(argv: list[str] | None = None) -> None:
	"""Run this module's command-line workflow and return its process status."""
	args = build_parser().parse_args(argv)
	root = args.dataset_root.resolve()
	if not args.skip_global_validation:
		validate_dataset(root)
	config = load_config(root / "config.lock.json", strict_revision=True)
	jobs = list(iter_jsonl(root / "jobs.jsonl"))
	levels = set(args.action_level or (1, 2, 3))
	selected = [
		job for job in jobs
		if int(job["action_level"]) in levels and job.get("executor_family") == "flux2"
	]
	selected = selected[max(0, args.start) :]
	if args.limit is not None:
		selected = selected[: max(0, args.limit)]
	pending = []
	for job in selected:
		if _completed_job(root, job):
			continue
		result_path = _artifact_path(root, str(job["result_path"]))
		if result_path.is_file():
			try:
				existing = read_json(result_path)
			except Exception as exc:
				raise DataValidationError(f"corrupt existing result {result_path}: {exc}") from exc
			identity_ok = (
				isinstance(existing, Mapping)
				and existing.get("job_id") == job["job_id"]
				and existing.get("request_hash") == job["request_hash"]
				and existing.get("policy_signature") == job["policy_signature"]
				and existing.get("flux2_policy_digest") == job["flux2_policy_digest"]
				and existing.get("l3_policy_digest") == job["l3_policy_digest"]
				and existing.get("executor_family") == "flux2"
			)
			if not identity_ok:
				raise DataValidationError(f"stale result identity at {result_path}")
			if existing.get("status") != "error" and not args.retry_errors:
				raise DataValidationError(f"invalid successful result/output pair at {result_path}")
			if not args.retry_errors:
				continue
		pending.append(job)
	print(
		{
			"selected": len(selected),
			"pending": len(pending),
			"skipped": len(selected) - len(pending),
			"levels": sorted(levels),
			"dry_run": args.dry_run,
		}
	)
	if args.dry_run or not pending:
		return

	# Heavy imports intentionally begin here.
	import torch
	from PIL import Image
	from inference.main_method.modules.flux_service import FluxService

	dtype = torch.bfloat16 if config.policy.dtype == "bfloat16" else torch.float16
	model_path = _resolve_pinned_model(config.policy.model_id, config.policy.model_revision)
	service = FluxService(str(model_path), args.device, dtype)
	# Compatible jobs are grouped for utilization.  FLUX.2 diffusers 0.37.1
	# repeats one flattened reference set across a prompt batch, so L1/L2 must not
	# be naively batched: doing so would leak images between samples.
	pending.sort(key=lambda job: (int(job["action_level"]), len(job.get("edits") or []), job["job_id"]))
	counts = {"ok": 0, "error": 0}
	completed = 0
	for level in (1, 2):
		for job in [row for row in pending if int(row["action_level"]) == level]:
			archive_existing(_artifact_path(root, str(job["result_path"])))
			result = _run_single_job(service, torch, Image, root, job)
			counts[result["status"]] += 1
			completed += 1
			print({"progress": f"{completed}/{len(pending)}", "job_id": job["job_id"], "status": result["status"], "effective_batch_size": 1}, flush=True)
	l3_jobs = [row for row in pending if int(row["action_level"]) == 3]
	step = max(1, int(args.batch_size))
	for start in range(0, len(l3_jobs), step):
		chunk = l3_jobs[start : start + step]
		for job in chunk:
			archive_existing(_artifact_path(root, str(job["result_path"])))
		started = time.monotonic()
		policy = chunk[0]["policy"]
		try:
			paths = list(
				service.generate_batch(
					{
						"items": [
							{"prompt": job["prompt"], "output_image": str(_artifact_path(root, str(job["output_image"]))), "seed": int(job["seed"])}
							for job in chunk
						],
						"resolution": int(policy["resolution"]),
						"steps": int(policy["steps"]),
						"guidance_scale": float(policy["guidance_scale"]),
						"max_sequence_length": int(policy["max_sequence_length"]),
					}
				)
			)
			if len(paths) != len(chunk):
				raise RuntimeError(
					f"Flux.2 L3 batch output mismatch: expected={len(chunk)} actual={len(paths)}"
				)
			for job, output_path in zip(chunk, paths):
				payload = {
					**_base_result(job),
					"status": "ok",
					"output_image": str(Path(output_path).relative_to(root)),
					"output_sha256": sha256_file(Path(output_path)),
					"call_count": 1,
					"flux_call_count": 1,
					"reference_images": [],
					"rois_xyxy": [],
					"reference_records": [],
					"duration_seconds": time.monotonic() - started,
					"effective_batch_size": len(chunk),
				}
				write_json_atomic(_artifact_path(root, str(job["result_path"])), payload)
				counts["ok"] += 1
		except Exception:
			for job in chunk:
				result = _run_single_job(service, torch, Image, root, job)
				counts[result["status"]] += 1
		completed += len(chunk)
		print({"progress": f"{completed}/{len(pending)}", "level": 3, "requested_batch_size": len(chunk)}, flush=True)
	print(counts)


if __name__ == "__main__":
	main()
