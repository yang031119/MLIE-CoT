"""Execute non-FLUX L3 jobs with the frozen initial-image generator.

FLUX-initial L3 jobs remain in ``execute_flux2`` so they retain true text-only
batching.  A dataset is homogeneous in its L3 policy; this executor therefore
loads exactly one BAGEL or SD3.5 model and rejects every identity mismatch
before allocating GPU memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

from .aggregate import _load_and_validate_result
from .canonical import archive_existing, iter_jsonl, read_json, sha256_file, write_json_atomic
from .planner import load_config
from .schema import DataValidationError
from .validate import validate_dataset
from inference.main_method.modules.l3_policy import resolve_initial_base_snapshot


def _path(root: Path, value: str) -> Path:
	path = Path(value)
	return path if path.is_absolute() else root / path


def _resolve_pinned_model(model_id: str, revision: str, explicit_path: str = "") -> Path:
	try:
		resolved, _identity, _revision = resolve_initial_base_snapshot(
			explicit_path or model_id,
			model_id=model_id,
			model_revision=revision,
		)
	except (ValueError, FileNotFoundError) as exc:
		raise DataValidationError(f"invalid pinned initial-base L3 model: {exc}") from exc
	return resolved


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
		"action_level": 3,
		"rollout_index": job["rollout_index"],
		"seed": job["seed"],
		"round_index": job["round_index"],
		"action_seed": job["action_seed"],
	}


def _write_result(root: Path, job: Mapping[str, Any], client: Any, family: str, runtime_args: Any) -> dict[str, Any]:
	"""Generate one L3 base image and persist its content-addressed result record."""
	result_path = _path(root, str(job["result_path"]))
	output_path = _path(root, str(job["output_image"]))
	started = time.monotonic()
	base = _base_result(job)
	try:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		if family == "bagel":
			client.generate(str(job["prompt"]), output_path, runtime_args, seed=int(job["seed"]))
		else:
			client.generate(str(job["prompt"]), output_path, int(job["seed"]), runtime_args)
		payload = {
			**base,
			"status": "ok",
			"output_image": str(output_path.relative_to(root)),
			"output_sha256": sha256_file(output_path),
			"call_count": 1,
			"executor_call_count": 1,
			"reference_images": [],
			"rois_xyxy": [],
			"reference_records": [],
			"duration_seconds": time.monotonic() - started,
		}
	except Exception as exc:
		payload = {
			**base,
			"status": "error",
			"error_type": type(exc).__name__,
			"error_message": str(exc),
			"duration_seconds": time.monotonic() - started,
		}
	write_json_atomic(result_path, payload)
	return payload


def build_parser() -> argparse.ArgumentParser:
	"""Construct the CLI parser for L3 base-image regeneration."""
	parser = argparse.ArgumentParser(description="Execute frozen BAGEL/SD3.5 initial-base L3 jobs")
	parser.add_argument("--dataset-root", type=Path, required=True)
	parser.add_argument("--gpu-index", type=int, default=0)
	parser.add_argument("--python", default=sys.executable)
	parser.add_argument("--bagel-service-script", type=Path, default=Path(__file__).parents[1] / "V10/modules/bagel_service.py")
	parser.add_argument("--sd35-service-script", type=Path, default=Path(__file__).parents[1] / "V10/modules/sd35_service.py")
	parser.add_argument("--unicot-root", default="third_party/UniCoT")
	parser.add_argument("--model-path", default="", help="Optional exact snapshots/<revision> path for the locked primary model.")
	parser.add_argument("--auxiliary-model-path", default="", help="Optional exact BAGEL fine-tuned snapshots/<revision> path.")
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--limit", type=int)
	parser.add_argument("--retry-errors", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	return parser


def main(argv: list[str] | None = None) -> None:
	"""Run this module's command-line workflow and return its process status."""
	args = build_parser().parse_args(argv)
	root = args.dataset_root.resolve()
	validate_dataset(root)
	config = load_config(root / "config.lock.json", strict_revision=True)
	policy = config.l3_policy.contract
	family = str(policy["executor"]["family"])
	if family == "flux2":
		print({"family": family, "selected": 0, "pending": 0, "reason": "handled_by_execute_flux2"})
		return
	jobs = [
		job for job in iter_jsonl(root / "jobs.jsonl")
		if int(job["action_level"]) == 3 and job.get("executor_family") == family
	]
	jobs = jobs[max(0, args.start) :]
	if args.limit is not None:
		jobs = jobs[: max(0, args.limit)]
	pending: list[dict[str, Any]] = []
	for job in jobs:
		result, _error = _load_and_validate_result(root, job)
		if result is not None:
			continue
		result_path = _path(root, str(job["result_path"]))
		if result_path.is_file():
			existing = read_json(result_path)
			identity_ok = all(
				existing.get(key) == job.get(key)
				for key in ("job_id", "request_hash", "policy_signature", "l3_policy_digest", "executor_family")
			)
			if not identity_ok:
				raise DataValidationError(f"stale L3 result identity at {result_path}")
			if existing.get("status") != "error" and not args.retry_errors:
				raise DataValidationError(f"invalid successful L3 result/output pair at {result_path}")
			if not args.retry_errors:
				continue
		pending.append(job)
	print({"family": family, "selected": len(jobs), "pending": len(pending), "dry_run": args.dry_run})
	if args.dry_run or not pending:
		return

	executor = policy["executor"]
	generation = policy["generation"]
	model_path = _resolve_pinned_model(executor["model_id"], executor["model_revision"], args.model_path)
	if family == "bagel":
		auxiliary_path = _resolve_pinned_model(
			executor["auxiliary_model_id"],
			executor["auxiliary_model_revision"],
			args.auxiliary_model_path,
		)
		from inference.main_method.modules.vcot_branch import BagelClient

		runtime_args = SimpleNamespace(
			bagel_gpu_index=args.gpu_index,
			bagel_python=args.python,
			bagel_service_script=str(args.bagel_service_script.resolve()),
			unicot_root=args.unicot_root,
			bagel_model_path=str(model_path),
			bagel_fine_tuned_path=str(auxiliary_path),
			max_latent_size=int(generation["max_latent_size"]),
			resolution=int(generation["resolution"]),
			bagel_steps=int(generation["steps"]),
			bagel_cfg_scale=float(generation["guidance_scale"]),
		)
		client = BagelClient(runtime_args)
	else:
		from inference.main_method.modules.vcot_branch import SD35TurboClient

		runtime_args = SimpleNamespace(
			sd35_gpu_index=args.gpu_index,
			vcot_python=args.python,
			sd35_service_script=str(args.sd35_service_script.resolve()),
			sd35_model=str(model_path),
			sd35_dtype=executor["dtype"],
			resolution=int(generation["resolution"]),
			sd35_steps=int(generation["steps"]),
			sd35_guidance_scale=float(generation["guidance_scale"]),
			sd35_max_sequence_length=int(generation["max_sequence_length"]),
		)
		client = SD35TurboClient(runtime_args)
	counts = {"ok": 0, "error": 0}
	try:
		for index, job in enumerate(pending, start=1):
			archive_existing(_path(root, str(job["result_path"])))
			result = _write_result(root, job, client, family, runtime_args)
			counts[result["status"]] += 1
			print({"progress": f"{index}/{len(pending)}", "job_id": job["job_id"], "status": result["status"]}, flush=True)
	finally:
		client.close()
	print(counts)


if __name__ == "__main__":
	main()
