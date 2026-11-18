"""GPU executor for frozen bootstrap initial-generation jobs."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Mapping

from .bootstrap import (
	INITIAL_STAGE,
	_relative,
	validate_initial_job_contract,
	validate_initial_result,
	validate_initial_prompt_mode_manifest,
	validate_prompt_row_contract,
	validate_source_prompts_snapshot,
)
from .canonical import archive_existing, iter_jsonl, read_json, sha256_file, sha256_json, write_json_atomic
from .execute_flux2 import _resolve_pinned_model
from .planner import load_config
from .schema import DataValidationError


def _valid_completed(root: Path, job: Mapping[str, Any]) -> bool:
	try:
		validate_initial_result(root, job)
	except Exception:
		return False
	return True


def _validate_jobs(root: Path, jobs: list[dict[str, Any]], config: Any) -> None:
	"""Validate jobs and reject contract violations."""
	if len({job.get("job_id") for job in jobs}) != len(jobs):
		raise DataValidationError("duplicate initial job_id")
	if len({job.get("sample_key") for job in jobs}) != len(jobs):
		raise DataValidationError("duplicate initial sample_key")
	for job in jobs:
		validate_initial_job_contract(job, config)
		_relative(root, str(job["result_path"]))
		_relative(root, str(job["output_image"]))
	manifest = read_json(root / "bootstrap_manifest.json")
	prompt_rows = list(iter_jsonl(root / "prompt_rows.jsonl"))
	if manifest.get("stage") != INITIAL_STAGE:
		raise DataValidationError("invalid bootstrap manifest stage")
	if manifest.get("policy_signature") != config.policy.signature:
		raise DataValidationError("bootstrap policy signature mismatch")
	if (
		manifest.get("flux2_policy_contract") != config.policy.collector_contract
		or manifest.get("flux2_policy_digest") != config.policy.collector_contract_digest
	):
		raise DataValidationError("bootstrap Flux.2 collector-policy lock mismatch")
	if manifest.get("config_sha256") != sha256_json(config.as_dict()):
		raise DataValidationError("bootstrap config differs from manifest")
	validate_source_prompts_snapshot(root, manifest)
	validate_initial_prompt_mode_manifest(manifest, prompt_rows)
	if manifest.get("prompt_count") != len(prompt_rows) or manifest.get("job_count") != len(jobs):
		raise DataValidationError("bootstrap manifest count mismatch")
	if manifest.get("frozen_digests", {}).get("prompts") != sha256_json(prompt_rows):
		raise DataValidationError("prompt rows differ from bootstrap manifest")
	if manifest.get("frozen_digests", {}).get("initial_jobs") != sha256_json(jobs):
		raise DataValidationError("initial jobs differ from bootstrap manifest")
	for row in prompt_rows:
		validate_prompt_row_contract(row)
	if len({row.get("sample_key") for row in prompt_rows}) != len(prompt_rows):
		raise DataValidationError("duplicate prompt sample_key")
	if {row.get("sample_key") for row in prompt_rows} != {job.get("sample_key") for job in jobs}:
		raise DataValidationError("prompt rows and initial jobs do not contain the same sample_key set")


def build_parser() -> argparse.ArgumentParser:
	"""Construct the CLI parser for resumable FLUX.2 initial generation."""
	parser = argparse.ArgumentParser(description="Execute pinned Flux.2 initial-image jobs")
	parser.add_argument("--bootstrap-root", type=Path, required=True)
	parser.add_argument("--device", default="cuda:0")
	parser.add_argument("--batch-size", type=int, default=2)
	parser.add_argument("--start", type=int, default=0)
	parser.add_argument("--limit", type=int)
	parser.add_argument("--retry-errors", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	return parser


def main(argv: list[str] | None = None) -> None:
	"""Run this module's command-line workflow and return its process status."""
	args = build_parser().parse_args(argv)
	root = args.bootstrap_root.resolve()
	config = load_config(root / "bootstrap_config.lock.json", strict_revision=True)
	jobs = list(iter_jsonl(root / "initial_jobs.jsonl"))
	_validate_jobs(root, jobs, config)
	selected = jobs[max(0, args.start) :]
	if args.limit is not None:
		selected = selected[: max(0, args.limit)]
	pending: list[dict[str, Any]] = []
	for job in selected:
		if _valid_completed(root, job):
			continue
		result_path = _relative(root, str(job["result_path"]))
		if result_path.is_file():
			try:
				existing = read_json(result_path)
			except Exception as exc:
				raise DataValidationError(f"corrupt initial result {result_path}: {exc}") from exc
			identity_ok = (
				isinstance(existing, Mapping)
				and existing.get("job_id") == job["job_id"]
				and existing.get("request_hash") == job["request_hash"]
				and existing.get("policy_signature") == job["policy_signature"]
				and existing.get("flux2_policy_digest") == job["flux2_policy_digest"]
			)
			if not identity_ok:
				raise DataValidationError(f"stale initial result: {result_path}")
			if existing.get("status") != "error":
				raise DataValidationError(f"invalid initial output/result pair: {result_path}")
			if not args.retry_errors:
				continue
		pending.append(job)
	print({"selected": len(selected), "pending": len(pending), "dry_run": args.dry_run})
	if args.dry_run or not pending:
		return

	import torch
	from inference.main_method.modules.flux_service import FluxService

	model_path = _resolve_pinned_model(config.policy.model_id, config.policy.model_revision)
	dtype = torch.bfloat16 if config.policy.dtype == "bfloat16" else torch.float16
	service = FluxService(str(model_path), args.device, dtype)
	counts = {"ok": 0, "error": 0}
	step = max(1, int(args.batch_size))
	for start in range(0, len(pending), step):
		chunk = pending[start : start + step]
		for job in chunk:
			archive_existing(_relative(root, str(job["result_path"])))
		started = time.monotonic()
		policy = chunk[0]["policy"]
		items = [
			{
				"prompt": job["generation_prompt"],
				"output_image": str(_relative(root, str(job["output_image"]))),
				"seed": job["seed"],
			}
			for job in chunk
		]
		outputs: list[Path | Exception]
		try:
			outputs = list(
				service.generate_batch(
					{
						"items": items,
						"resolution": policy["resolution"],
						"steps": policy["steps"],
						"guidance_scale": policy["guidance_scale"],
						"max_sequence_length": policy["max_sequence_length"],
					}
				)
			)
			if len(outputs) != len(chunk):
				raise RuntimeError(
					f"Flux.2 initial batch output mismatch: expected={len(chunk)} actual={len(outputs)}"
				)
		except Exception:
			# Text-only scalar fallback cannot mix image conditions across samples.
			outputs = []
			for job, item in zip(chunk, items):
				try:
					outputs.append(
						Path(service.generate({**item, "resolution": policy["resolution"], "steps": policy["steps"], "guidance_scale": policy["guidance_scale"], "max_sequence_length": policy["max_sequence_length"]}))
					)
				except Exception as exc:
					outputs.append(exc)
		for job, output in zip(chunk, outputs):
			base = {
				"schema_version": job["schema_version"],
				"stage": INITIAL_STAGE,
				"job_id": job["job_id"],
				"request_hash": job["request_hash"],
				"policy_signature": job["policy_signature"],
				"flux2_policy_digest": job["flux2_policy_digest"],
				"executor_family": "flux2",
				"model_id": job["policy"]["model_id"],
				"model_revision": job["policy"]["model_revision"],
				"seed": job["seed"],
			}
			if isinstance(output, Exception):
				payload = {**base, "status": "error", "error_type": type(output).__name__, "error_message": str(output), "duration_seconds": time.monotonic() - started}
			else:
				payload = {**base, "status": "ok", "output_image": job["output_image"], "output_sha256": sha256_file(Path(output)), "call_count": 1, "duration_seconds": time.monotonic() - started}
			write_json_atomic(_relative(root, str(job["result_path"])), payload)
			counts[payload["status"]] += 1
		print({"progress": f"{min(start + len(chunk), len(pending))}/{len(pending)}", **counts}, flush=True)
	print(counts)


if __name__ == "__main__":
	main()
