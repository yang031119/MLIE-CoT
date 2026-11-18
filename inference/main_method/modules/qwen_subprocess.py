"""Serialize Qwen batches into short-lived, restartable worker processes.

PIL objects cannot be passed reliably across process boundaries, so each job
directory contains message JSON and PNG copies addressed by stable original
item ids.  Workers rewrite ``output.json`` after partial progress.  The parent
polls that file, delivers each completed id at most once, and sends only
unresolved ids through fallback GPU-group strategies.

Successful temporary job directories are removed unless audit retention is
requested.  Failed directories and stdout logs are preserved for diagnosis.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image


MODULE_DIR = Path(__file__).resolve().parent
WORKER_PATH = MODULE_DIR / "qwen_worker.py"


class QwenWorkerError(RuntimeError):
	"""Raised when the Qwen worker process fails."""


def _serialize_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Replace in-memory image objects with positional indices for job JSON."""
	serialized: List[Dict[str, Any]] = []
	image_index = 0
	for message in messages:
		content: List[Dict[str, Any]] = []
		for part in message.get("content", []):
			if part.get("type") == "image":
				content.append({"type": "image", "image_index": image_index})
				image_index += 1
			else:
				clean = dict(part)
				clean.pop("image", None)
				content.append(clean)
		serialized.append({"role": message.get("role", "user"), "content": content})
	return serialized


def _save_images(images: Sequence[Image.Image], item_dir: Path) -> List[str]:
	"""Materialize ordered RGB PNG inputs and return their absolute job paths."""
	item_dir.mkdir(parents=True, exist_ok=True)
	paths: List[str] = []
	for index, image in enumerate(images):
		path = item_dir / f"image_{index:02d}.png"
		image.convert("RGB").save(path)
		paths.append(str(path))
	return paths


def _make_job_dir(output_dir: Path, batch_kind: str) -> Path:
	"""Create a collision-safe job directory below the run output."""
	root = output_dir / "_qwen_jobs"
	root.mkdir(parents=True, exist_ok=True)
	stamp = f"{int(time.time())}_{os.getpid()}_{batch_kind}"
	job_dir = root / stamp
	suffix = 1
	while job_dir.exists():
		suffix += 1
		job_dir = root / f"{stamp}_{suffix}"
	job_dir.mkdir(parents=True)
	return job_dir


def _load_worker_outputs(output_path: Path) -> Dict[int, str]:
	"""Best-effort read currently committed worker outputs keyed by original id."""
	if not output_path.exists():
		return {}
	try:
		result = json.loads(output_path.read_text(encoding="utf-8"))
	except Exception:
		return {}
	raw_by_id: Dict[int, str] = {}
	for item in result.get("outputs", []):
		try:
			raw_by_id[int(item["id"])] = str(item.get("raw", ""))
		except Exception:
			continue
	return raw_by_id


def _read_text_if_exists(path: Path) -> str:
	"""Read a diagnostic log without allowing I/O/encoding errors to mask failure."""
	try:
		return path.read_text(encoding="utf-8", errors="ignore")
	except Exception:
		return ""


def _parse_gpu_groups(value: Any) -> List[str]:
	"""Parse semicolon-separated ``CUDA_VISIBLE_DEVICES`` group specifications."""
	text = str(value or "").strip()
	if not text:
		return []
	groups: List[str] = []
	for raw_group in text.split(";"):
		parts = [part.strip() for part in raw_group.split(",") if part.strip()]
		if parts:
			groups.append(",".join(parts))
	return groups


def _split_indices(indices: Sequence[int], group_count: int) -> List[List[int]]:
	"""Round-robin item ids across worker groups while retaining original ids."""
	chunks: List[List[int]] = [[] for _ in range(max(1, group_count))]
	for offset, index in enumerate(indices):
		chunks[offset % len(chunks)].append(index)
	return chunks


def _ingest_worker_outputs(
	output_path: Path,
	remaining_indices: Sequence[int],
	raw_by_id: Dict[int, str],
	delivered_ids: Set[int],
	partial_callback: Optional[Callable[[int, str], None]],
) -> int:
	"""Merge newly committed outputs and invoke each partial callback once."""
	before_count = len(raw_by_id)
	remaining = set(remaining_indices)
	partial_by_id = _load_worker_outputs(output_path)
	for item_id, raw in partial_by_id.items():
		if item_id not in remaining:
			continue
		if item_id not in raw_by_id:
			raw_by_id[item_id] = raw
		if partial_callback is not None and item_id not in delivered_ids:
			partial_callback(item_id, raw)
			delivered_ids.add(item_id)
	return len(raw_by_id) - before_count


def _write_worker_job(
	prepared: Sequence[Dict[str, Any]],
	remaining_indices: Sequence[int],
	args: Any,
	max_new_tokens: int,
	batch_kind: str,
	batch_size: int,
) -> Tuple[Path, Path, Path]:
	"""Write one self-contained worker job for unresolved original item ids."""
	job_dir = _make_job_dir(Path(args.output_dir), batch_kind)
	items: List[Dict[str, Any]] = []
	for original_index in remaining_indices:
		item = prepared[original_index]
		item_dir = job_dir / f"item_{original_index:03d}"
		image_paths = _save_images(item.get("images", []), item_dir)
		route_image_path = ""
		if item.get("route_image") is not None:
			route_image_path = _save_images([item["route_image"]], item_dir / "route")[0]
		items.append(
			{
				"id": original_index,
				"messages": _serialize_messages(item["messages"]),
				"image_paths": image_paths,
				"route_image_path": route_image_path,
				"route_prompt": str(item.get("route_prompt") or ""),
			}
		)
	job = {
		"qwen_args": {
			"model": str(args.qwen_model),
			"dtype": str(args.qwen_dtype),
			"quantization": str(args.qwen_quantization),
			"max_memory_per_gpu": str(args.qwen_max_memory_per_gpu),
			"min_free_gib": float(args.qwen_min_free_gib),
			"allow_cpu_offload": bool(getattr(args, "qwen_allow_cpu_offload", False)),
			"route_head": str(getattr(args, "qwen_route_head", "") or ""),
		},
		"max_new_tokens": int(max_new_tokens),
		"batch_size": max(1, int(batch_size)),
		"items": items,
	}
	job_path = job_dir / "job.json"
	output_path = job_dir / "output.json"
	job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
	return job_dir, job_path, output_path


def _launch_worker_job(
	prepared: Sequence[Dict[str, Any]],
	indices: Sequence[int],
	args: Any,
	max_new_tokens: int,
	batch_kind: str,
	batch_size: int,
	cuda_visible_devices: str,
) -> Dict[str, Any]:
	"""Start one isolated worker with an optional device-visibility override."""
	job_dir, job_path, output_path = _write_worker_job(
		prepared,
		indices,
		args,
		max_new_tokens,
		batch_kind,
		batch_size,
	)
	cmd = [sys.executable, str(WORKER_PATH), "--job", str(job_path), "--output", str(output_path)]
	stdout_path = job_dir / "worker_stdout.log"
	env = os.environ.copy()
	if cuda_visible_devices:
		env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
	stdout_handle = stdout_path.open("w", encoding="utf-8", errors="ignore")
	try:
		proc = subprocess.Popen(
			cmd,
			text=True,
			stdout=stdout_handle,
			stderr=subprocess.STDOUT,
			env=env,
		)
	except Exception:
		stdout_handle.close()
		raise
	return {
		"job_dir": job_dir,
		"output_path": output_path,
		"stdout_path": stdout_path,
		"stdout_handle": stdout_handle,
		"proc": proc,
		"indices": list(indices),
		"cuda_visible_devices": cuda_visible_devices,
	}


def _terminate_worker_jobs(jobs: Sequence[Dict[str, Any]]) -> None:
	"""Terminate, then kill if necessary, all live jobs and close log handles."""
	for job in jobs:
		proc = job.get("proc")
		if proc is None or proc.poll() is not None:
			continue
		try:
			proc.terminate()
			proc.wait(timeout=10)
		except Exception:
			try:
				proc.kill()
				proc.wait(timeout=10)
			except Exception:
				pass
	for job in jobs:
		handle = job.get("stdout_handle")
		try:
			if handle is not None:
				handle.close()
		except Exception:
			pass


def _wait_worker_jobs(
	jobs: Sequence[Dict[str, Any]],
	raw_by_id: Dict[int, str],
	delivered_ids: Set[int],
	partial_callback: Optional[Callable[[int, str], None]],
) -> None:
	"""Poll partial outputs until workers exit, then print their captured logs."""
	try:
		while any(job["proc"].poll() is None for job in jobs):
			for job in jobs:
				_ingest_worker_outputs(
					job["output_path"],
					job["indices"],
					raw_by_id,
					delivered_ids,
					partial_callback,
				)
			time.sleep(5.0)
	except Exception:
		_terminate_worker_jobs(jobs)
		raise
	finally:
		for job in jobs:
			handle = job.get("stdout_handle")
			try:
				if handle is not None:
					handle.close()
			except Exception:
				pass
	for job in jobs:
		_ingest_worker_outputs(job["output_path"], job["indices"], raw_by_id, delivered_ids, partial_callback)
		proc_stdout = _read_text_if_exists(job["stdout_path"])
		if proc_stdout:
			group = job["cuda_visible_devices"] or "all"
			print(
				f"qwen-worker stdout group={group} job_dir={job['job_dir']}",
				flush=True,
			)
			print(proc_stdout, end="" if proc_stdout.endswith("\n") else "\n", flush=True)


def _run_worker_strategy(
	prepared: Sequence[Dict[str, Any]],
	args: Any,
	max_new_tokens: int,
	batch_size: int,
	batch_kind: str,
	remaining_indices: Sequence[int],
	groups: Sequence[str],
	raw_by_id: Dict[int, str],
	delivered_ids: Set[int],
	partial_callback: Optional[Callable[[int, str], None]],
	strategy_name: str,
) -> Tuple[List[Dict[str, Any]], int]:
	"""Run one GPU grouping strategy and describe incomplete worker shards."""
	target_indices = [index for index in remaining_indices if index not in raw_by_id]
	if not target_indices:
		return [], 0
	active_groups = list(groups) or [""]
	splits = _split_indices(target_indices, len(active_groups))
	jobs: List[Dict[str, Any]] = []
	for group, indices in zip(active_groups, splits):
		if not indices:
			continue
		jobs.append(
			_launch_worker_job(
				prepared,
				indices,
				args,
				max_new_tokens,
				batch_kind,
				batch_size,
				group,
			)
		)
	before_count = len(raw_by_id)
	print(
		f"qwen-worker strategy={strategy_name} groups={active_groups} "
		f"items={len(target_indices)} workers={len(jobs)} batch_size={batch_size}",
		flush=True,
	)
	_wait_worker_jobs(jobs, raw_by_id, delivered_ids, partial_callback)
	failures: List[Dict[str, Any]] = []
	for job in jobs:
		missing = [index for index in job["indices"] if index not in raw_by_id]
		if job["proc"].returncode != 0 or missing:
			failures.append(
				{
					"job_dir": job["job_dir"],
					"group": job["cuda_visible_devices"] or "all",
					"exit_code": job["proc"].returncode,
					"missing": missing,
					"stdout_tail": _read_text_if_exists(job["stdout_path"])[-4000:],
				}
			)
			continue
		if not bool(getattr(args, "qwen_keep_job_files", False)):
			shutil.rmtree(job["job_dir"], ignore_errors=True)
	return failures, len(raw_by_id) - before_count


def run_qwen_worker_batch(
	prepared: Sequence[Dict[str, Any]],
	args: Any,
	max_new_tokens: int,
	batch_size: int = 1,
	partial_callback: Optional[Callable[[int, str], None]] = None,
) -> List[str]:
	"""Resolve a batch through primary, fallback, and default worker strategies.

	Already recovered ids are excluded from every later strategy.  The returned
	list is restored to input order; exhaustion raises ``QwenWorkerError`` with
	the missing ids and retained job/log locations.
	"""
	if not prepared:
		return []
	batch_kind = str(prepared[0].get("batch_kind", "qwen"))
	raw_by_id: Dict[int, str] = {}
	delivered_ids: Set[int] = set()
	remaining_indices = list(range(len(prepared)))
	primary_groups = _parse_gpu_groups(getattr(args, "qwen_worker_gpu_groups", ""))
	fallback_groups = _parse_gpu_groups(getattr(args, "qwen_worker_fallback_gpu_groups", ""))
	strategies: List[Tuple[str, List[str]]] = []
	if primary_groups:
		strategies.append(("primary", primary_groups))
	if fallback_groups and fallback_groups != primary_groups:
		strategies.append(("fallback", fallback_groups))
	strategies.append(("default", [""]))
	last_failures: List[Dict[str, Any]] = []
	for strategy_name, groups in strategies:
		remaining_indices = [index for index in remaining_indices if index not in raw_by_id]
		if not remaining_indices:
			break
		failures, recovered = _run_worker_strategy(
			prepared,
			args,
			max_new_tokens,
			batch_size,
			batch_kind,
			remaining_indices,
			groups,
			raw_by_id,
			delivered_ids,
			partial_callback,
			strategy_name,
		)
		remaining_indices = [index for index in remaining_indices if index not in raw_by_id]
		if not failures and not remaining_indices:
			break
		if failures:
			last_failures = failures
			print(
				f"qwen-worker strategy={strategy_name} incomplete "
				f"recovered={recovered} remaining={len(remaining_indices)}",
				flush=True,
			)
			continue
		if remaining_indices:
			last_failures = [
				{
					"job_dir": "",
					"group": ",".join(groups),
					"exit_code": 0,
					"missing": remaining_indices,
					"stdout_tail": "",
				}
			]
	if remaining_indices:
		failure_text = json.dumps(last_failures, ensure_ascii=False, default=str, indent=2)
		raise QwenWorkerError(
			f"Qwen worker strategies exhausted; recovered={len(raw_by_id)}/{len(prepared)} "
			f"missing={remaining_indices[:20]} total_missing={len(remaining_indices)}\n{failure_text}"
		)
	if len(raw_by_id) != len(prepared):
		missing = [index for index in range(len(prepared)) if index not in raw_by_id]
		raise QwenWorkerError(f"Qwen worker output count mismatch; missing={missing[:20]}")
	return [raw_by_id[index] for index in range(len(prepared))]
