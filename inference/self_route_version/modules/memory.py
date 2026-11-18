"""GPU, memory, and model-path helpers for MLIE-CoT."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Sequence

from .vcot_branch import HeavyRuntimeManager

from .logging_utils import log_stage

def force_cuda_gc() -> None:
	"""Best-effort Python and CUDA cache collection across visible devices."""
	gc.collect()
	try:
		import torch

		if torch.cuda.is_available():
			for device_index in range(torch.cuda.device_count()):
				with torch.cuda.device(device_index):
					torch.cuda.empty_cache()
					torch.cuda.ipc_collect()
			torch.cuda.synchronize()
	except Exception:
		pass
	gc.collect()


def release_qwen_hard(runtime: HeavyRuntimeManager) -> None:
	"""Drop the lazy Qwen runtime and aggressively return cached CUDA memory."""
	runtime.release_qwen()
	force_cuda_gc()


def resolve_hf_snapshot(path_or_repo: str) -> str:
	"""Resolve a local path or cached Hugging Face repo without downloading."""
	candidate = Path(path_or_repo)
	if candidate.exists():
		return str(candidate)

	roots = [
		Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub",
		Path.home() / ".cache" / "huggingface" / "hub",
	]
	cache_name = "models--" + path_or_repo.replace("/", "--")
	for root in roots:
		base = root / cache_name
		if not base.exists():
			continue
		snaps = base / "snapshots"
		refs = base / "refs" / "main"
		if refs.exists() and snaps.exists():
			sha = refs.read_text(encoding="utf-8").strip()
			resolved = snaps / sha
			if resolved.exists():
				return str(resolved)
		if (snaps / "main").exists():
			return str(snaps / "main")
		if snaps.exists():
			dirs = sorted([item for item in snaps.iterdir() if item.is_dir()])
			if dirs:
				return str(dirs[-1])
	raise FileNotFoundError(f"Cannot resolve local Hugging Face snapshot: {path_or_repo}")


def resolve_qwen_model_path(path_or_repo: str) -> str:
	"""Resolve Qwen paths while tolerating known dot/underscore cache aliases."""
	candidate = Path(path_or_repo)
	if candidate.exists():
		return str(candidate)

	aliases = [path_or_repo]
	if path_or_repo.endswith("Qwen3___5-122B-A10B"):
		aliases.append("Qwen/Qwen3.5-122B-A10B")
	if path_or_repo.endswith("Qwen3.5-122B-A10B"):
		aliases.append("Qwen/Qwen3___5-122B-A10B")
	tried: List[str] = []
	for alias in aliases:
		if alias in tried:
			continue
		tried.append(alias)
		alias_path = Path(alias)
		if alias_path.exists():
			return str(alias_path)
		try:
			return resolve_hf_snapshot(alias)
		except Exception:
			continue
	raise FileNotFoundError(f"Cannot resolve Qwen model path from: {path_or_repo}")


def query_gpu_free_gib() -> Dict[int, float]:
	"""Return physical GPU free memory in GiB, or an empty mapping on failure."""
	try:
		out = subprocess.check_output(
			["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
			text=True,
			timeout=10,
		)
	except Exception:
		return {}
	result: Dict[int, float] = {}
	for line in out.strip().splitlines():
		parts = [item.strip() for item in line.split(",")]
		if len(parts) != 2:
			continue
		try:
			result[int(parts[0])] = float(parts[1]) / 1024.0
		except Exception:
			continue
	return result


def parse_gpu_indices(text: str) -> List[int]:
	"""Parse a comma-separated GPU list and reject an empty specification."""
	items: List[int] = []
	for chunk in text.split(","):
		chunk = chunk.strip()
		if not chunk:
			continue
		items.append(int(chunk))
	if not items:
		raise ValueError("gpu list is empty")
	return items


def infer_visible_gpu_indices(limit: int) -> List[int]:
	"""Return process-visible CUDA indices, capped by the requested worker count."""
	visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
	if visible.strip():
		count = len([chunk for chunk in visible.split(",") if chunk.strip()])
		return list(range(min(max(1, limit), count)))
	return list(range(max(1, limit)))


def resolve_parallel_service_gpu_indices(spec: str, fallback_index: int, max_workers: int) -> List[int]:
	"""Resolve explicit or inferred device indices for a service worker pool."""
	if spec.strip():
		items = parse_gpu_indices(spec)
	else:
		items = infer_visible_gpu_indices(max_workers)
	if not items:
		return [fallback_index]
	return items[: max(1, max_workers)]


def memory_retry_allowed(retry_count: int, max_retries: int) -> bool:
	"""Interpret non-positive retry limits as unbounded operational waiting."""
	return max_retries <= 0 or retry_count < max_retries


def wait_for_gpu_free_gib(
	gpu_index: int,
	min_free_gib: float,
	stage_name: str,
	wait_sec: float,
	max_wait_sec: float = 0.0,
) -> bool:
	"""Block until one GPU reaches a free-memory threshold or timeout.

	Returns ``False`` only on a configured timeout.  Periodic stage logs make
	long waits observable without flooding the terminal.
	"""
	if min_free_gib <= 0:
		return True
	wait_sec = max(1.0, float(wait_sec))
	started_at = time.time()
	last_log = 0.0
	while True:
		free_by_gpu = query_gpu_free_gib()
		free_gib = free_by_gpu.get(gpu_index, 0.0)
		if free_gib >= min_free_gib:
			return True
		elapsed = time.time() - started_at
		if max_wait_sec > 0 and elapsed >= max_wait_sec:
			return False
		now = time.time()
		if now - last_log >= max(10.0, wait_sec):
			log_stage(
				f"{stage_name} memory-wait gpu={gpu_index} free_gib={free_gib:.1f} "
				f"need_gib={min_free_gib:.1f} wait_sec={wait_sec:g}"
			)
			last_log = now
		time.sleep(wait_sec)


def select_ready_gpu_indices(
	gpu_indices: Sequence[int],
	min_free_gib: float,
	stage_name: str,
	wait_sec: float,
	max_wait_sec: float = 0.0,
) -> List[int]:
	"""Wait for and rank currently ready GPUs by free memory.

	On timeout the highest-memory requested GPU is returned so the caller can
	attempt initialization and obtain the model's authoritative error.
	"""
	items = list(gpu_indices)
	if min_free_gib <= 0:
		return items
	if not items:
		return []
	wait_sec = max(1.0, float(wait_sec))
	started_at = time.time()
	last_log = 0.0
	while True:
		free_by_gpu = query_gpu_free_gib()
		ready = [gpu for gpu in items if free_by_gpu.get(gpu, 0.0) >= min_free_gib]
		if ready:
			ready.sort(key=lambda gpu: (-free_by_gpu.get(gpu, 0.0), gpu))
			return ready
		elapsed = time.time() - started_at
		if max_wait_sec > 0 and elapsed >= max_wait_sec:
			best = max(items, key=lambda gpu: free_by_gpu.get(gpu, 0.0))
			log_stage(
				f"{stage_name} memory-wait timeout; using gpu={best} "
				f"free_gib={free_by_gpu.get(best, 0.0):.1f} need_gib={min_free_gib:.1f}"
			)
			return [best]
		now = time.time()
		if now - last_log >= max(10.0, wait_sec):
			log_stage(
				f"{stage_name} memory-wait no-ready-gpu requested={items} "
				f"free_gib={free_by_gpu} need_gib={min_free_gib:.1f} wait_sec={wait_sec:g}"
			)
			last_log = now
		time.sleep(wait_sec)


def resolve_local_edit_gpu_indices(spec: str, min_free_gib: float) -> List[int]:
	"""Resolve editor GPUs, selecting all eligible devices in ``auto`` mode."""
	if spec.strip().lower() != "auto":
		return parse_gpu_indices(spec)

	free_by_gpu = query_gpu_free_gib()
	if not free_by_gpu:
		return [0]

	eligible = sorted(
		[gpu for gpu, free in free_by_gpu.items() if free >= min_free_gib],
		key=lambda gpu: (-free_by_gpu[gpu], gpu),
	)
	if eligible:
		return eligible

	best_gpu = max(free_by_gpu.items(), key=lambda item: item[1])[0]
	return [best_gpu]


def is_memory_related_text(text: str) -> bool:
	"""Recognize CUDA/HIP allocation and illegal-address failure signatures."""
	value = str(text).lower()
	patterns = (
		"outofmemoryerror",
		"cuda out of memory",
		"torch.outofmemoryerror",
		"illegal memory access",
		"cuda error: an illegal memory access was encountered",
		"cudaerrorillegaladdress",
		"cublas_status_alloc_failed",
		"dispatched on the cpu or the disk",
		"enough gpu ram",
		"no gpu has enough free memory",
		"hip out of memory",
	)
	return any(pattern in value for pattern in patterns)


def is_memory_related_exception(exc: Exception) -> bool:
	"""Classify an exception using both its representation and message."""
	return is_memory_related_text(repr(exc)) or is_memory_related_text(str(exc))


def is_qwen_load_memory_wait_exception(exc: Exception) -> bool:
	"""Identify Qwen placement failures that may resolve when GPUs become free."""
	value = f"{type(exc).__name__} {exc!s}".lower()
	return (
		"no gpu has enough free memory" in value
		or "dispatched on the cpu or the disk" in value
		or "enough gpu ram" in value
	)


def next_retry_max_side(current_side: int, min_side: int, scale: float) -> int:
	"""Compute a strictly smaller OOM-retry image side above a fixed floor."""
	if current_side <= min_side:
		return current_side
	next_side = int(current_side * scale)
	if next_side >= current_side:
		next_side = current_side - 1
	return max(min_side, next_side)
