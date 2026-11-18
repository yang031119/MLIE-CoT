"""Threaded per-GPU worker pool helpers."""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, List, Sequence

from .io_utils import mark_state_error
from .logging_utils import describe_states_brief, log_stage
from .memory import is_memory_related_exception, memory_retry_allowed, wait_for_gpu_free_gib
from .state import SampleState

def clone_namespace(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
	"""Copy parsed arguments and apply worker-local overrides."""
	payload = dict(vars(args))
	payload.update(overrides)
	return argparse.Namespace(**payload)


def partition_states_round_robin(states: Sequence[SampleState], worker_count: int) -> List[List[SampleState]]:
	"""Distribute states deterministically while preserving order per worker."""
	partitions: List[List[SampleState]] = [[] for _ in range(max(1, worker_count))]
	for idx, state in enumerate(states):
		partitions[idx % len(partitions)].append(state)
	return partitions


def run_state_jobs_with_client_pool(
	states: Sequence[SampleState],
	gpu_indices: Sequence[int],
	stage_name: str,
	client_factory,
	per_state_fn,
	min_free_gib: float = 0.0,
	memory_wait_sec: float = 20.0,
	max_memory_retries: int = 0,
) -> None:
	"""Process states with one lazy client per GPU.

	Client construction and per-state requests have independent memory-retry
	counters.  A retry closes the old client before waiting, preventing leaked
	GPU allocations from making the next attempt deterministically fail.
	"""
	if not states:
		return
	if not gpu_indices:
		raise ValueError(f"{stage_name} requires at least one GPU index")
	partitions = partition_states_round_robin(states, len(gpu_indices))

	def worker_fn(worker_slot: int, gpu_index: int, worker_states: Sequence[SampleState]) -> None:
		"""Own one GPU/client and process its assigned states sequentially."""
		if not worker_states:
			return
		log_stage(
			f"{stage_name} worker_slot={worker_slot} gpu={gpu_index} count={len(worker_states)} samples={describe_states_brief(worker_states)}"
		)
		client = None
		init_retry_count = 0

		def ensure_client():
			"""Lazily construct or memory-retry this worker's service client."""
			nonlocal client, init_retry_count
			if client is not None:
				return client
			while True:
				if not wait_for_gpu_free_gib(gpu_index, min_free_gib, f"{stage_name} worker-init", memory_wait_sec):
					raise RuntimeError(f"{stage_name} timed out waiting for gpu {gpu_index}")
				try:
					client = client_factory(gpu_index)
					return client
				except Exception as exc:
					if is_memory_related_exception(exc) and memory_retry_allowed(init_retry_count, max_memory_retries):
						init_retry_count += 1
						log_stage(
							f"{stage_name} worker-init memory-retry gpu={gpu_index} "
							f"attempt={init_retry_count} reason={type(exc).__name__}"
						)
						time.sleep(max(1.0, float(memory_wait_sec)))
						continue
					raise

		try:
			for state in worker_states:
				state_retry_count = 0
				while True:
					try:
						per_state_fn(ensure_client(), state, gpu_index)
						break
					except Exception as exc:
						if is_memory_related_exception(exc) and memory_retry_allowed(state_retry_count, max_memory_retries):
							state_retry_count += 1
							log_stage(
								f"{stage_name} memory-retry sample={state.sample_id} gpu={gpu_index} "
								f"attempt={state_retry_count} reason={type(exc).__name__}"
							)
							with contextlib.suppress(Exception):
								if client is not None:
									client.close()
							client = None
							time.sleep(max(1.0, float(memory_wait_sec)))
							continue
						mark_state_error(state, exc, f"{stage_name}_error")
						break
		except Exception as exc:
			for state in worker_states:
				if state.done or state.error is not None:
					continue
				mark_state_error(state, exc, f"{stage_name}_worker_init_error")
		finally:
			with contextlib.suppress(Exception):
				if client is not None:
					client.close()

	with ThreadPoolExecutor(max_workers=len(gpu_indices)) as executor:
		futures = [
			executor.submit(worker_fn, slot, gpu_index, worker_states)
			for slot, (gpu_index, worker_states) in enumerate(zip(gpu_indices, partitions), start=1)
			if worker_states
		]
		for future in futures:
			future.result()


def run_state_batches_with_client_pool(
	states: Sequence[SampleState],
	gpu_indices: Sequence[int],
	stage_name: str,
	client_factory,
	per_batch_fn,
	per_state_fallback_fn,
	*,
	batch_size: int,
	min_free_gib: float = 0.0,
	memory_wait_sec: float = 20.0,
	max_memory_retries: int = 0,
) -> None:
	"""Run small per-GPU batches, falling back safely to scalar requests.

	This is deliberately separate from the general scalar pool: diffusion
	batching is an optimization, while per-state execution remains the recovery
	path for OOMs, unsupported batch shapes, and partial service failures.
	"""
	if not states:
		return
	if not gpu_indices:
		raise ValueError(f"{stage_name} requires at least one GPU index")
	batch_size = max(1, int(batch_size))
	partitions = partition_states_round_robin(states, len(gpu_indices))

	def worker_fn(worker_slot: int, gpu_index: int, worker_states: Sequence[SampleState]) -> None:
		"""Own one GPU/client and run adaptive batches for its state partition."""
		if not worker_states:
			return
		log_stage(
			f"{stage_name} worker_slot={worker_slot} gpu={gpu_index} count={len(worker_states)} "
			f"batch_size={batch_size} samples={describe_states_brief(worker_states)}"
		)
		client = None
		init_retry_count = 0

		def close_client() -> None:
			"""Close and clear the client before a memory recovery attempt."""
			nonlocal client
			with contextlib.suppress(Exception):
				if client is not None:
					client.close()
			client = None

		def ensure_client():
			"""Lazily construct the batch worker's service client."""
			nonlocal client, init_retry_count
			if client is not None:
				return client
			while True:
				if not wait_for_gpu_free_gib(gpu_index, min_free_gib, f"{stage_name} worker-init", memory_wait_sec):
					raise RuntimeError(f"{stage_name} timed out waiting for gpu {gpu_index}")
				try:
					client = client_factory(gpu_index)
					return client
				except Exception as exc:
					if is_memory_related_exception(exc) and memory_retry_allowed(init_retry_count, max_memory_retries):
						init_retry_count += 1
						log_stage(
							f"{stage_name} worker-init memory-retry gpu={gpu_index} "
							f"attempt={init_retry_count} reason={type(exc).__name__}"
						)
						time.sleep(max(1.0, float(memory_wait_sec)))
						continue
					raise

		def run_scalar(state: SampleState) -> None:
			"""Run one state with bounded memory retries and durable failure."""
			state_retry_count = 0
			while True:
				try:
					per_state_fallback_fn(ensure_client(), state, gpu_index)
					return
				except Exception as exc:
					if is_memory_related_exception(exc) and memory_retry_allowed(state_retry_count, max_memory_retries):
						state_retry_count += 1
						close_client()
						log_stage(
							f"{stage_name} scalar memory-retry sample={state.sample_id} gpu={gpu_index} "
							f"attempt={state_retry_count} reason={type(exc).__name__}"
						)
						time.sleep(max(1.0, float(memory_wait_sec)))
						continue
					mark_state_error(state, exc, f"{stage_name}_scalar_fallback_error")
					return

		def run_adaptive(batch: Sequence[SampleState]) -> None:
			"""Bisect a failed batch until the largest fitting size is found."""
			if len(batch) == 1:
				run_scalar(batch[0])
				return
			try:
				per_batch_fn(ensure_client(), batch, gpu_index)
				return
			except Exception as exc:
				left_size = max(1, len(batch) // 2)
				log_stage(
					f"{stage_name} batch-bisect gpu={gpu_index} count={len(batch)} "
					f"next={left_size}+{len(batch) - left_size} reason={type(exc).__name__}"
				)
				close_client()
				run_adaptive(batch[:left_size])
				run_adaptive(batch[left_size:])

		try:
			for start in range(0, len(worker_states), batch_size):
				run_adaptive(list(worker_states[start : start + batch_size]))
		except Exception as exc:
			for state in worker_states:
				if state.done or state.error is not None or state.current_image is not None:
					continue
				mark_state_error(state, exc, f"{stage_name}_worker_init_error")
		finally:
			close_client()

	with ThreadPoolExecutor(max_workers=len(gpu_indices)) as executor:
		futures = [
			executor.submit(worker_fn, slot, gpu_index, worker_states)
			for slot, (gpu_index, worker_states) in enumerate(zip(gpu_indices, partitions), start=1)
			if worker_states
		]
		for future in futures:
			future.result()
