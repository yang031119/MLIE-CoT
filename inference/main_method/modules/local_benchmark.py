"""Bridge routed L1 plans to the isolated local-edit executor.

V10 L1 is one joint FLUX.2 call with reference order
``full image, crop 1, ..., crop N``.  This module scales the route-time plan to
the generator resolution, derives a stable action seed, writes a self-contained
benchmark task, invokes the separate executor, and stores exactly one candidate
for blind retain review.

The batch cache is content-addressed in practice: reuse requires identical
``input.png`` and ``edit.json``, not merely the same sample id.  Partial cache
hits seed the executor's ordinary resume layout so already completed samples
are not regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

from .config import LOCAL_BENCHMARK_SCRIPT, LOCAL_MODEL_ORDER, LOCAL_REPAIR_MODEL_ORDER, OBJECT_REPAIR_MODEL_ORDER
from .contracts import build_level_action_plan
from .image_utils import copy_image_generation_sized, generation_resolution, scale_quad_points
from .io_utils import append_event, mark_state_error, save_json, save_status
from .logging_utils import describe_states_brief, log_stage
from .memory import (
	is_memory_related_text,
	memory_retry_allowed,
	query_gpu_free_gib,
	resolve_local_edit_gpu_indices,
)
from .local_edit_worker import (
	L1_JOINT_ACTION_SCHEMA,
	l1_reference_roles,
	stable_l1_action_seed,
)
from .state import SampleState, sample_round_dir


def _canonical_joint_edit(edit: Dict[str, Any], index: int) -> Dict[str, Any]:
	"""Validate and normalize one ordered region in a joint L1 request."""
	edit_prompt = str(edit.get("edit_prompt") or edit.get("instruction") or edit.get("whole_edit") or edit.get("issue") or "").strip()
	if not edit_prompt:
		raise ValueError(f"Joint L1 edit {index} has an empty instruction")
	quad_points = edit.get("quad_points") or []
	if len(quad_points) != 4:
		raise ValueError(f"Joint L1 edit {index} must have exactly four quad_points")
	return {
		"id": str(edit.get("id") or f"edit_{index:02d}").strip(),
		"target_entity": str(edit.get("target_entity") or edit.get("target_object") or "localized region").strip(),
		"target_object": str(edit.get("target_object") or edit.get("target_entity") or "localized region").strip(),
		"issue": str(edit.get("issue") or "diagnosed mismatch").strip(),
		"instruction": edit_prompt,
		"edit_prompt": edit_prompt,
		"whole_edit": str(edit.get("whole_edit") or edit_prompt).strip(),
		"quad_points": [[int(point[0]), int(point[1])] for point in quad_points],
	}


def _fallback_joint_diagnosis(edits: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
	"""Build minimal factual context when an old artifact lacks diagnosis JSON."""
	issues = " | ".join(str(edit.get("issue") or "diagnosed mismatch").strip() for edit in edits)
	fixes = " | ".join(str(edit.get("instruction") or edit.get("edit_prompt") or "correct it").strip() for edit in edits)
	return {
		"fully_satisfied": False,
		"reason": "Localized mismatches require a joint L1 edit.",
		"issue": issues,
		"fix": fixes,
	}


def _joint_action_context(
	state: SampleState,
	attempt_dir: Path,
	edits: Sequence[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
	"""Recover the route-time diagnosis so execution uses the shared action contract."""
	target_prompt = str(getattr(state, "prompt", "") or "").strip()
	diagnosis: Dict[str, Any] | None = None

	def inspect(payload: Any) -> None:
		"""Extract the newest target prompt and canonical diagnosis if present."""
		nonlocal target_prompt, diagnosis
		if not isinstance(payload, dict):
			return
		for candidate in (
			payload,
			payload.get("action_plan"),
			payload.get("analysis"),
			(payload.get("analysis") or {}).get("action_plan") if isinstance(payload.get("analysis"), dict) else None,
		):
			if not isinstance(candidate, dict):
				continue
			target_prompt = str(candidate.get("target_prompt") or target_prompt).strip()
			candidate_diagnosis = candidate.get("diagnosis") or candidate.get("canonical_diagnosis")
			if isinstance(candidate_diagnosis, dict):
				diagnosis = candidate_diagnosis

	for path in (attempt_dir / "l1_action_plan.json", attempt_dir / "repair_plan.json"):
		if not path.is_file():
			continue
		try:
			inspect(json.loads(path.read_text(encoding="utf-8")))
		except (OSError, json.JSONDecodeError):
			continue
	for event in reversed(list(getattr(state, "history", []) or [])):
		if diagnosis is not None:
			break
		inspect(event.get("analysis") if isinstance(event, dict) else None)
	return target_prompt, diagnosis or _fallback_joint_diagnosis(edits)


def build_local_benchmark_task_json(
	edits: Sequence[Dict[str, Any]] | Dict[str, Any],
	*,
	action_seed: int | None = None,
	round_index: int = 0,
	target_prompt: str = "",
	diagnosis: Dict[str, Any] | None = None,
	width: int = 1024,
	height: int = 1024,
) -> Dict[str, Any]:
	"""Build the one-call L1 task from the same contract as router-data collection."""
	if isinstance(edits, dict):
		nested = edits.get("edits")
		raw_edits = nested if isinstance(nested, list) else [edits]
	else:
		raw_edits = list(edits)
	if not 1 <= len(raw_edits) <= 3:
		raise ValueError(f"{L1_JOINT_ACTION_SCHEMA} requires 1-3 edits, got {len(raw_edits)}")
	canonical_edits = [_canonical_joint_edit(dict(edit), index) for index, edit in enumerate(raw_edits, start=1)]
	action_plan = build_level_action_plan(
		1,
		target_prompt,
		diagnosis or _fallback_joint_diagnosis(canonical_edits),
		edits=canonical_edits,
		width=int(width),
		height=int(height),
	)
	return {
		**action_plan,
		"action_schema": L1_JOINT_ACTION_SCHEMA,
		"operation": "edit_multi_reference",
		"reference_roles": l1_reference_roles(len(canonical_edits)),
		"action_seed": int(action_seed) if action_seed is not None else None,
		"round_index": int(round_index),
		"whole_edit": " ".join(edit["whole_edit"] for edit in canonical_edits),
	}


def prepare_local_benchmark_retry_inputs(
	state: SampleState,
	batch_sample_dir: Path,
	attempt_dir: Path,
	artifact_prefix: str,
	resolution: int,
	_legacy_max_local_attempts: int | None = None,
	*,
	base_seed: int = 0,
) -> Dict[str, Any]:
	"""Prepare one joint L1 action. The legacy argument is accepted but intentionally ignored."""
	assert state.current_image is not None
	raw_edits = [dict(edit) for edit in state.micro_plan]
	if not 1 <= len(raw_edits) <= 3:
		raise ValueError(f"{L1_JOINT_ACTION_SCHEMA} requires 1-3 edits, got {len(raw_edits)}")
	image = Image.open(state.current_image).convert("RGB")
	original_width, original_height = image.size
	resized = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
	scale_x = resized.width / float(original_width)
	scale_y = resized.height / float(original_height)
	scaled_edits: List[Dict[str, Any]] = []
	for index, raw_edit in enumerate(raw_edits, start=1):
		edit = dict(raw_edit)
		edit["quad_points"] = scale_quad_points(edit.get("quad_points") or [], scale_x, scale_y)
		scaled_edits.append(_canonical_joint_edit(edit, index))
	round_index = int(getattr(state, "global_round", 0))
	action_seed = stable_l1_action_seed(base_seed, state.sample_id, scaled_edits, round_index)
	target_prompt, diagnosis = _joint_action_context(state, attempt_dir, scaled_edits)
	task_payload = build_local_benchmark_task_json(
		scaled_edits,
		action_seed=action_seed,
		round_index=round_index,
		target_prompt=target_prompt,
		diagnosis=diagnosis,
		width=resized.width,
		height=resized.height,
	)
	batch_sample_dir.mkdir(parents=True, exist_ok=True)
	attempt_dir.mkdir(parents=True, exist_ok=True)
	resized.save(batch_sample_dir / "input.png")
	input_path = attempt_dir / f"{artifact_prefix}_initial.png"
	resized.save(input_path)
	save_json(batch_sample_dir / "edit.json", task_payload)
	plan_path = attempt_dir / f"{artifact_prefix}_plan.json"
	save_json(plan_path, task_payload)
	return {
		"action_schema": L1_JOINT_ACTION_SCHEMA,
		"operation": task_payload["operation"],
		"reference_roles": task_payload["reference_roles"],
		"action_seed": action_seed,
		"edits": scaled_edits,
		"edit": scaled_edits[0],
		"input_path": str(input_path),
		"plan_path": str(plan_path),
		"input_size": [resized.width, resized.height],
		"input_max_side": max(resized.size),
	}


def collect_local_benchmark_retry_text(proc_stdout: str, output_root: Path) -> str:
	"""Combine process and result summaries for robust memory-error detection."""
	parts = [proc_stdout]
	for extra_path in (output_root / "summary.jsonl", output_root / "run_summary.json"):
		if extra_path.exists():
			parts.append(extra_path.read_text(encoding="utf-8", errors="ignore"))
	return "\n".join(parts)


def clear_local_benchmark_outputs(batch_sample_dir: Path) -> None:
	"""Remove only generated model images before an exact-request retry."""
	for model_name in LOCAL_MODEL_ORDER:
		path = batch_sample_dir / f"{model_name}.png"
		if path.exists():
			path.unlink()


def local_model_order_for_edit(edit: Dict[str, Any], args: argparse.Namespace | None = None) -> List[str]:
	"""Choose the configured L1 executor order from the plan's repair scope."""
	repair_scope = str(edit.get("repair_scope") or "").strip().lower()
	if repair_scope == "object level":
		return list(OBJECT_REPAIR_MODEL_ORDER)
	return list(LOCAL_REPAIR_MODEL_ORDER)


def _joint_l1_tasks(states: Sequence[SampleState], max_actions: int, max_crops: int = 3) -> List[SampleState]:
	"""Filter ready L1 states and mark policy-invalid crop counts as errors."""
	candidates = [
		state
		for state in states
		if state.phase == "micro"
		and state.current_image
		and not state.pending_local_review
		and not state.done
		and not state.error
		and state.micro_plan
		and int(state.global_round) < int(max_actions)
	]
	tasks: List[SampleState] = []
	for state in candidates:
		if not 1 <= len(state.micro_plan) <= max_crops:
			mark_state_error(
				state,
				f"{L1_JOINT_ACTION_SCHEMA} requires 1-{max_crops} edits under the locked Flux policy, got {len(state.micro_plan)}",
				"l1_joint_invalid_plan",
			)
			continue
		tasks.append(state)
	return tasks


def run_local_edit_benchmark_batch(args: argparse.Namespace, states: Sequence[SampleState]) -> None:
	"""Group ready L1 states by executor set and dispatch joint edit batches."""
	tasks = _joint_l1_tasks(states, int(args.max_global_rounds), int(getattr(args, "flux_l1_max_crops", 3)))
	if not tasks:
		return
	grouped: Dict[Tuple[str, ...], List[SampleState]] = {}
	for state in tasks:
		key = tuple(local_model_order_for_edit(state.micro_plan[0], args))
		grouped.setdefault(key, []).append(state)
	for model_tuple, group_states in grouped.items():
		_run_local_edit_benchmark_batch_for_models(args, group_states, list(model_tuple))


def _cached_local_batch_roots(args: argparse.Namespace, model_suffix: str, current_batch_root: Path) -> List[Path]:
	"""List older compatible batch roots from newest to oldest."""
	batches_root = Path(args.output_dir) / "_local_batches"
	if not batches_root.exists():
		return []
	roots = [
		path
		for path in batches_root.glob(f"micro_batch_*_{model_suffix}")
		if path.is_dir() and path.resolve() != current_batch_root.resolve()
	]
	roots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
	return roots


def _cached_model_result(batch_root: Path, model_name: str, sample_id: str) -> Dict[str, Any]:
	"""Load the newest parseable result metadata for one cached model/sample."""
	for output_root in sorted(batch_root.glob("outputs_try_*"), key=lambda path: path.stat().st_mtime, reverse=True):
		result_src = output_root / model_name / sample_id / "result.json"
		if not result_src.exists():
			continue
		try:
			return json.loads(result_src.read_text(encoding="utf-8"))
		except Exception:
			return {"result_json": str(result_src), "parse_error": True}
	return {}


def _same_file_content(left: Path, right: Path) -> bool:
	"""Return whether two cache-key files have identical content."""
	if not left.is_file() or not right.is_file():
		return False
	if left.stat().st_size != right.stat().st_size:
		return False
	left_hash = hashlib.sha256()
	right_hash = hashlib.sha256()
	with left.open("rb") as left_handle, right.open("rb") as right_handle:
		while True:
			left_chunk = left_handle.read(1024 * 1024)
			right_chunk = right_handle.read(1024 * 1024)
			if not left_chunk and not right_chunk:
				break
			left_hash.update(left_chunk)
			right_hash.update(right_chunk)
	return left_hash.digest() == right_hash.digest()


def _cached_sample_matches_current_inputs(
	current_batch_root: Path,
	cached_batch_root: Path,
	sample_id: str,
) -> bool:
	"""Only reuse a local result for the exact same base image and edit request.

	A sample id alone is not a valid cache key: retries can change both the
	current image and the edit instruction while retaining the same sample id.
	"""
	current_dir = current_batch_root / "data" / sample_id
	cached_dir = cached_batch_root / "data" / sample_id
	return _same_file_content(current_dir / "input.png", cached_dir / "input.png") and _same_file_content(
		current_dir / "edit.json", cached_dir / "edit.json"
	)


def _copy_cached_local_batch_outputs(
	args: argparse.Namespace,
	tasks: Sequence[SampleState],
	selected_model_order: Sequence[str],
	model_suffix: str,
	batch_root: Path,
) -> bool:
	"""Reuse exact input/edit matches, including useful partial cache hits.

	Returns ``True`` only when every requested sample/model output was fully
	materialized into its review hand-off.  Partial matches are instead copied to
	the first executor attempt and the caller continues normal execution.
	"""
	cached_roots = _cached_local_batch_roots(args, model_suffix, batch_root)
	if not cached_roots:
		return False
	resolved: List[Tuple[SampleState, Dict[str, str], Dict[str, Dict[str, Any]], Dict[str, Path]]] = []
	all_resolved = True
	for state in tasks:
		pending = state.pending_local_review
		if pending is None:
			all_resolved = False
			continue
		source_paths: Dict[str, Path] = {}
		model_results: Dict[str, Dict[str, Any]] = {}
		sample_resolved = True
		for model_name in selected_model_order:
			source_path = None
			source_root = None
			for cached_root in cached_roots:
				if not _cached_sample_matches_current_inputs(batch_root, cached_root, state.sample_id):
					continue
				candidate = cached_root / "data" / state.sample_id / f"{model_name}.png"
				if candidate.exists() and candidate.stat().st_size > 0:
					source_path = candidate
					source_root = cached_root
					break
			if source_path is None or source_root is None:
				sample_resolved = False
				break
			source_paths[model_name] = source_path
			result_payload = _cached_model_result(source_root, model_name, state.sample_id)
			if not result_payload:
				sample_resolved = False
				break
			model_results[model_name] = result_payload
		if not sample_resolved:
			all_resolved = False
			continue
		resolved.append((state, {}, model_results, source_paths))

	if not resolved:
		return False

	# A resumed batch commonly has only a subset of its samples completed. Seed
	# those exact input/edit matches into the first worker output directory so
	# the benchmark's ordinary per-sample resume path skips them. Previously the
	# cache was all-or-nothing, causing every completed sample to be regenerated
	# whenever even one task was missing.
	if not all_resolved or len(resolved) != len(tasks):
		seeded_output_root = batch_root / "outputs_try_01"
		for state, _, model_results, source_paths in resolved:
			for model_name, source_path in source_paths.items():
				out_dir = seeded_output_root / model_name / state.sample_id
				out_dir.mkdir(parents=True, exist_ok=True)
				final_path = out_dir / "final.png"
				if not final_path.exists():
					shutil.copy2(source_path, final_path)
				result_path = out_dir / "result.json"
				if not result_path.exists():
					save_json(result_path, model_results[model_name])
		log_stage(
			f"micro-batch partial-cache models={','.join(selected_model_order)} "
			f"reused={len(resolved)} pending={len(tasks) - len(resolved)} cached_roots={len(cached_roots)}"
		)
		return False

	for state, model_paths, model_results, source_paths in resolved:
		pending = state.pending_local_review
		assert pending is not None
		attempt_dir = Path(pending["attempt_dir"])
		artifact_prefix = str(pending["artifact_prefix"])
		for model_name, source_path in source_paths.items():
			dst = copy_image_generation_sized(source_path, attempt_dir / f"{artifact_prefix}_repaired_{model_name}.png", args)
			model_paths[model_name] = str(dst)
			if model_name in model_results:
				save_json(attempt_dir / f"{artifact_prefix}_{model_name}_result.json", model_results[model_name])
		pending["model_paths"] = model_paths
		pending["model_results"] = model_results
		append_event(
			state,
			{
				"event": "l1_joint_generation_outputs",
				"action_schema": pending.get("action_schema", L1_JOINT_ACTION_SCHEMA),
				"action_seed": pending.get("action_seed"),
				"edit_count": len(pending.get("edits") or []),
				"models": model_paths,
				"model_results": model_results,
				"input_max_side": pending.get("input_max_side"),
				"cache_hit": True,
			},
		)
		save_status(state)
	log_stage(
		f"micro-batch cache-hit models={','.join(selected_model_order)} count={len(tasks)} "
		f"cached_roots={len(cached_roots)}"
	)
	return True


def _run_local_edit_benchmark_batch_for_models(
	args: argparse.Namespace,
	states: Sequence[SampleState],
	selected_model_order: Sequence[str],
) -> None:
	"""Materialize tasks, run/retry the isolated executor, and commit candidates."""
	tasks = _joint_l1_tasks(states, int(args.max_global_rounds), int(getattr(args, "flux_l1_max_crops", 3)))
	if not tasks:
		return
	selected_model_order = [model_name for model_name in selected_model_order if model_name in LOCAL_MODEL_ORDER]
	if not selected_model_order:
		selected_model_order = list(LOCAL_REPAIR_MODEL_ORDER)
	log_stage(
		f"dispatch step=edit-micro models={','.join(selected_model_order)} count={len(tasks)} samples={describe_states_brief(tasks)}"
	)
	model_suffix = "_".join(selected_model_order)
	batch_root = Path(args.output_dir) / "_local_batches" / f"micro_batch_{int(time.time() * 1000)}_{model_suffix}"
	data_root = batch_root / "data"
	output_root = batch_root / "outputs"
	data_root.mkdir(parents=True, exist_ok=True)
	selected_gpu_indices = resolve_local_edit_gpu_indices(args.local_edit_gpu_indices, args.local_edit_min_free_gib)
	save_json(
		batch_root / "selected_gpus.json",
		{
			"requested": args.local_edit_gpu_indices,
			"min_free_gib": args.local_edit_min_free_gib,
			"selected": selected_gpu_indices,
			"free_gib": query_gpu_free_gib(),
		},
	)
	current_resolution = generation_resolution(args)
	for state in tasks:
		log_stage(
			f"sample={state.sample_id} step=edit-micro-joint mode=micro edits={len(state.micro_plan)} "
			f"models={','.join(selected_model_order)}"
		)
		batch_sample_dir = data_root / state.sample_id
		attempt_dir = sample_round_dir(state.sample_dir, state.global_round)
		artifact_prefix = "repair_l1_joint"
		attempt_dir.mkdir(parents=True, exist_ok=True)
		prepared_input = prepare_local_benchmark_retry_inputs(
			state,
			batch_sample_dir,
			attempt_dir,
			artifact_prefix,
			current_resolution,
			base_seed=int(args.seed),
		)
		state.pending_local_review = {
			"action_schema": prepared_input["action_schema"],
			"operation": prepared_input["operation"],
			"reference_roles": prepared_input["reference_roles"],
			"action_seed": prepared_input["action_seed"],
			"attempt_dir": str(attempt_dir),
			"artifact_prefix": artifact_prefix,
			"input_path": prepared_input["input_path"],
			"plan_path": prepared_input["plan_path"],
			"batch_sample_dir": str(batch_sample_dir),
			"edit": prepared_input["edit"],
			"edits": prepared_input["edits"],
			"input_size": prepared_input["input_size"],
			"input_max_side": prepared_input["input_max_side"],
			"selected_models": list(selected_model_order),
		}
		append_event(
			state,
			{
				"event": "l1_joint_generation_start",
				"action_schema": prepared_input["action_schema"],
				"action_seed": prepared_input["action_seed"],
				"edits": prepared_input["edits"],
				"models": list(selected_model_order),
			},
		)

	if _copy_cached_local_batch_outputs(args, tasks, selected_model_order, model_suffix, batch_root):
		return

	retry_index = 1
	while True:
		output_root = batch_root / f"outputs_try_{retry_index:02d}"
		output_root.mkdir(parents=True, exist_ok=True)
		for state in tasks:
			pending = state.pending_local_review
			assert pending is not None
			batch_sample_dir = Path(pending["batch_sample_dir"])
			attempt_dir = Path(pending["attempt_dir"])
			clear_local_benchmark_outputs(batch_sample_dir)
			prepared_input = prepare_local_benchmark_retry_inputs(
				state,
				batch_sample_dir,
				attempt_dir,
				str(pending["artifact_prefix"]),
				current_resolution,
				base_seed=int(args.seed),
			)
			pending["edit"] = prepared_input["edit"]
			pending["edits"] = prepared_input["edits"]
			pending["operation"] = prepared_input["operation"]
			pending["reference_roles"] = prepared_input["reference_roles"]
			pending["action_seed"] = prepared_input["action_seed"]
			pending["input_size"] = prepared_input["input_size"]
			pending["input_max_side"] = prepared_input["input_max_side"]
			pending["input_path"] = prepared_input["input_path"]
			pending["plan_path"] = prepared_input["plan_path"]

		cmd = [
			args.vcot_python,
			str(LOCAL_BENCHMARK_SCRIPT),
			"run",
			"--data-root",
			str(data_root),
			"--output-root",
			str(output_root),
			"--gpu-indices",
			",".join(str(item) for item in selected_gpu_indices),
			"--workers-per-model",
			str(args.local_edit_workers_per_model),
			"--models",
			",".join(selected_model_order),
			"--model-min-free-gib",
			args.local_edit_model_min_free_gib,
			"--model-max-parallel",
			args.local_edit_model_max_parallel,
			"--max-memory-retries",
			str(args.local_edit_max_memory_retries),
			"--memory-wait-sec",
			str(args.local_edit_memory_wait_sec),
			"--vcot-python",
			args.vcot_python,
			"--flux2-model",
			args.flux_model,
			"--flux2-steps",
			str(args.flux_steps),
			"--flux2-guidance-scale",
			str(args.flux_guidance_scale),
			"--flux2-max-sequence-length",
			str(args.flux_max_sequence_length),
			"--flux2-dtype",
			str(args.flux_dtype),
			"--l1-roi-pad-ratio",
			str(args.flux_l1_roi_pad_ratio),
			"--l1-min-span",
			str(args.flux_l1_min_span),
			"--l1-max-crops",
			str(args.flux_l1_max_crops),
			"--seed",
			str(args.seed),
			"--resolution",
			str(current_resolution),
		]
		proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
		log_path = batch_root / f"run_try_{retry_index:02d}.log"
		log_path.write_text(proc.stdout, encoding="utf-8")
		retry_text = collect_local_benchmark_retry_text(proc.stdout, output_root)
		log_stage(
			f"micro-batch completed exit_code={proc.returncode} retry={retry_index} resolution={current_resolution} log={log_path}"
		)
		if proc.returncode != 0:
			if is_memory_related_text(retry_text) and memory_retry_allowed(retry_index - 1, args.local_edit_outer_max_memory_retries):
				log_stage(
					f"micro-batch memory-retry exit_code={proc.returncode} retry={retry_index} "
					f"wait_sec={args.local_edit_memory_wait_sec:g}"
				)
				time.sleep(max(1.0, float(args.local_edit_memory_wait_sec)))
				retry_index += 1
				continue
			raise RuntimeError(f"Local edit benchmark failed with code {proc.returncode}. See {log_path}")

		attempt_results: List[Tuple[SampleState, Dict[str, Any], Path, Dict[str, str], List[str]]] = []
		memory_retry_needed = False
		for state in tasks:
			pending = state.pending_local_review
			assert pending is not None
			batch_sample_dir = Path(pending["batch_sample_dir"])
			attempt_dir = Path(pending["attempt_dir"])
			artifact_prefix = str(pending["artifact_prefix"])
			model_paths: Dict[str, str] = {}
			model_results: Dict[str, Any] = {}
			for model_name in selected_model_order:
				src = batch_sample_dir / f"{model_name}.png"
				if src.exists():
					dst = copy_image_generation_sized(src, attempt_dir / f"{artifact_prefix}_repaired_{model_name}.png", args)
					model_paths[model_name] = str(dst)
				result_src = output_root / model_name / state.sample_id / "result.json"
				if result_src.exists():
					try:
						result_payload = json.loads(result_src.read_text(encoding="utf-8"))
					except Exception:
						result_payload = {"result_json": str(result_src), "parse_error": True}
					model_results[model_name] = result_payload
					save_json(attempt_dir / f"{artifact_prefix}_{model_name}_result.json", result_payload)
			missing = [model_name for model_name in selected_model_order if model_name not in model_paths]
			pending["model_results"] = model_results
			attempt_results.append((state, pending, attempt_dir, model_paths, missing))
			if missing and is_memory_related_text(retry_text):
				memory_retry_needed = True
		if memory_retry_needed:
			log_stage(
				f"micro-batch memory issue detected with fixed resolution={current_resolution}; not reducing generation size"
			)
			if memory_retry_allowed(retry_index - 1, args.local_edit_outer_max_memory_retries):
				log_stage(
					f"micro-batch memory-retry missing_outputs retry={retry_index} "
					f"wait_sec={args.local_edit_memory_wait_sec:g}"
				)
				time.sleep(max(1.0, float(args.local_edit_memory_wait_sec)))
				retry_index += 1
				continue

		for state, pending, attempt_dir, model_paths, missing in attempt_results:
			if missing:
				mark_state_error(state, f"Missing local-edit outputs: {missing}", "l1_joint_generation_missing_outputs")
				state.pending_local_review = None
				continue
			pending["model_paths"] = model_paths
			append_event(
				state,
				{
					"event": "l1_joint_generation_outputs",
					"action_schema": pending.get("action_schema", L1_JOINT_ACTION_SCHEMA),
					"action_seed": pending.get("action_seed"),
					"edit_count": len(pending.get("edits") or []),
					"models": model_paths,
					"model_results": pending.get("model_results") or {},
					"input_max_side": pending.get("input_max_side"),
				},
			)
			save_status(state)
		break
