"""Execute initial generation and the V10 L2/L3 whole-image actions.

This module owns image-producing calls outside L1.  Before loading an editor it
releases Qwen and other heavy clients, selects GPUs with sufficient capacity,
and uses one independent service client per worker.  A successful repair call
always produces exactly one candidate; acceptance is deferred to the Qwen
review stages.

Durability follows a prepare/call/commit pattern.  The action plan and pending
review record are persisted around the model call so resume logic can
distinguish an undispatched action from a generated candidate awaiting review.
Operational memory retries repeat the exact request and do not create a new
semantic candidate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from .vcot_branch import (
	BagelClient,
	FluxClient,
	HeavyRuntimeManager,
	SD35TurboClient,
)

from .io_utils import append_event, ensure_round_initial, mark_state_error, save_json, save_status
from .logging_utils import describe_states_brief, log_stage
from .memory import (
	is_memory_related_exception,
	release_qwen_hard,
	resolve_parallel_service_gpu_indices,
	select_ready_gpu_indices,
)
from .state import SampleState, round_action_plan_path, round_error_analysis_path, sample_round_dir
from .workers import clone_namespace, run_state_batches_with_client_pool, run_state_jobs_with_client_pool


def _require_action_instruction(analysis: dict, expected_level: int) -> str:
	"""Return a non-empty instruction from an action plan of the expected level."""
	plan = analysis.get("action_plan")
	if not isinstance(plan, dict) or int(plan.get("level", -1)) != int(expected_level):
		raise ValueError(f"missing V10 L{expected_level} action plan in diagnosis artifact")
	instruction = str(plan.get("instruction") or "").strip()
	if not instruction:
		raise ValueError(f"empty V10 L{expected_level} action instruction")
	return instruction


def _load_round_diagnosis_and_plan(state: SampleState) -> tuple[Path, dict]:
	"""Load separate diagnosis/plan artifacts and combine them for execution."""
	analysis_path = _analysis_path_for_round(state)
	analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
	plan_path = round_action_plan_path(state.sample_dir, state.global_round)
	plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
	action_plan = plan_payload.get("action_plan") if isinstance(plan_payload, dict) else None
	if not isinstance(action_plan, dict):
		raise ValueError(f"missing V10 action_plan in {plan_path}")
	analysis["action_plan"] = action_plan
	return analysis_path, analysis


def _analysis_path_for_round(state: SampleState) -> Path:
	"""Resolve the canonical diagnosis path with legacy read compatibility."""
	path = round_error_analysis_path(state.sample_dir, state.global_round)
	if path.exists():
		return path
	return state.sample_dir / f"round_{state.global_round:02d}_triage.json"


def build_runtime(args: argparse.Namespace) -> HeavyRuntimeManager:
	"""Create the lazy heavyweight-runtime owner used by all pipeline stages."""
	return HeavyRuntimeManager(args, per_gpu_extra={})


def run_initial_generation_batch(runtime: HeavyRuntimeManager, args: argparse.Namespace, states: Sequence[SampleState]) -> None:
	"""Generate one deterministic initial image for each fresh sample.

	The selected backend may run across several GPUs; seeds are derived from the
	base seed and prompt index.  Completion stores ``initial.png`` and schedules
	canonical diagnosis without consuming a repair-action round.
	"""
	if not states:
		return
	pending_states = [state for state in states if state.current_image is None]
	if not pending_states:
		return
	generator = str(getattr(args, "initial_generator", "bagel") or "bagel").strip().lower()
	release_qwen_hard(runtime)
	runtime.release_bagel()
	runtime.release_flux()

	if generator == "bagel":
		bagel_gpu_indices = resolve_parallel_service_gpu_indices(
			args.bagel_gpu_indices,
			args.bagel_gpu_index,
			args.bagel_parallel_workers,
		)
		log_stage(
			f"initial-generation model=BAGEL mode=global count={len(pending_states)} gpus={bagel_gpu_indices} samples={describe_states_brief(pending_states)}"
		)
		bagel_gpu_indices = select_ready_gpu_indices(
			bagel_gpu_indices,
			args.edit_service_min_free_gib,
			"initial_generation",
			args.edit_service_memory_wait_sec,
			args.edit_service_max_wait_sec,
		)

		def create_bagel_client(gpu_index: int) -> BagelClient:
			"""Bind a BAGEL service client to this worker GPU."""
			return BagelClient(clone_namespace(args, bagel_gpu_index=gpu_index))

		def run_one(client: BagelClient, state: SampleState, gpu_index: int) -> None:
			"""Generate and commit one BAGEL initial image."""
			try:
				log_stage(f"sample={state.sample_id} step=generate-initial model=BAGEL mode=global gpu={gpu_index}")
				out_path = state.sample_dir / "initial.png"
				generation_prompt = state.generation_prompt or state.prompt
				seed = int(args.seed) + int(state.index)
				started_at = time.time()
				client.generate(generation_prompt, out_path, args, seed=seed)
				duration_sec = round(time.time() - started_at, 3)
				state.current_image = out_path
				ensure_round_initial(state)
				state.needs_global_analysis = True
				state.pending_global_analysis_reason = "initial_generation"
				append_event(
					state,
					{
						"event": "initial_generate",
						"model": "BAGEL",
						"image": str(out_path),
						"gpu": gpu_index,
						"seed": seed,
						"generation_prompt": generation_prompt,
						"duration_sec": duration_sec,
					},
				)
				save_status(state)
			except Exception as exc:
				mark_state_error(state, exc, "initial_generate_error")

		run_state_jobs_with_client_pool(pending_states, bagel_gpu_indices, "initial_generation", create_bagel_client, run_one)
		return

	if generator == "flux":
		flux_gpu_indices = resolve_parallel_service_gpu_indices(
			args.flux_gpu_indices,
			args.flux_gpu_index,
			args.flux_parallel_workers,
		)
		log_stage(
			f"initial-generation model=FLUX2 mode=text-only count={len(pending_states)} gpus={flux_gpu_indices} samples={describe_states_brief(pending_states)}"
		)
		flux_gpu_indices = select_ready_gpu_indices(
			flux_gpu_indices,
			args.edit_service_min_free_gib,
			"initial_generation",
			args.edit_service_memory_wait_sec,
			args.edit_service_max_wait_sec,
		)

		def create_flux_client(gpu_index: int) -> FluxClient:
			"""Bind a FLUX.2 service client to this worker GPU."""
			return FluxClient(clone_namespace(args, flux_gpu_index=gpu_index))

		def run_one(client: FluxClient, state: SampleState, gpu_index: int) -> None:
			"""Generate and commit one scalar FLUX.2 initial image."""
			try:
				log_stage(f"sample={state.sample_id} step=generate-initial model=FLUX2 mode=text-only gpu={gpu_index}")
				out_path = state.sample_dir / "initial.png"
				generation_prompt = state.generation_prompt or state.prompt
				seed = int(args.seed) + int(state.index)
				started_at = time.time()
				client.generate(generation_prompt, out_path, seed, args)
				duration_sec = round(time.time() - started_at, 3)
				state.current_image = out_path
				ensure_round_initial(state)
				state.needs_global_analysis = True
				state.pending_global_analysis_reason = "initial_generation"
				append_event(
					state,
					{
						"event": "initial_generate",
						"model": "FLUX2",
						"image": str(out_path),
						"gpu": gpu_index,
						"seed": seed,
						"generation_prompt": generation_prompt,
						"duration_sec": duration_sec,
					},
				)
				save_status(state)
			except Exception as exc:
				if is_memory_related_exception(exc):
					raise
				mark_state_error(state, exc, "initial_generate_error")

		def run_batch(client: FluxClient, batch: Sequence[SampleState], gpu_index: int) -> None:
			"""Generate a same-settings FLUX.2 batch and commit each item."""
			items = []
			for state in batch:
				out_path = state.sample_dir / "initial.png"
				generation_prompt = state.generation_prompt or state.prompt
				seed = int(args.seed) + int(state.index)
				items.append((generation_prompt, out_path, seed))
			log_stage(
				f"samples={describe_states_brief(batch)} step=generate-initial model=FLUX2 "
				f"mode=text-only-batch batch={len(batch)} gpu={gpu_index}"
			)
			started_at = time.time()
			client.generate_batch(items, args)
			duration_sec = round(time.time() - started_at, 3)
			for state, (generation_prompt, out_path, seed) in zip(batch, items):
				state.current_image = out_path
				ensure_round_initial(state)
				state.needs_global_analysis = True
				state.pending_global_analysis_reason = "initial_generation"
				append_event(
					state,
					{
						"event": "initial_generate",
						"model": "FLUX2",
						"image": str(out_path),
						"gpu": gpu_index,
						"seed": seed,
						"generation_prompt": generation_prompt,
						"batch_size": len(batch),
						"duration_sec": duration_sec,
					},
				)
				save_status(state)

		flux_batch_size = max(1, int(getattr(args, "flux_batch_size", 1)))
		if flux_batch_size > 1:
			run_state_batches_with_client_pool(
				pending_states,
				flux_gpu_indices,
				"initial_generation",
				create_flux_client,
				run_batch,
				run_one,
				batch_size=flux_batch_size,
				min_free_gib=args.edit_service_min_free_gib,
				memory_wait_sec=args.edit_service_memory_wait_sec,
				max_memory_retries=args.edit_service_max_memory_retries,
			)
		else:
			run_state_jobs_with_client_pool(
				pending_states,
				flux_gpu_indices,
				"initial_generation",
				create_flux_client,
				run_one,
				min_free_gib=args.edit_service_min_free_gib,
				memory_wait_sec=args.edit_service_memory_wait_sec,
				max_memory_retries=args.edit_service_max_memory_retries,
			)
		return

	if generator == "sd35":
		sd35_gpu_indices = resolve_parallel_service_gpu_indices(
			args.sd35_gpu_indices,
			args.sd35_gpu_index,
			args.sd35_parallel_workers,
		)
		log_stage(
			f"initial-generation model=SD35_TURBO mode=text-only count={len(pending_states)} gpus={sd35_gpu_indices} samples={describe_states_brief(pending_states)}"
		)
		sd35_gpu_indices = select_ready_gpu_indices(
			sd35_gpu_indices,
			args.edit_service_min_free_gib,
			"initial_generation",
			args.edit_service_memory_wait_sec,
			args.edit_service_max_wait_sec,
		)

		def create_sd35_client(gpu_index: int) -> SD35TurboClient:
			"""Bind an SD3.5 service client to this worker GPU."""
			return SD35TurboClient(clone_namespace(args, sd35_gpu_index=gpu_index))

		def run_one(client: SD35TurboClient, state: SampleState, gpu_index: int) -> None:
			"""Generate and commit one SD3.5 initial image."""
			try:
				log_stage(f"sample={state.sample_id} step=generate-initial model=SD35_TURBO mode=text-only gpu={gpu_index}")
				out_path = state.sample_dir / "initial.png"
				generation_prompt = state.generation_prompt or state.prompt
				seed = int(args.seed) + int(state.index)
				started_at = time.time()
				client.generate(generation_prompt, out_path, seed, args)
				duration_sec = round(time.time() - started_at, 3)
				state.current_image = out_path
				ensure_round_initial(state)
				state.needs_global_analysis = True
				state.pending_global_analysis_reason = "initial_generation"
				append_event(
					state,
					{
						"event": "initial_generate",
						"model": "SD35_TURBO",
						"image": str(out_path),
						"gpu": gpu_index,
						"seed": seed,
						"generation_prompt": generation_prompt,
						"duration_sec": duration_sec,
					},
				)
				save_status(state)
			except Exception as exc:
				if is_memory_related_exception(exc):
					raise
				mark_state_error(state, exc, "initial_generate_error")

		run_state_jobs_with_client_pool(
			pending_states,
			sd35_gpu_indices,
			"initial_generation",
			create_sd35_client,
			run_one,
			min_free_gib=args.edit_service_min_free_gib,
			memory_wait_sec=args.edit_service_memory_wait_sec,
			max_memory_retries=args.edit_service_max_memory_retries,
		)
		return

	raise ValueError(f"Unsupported initial generator: {args.initial_generator}")


def run_regenerate_generation_batch(runtime: HeavyRuntimeManager, args: argparse.Namespace, states: Sequence[SampleState]) -> None:
	"""Execute L3 with the exact same generator family as the initial image.

	The current image is deliberately not supplied to L3.  It remains the
	``original`` in a pending blind comparison, while one prompt-only candidate
	is generated with the policy-bound backend and deterministic round seed.
	"""
	tasks = [
		state
		for state in states
		if state.phase == "regenerate"
		and state.current_image
		and not state.needs_global_analysis
		and not state.done
		and not state.error
		and int(state.global_round) < int(args.max_global_rounds)
	]
	if not tasks:
		return
	from .l3_policy import normalize_initial_generator_family

	family = normalize_initial_generator_family(getattr(args, "initial_generator", ""))
	for state in tasks:
		state_family = normalize_initial_generator_family(state.initial_generator_family or family)
		if state_family != family:
			raise ValueError(
				f"L3 initial-generator mismatch for {state.sample_id}: state={state_family} run={family}"
			)
	if family == "bagel":
		generator = "bagel"
		model_label = "BAGEL"
		output_name = "repaired_regenerate_bagel.png"
		gpu_indices = resolve_parallel_service_gpu_indices(
			args.bagel_gpu_indices, args.bagel_gpu_index, args.bagel_parallel_workers
		)

		def create_client(gpu_index: int):
			"""Create the policy-bound BAGEL L3 client."""
			return BagelClient(clone_namespace(args, bagel_gpu_index=gpu_index))
	elif family == "flux2":
		generator = "flux"
		model_label = "FLUX2"
		output_name = "repaired_regenerate_flux2.png"
		gpu_indices = resolve_parallel_service_gpu_indices(
			args.flux_gpu_indices, args.flux_gpu_index, args.flux_parallel_workers
		)

		def create_client(gpu_index: int):
			"""Create the policy-bound FLUX.2 L3 client."""
			return FluxClient(clone_namespace(args, flux_gpu_index=gpu_index))
	else:
		generator = "sd35"
		model_label = "SD35_TURBO"
		output_name = "repaired_regenerate_sd35.png"
		gpu_indices = resolve_parallel_service_gpu_indices(
			args.sd35_gpu_indices, args.sd35_gpu_index, args.sd35_parallel_workers
		)

		def create_client(gpu_index: int):
			"""Create the policy-bound SD3.5 L3 client."""
			return SD35TurboClient(clone_namespace(args, sd35_gpu_index=gpu_index))

	log_stage(
		f"dispatch step=regenerate model={model_label} mode=prompt-only count={len(tasks)} gpus={gpu_indices} samples={describe_states_brief(tasks)}"
	)
	release_qwen_hard(runtime)
	runtime.release_bagel()
	runtime.release_flux()

	def run_one(client, state: SampleState, gpu_index: int) -> None:
		"""Generate one L3 candidate and commit its pending blind review."""
		try:
			analysis_path, analysis = _load_round_diagnosis_and_plan(state)
			action_plan = analysis.get("action_plan") or {}
			if str(action_plan.get("executor_family") or "") != family:
				raise ValueError(
					"L3 action-plan executor does not match this sample's initial generator: "
					f"plan={action_plan.get('executor_family')!r} initial={family!r}"
				)
			previous_image = state.current_image
			next_round = state.global_round + 1
			remaining_attempts = max(0, int(args.max_global_rounds) - int(state.global_round))
			round_dir = sample_round_dir(state.sample_dir, state.global_round)
			round_dir.mkdir(parents=True, exist_ok=True)
			regenerate_prompt = _require_action_instruction(analysis, 3)
			out_path = round_dir / output_name
			seed = int(args.seed) + int(state.index) * 100 + int(next_round)
			save_json(
				round_dir / "repair_regenerate_prompt.json",
				{
					"target_prompt": state.prompt,
					"generator": generator,
					"model": model_label,
					"regenerate_prompt": regenerate_prompt,
					"analysis_path": str(analysis_path),
					"previous_image_discarded": str(previous_image),
					"remaining_attempts": remaining_attempts,
					"seed": seed,
				},
			)
			log_stage(f"sample={state.sample_id} step=regenerate model={model_label} mode=prompt-only round={next_round} gpu={gpu_index}")
			started_at = time.time()
			if family == "bagel":
				client.generate(regenerate_prompt, out_path, args, seed=seed)
			else:
				client.generate(regenerate_prompt, out_path, seed, args)
			duration_sec = round(time.time() - started_at, 3)
			state.current_image = previous_image
			state.phase = "global"
			state.needs_global_analysis = False
			state.pending_global_analysis_reason = "l3_candidate_pending_retain_review"
			state.pending_global_compare = {
				"round": next_round,
				"round_dir": str(round_dir),
				"analysis_path": str(analysis_path),
				"original": str(previous_image),
				"generated_candidates": [
					{
						"role": "regenerate",
						"tool": f"{model_label.lower()}_generate",
						"route_family": "prompt_only_regeneration",
						"path": str(out_path),
						"instruction": regenerate_prompt,
					}
				],
				"action_level": 3,
				"executor_family": family,
			}
			state.pending_local_review = None
			append_event(
				state,
				{
					"event": "prompt_only_regenerate_candidate",
					"model": model_label,
					"executor_family": family,
					"round": next_round,
					"remaining_attempts": remaining_attempts,
					"previous_image_discarded": str(previous_image),
					"output": str(out_path),
					"retained_pending_review": True,
					"gpu": gpu_index,
					"seed": seed,
					"regenerate_prompt": regenerate_prompt,
					"duration_sec": duration_sec,
				},
			)
			save_status(state)
		except Exception as exc:
			if is_memory_related_exception(exc):
				raise
			mark_state_error(state, exc, "prompt_only_regenerate_error")

	run_state_jobs_with_client_pool(
		tasks,
		gpu_indices,
		"prompt_only_regenerate",
		create_client,
		run_one,
		min_free_gib=args.edit_service_min_free_gib,
		memory_wait_sec=args.edit_service_memory_wait_sec,
		max_memory_retries=args.edit_service_max_memory_retries,
	)


def _append_generated_global_candidate(pending: dict, role: str, tool: str, route_family: str, path: Path, instruction: str) -> None:
	"""Append one auditable candidate record to a prepared comparison hand-off."""
	candidates = pending.setdefault("generated_candidates", [])
	candidates.append(
		{
			"role": role,
			"tool": tool,
			"route_family": route_family,
			"path": str(path),
			"instruction": instruction,
		}
	)


def _prepare_global_edit_state(args: argparse.Namespace, state: SampleState) -> None:
	"""Persist the L2 review hand-off before dispatching the FLUX.2 call."""
	analysis_path, analysis = _load_round_diagnosis_and_plan(state)
	next_round = state.global_round + 1
	remaining_attempts = max(0, int(args.max_global_rounds) - int(state.global_round))
	round_dir = sample_round_dir(state.sample_dir, state.global_round)
	round_dir.mkdir(parents=True, exist_ok=True)
	state.pending_global_compare = {
		"round": next_round,
		"remaining_attempts": remaining_attempts,
		"round_dir": str(round_dir),
		"analysis_path": str(analysis_path),
		"original": str(state.current_image),
		"instruction": _require_action_instruction(analysis, 2),
		"generated_candidates": [],
		"action_level": 2,
		"executor_family": "flux2",
	}
	save_status(state)


def run_global_edit_generation(runtime: HeavyRuntimeManager, args: argparse.Namespace, states: Sequence[SampleState]) -> None:
	"""Execute one FLUX.2 whole-image edit for every ready L2 state.

	The generated path is appended to the previously persisted pending
	comparison.  ``current_image`` and ``global_round`` are not advanced here;
	the blind comparison stage owns retention and budget advancement.
	"""
	tasks = [
		state
		for state in states
		if state.phase == "global"
		and state.current_image
		and not state.needs_global_analysis
		and not state.done
		and not state.error
		and int(state.global_round) < int(args.max_global_rounds)
	]
	if not tasks:
		return
	for state in tasks:
		if state.pending_global_compare is None:
			try:
				_prepare_global_edit_state(args, state)
			except Exception as exc:
				mark_state_error(state, exc, "global_edit_prepare_error")

	tasks = [state for state in tasks if state.error is None and state.pending_global_compare is not None]
	if not tasks:
		return
	release_qwen_hard(runtime)
	runtime.release_bagel()
	runtime.release_flux()

	flux_tasks = [state for state in tasks if state.error is None]
	flux_gpu_indices = resolve_parallel_service_gpu_indices(
		args.flux_gpu_indices,
		args.flux_gpu_index,
		args.flux_parallel_workers,
	)
	log_stage(
		f"dispatch step=edit-global model=FLUX2 count={len(flux_tasks)} "
		f"flux_gpus={flux_gpu_indices} samples={describe_states_brief(flux_tasks)}"
	)

	def create_flux_client(gpu_index: int) -> FluxClient:
		"""Bind a FLUX.2 L2 editor client to this worker GPU."""
		return FluxClient(clone_namespace(args, flux_gpu_index=gpu_index))

	def run_flux_ops(client: FluxClient, state: SampleState, gpu_index: int) -> None:
		"""Generate one L2 candidate and append it to the prepared hand-off."""
		pending = state.pending_global_compare
		assert pending is not None
		try:
			round_dir = Path(pending["round_dir"])
			remaining_attempts = int(
				pending.get(
					"remaining_attempts",
					max(0, int(args.max_global_rounds) - int(state.global_round)),
				)
			)
			round_idx = int(pending["round"])
			flux_prompt = str(pending["instruction"])
			flux_out = round_dir / "repaired_flux2.png"
			flux_seed = int(args.seed) + state.index * 100 + round_idx + 30000
			log_stage(
				f"sample={state.sample_id} step=edit-global model=FLUX2 "
				f"mode=global round={round_idx} gpu={gpu_index}"
			)
			started_at = time.time()
			client.edit(state.current_image, flux_prompt, flux_out, flux_seed, args)
			duration_sec = round(time.time() - started_at, 3)
			_append_generated_global_candidate(
				pending,
				"flux2",
				"flux2_edit",
				"image_global",
				flux_out,
				flux_prompt,
			)
			append_event(
				state,
				{
					"event": "global_flux2_edit",
					"round": round_idx,
					"remaining_attempts": remaining_attempts,
					"output": str(flux_out),
					"gpu": gpu_index,
					"seed": flux_seed,
					"duration_sec": duration_sec,
				},
			)
			save_status(state)
		except Exception as exc:
			if is_memory_related_exception(exc):
				raise
			mark_state_error(state, exc, "global_flux_edit_error")

	run_state_jobs_with_client_pool(
		flux_tasks,
		flux_gpu_indices,
		"global_flux_edit",
		create_flux_client,
		run_flux_ops,
		min_free_gib=args.edit_service_min_free_gib,
		memory_wait_sec=args.edit_service_memory_wait_sec,
		max_memory_retries=args.edit_service_max_memory_retries,
	)
