"""Validate contracts and drive the resumable MLIE-CoT state machine.

This module is the control plane; it does not implement model inference.
Startup first binds all requested model settings to immutable Qwen, FLUX.2,
and initial/L3 policy contracts.  Only after those checks pass does it load
sample state and create the lazy heavyweight runtime.

The main loop repeatedly dispatches ready samples to one stage:

``diagnosis -> route/plan -> generate -> blind retain review -> diagnosis``.

Stages persist their hand-off before returning, so the scheduler can resume
without repeating a successful model call.  ``global_round`` advances only
when a repair candidate is generated.  A final diagnosis is still allowed
after the action budget is exhausted, but no further route is dispatched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .edit_stages import build_runtime, run_global_edit_generation, run_initial_generation_batch, run_regenerate_generation_batch
from .contracts import build_qwen_analyzer_l1_distribution_contract
from .flux_policy import build_online_flux2_policy, flux2_policy_digest
from .l3_policy import (
	l3_policy_digest,
	normalize_initial_generator_family,
	resolve_online_l3_policy,
	validate_l3_policy_contract,
)
from .image_utils import generation_resolution
from .io_utils import finalize_state, load_states, normalize_v10_action_budget_state, save_json, save_status, select_prompts
from .local_benchmark import run_local_edit_benchmark_batch
from .memory import release_qwen_hard, resolve_qwen_model_path
from .qwen_stages import (
	run_global_compare_batch,
	run_global_triage_batch,
	run_local_review_batch,
	run_micro_box_review_batch,
	run_prompt_reasoning_batch,
)
from .state import SampleState

def apply_single_model_defaults(args: argparse.Namespace) -> None:
	"""Translate compatibility ``single_model`` selections into V10 backends."""
	single_model = str(getattr(args, "single_model", "off") or "off").strip().lower()
	if single_model == "bagel":
		args.initial_generator = "bagel"
	elif single_model == "flux2":
		args.initial_generator = "flux"


def maybe_finalize(states: Sequence[SampleState], summary_jsonl: Path) -> None:
	"""Persist terminal states that have not yet been written to the summary.

	Error states retain their latest current image as the best recoverable
	output.  Successful/partial states are finalized only after a concrete final
	image has been selected.
	"""
	for state in states:
		if state.done:
			continue
		if state.error is not None:
			state.status = "error"
			state.final_reason = state.final_reason or "exception"
			state.final_image = state.current_image
			finalize_state(state, summary_jsonl)
			continue
		if state.status in {"ok", "partial"} and state.final_image is not None:
			finalize_state(state, summary_jsonl)


def flush_exceptions(states: Sequence[SampleState], summary_jsonl: Path) -> None:
	"""Compatibility alias used by callers that flush per-sample failures."""
	maybe_finalize(states, summary_jsonl)


def _persisted_initial_generator_family(state: SampleState) -> str | None:
	"""Recover pre-field V10 status from the initial-generation timeline."""
	if state.initial_generator_family:
		return normalize_initial_generator_family(state.initial_generator_family)
	if state.current_image is None:
		return None
	for event in state.history:
		if event.get("event") == "initial_generate":
			return normalize_initial_generator_family(event.get("model"))
	return None


def _apply_resolved_l3_paths(
	args: argparse.Namespace,
	resolved_paths: dict[str, Path],
	policy: dict,
) -> None:
	"""Point both initial generation and L3 at the policy-validated bytes."""
	executor = policy["executor"]
	family = executor["family"]
	if family == "bagel":
		args.bagel_model_path = str(resolved_paths["model"])
		args.bagel_fine_tuned_path = str(resolved_paths["auxiliary_model"])
		args.bagel_model_id = executor["model_id"]
		args.bagel_model_revision = executor["model_revision"]
		args.bagel_fine_tuned_model_id = executor["auxiliary_model_id"]
		args.bagel_fine_tuned_model_revision = executor["auxiliary_model_revision"]
	elif family == "sd35":
		args.sd35_model = str(resolved_paths["model"])
		args.sd35_model_id = executor["model_id"]
		args.sd35_model_revision = executor["model_revision"]


def _explicit_initial_base_policy_requested(args: argparse.Namespace) -> bool:
	"""Enable exact fixed-level collection without requiring a pre-existing head."""
	family = normalize_initial_generator_family(getattr(args, "initial_generator", ""))
	if family == "bagel":
		return bool(
			str(getattr(args, "bagel_model_revision", "") or "").strip()
			or str(getattr(args, "bagel_fine_tuned_model_revision", "") or "").strip()
		)
	if family == "sd35":
		return bool(str(getattr(args, "sd35_model_revision", "") or "").strip())
	return bool(str(getattr(args, "flux_model_revision", "") or "").strip())


def run_pipeline(args: argparse.Namespace) -> int:
	"""Execute a validated, resumable V10 run.

	The function performs five high-level operations: normalize deprecated
	options, validate immutable runtime policies, persist ``run_config.json``,
	restore and validate sample states, and schedule ready stages until every
	sample is terminal or the selected partial ``stage_mode`` has no work.

	Raises:
		ValueError: If CLI settings, checkpoint contracts, or resumed artifacts
			would change the semantics of an existing/formal run.

	Returns:
		Zero after all selected work has been durably finalized.
	"""
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
	if int(args.max_global_rounds) < 0:
		raise ValueError(f"max_global_rounds must be non-negative, got {args.max_global_rounds}")
	# Accepted only so old launch scripts fail soft; V10 has no action-level
	# local retry and always emits exactly one L1 candidate.
	args.max_local_attempts = 1
	# The official V10 online/collection policy executes the single post-route
	# L1 plan directly. Extra bbox replanning would change L1 utility at deploy
	# time relative to the collector, so the legacy knob is ignored.
	args.qwen_box_review_max_rounds = 0
	args.enable_regenerate_level = True
	if int(getattr(args, "fixed_repair_level", 0) or 0) in {1, 2, 3}:
		args.qwen_route_head = ""
	apply_single_model_defaults(args)
	online_flux2_policy = None
	online_l3_policy = None
	route_config = None
	if str(getattr(args, "qwen_route_head", "") or ""):
		# Bind the actual CLI settings and resolved local Flux bytes before the
		# route head or any large Qwen/Flux backbone is allocated.
		from .route_head import (
			peek_route_head_config,
			validate_route_flux2_runtime_policy,
			validate_route_l3_runtime_policy,
		)

		route_config = peek_route_head_config(args.qwen_route_head)
		resolved_flux2, online_flux2_policy = build_online_flux2_policy(args, require_immutable=True)
		validate_route_flux2_runtime_policy(route_config, online_flux2_policy)
		args.flux_model = str(resolved_flux2)
		args.flux_model_id = online_flux2_policy["executor"]["model_id"]
		args.flux_model_revision = online_flux2_policy["executor"]["model_revision"]
		resolved_l3_paths, online_l3_policy = resolve_online_l3_policy(
			args,
			flux2_policy=online_flux2_policy,
		)
		# Merely recording a declared revision while leaving a mutable path in the
		# service arguments would not establish the claimed same-base binding.
		_apply_resolved_l3_paths(args, resolved_l3_paths, online_l3_policy)
		validate_route_l3_runtime_policy(route_config, online_l3_policy)
	elif _explicit_initial_base_policy_requested(args):
		# Bootstrap an exact BAGEL/SD3.5 (or Flux) fixed-level collection run
		# before a route head exists.  These run_config/status contracts can then
		# be exported and used to train the first head for that initial base.
		if normalize_initial_generator_family(args.initial_generator) == "flux2":
			resolved_flux2, online_flux2_policy = build_online_flux2_policy(args, require_immutable=True)
			args.flux_model = str(resolved_flux2)
			args.flux_model_id = online_flux2_policy["executor"]["model_id"]
			args.flux_model_revision = online_flux2_policy["executor"]["model_revision"]
		resolved_l3_paths, online_l3_policy = resolve_online_l3_policy(
			args,
			flux2_policy=online_flux2_policy,
		)
		_apply_resolved_l3_paths(args, resolved_l3_paths, online_l3_policy)
	if int(getattr(args, "fixed_repair_level", 0) or 0) == 3 and online_l3_policy is None:
		raise ValueError(
			"A fixed L3 ablation must bind the initial/L3 generator to immutable model revisions. "
			"Provide --flux-model-revision, --sd35-model-revision, or both BAGEL revision arguments."
		)
	args.qwen_model = resolve_qwen_model_path(str(args.qwen_model))
	qwen_distribution = build_qwen_analyzer_l1_distribution_contract(
		model=args.qwen_model_id,
		model_revision=args.qwen_model_revision,
		dtype=args.qwen_dtype,
		quantization=args.qwen_quantization,
		allow_cpu_offload=bool(getattr(args, "qwen_allow_cpu_offload", False)),
		image_shrink_retry=not bool(getattr(args, "qwen_disable_image_shrink_retry", True)),
		input_max_side=args.qwen_input_max_side,
		max_new_tokens=args.qwen_max_new_tokens,
		require_immutable_revision=True,
	)
	if route_config is not None:
		from .route_head import validate_route_qwen_distribution

		validate_route_qwen_distribution(route_config, qwen_distribution)
	can_run_qwen = args.stage_mode in {"full", "qwen"}
	can_run_edit = args.stage_mode in {"full", "edit"}
	selected = select_prompts(Path(args.data_path), args.start_index, args.limit)
	output_root = Path(args.output_dir)
	output_root.mkdir(parents=True, exist_ok=True)
	previous_run_l3_policy_digest = None
	previous_run_config_path = output_root / "run_config.json"
	if bool(args.resume) and previous_run_config_path.is_file():
		try:
			previous_run_config = json.loads(previous_run_config_path.read_text(encoding="utf-8"))
			previous_l3_policy = previous_run_config.get("l3_policy_contract")
			if previous_l3_policy is not None:
				previous_run_l3_policy_digest = l3_policy_digest(
					validate_l3_policy_contract(previous_l3_policy)
				)
				declared_previous_digest = str(previous_run_config.get("l3_policy_digest") or "").strip().lower()
				if declared_previous_digest and declared_previous_digest != previous_run_l3_policy_digest:
					raise ValueError("prior run_config.json L3 policy digest mismatch")
		except Exception as exc:
			raise ValueError(f"Cannot resume: invalid prior run_config.json L3 policy: {exc}") from exc
	save_json(
		previous_run_config_path,
		{
			"data_path": str(args.data_path),
			"output_dir": str(args.output_dir),
			"start_index": args.start_index,
			"limit": args.limit,
			"resolution": generation_resolution(args),
				"qwen_review_input_max_side": args.qwen_review_input_max_side,
				"qwen_compare_input_max_side": args.qwen_compare_input_max_side,
				"qwen_prompt_reasoning_max_new_tokens": args.qwen_prompt_reasoning_max_new_tokens,
			"qwen_route_head": str(getattr(args, "qwen_route_head", "") or ""),
			"qwen_analyzer_l1_distribution": qwen_distribution,
			"flux2_collector_policy": online_flux2_policy,
			"flux2_policy_digest": flux2_policy_digest(online_flux2_policy) if online_flux2_policy else None,
			"l3_policy_contract": online_l3_policy,
			"l3_policy_digest": l3_policy_digest(online_l3_policy) if online_l3_policy else None,
				"initial_generator": args.initial_generator,
				"single_model": getattr(args, "single_model", "off"),
				"language_backend": getattr(args, "language_backend", "auto"),
			"max_global_rounds": args.max_global_rounds,
			"l1_candidates_per_action": 1,
			"qwen_box_review_max_rounds": args.qwen_box_review_max_rounds,
			"enable_regenerate_level": getattr(args, "enable_regenerate_level", False),
			"skip_prompt_reasoning": getattr(args, "skip_prompt_reasoning", False),
			"fixed_repair_level": getattr(args, "fixed_repair_level", None),
			"bagel_gpu_indices": args.bagel_gpu_indices,
			"bagel_parallel_workers": args.bagel_parallel_workers,
			"local_edit_gpu_indices": args.local_edit_gpu_indices,
			"local_edit_workers_per_model": args.local_edit_workers_per_model,
			"local_edit_min_free_gib": args.local_edit_min_free_gib,
			"local_edit_model_min_free_gib": args.local_edit_model_min_free_gib,
			"local_edit_model_max_parallel": args.local_edit_model_max_parallel,
			"local_edit_max_memory_retries": args.local_edit_max_memory_retries,
			"local_edit_outer_max_memory_retries": args.local_edit_outer_max_memory_retries,
			"local_edit_memory_wait_sec": args.local_edit_memory_wait_sec,
			"edit_service_min_free_gib": args.edit_service_min_free_gib,
			"edit_service_memory_wait_sec": args.edit_service_memory_wait_sec,
			"edit_service_max_memory_retries": args.edit_service_max_memory_retries,
			"edit_service_max_wait_sec": args.edit_service_max_wait_sec,
			"flux_gpu_indices": args.flux_gpu_indices,
			"flux_parallel_workers": args.flux_parallel_workers,
			"flux_batch_size": args.flux_batch_size,
			"flux_model": args.flux_model,
			"flux_steps": args.flux_steps,
				"flux_guidance_scale": args.flux_guidance_scale,
			"sd35_model": getattr(args, "sd35_model", ""),
			"sd35_gpu_indices": getattr(args, "sd35_gpu_indices", ""),
			"sd35_parallel_workers": getattr(args, "sd35_parallel_workers", 0),
			"sd35_steps": getattr(args, "sd35_steps", 0),
			"sd35_guidance_scale": getattr(args, "sd35_guidance_scale", 0.0),
			"resume": args.resume,
				"stage_mode": args.stage_mode,
				"qwen_batch_size": args.qwen_batch_size,
				"qwen_min_free_gib": args.qwen_min_free_gib,
				"qwen_memory_wait_sec": args.qwen_memory_wait_sec,
				"qwen_inprocess": args.qwen_inprocess,
			},
		)
	states = load_states(selected, output_root, args.resume, int(args.max_global_rounds))
	selected_initial_family = normalize_initial_generator_family(args.initial_generator)
	selected_initial_policy_digest = l3_policy_digest(online_l3_policy) if online_l3_policy else None
	for state in states:
		persisted_family = _persisted_initial_generator_family(state)
		if persisted_family is None and state.current_image is not None:
			raise ValueError(
				f"Cannot resume {state.sample_id}: the initial generator is absent from status/timeline"
			)
		if persisted_family is not None and persisted_family != selected_initial_family:
			raise ValueError(
				f"Cannot resume {state.sample_id} with a different initial generator: "
				f"persisted={persisted_family} requested={selected_initial_family}"
			)
		persisted_policy_digest = (
			str(state.initial_generator_policy_digest or "").strip().lower()
			or previous_run_l3_policy_digest
			or None
		)
		if state.current_image is not None and selected_initial_policy_digest is not None:
			if persisted_policy_digest is None:
				raise ValueError(
					f"Cannot resume {state.sample_id}: status.json does not bind the existing initial image "
					"to an immutable initial-generator policy"
				)
			if persisted_policy_digest != selected_initial_policy_digest:
				raise ValueError(
					f"Cannot resume {state.sample_id} with different initial-generator bytes/settings: "
					f"persisted={persisted_policy_digest} requested={selected_initial_policy_digest}"
				)
		state.initial_generator_family = selected_initial_family
		if selected_initial_policy_digest is not None:
			state.initial_generator_policy_digest = selected_initial_policy_digest
		save_status(state)
	summary_jsonl = output_root / "summary.jsonl"
	for state in states:
		normalize_v10_action_budget_state(
			state,
			int(args.max_global_rounds),
			source="pipeline_start",
		)
	maybe_finalize(states, summary_jsonl)
	runtime = build_runtime(args)
	try:
		if args.stage_mode == "full":
			run_prompt_reasoning_batch(runtime, args, states)
		if can_run_edit:
			run_initial_generation_batch(runtime, args, states)
		maybe_finalize(states, summary_jsonl)

		# The scheduler normally terminates as soon as no active state remains.
		# This upper bound is a corruption/deadlock guard, not an action budget.
		max_loop_iters = max(
			64,
			(int(args.max_global_rounds) + 1)
			* (int(args.qwen_box_review_max_rounds) + 9),
		)
		for _loop_idx in range(max_loop_iters):
			for state in states:
				normalize_v10_action_budget_state(
					state,
					int(args.max_global_rounds),
					source="pipeline_loop",
				)
			maybe_finalize(states, summary_jsonl)
			active = [state for state in states if not state.done and state.error is None]
			if not active:
				break
			made_progress = False

			if can_run_qwen and args.stage_mode == "qwen":
				prompt_reasoning_candidates = [state for state in active if state.current_image is None and state.generation_prompt is None]
				if prompt_reasoning_candidates:
					run_prompt_reasoning_batch(runtime, args, prompt_reasoning_candidates)
					maybe_finalize(states, summary_jsonl)
					made_progress = True
					continue

			# Reviews are intentionally handled before dispatching another edit:
			# a generated candidate must be accepted or rolled back exactly once.
			triage_candidates = [state for state in active if state.needs_global_analysis and state.current_image]
			if can_run_qwen and triage_candidates:
				run_global_triage_batch(runtime, args, triage_candidates)
				maybe_finalize(states, summary_jsonl)
				made_progress = True

			active = [state for state in states if not state.done and state.error is None]
			pending_global_compare = [state for state in active if state.pending_global_compare]
			if can_run_qwen and pending_global_compare:
				run_global_compare_batch(runtime, args, pending_global_compare)
				maybe_finalize(states, summary_jsonl)
				made_progress = True

			active = [state for state in states if not state.done and state.error is None]
			pending_local_review = [state for state in active if state.pending_local_review]
			if can_run_qwen and pending_local_review:
				run_local_review_batch(runtime, args, pending_local_review)
				maybe_finalize(states, summary_jsonl)
				made_progress = True

			active = [state for state in states if not state.done and state.error is None]
			pending_box_review = [state for state in active if state.phase == "micro" and state.current_image and state.micro_plan and not state.micro_box_review_done and not state.pending_local_review]
			if can_run_qwen and pending_box_review:
				run_micro_box_review_batch(runtime, args, pending_box_review)
				maybe_finalize(states, summary_jsonl)
				made_progress = True

			active = [state for state in states if not state.done and state.error is None]
			regenerate_tasks = [
				state
				for state in active
				if state.phase == "regenerate" and state.current_image and not state.needs_global_analysis
				and int(state.global_round) < int(args.max_global_rounds)
			]
			if regenerate_tasks:
				if can_run_edit and regenerate_tasks:
					run_regenerate_generation_batch(runtime, args, regenerate_tasks)
					maybe_finalize(states, summary_jsonl)
					made_progress = True

			active = [state for state in states if not state.done and state.error is None]
			global_tasks = [
				state
				for state in active
				if state.phase == "global" and state.current_image and not state.needs_global_analysis and not state.pending_global_compare
				and int(state.global_round) < int(args.max_global_rounds)
			]
			if global_tasks:
				if can_run_edit and global_tasks:
					run_global_edit_generation(runtime, args, global_tasks)
					maybe_finalize(states, summary_jsonl)
					made_progress = True

			active = [state for state in states if not state.done and state.error is None]
			micro_tasks = [
				state
				for state in active
				if state.phase == "micro"
				and state.current_image
				and not state.pending_local_review
				and (state.micro_box_review_done or int(args.qwen_box_review_max_rounds) <= 0)
				and state.micro_edit_index < len(state.micro_plan)
				and int(state.global_round) < int(args.max_global_rounds)
			]
			if can_run_edit and micro_tasks:
				release_qwen_hard(runtime)
				run_local_edit_benchmark_batch(args, micro_tasks)
				maybe_finalize(states, summary_jsonl)
				made_progress = True

			if not made_progress:
				if args.stage_mode != "full":
					break
				for state in active:
					if state.current_image is not None and not state.needs_global_analysis and state.phase == "global":
						state.needs_global_analysis = True
						state.pending_global_analysis_reason = "idle_reanalysis"
				if not any(state.needs_global_analysis for state in active):
					break

		if args.stage_mode == "full":
			for state in states:
				if state.done:
					continue
				if state.error is None:
					state.status = "partial"
					state.final_reason = state.final_reason or "loop_exhausted"
					state.final_image = state.current_image
				finalize_state(state, summary_jsonl)
	finally:
		runtime.close()
	return 0
