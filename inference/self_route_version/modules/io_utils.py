"""Filesystem persistence and resume helpers for sample state."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import CATEGORY_RANK, LOCAL_MODEL_ORDER
from .image_utils import copy_image
from .state import STATE_SCHEMA_VERSION, SampleState, round_error_analysis_path, safe_prompt_id, sample_round_dir

def save_json(path: Path, payload: Any) -> None:
	"""Write deterministic, UTF-8, human-readable JSON with a trailing newline."""
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
	"""Append one compact JSON event to a JSONL audit stream."""
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def canonicalize_summary_jsonl(path: Path) -> List[Dict[str, Any]]:
	"""Keep the latest run-summary row for every sample index.

	A budget-extended resume intentionally finalizes a previously terminal
	sample again.  Its per-sample timeline remains the append-only provenance;
	the run-level summary, however, is a current-state index and must contain
	exactly one row per sample.  Rewriting only at pipeline/validator boundaries
	avoids an O(N^2) rewrite during ordinary batch finalization.
	"""
	rows = _load_jsonl_if_exists(path)
	if not rows:
		return []
	latest_by_index: Dict[int, Dict[str, Any]] = {}
	for row in rows:
		if "index" not in row:
			raise ValueError(f"summary row missing index: {path}")
		latest_by_index[int(row["index"])] = row
	canonical = [latest_by_index[index] for index in sorted(latest_by_index)]
	tmp_path = path.with_name(f".{path.name}.tmp")
	with tmp_path.open("w", encoding="utf-8") as handle:
		for row in canonical:
			handle.write(json.dumps(row, ensure_ascii=False) + "\n")
	tmp_path.replace(path)
	return canonical


def append_event(state: SampleState, event: Dict[str, Any]) -> None:
	"""Add timestamp/phase defaults and persist an event in memory and on disk."""
	payload = dict(event)
	payload.setdefault("timestamp", time.time())
	payload.setdefault("phase", state.phase)
	state.history.append(payload)
	append_jsonl(state.sample_dir / "timeline.jsonl", payload)


def mark_state_error(state: SampleState, error: Exception | str, event_name: str) -> None:
	"""Make a per-sample failure terminal and durably record its cause."""
	state.error = repr(error) if not isinstance(error, str) else error
	state.status = "error"
	state.final_reason = state.final_reason or event_name
	append_event(state, {"event": event_name, "error": state.error})
	save_status(state)


def update_best_round(state: SampleState, image_path: Path, analysis: Dict[str, Any], tag: str) -> None:
	"""Track the best diagnosed image using the configured category ordering."""
	rank = CATEGORY_RANK.get(analysis.get("category", "Unknown"), 4)
	if rank < state.best_rank:
		state.best_rank = rank
		state.best_round = {
			"tag": tag,
			"image": str(image_path),
			"analysis": analysis,
		}


def save_status(state: SampleState) -> None:
	"""Serialize the current resumable state to ``status.json``."""
	payload = {
		"state_schema_version": STATE_SCHEMA_VERSION,
		"index": state.index,
		"sample_id": state.sample_id,
		"prompt": state.prompt,
		"initial_generator_family": state.initial_generator_family,
		"initial_generator_policy_digest": state.initial_generator_policy_digest,
		"generation_prompt": state.generation_prompt,
		"phase": state.phase,
		"status": state.status,
		"done": state.done,
		"global_round": state.global_round,
		"micro_edit_index": state.micro_edit_index,
		"current_image": str(state.current_image) if state.current_image else None,
		"final_image": str(state.final_image) if state.final_image else None,
		"final_reason": state.final_reason,
		"needs_global_analysis": state.needs_global_analysis,
		"pending_global_analysis_reason": state.pending_global_analysis_reason,
		"pending_global_compare": state.pending_global_compare,
		"pending_local_review": state.pending_local_review,
		"micro_plan": state.micro_plan,
		"micro_box_review_done": state.micro_box_review_done,
		"repair_memory": state.repair_memory,
		"best_round": state.best_round,
		"best_rank": state.best_rank,
		"error": state.error,
		"started_at": state.started_at,
		"updated_at": time.time(),
	}
	save_json(state.sample_dir / "status.json", payload)


def finalize_state(state: SampleState, summary_jsonl: Path) -> Dict[str, Any]:
	"""Materialize ``final.png`` and write per-sample plus run-level summaries.

	The selected image is copied rather than moved because round artifacts are
	immutable provenance.  Callers should finalize a state once; ``state.done``
	prevents normal scheduler paths from appending duplicate summary rows.
	"""
	state.done = True
	if state.final_image is None:
		state.final_image = state.current_image
	if state.final_image is not None:
		final_copy = copy_image(state.final_image, state.sample_dir / "final.png")
	else:
		final_copy = None
	summary = {
		"index": state.index,
		"sample_id": state.sample_id,
		"prompt": state.prompt,
		"generation_prompt": state.generation_prompt,
		"sample_dir": str(state.sample_dir),
		"phase": state.phase,
		"status": state.status,
		"final_reason": state.final_reason,
		"final_image": str(final_copy) if final_copy else None,
		"best_round": state.best_round,
		"elapsed_sec": round(time.time() - state.started_at, 3),
		"history": state.history,
		"repair_memory": state.repair_memory,
		"micro_plan": state.micro_plan,
		"micro_box_review_done": state.micro_box_review_done,
		"error": state.error,
	}
	save_json(state.sample_dir / "summary.json", summary)
	append_jsonl(summary_jsonl, summary)
	save_status(state)
	return summary


def ensure_round_initial(state: SampleState) -> Optional[Path]:
	"""Archive the image entering the current round under its round directory."""
	if state.current_image is None:
		return None
	dst = sample_round_dir(state.sample_dir, state.global_round) / "initial.png"
	if not dst.exists():
		copy_image(state.current_image, dst)
	return dst


def select_prompts(data_path: Path, start_index: int, limit: Optional[int]) -> List[Tuple[int, str]]:
	"""Load non-empty prompts and apply row-preserving slice semantics."""
	prompts = [line.strip() for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
	indexed = list(enumerate(prompts))
	if start_index > 0:
		indexed = indexed[start_index:]
	if limit is not None:
		indexed = indexed[:limit]
	return indexed


def init_states(selected: Sequence[Tuple[int, str]], output_root: Path) -> List[SampleState]:
	"""Create fresh sample directories and their initial pending status."""
	states: List[SampleState] = []
	for index, prompt in selected:
		sample_id = safe_prompt_id(index, prompt)
		sample_dir = output_root / sample_id
		sample_dir.mkdir(parents=True, exist_ok=True)
		(sample_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
		state = SampleState(index=index, prompt=prompt, sample_dir=sample_dir, sample_id=sample_id)
		save_status(state)
		states.append(state)
	return states


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
	"""Best-effort load a JSON object; malformed/non-object files return ``None``."""
	if not path.exists():
		return None
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return None
	return payload if isinstance(payload, dict) else None


def _load_jsonl_if_exists(path: Path) -> List[Dict[str, Any]]:
	"""Best-effort load valid object rows while ignoring torn JSONL lines."""
	if not path.exists():
		return []
	items: List[Dict[str, Any]] = []
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			payload = json.loads(line)
		except Exception:
			continue
		if isinstance(payload, dict):
			items.append(payload)
	return items


def _infer_current_image(sample_dir: Path, status_payload: Optional[Dict[str, Any]], global_round: int, micro_edit_index: int) -> Optional[Path]:
	"""Recover the latest durable image using status first, then artifact order."""
	if status_payload:
		current_image = status_payload.get("current_image")
		if current_image:
			candidate = Path(str(current_image))
			if candidate.exists():
				return candidate
	final_image = sample_dir / "final.png"
	if final_image.exists():
		return final_image
	initial_image = sample_dir / "initial.png"
	if initial_image.exists() and global_round == 0:
		return initial_image
	round_initial = sample_round_dir(sample_dir, global_round) / "initial.png"
	if round_initial.exists():
		return round_initial
	micro_selected = sample_dir / f"micro_{micro_edit_index + 1:02d}_selected.png"
	if micro_selected.exists():
		return micro_selected
	for regenerate_name in ("flux_regenerate.png", "bagel_regenerate.png"):
		regenerate_image = sample_dir / f"round_{global_round:02d}_regenerate" / regenerate_name
		if regenerate_image.exists():
			return regenerate_image
	selected_global = sample_dir / f"round_{global_round:02d}_selected.png"
	if selected_global.exists():
		return selected_global
	for round_zero_name in ("round_00_flux_generate.png", "round_00_bagel_generate.png"):
		round_zero = sample_dir / round_zero_name
		if round_zero.exists():
			return round_zero
	return None


def _infer_needs_global_analysis(state: SampleState, status_payload: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
	"""Determine whether the resumed image still needs a canonical diagnosis."""
	if status_payload and isinstance(status_payload.get("needs_global_analysis"), bool):
		return bool(status_payload["needs_global_analysis"]), str(status_payload.get("pending_global_analysis_reason") or "resume_status")
	if state.done or state.error or state.current_image is None:
		return False, state.pending_global_analysis_reason
	if state.phase == "micro":
		return False, state.pending_global_analysis_reason
	analysis_path = round_error_analysis_path(state.sample_dir, state.global_round)
	legacy_path = state.sample_dir / f"round_{state.global_round:02d}_triage.json"
	if not analysis_path.exists() and not legacy_path.exists():
		return True, "resume_missing_triage"
	return False, state.pending_global_analysis_reason


def _pending_global_compare_complete(payload: Optional[Dict[str, Any]]) -> bool:
	"""Check that a pending whole-image comparison has all required assets."""
	if not isinstance(payload, dict):
		return False
	required = ("round", "round_dir", "analysis_path", "original")
	if not all(payload.get(key) for key in required):
		return False
	generated = payload.get("generated_candidates")
	if not isinstance(generated, list) or len(generated) != 1 or not isinstance(generated[0], dict):
		return False
	path = generated[0].get("path")
	return bool(path) and Path(str(path)).is_file()


def _pending_local_review_complete(payload: Optional[Dict[str, Any]]) -> bool:
	"""Check that a pending L1 retain review has all required assets."""
	if not isinstance(payload, dict):
		return False
	model_paths = payload.get("model_paths")
	if not isinstance(model_paths, dict):
		return False
	selected_models = payload.get("selected_models")
	if isinstance(selected_models, list) and selected_models:
		required_models = [str(model_name) for model_name in selected_models]
	else:
		required_models = [model_name for model_name in LOCAL_MODEL_ORDER if model_name in model_paths]
	input_path = payload.get("input_path")
	return (
		bool(required_models)
		and bool(input_path)
		and Path(str(input_path)).is_file()
		and all(model_paths.get(model_name) and Path(str(model_paths[model_name])).is_file() for model_name in required_models)
	)


def _has_resumable_work(state: SampleState) -> bool:
	"""Return whether durable artifacts imply unfinished work despite terminal flags."""
	if state.pending_global_compare or state.pending_local_review:
		return True
	if state.needs_global_analysis:
		return True
	if state.phase == "global":
		return True
	if state.phase == "regenerate":
		return True
	if state.phase == "micro" and state.micro_edit_index < len(state.micro_plan):
		return True
	return False


def _should_reopen_resumed_state(state: SampleState, max_actions: int) -> bool:
	"""Decide whether a previously finalized state must be reopened on resume."""
	if not state.done:
		return False
	if state.error:
		return _has_resumable_work(state)
	if state.final_reason == "max_repair_actions_reached_after_final_diagnosis":
		return state.current_image is not None and int(state.global_round) < int(max_actions)
	if state.final_reason != "loop_exhausted":
		return False
	return _has_resumable_work(state)


def _clear_action_dispatch_state(state: SampleState) -> None:
	"""Clear transient plan/review fields after crossing an action boundary."""
	state.pending_global_compare = None
	state.pending_local_review = None
	state.micro_plan = []
	state.micro_box_review_done = False
	state.micro_edit_index = 0
	state.micro_attempt = 0


def normalize_v10_action_budget_state(
	state: SampleState,
	max_actions: int,
	*,
	source: str,
	persist: bool = True,
) -> bool:
	"""Normalize legacy/corrupt action phases without ever dispatching over budget.

	``global_round`` counts completed repair actions.  A state may dispatch L1,
	L2, or L3 only while it is strictly below ``max_actions``.  At the boundary
	the only legal work is one fresh global diagnosis of the retained image.
	"""
	max_actions = int(max_actions)
	if max_actions < 0:
		raise ValueError(f"max_actions must be non-negative, got {max_actions}")
	if int(state.global_round) < 0:
		raise ValueError(f"Cannot normalize {state.sample_id}: global_round is negative")
	if state.phase not in {"global", "micro", "regenerate", "micro_final_review"}:
		raise ValueError(f"Cannot normalize {state.sample_id}: unsupported V10 phase {state.phase!r}")

	before = {
		"phase": state.phase,
		"global_round": int(state.global_round),
		"needs_global_analysis": bool(state.needs_global_analysis),
		"status": state.status,
		"pending_global_compare": bool(state.pending_global_compare),
		"pending_local_review": bool(state.pending_local_review),
		"micro_plan_size": len(state.micro_plan),
	}
	reasons: List[str] = []
	terminal_ready = bool(
		state.error
		or (state.done and state.final_reason != "loop_exhausted")
		or (
			not state.done
			and state.status in {"ok", "partial"}
			and state.final_image is not None
			and state.final_reason != "loop_exhausted"
		)
	)

	def request_global_diagnosis(reason: str) -> None:
		"""Move the normalized state to a fresh, non-action diagnosis boundary."""
		state.phase = "global"
		_clear_action_dispatch_state(state)
		state.needs_global_analysis = True
		state.pending_global_analysis_reason = reason
		if not state.done:
			state.status = "pending"
			state.final_reason = None
			state.final_image = None
		reasons.append(reason)

	if state.phase == "micro_final_review":
		state.phase = "global"
		_clear_action_dispatch_state(state)
		if not terminal_ready:
			request_global_diagnosis("legacy_micro_final_review_migrated")
		reasons.append("remove_legacy_micro_final_review")

	if int(state.global_round) > max_actions:
		state.phase = "global"
		_clear_action_dispatch_state(state)
		if not terminal_ready:
			state.needs_global_analysis = False
			state.pending_global_analysis_reason = "resume_round_exceeds_action_budget"
			state.status = "partial"
			state.final_reason = "resume_round_exceeds_action_budget"
			state.final_image = state.current_image
		reasons.append("round_exceeds_action_budget")
	elif (
		not terminal_ready
		and int(state.global_round) == max_actions
		and state.current_image is not None
		and (
			state.phase != "global"
			or bool(state.pending_global_compare)
			or bool(state.pending_local_review)
			or not state.needs_global_analysis
		)
	):
		# Pending reviews at this round can only come from an illegal extra action:
		# a valid final action is pending while global_round == max_actions - 1.
		request_global_diagnosis("action_budget_exhausted_final_diagnosis")
	elif (
		not terminal_ready
		and state.phase == "micro"
		and state.current_image is not None
		and not state.micro_plan
		and not state.pending_local_review
	):
		request_global_diagnosis("empty_l1_plan_reanalysis")

	after = {
		"phase": state.phase,
		"global_round": int(state.global_round),
		"needs_global_analysis": bool(state.needs_global_analysis),
		"status": state.status,
		"pending_global_compare": bool(state.pending_global_compare),
		"pending_local_review": bool(state.pending_local_review),
		"micro_plan_size": len(state.micro_plan),
	}
	changed = before != after
	if changed:
		append_event(
			state,
			{
				"event": "v10_action_budget_state_normalized",
				"source": source,
				"max_actions": max_actions,
				"reasons": list(dict.fromkeys(reasons)),
				"before": before,
				"after": after,
			},
		)
		if persist:
			save_status(state)
	return changed


def load_states(
	selected: Sequence[Tuple[int, str]],
	output_root: Path,
	resume: bool,
	max_actions: int,
) -> List[SampleState]:
	"""Create fresh states or reconstruct validated V10 states from disk.

	Resume rejects incompatible schema versions, restores only paths that still
	exist, verifies pending hand-offs, derives missing safe fields from round
	artifacts, and removes stale final outputs when unfinished work is detected.
	"""
	if int(max_actions) < 0:
		raise ValueError(f"max_actions must be non-negative, got {max_actions}")
	if not resume:
		return init_states(selected, output_root)

	states: List[SampleState] = []
	for index, prompt in selected:
		sample_id = safe_prompt_id(index, prompt)
		sample_dir = output_root / sample_id
		sample_dir.mkdir(parents=True, exist_ok=True)
		(sample_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
		status_payload = _load_json_if_exists(sample_dir / "status.json")
		state = SampleState(index=index, prompt=prompt, sample_dir=sample_dir, sample_id=sample_id)
		if status_payload:
			if int(status_payload.get("state_schema_version", -1)) != STATE_SCHEMA_VERSION:
				raise ValueError(
					f"Cannot resume {sample_id}: status.json is not V10 state schema {STATE_SCHEMA_VERSION}. "
					"Use a new output directory or migrate the run explicitly."
				)
			generation_prompt = status_payload.get("generation_prompt")
			initial_generator_family = status_payload.get("initial_generator_family")
			if initial_generator_family:
				state.initial_generator_family = str(initial_generator_family)
			initial_generator_policy_digest = status_payload.get("initial_generator_policy_digest")
			if initial_generator_policy_digest:
				state.initial_generator_policy_digest = str(initial_generator_policy_digest)
			if generation_prompt:
				state.generation_prompt = str(generation_prompt)
			state.phase = str(status_payload.get("phase") or state.phase)
			state.status = str(status_payload.get("status") or state.status)
			state.done = bool(status_payload.get("done", state.done))
			state.global_round = int(status_payload.get("global_round", state.global_round) or 0)
			state.micro_edit_index = int(status_payload.get("micro_edit_index", state.micro_edit_index) or 0)
			state.final_reason = status_payload.get("final_reason")
			pending_global_compare = status_payload.get("pending_global_compare")
			if isinstance(pending_global_compare, dict):
				state.pending_global_compare = pending_global_compare
			pending_local_review = status_payload.get("pending_local_review")
			if isinstance(pending_local_review, dict):
				state.pending_local_review = pending_local_review
			micro_plan = status_payload.get("micro_plan")
			if isinstance(micro_plan, list):
				state.micro_plan = [item for item in micro_plan if isinstance(item, dict)]
			repair_memory = status_payload.get("repair_memory")
			if isinstance(repair_memory, list):
				state.repair_memory = [item for item in repair_memory if isinstance(item, dict)][-5:]
			state.micro_box_review_done = bool(status_payload.get("micro_box_review_done", state.micro_box_review_done))
			best_round = status_payload.get("best_round")
			if isinstance(best_round, dict):
				state.best_round = best_round
			best_rank = status_payload.get("best_rank")
			if isinstance(best_rank, int):
				state.best_rank = best_rank
			state.error = status_payload.get("error")
			final_image = status_payload.get("final_image")
			if final_image:
				candidate = Path(str(final_image))
				if candidate.exists():
					state.final_image = candidate
			started_at = status_payload.get("started_at")
			if isinstance(started_at, (int, float)):
				state.started_at = float(started_at)
		state.history = _load_jsonl_if_exists(sample_dir / "timeline.jsonl")
		if state.generation_prompt is None:
			prompt_reasoning_payload = _load_json_if_exists(sample_dir / "generation_prompt.json")
			if prompt_reasoning_payload and prompt_reasoning_payload.get("generation_prompt"):
				state.generation_prompt = str(prompt_reasoning_payload["generation_prompt"])
		state.current_image = _infer_current_image(sample_dir, status_payload, state.global_round, state.micro_edit_index)
		if state.pending_global_compare and not _pending_global_compare_complete(state.pending_global_compare):
			state.pending_global_compare = None
		if state.pending_local_review and not _pending_local_review_complete(state.pending_local_review):
			state.pending_local_review = None
		micro_plan_payload = _load_json_if_exists(sample_round_dir(sample_dir, state.global_round) / "repair_plan.json")
		if micro_plan_payload is None:
			micro_plan_payload = _load_json_if_exists(sample_dir / f"round_{state.global_round:02d}_micro_plan.json")
		if not state.micro_plan and micro_plan_payload and isinstance(micro_plan_payload.get("plan"), list):
			state.micro_plan = [item for item in micro_plan_payload["plan"] if isinstance(item, dict)]
		box_review_final = sample_round_dir(sample_dir, state.global_round) / "bbox_review_final.json"
		legacy_box_review_final = sample_dir / f"round_{state.global_round:02d}_micro_box_review_final.json"
		if not state.micro_box_review_done and (box_review_final.exists() or legacy_box_review_final.exists()):
			state.micro_box_review_done = True
		best_round_payload = None
		if state.best_round is None and state.done:
			summary_payload = _load_json_if_exists(sample_dir / "summary.json")
			if summary_payload:
				best_round_payload = summary_payload.get("best_round")
		if isinstance(best_round_payload, dict):
			state.best_round = best_round_payload
			analysis = best_round_payload.get("analysis")
			if isinstance(analysis, dict):
				state.best_rank = CATEGORY_RANK.get(str(analysis.get("category", "Unknown")), state.best_rank)
		state.needs_global_analysis, state.pending_global_analysis_reason = _infer_needs_global_analysis(state, status_payload)
		normalized = normalize_v10_action_budget_state(
			state,
			max_actions,
			source="resume",
			persist=False,
		)
		reopen_reason = state.final_reason
		if _should_reopen_resumed_state(state, max_actions):
			state.done = False
			state.status = "pending"
			state.error = None
			state.final_reason = None
			state.final_image = None
			if reopen_reason == "max_repair_actions_reached_after_final_diagnosis":
				state.phase = "global"
				_clear_action_dispatch_state(state)
				state.needs_global_analysis = True
				state.pending_global_analysis_reason = "resume_action_budget_extended"
				append_event(
					state,
					{
						"event": "resume_action_budget_extended",
						"completed_actions": int(state.global_round),
						"new_max_actions": int(max_actions),
					},
				)
			for stale_name in ("summary.json", "final.png"):
				stale_path = sample_dir / stale_name
				if stale_path.exists():
					stale_path.unlink()
			normalized = True
		states.append(state)
		if normalized or not status_payload:
			save_status(state)
	return states
