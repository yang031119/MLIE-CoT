"""Qwen batch execution and retry logic."""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Sequence

from .vcot_branch import HeavyRuntimeManager

from .config import QWEN_MULTI_IMAGE_INPUT_MAX_SIDE
from .image_utils import next_ladder_side, set_prepared_qwen_image
from .logging_utils import log_stage
from .memory import (
	is_memory_related_exception,
	is_qwen_load_memory_wait_exception,
	next_retry_max_side,
	release_qwen_hard,
)
from .qwen_subprocess import run_qwen_worker_batch
from .contracts import model_visible_diagnosis_text, qwen_response_has_degenerate_repetition
from .route_head import append_route_marker


def _attach_route_predictions(
	planner: Any,
	items: Sequence[Dict[str, Any]],
	results: Sequence[Dict[str, Any]],
) -> None:
	"""Route only validated diagnoses and expose factual fields to the encoder."""
	indices = [
		index
		for index, (item, result) in enumerate(zip(items, results))
		if item.get("route_prompt")
		and item.get("images")
		and planner.route_head is not None
		and result.get("valid_json")
		and not result.get("fully_satisfied")
	]
	if not indices:
		return
	predictions = planner.predict_routes(
		[str(items[index]["route_prompt"]) for index in indices],
		[
			model_visible_diagnosis_text(results[index].get("canonical_diagnosis") or results[index])
			for index in indices
		],
		[items[index].get("route_image") or items[index]["images"][0] for index in indices],
	)
	for index, prediction in zip(indices, predictions):
		result = results[index]
		result["route_prediction"] = prediction
		result["raw_output"] = append_route_marker(str(result.get("raw_output") or ""), prediction)

def _is_nf4_illegal_memory_error(exc: Exception, args: argparse.Namespace) -> bool:
	"""Detect the narrow NF4 CUDA-address failure eligible for int8 fallback."""
	quant = str(getattr(args, "qwen_quantization", "") or "").lower()
	if quant not in {"4bit", "int4", "nf4"}:
		return False
	text = str(exc).lower()
	return "illegal memory access" in text or "cudaerrorillegaladdress" in text


def run_qwen_batch(
	planner: Any,
	prepared: Sequence[Dict[str, Any]],
	max_new_tokens: int,
	validator,
	retry_instruction: str,
	batch_size: int,
) -> List[Dict[str, Any]]:
	"""Generate, validate, optionally JSON-retry, and route an in-process batch."""
	results: List[Dict[str, Any]] = []
	step = max(1, int(batch_size))
	batch_kind = str(prepared[0].get("batch_kind", "qwen")) if prepared else "qwen"
	if prepared:
		log_stage(f"qwen-batch kind={batch_kind} count={len(prepared)} batch_size={step}")
	if step == 1:
		for item in prepared:
			if item.get("log_label"):
				log_stage(item["log_label"])
			try:
				raw = planner._generate(
					item["messages"],
					item["images"],
					max_new_tokens,
				)
			except Exception:
				if item.get("log_label"):
					log_stage(f"{item['log_label']} failed")
				raise
			result = validator(raw, item)
			if not result.get("valid_json"):
				if item.get("retry_log_label"):
					log_stage(item["retry_log_label"])
				retry_messages = append_qwen_retry_messages(item["messages"], raw, retry_instruction)
				retry_generate = (
					getattr(planner, "_generate_retry", planner._generate)
					if qwen_response_has_degenerate_repetition(raw)
					else planner._generate
				)
				retry_raw = retry_generate(retry_messages, item["images"], max_new_tokens)
				result = validator(retry_raw, item)
				result["json_retry_count"] = 1
			_attach_route_predictions(planner, [item], [result])
			results.append(result)
		return results
	for start in range(0, len(prepared), step):
		chunk = prepared[start : start + step]
		log_stage(f"qwen-batch kind={batch_kind} chunk={start // step + 1} size={len(chunk)}")
		raw_outputs = planner._generate_batch(
			[item["messages"] for item in chunk],
			[item.get("images", []) for item in chunk],
			max_new_tokens,
		)
		chunk_results: List[Dict[str, Any]] = []
		for item, raw in zip(chunk, raw_outputs):
			result = validator(raw, item)
			chunk_results.append(result)
		invalid_indices = [index for index, result in enumerate(chunk_results) if not result.get("valid_json")]
		for retry_decoding in (False, True):
			group = [
				index for index in invalid_indices
				if qwen_response_has_degenerate_repetition(raw_outputs[index]) is retry_decoding
			]
			if not group:
				continue
			retry_messages_batch = []
			retry_images_batch = []
			for index in group:
				item = chunk[index]
				if item.get("retry_log_label"):
					log_stage(item["retry_log_label"])
				retry_messages_batch.append(
					append_qwen_retry_messages(item["messages"], raw_outputs[index], retry_instruction)
				)
				retry_images_batch.append(item["images"])
			if len(group) == 1:
				retry_generate = (
					getattr(planner, "_generate_retry", planner._generate)
					if retry_decoding else planner._generate
				)
				retry_outputs = [retry_generate(retry_messages_batch[0], retry_images_batch[0], max_new_tokens)]
			else:
				retry_generate_batch = (
					getattr(planner, "_generate_batch_retry", planner._generate_batch)
					if retry_decoding else planner._generate_batch
				)
				retry_outputs = retry_generate_batch(retry_messages_batch, retry_images_batch, max_new_tokens)
			for index, retry_raw in zip(group, retry_outputs):
				result = validator(retry_raw, chunk[index])
				result["json_retry_count"] = 1
				chunk_results[index] = result
		_attach_route_predictions(planner, chunk, chunk_results)
		results.extend(chunk_results)
	return results


def append_qwen_retry_messages(
	messages: Sequence[Dict[str, Any]],
	raw: str,
	retry_instruction: str,
) -> List[Dict[str, Any]]:
	"""Append one correction turn while bounding echoed malformed output."""
	return list(messages) + [
		{"role": "assistant", "content": [{"type": "text", "text": str(raw)[:1200]}]},
		{"role": "user", "content": [{"type": "text", "text": retry_instruction}]},
	]


def run_qwen_batch_subprocess(
	args: argparse.Namespace,
	prepared: Sequence[Dict[str, Any]],
	max_new_tokens: int,
	validator,
	retry_instruction: str,
	batch_size: int,
) -> List[Dict[str, Any]]:
	"""Execute serialized jobs out of process and preserve partial valid outputs.

	A worker may finish some items before another item or GPU fails.  Valid
	partial callbacks are committed into their original slots so fallback
	strategies process only the unresolved subset.
	"""
	results: List[Optional[Dict[str, Any]]] = [None] * len(prepared)
	step = max(1, int(batch_size))
	batch_kind = str(prepared[0].get("batch_kind", "qwen")) if prepared else "qwen"
	if prepared:
		log_stage(f"qwen-batch kind={batch_kind} count={len(prepared)} batch_size={step} worker=subprocess")
	for item in prepared:
		if item.get("log_label"):
			log_stage(item["log_label"])

	partial_seen: set[int] = set()

	def handle_partial(index: int, raw: str) -> None:
		"""Validate and commit one first-attempt worker result at most once."""
		if index in partial_seen:
			return
		partial_seen.add(index)
		item = prepared[index]
		result = validator(raw, item)
		callback = item.get("partial_result_callback")
		if callable(callback):
			callback(result)
		if not result.get("valid_json"):
			return
		results[index] = result

	raw_outputs = run_qwen_worker_batch(prepared, args, max_new_tokens, step, partial_callback=handle_partial)
	retry_items: List[Dict[str, Any]] = []
	retry_indices: List[int] = []
	for index, (item, raw) in enumerate(zip(prepared, raw_outputs)):
		if results[index] is not None:
			continue
		result = validator(raw, item)
		if result.get("valid_json"):
			results[index] = result
			continue
		if item.get("retry_log_label"):
			log_stage(item["retry_log_label"])
		retry_item = dict(item)
		retry_item["messages"] = append_qwen_retry_messages(item["messages"], raw, retry_instruction)
		retry_items.append(retry_item)
		retry_indices.append(index)
	if retry_items:
		retry_partial_seen: set[int] = set()

		def handle_retry_partial(retry_index: int, raw: str) -> None:
			"""Map a correction-worker slot back to its original batch slot."""
			if retry_index in retry_partial_seen:
				return
			retry_partial_seen.add(retry_index)
			original_index = retry_indices[retry_index]
			retry_item = retry_items[retry_index]
			result = validator(raw, retry_item)
			result["json_retry_count"] = 1
			callback = retry_item.get("partial_result_callback")
			if callable(callback):
				callback(result)
			if not result.get("valid_json"):
				return
			results[original_index] = result

		retry_raw_outputs = run_qwen_worker_batch(
			retry_items,
			args,
			max_new_tokens,
			step,
			partial_callback=handle_retry_partial,
		)
		for retry_index, (original_index, retry_item, raw) in enumerate(zip(retry_indices, retry_items, retry_raw_outputs)):
			if results[original_index] is not None:
				continue
			result = validator(raw, retry_item)
			result["json_retry_count"] = 1
			results[original_index] = result
	return [result or {"valid_json": False, "raw": ""} for result in results]


def run_qwen_batch_with_retries(
	runtime: HeavyRuntimeManager,
	args: argparse.Namespace,
	prepared: Sequence[Dict[str, Any]],
	max_new_tokens: int,
	validator,
	retry_instruction: str,
	batch_size: int,
) -> List[Dict[str, Any]]:
	"""Run Qwen with ordered memory fallbacks and no semantic candidate replay.

	Recovery order is: release editor models, retry a known NF4 illegal-address
	load in int8, wait for placement capacity, reduce batch size to one, then
	optionally step down the configured image-size ladder.  Non-memory failures
	propagate immediately.
	"""
	if not prepared:
		return []
	current_batch_size = max(1, int(batch_size))
	min_side = max(1, int(args.retry_min_side))
	tried_int8_fallback = False
	while True:
		try:
			runtime.release_bagel()
			runtime.release_flux()
			if bool(getattr(args, "qwen_inprocess", False)):
				return run_qwen_batch(
					runtime.qwen(),
					prepared,
					max_new_tokens,
					validator,
					retry_instruction,
					current_batch_size,
				)
			release_qwen_hard(runtime)
			return run_qwen_batch_subprocess(
				args,
				prepared,
				max_new_tokens,
				validator,
				retry_instruction,
				current_batch_size,
			)
		except Exception as exc:
			if not is_memory_related_exception(exc):
				raise
			release_qwen_hard(runtime)
			if (not tried_int8_fallback) and _is_nf4_illegal_memory_error(exc, args):
				log_stage("qwen-batch memory-retry quantization=nf4->8bit reason=illegal-memory-access")
				args.qwen_quantization = "8bit"
				tried_int8_fallback = True
				continue
			if is_qwen_load_memory_wait_exception(exc):
				log_stage(
					f"qwen-batch memory-retry load-wait reason={type(exc).__name__} "
					f"wait_sec={args.qwen_memory_wait_sec:g}"
				)
				time.sleep(max(1.0, float(args.qwen_memory_wait_sec)))
				continue
			if current_batch_size > 1:
				log_stage(
					f"qwen-batch memory-retry reason={type(exc).__name__} batch_size={current_batch_size}->1"
				)
				current_batch_size = 1
				continue
			if bool(getattr(args, "qwen_disable_image_shrink_retry", False)):
				log_stage(f"qwen-batch memory-retry reason={type(exc).__name__} shrink=disabled")
				raise
			shrunk = False
			for item in prepared:
				if item.get("qwen_side_ladder"):
					current_side = int(item.get("current_qwen_max_side") or QWEN_MULTI_IMAGE_INPUT_MAX_SIDE)
					next_side = next_ladder_side(current_side, item["qwen_side_ladder"])
				elif item.get("qwen_image") is None:
					continue
				else:
					current_side = int(item.get("current_qwen_max_side") or max(item["qwen_image"].size))
					next_side = next_retry_max_side(current_side, min_side, args.retry_scale)
				if next_side < current_side:
					set_prepared_qwen_image(item, next_side)
					shrunk = True
			if shrunk:
				def describe_shrunk_item(item: Dict[str, Any]) -> str:
					"""Format the current retry side for one queued sample."""
					side = item.get("current_qwen_max_side")
					if side is None and item.get("qwen_image") is not None:
						side = max(item["qwen_image"].size)
					if side is None:
						side = "no-image"
					return f"{item['state'].sample_id}:{side}"

				log_stage(
					"qwen-batch memory-retry shrink="
					+ ", ".join(describe_shrunk_item(item) for item in prepared)
				)
				continue
			log_stage(f"qwen-batch memory-retry reason={type(exc).__name__} shrink=unavailable")
			raise
