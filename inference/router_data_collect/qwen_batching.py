"""Small, runtime-agnostic helpers for reliable Qwen stage batches."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from inference.main_method.modules.contracts import qwen_response_has_degenerate_repetition


def append_json_retry_messages(
	messages: Sequence[Mapping[str, Any]],
	raw: str,
	retry_instruction: str,
) -> list[dict[str, Any]]:
	"""Match V10's one-turn schema-correction conversation exactly."""

	return [dict(message) for message in messages] + [
		{"role": "assistant", "content": [{"type": "text", "text": str(raw)[:1200]}]},
		{"role": "user", "content": [{"type": "text", "text": retry_instruction}]},
	]


def adaptive_generate_batch(
	planner: Any,
	messages_batch: Sequence[Sequence[Mapping[str, Any]]],
	images_batch: Sequence[Sequence[Any]],
	max_new_tokens: int,
	*,
	retry_decoding: bool = False,
) -> list[str | Exception]:
	"""Generate in one batch, recursively bisecting failures down to scalars.

	The output remains index-aligned.  A scalar failure is returned in its own
	slot so one corrupt image/OOM does not discard successful neighbors.
	"""

	if len(messages_batch) != len(images_batch):
		raise ValueError("Qwen messages/images batch lengths differ")
	if not messages_batch:
		return []
	try:
		if len(messages_batch) == 1:
			generate = getattr(planner, "_generate_retry", planner._generate) if retry_decoding else planner._generate
			return [
				generate(
					list(messages_batch[0]),
					list(images_batch[0]),
					int(max_new_tokens),
				)
			]
		generate_batch = getattr(planner, "_generate_batch_retry", planner._generate_batch) if retry_decoding else planner._generate_batch
		outputs = generate_batch(
			[list(messages) for messages in messages_batch],
			[list(images) for images in images_batch],
			int(max_new_tokens),
		)
		if len(outputs) != len(messages_batch):
			raise RuntimeError(
				f"Qwen batch output mismatch: expected={len(messages_batch)} actual={len(outputs)}"
			)
		return [str(output) for output in outputs]
	except Exception as exc:
		if len(messages_batch) == 1:
			return [exc]
		midpoint = len(messages_batch) // 2
		return adaptive_generate_batch(
			planner,
			messages_batch[:midpoint],
			images_batch[:midpoint],
			max_new_tokens,
			retry_decoding=retry_decoding,
		) + adaptive_generate_batch(
			planner,
			messages_batch[midpoint:],
			images_batch[midpoint:],
			max_new_tokens,
			retry_decoding=retry_decoding,
		)


def generate_with_one_json_retry(
	planner: Any,
	messages_batch: Sequence[Sequence[Mapping[str, Any]]],
	images_batch: Sequence[Sequence[Any]],
	max_new_tokens: int,
	validator: Callable[[str, int], Mapping[str, Any]],
	retry_instruction: str,
) -> list[dict[str, Any]]:
	"""Validate first responses and retry only schema-invalid rows once."""

	raw_outputs = adaptive_generate_batch(planner, messages_batch, images_batch, max_new_tokens)
	records: list[dict[str, Any]] = []
	invalid_indices: list[int] = []
	for index, raw in enumerate(raw_outputs):
		if isinstance(raw, Exception):
			records.append(
				{
					"valid_json": False,
					"json_retry_count": 0,
					"raw_response": "",
					"generation_exception": raw,
				}
			)
			continue
		try:
			normalized = dict(validator(raw, index))
		except Exception as exc:
			normalized = {"valid_json": False, "validation_exception": exc}
		record = {
			**normalized,
			"valid_json": bool(normalized.get("valid_json")),
			"json_retry_count": 0,
			"raw_response": raw,
		}
		records.append(record)
		if not record["valid_json"]:
			invalid_indices.append(index)

	if not invalid_indices:
		return records
	for retry_decoding in (False, True):
		group = [
			index for index in invalid_indices
			if qwen_response_has_degenerate_repetition(raw_outputs[index]) is retry_decoding
		]
		if not group:
			continue
		retry_messages = [
			append_json_retry_messages(messages_batch[index], str(raw_outputs[index]), retry_instruction)
			for index in group
		]
		retry_images = [images_batch[index] for index in group]
		retry_outputs = adaptive_generate_batch(
			planner,
			retry_messages,
			retry_images,
			max_new_tokens,
			retry_decoding=retry_decoding,
		)
		for index, retry_raw in zip(group, retry_outputs):
			initial_raw = str(raw_outputs[index])
			if isinstance(retry_raw, Exception):
				records[index] = {
					"valid_json": False,
					"json_retry_count": 1,
					"raw_response": "",
					"initial_raw_response": initial_raw,
					"generation_exception": retry_raw,
				}
				continue
			try:
				normalized = dict(validator(retry_raw, index))
			except Exception as exc:
				normalized = {"valid_json": False, "validation_exception": exc}
			records[index] = {
				**normalized,
				"valid_json": bool(normalized.get("valid_json")),
				"json_retry_count": 1,
				"raw_response": retry_raw,
				"initial_raw_response": initial_raw,
			}
	return records


__all__ = ["adaptive_generate_batch", "append_json_retry_messages", "generate_with_one_json_retry"]
