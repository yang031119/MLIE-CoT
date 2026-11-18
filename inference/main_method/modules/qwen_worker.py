#!/usr/bin/env python3
"""Qwen generation worker for MLIE-CoT.

This process intentionally owns the Qwen model lifetime. When the process exits,
CUDA contexts and accelerate device maps are released by the OS instead of relying
on in-process cache cleanup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = SCRIPT_DIR.parent
if str(INFERENCE_DIR) not in sys.path:
	sys.path.insert(0, str(INFERENCE_DIR))

from modules.vcot_branch import QwenPlanner, parse_dtype  # noqa: E402
from modules.contracts import canonical_diagnosis_text_from_raw  # noqa: E402
from modules.route_head import append_route_marker  # noqa: E402


def load_json(path: Path) -> Dict[str, Any]:
	"""Load the self-contained job description written by the parent process."""
	return json.loads(path.read_text(encoding="utf-8"))


def save_outputs(path: Path, outputs: List[Dict[str, Any]], complete: bool) -> None:
	"""Checkpoint accumulated outputs so the parent can recover partial progress."""
	payload = {"complete": complete, "outputs": outputs}
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rebuild_messages(serialized_messages: List[Dict[str, Any]], images: List[Image.Image]) -> List[Dict[str, Any]]:
	"""Replace serialized image indices with this process's decoded PIL objects."""
	messages: List[Dict[str, Any]] = []
	for message in serialized_messages:
		content: List[Dict[str, Any]] = []
		for part in message.get("content", []):
			if part.get("type") == "image":
				image_index = int(part["image_index"])
				content.append({"type": "image", "image": images[image_index]})
			else:
				content.append(dict(part))
		messages.append({"role": message.get("role", "user"), "content": content})
	return messages


def _open_item_images(item: Dict[str, Any]) -> List[Image.Image]:
	"""Open ordered multimodal inputs belonging to one serialized item."""
	return [Image.open(path).convert("RGB") for path in item.get("image_paths", [])]


def _open_route_image(item: Dict[str, Any]) -> Image.Image | None:
	"""Open the unannotated route-feature image when routing is requested."""
	path = str(item.get("route_image_path") or "")
	return Image.open(path).convert("RGB") if path else None


def _close_images(images: List[Image.Image]) -> None:
	"""Release decoded image buffers deterministically after generation."""
	for image in images:
		image.close()


def _can_batch_items(items: List[Dict[str, Any]]) -> bool:
	"""Return whether items have compatible equal-cardinality image layouts."""
	if len(items) <= 1:
		return False
	image_counts = {len(item.get("image_paths", [])) for item in items}
	# Qwen3VLProcessor accepts a flat image list in conversation/placeholder
	# order, so equal-cardinality paired reviews are safe to batch.
	return len(image_counts) == 1


def _generate_item(planner: QwenPlanner, item: Dict[str, Any], max_new_tokens: int) -> str:
	"""Generate one response and append a route only for a valid problem diagnosis."""
	images = _open_item_images(item)
	route_image = _open_route_image(item)
	try:
		messages = rebuild_messages(item["messages"], images)
		raw = planner._generate(messages, images, max_new_tokens)
		route_prompt = str(item.get("route_prompt") or "")
		canonical = canonical_diagnosis_text_from_raw(raw) if route_prompt else None
		feature_image = route_image or (images[0] if images else None)
		if canonical and not json.loads(canonical)["fully_satisfied"] and feature_image is not None and planner.route_head is not None:
			prediction = planner.predict_routes([route_prompt], [canonical], [feature_image])[0]
			raw = append_route_marker(raw, prediction)
		return raw
	finally:
		_close_images(images)
		if route_image is not None:
			route_image.close()


def _generate_item_batch(planner: QwenPlanner, items: List[Dict[str, Any]], max_new_tokens: int) -> List[str]:
	"""Generate compatible items together and batch their route-feature encoding."""
	images_batch: List[List[Image.Image]] = []
	route_images: List[Image.Image | None] = []
	messages_batch: List[List[Dict[str, Any]]] = []
	try:
		for item in items:
			images = _open_item_images(item)
			images_batch.append(images)
			route_images.append(_open_route_image(item))
			messages_batch.append(rebuild_messages(item["messages"], images))
		raw_outputs = planner._generate_batch(messages_batch, images_batch, max_new_tokens)
		if len(raw_outputs) != len(items):
			raise RuntimeError(
				f"Qwen batch output mismatch: expected={len(items)} actual={len(raw_outputs)}"
			)
		canonical_by_index = {
			index: canonical
			for index, item in enumerate(items)
			if str(item.get("route_prompt") or "")
			for canonical in [canonical_diagnosis_text_from_raw(raw_outputs[index])]
			if canonical is not None and not json.loads(canonical)["fully_satisfied"]
		}
		route_indices = [
			index
			for index, (item, images) in enumerate(zip(items, images_batch))
			if index in canonical_by_index
			and (route_images[index] is not None or images)
			and planner.route_head is not None
		]
		if route_indices:
			predictions = planner.predict_routes(
				[str(items[index]["route_prompt"]) for index in route_indices],
				[canonical_by_index[index] for index in route_indices],
				[route_images[index] or images_batch[index][0] for index in route_indices],
			)
			for index, prediction in zip(route_indices, predictions):
				raw_outputs[index] = append_route_marker(raw_outputs[index], prediction)
		return raw_outputs
	finally:
		for images in images_batch:
			_close_images(images)
		for route_image in route_images:
			if route_image is not None:
				route_image.close()


def run_job(job: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
	"""Own one Qwen lifetime and checkpoint outputs after every completed unit.

	A compatible chunk is generated as a batch; incompatible image layouts fall
	back to scalar calls.  ``output_path`` remains readable if a later item
	fails, allowing the parent to retain completed responses.
	"""
	qwen_args = job["qwen_args"]
	planner = QwenPlanner(
		model_path=qwen_args["model"],
		dtype=parse_dtype(qwen_args["dtype"]),
		max_memory_per_gpu=qwen_args["max_memory_per_gpu"],
		min_free_gib=float(qwen_args["min_free_gib"]),
		per_gpu_extra_reserve={},
		quantization=qwen_args["quantization"],
		allow_cpu_offload=bool(qwen_args.get("allow_cpu_offload", False)),
		route_head_path=str(qwen_args.get("route_head") or "") or None,
	)
	max_new_tokens = int(job["max_new_tokens"])
	batch_size = max(1, int(job.get("batch_size", 1)))
	items = list(job.get("items", []))
	outputs: List[Dict[str, Any]] = []
	try:
		for start in range(0, len(items), batch_size):
			chunk = items[start : start + batch_size]
			if _can_batch_items(chunk):
				raw_outputs = _generate_item_batch(planner, chunk, max_new_tokens)
				for item, raw in zip(chunk, raw_outputs):
					outputs.append({"id": item["id"], "raw": raw})
				save_outputs(output_path, outputs, complete=False)
				continue
			for item in chunk:
				raw = _generate_item(planner, item, max_new_tokens)
				outputs.append({"id": item["id"], "raw": raw})
				save_outputs(output_path, outputs, complete=False)
	finally:
		# Drop references before process teardown. The process exit is the real cleanup.
		planner.model = None
		planner.processor = None
	return {"complete": True, "outputs": outputs}


def main(argv: List[str] | None = None) -> int:
	"""CLI entry point for exactly one serialized parent-process job."""
	parser = argparse.ArgumentParser()
	parser.add_argument("--job", required=True)
	parser.add_argument("--output", required=True)
	args = parser.parse_args(argv)
	job_path = Path(args.job)
	output_path = Path(args.output)
	result = run_job(load_json(job_path), output_path)
	save_outputs(output_path, result["outputs"], complete=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
