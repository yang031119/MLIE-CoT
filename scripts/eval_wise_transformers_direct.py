#!/usr/bin/env python3
"""Run the repository WISE prompt with transformers/Qwen workers.

This is a deterministic, resumable artifact wrapper around the V10 Qwen worker.
Use ``prepare``, launch the emitted shard jobs, then use ``finalize``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/wise"
MODEL = "Qwen/Qwen3.5-35B-A3B"
DATA_FILES = (
	"cultural_common_sense_verified.json",
	"spatio-temporal_reasoning_verified.json",
	"natural_science_verified.json",
)
CATEGORIES = ("CULTURE", "TIME", "SPACE", "BIOLOGY", "PHYSICS", "CHEMISTRY")


def load_wise_module() -> Any:
	path = Path(os.environ.get("WISE_ROOT", str(ROOT / "third_party/WISE"))) / "vllm_eval.py"
	spec = importlib.util.spec_from_file_location("wise_vllm_eval", path)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"cannot import {path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def load_prompts() -> dict[int, dict[str, Any]]:
	rows: dict[int, dict[str, Any]] = {}
	for name in DATA_FILES:
		path = Path(os.environ.get("WISE_ROOT", str(ROOT / "third_party/WISE"))) / "data_verified" / name
		for row in json.loads(path.read_text(encoding="utf-8")):
			rows[int(row["prompt_id"])] = row
	if set(rows) != set(range(1, 1001)):
		raise ValueError(f"WISE prompt set incomplete: {len(rows)}")
	return rows


def worker_messages(prompt: dict[str, Any], wise: Any) -> list[dict[str, Any]]:
	messages = wise.build_evaluation_messages(prompt, "")
	converted: list[dict[str, Any]] = []
	for message in messages:
		content = []
		for part in message["content"]:
			if part["type"] == "image_url":
				content.append({"type": "image", "image_index": 0})
			else:
				content.append(dict(part))
		converted.append({"role": message["role"], "content": content})
	return converted


def prepare(args: argparse.Namespace) -> None:
	run = args.run.resolve()
	image_dir = (args.image_dir or run / "wise_eval_images/selected").resolve()
	out = (args.output or run / "wise_eval_direct_qwen35").resolve()
	out.mkdir(parents=True, exist_ok=True)
	prompts = load_prompts()
	wise = load_wise_module()
	items = []
	for prompt_id in range(1, 1001):
		image = image_dir / f"{prompt_id}.png"
		if not image.is_file():
			raise FileNotFoundError(image)
		items.append(
			{
				"id": str(prompt_id),
				"image_paths": [str(image)],
				"messages": worker_messages(prompts[prompt_id], wise),
			}
		)
	for shard in range(args.num_shards):
		job = {
			"qwen_args": {
				"model": str(args.model),
				"dtype": "bfloat16",
				"max_memory_per_gpu": args.max_memory_per_gpu,
				"min_free_gib": 16.0,
				"quantization": "none",
				"allow_cpu_offload": False,
				"route_head": "",
			},
			"max_new_tokens": 8,
			"batch_size": args.batch_size,
			"seed": args.seed,
			"decoding_contract": (
				{
					"do_sample": True,
					"temperature": args.temperature,
					"top_p": args.top_p,
					"top_k": args.top_k,
				}
				if args.do_sample
				else {"do_sample": False}
			),
			"items": [item for item in items if (int(item["id"]) - 1) % args.num_shards == shard],
		}
		(out / f"job_{shard}.json").write_text(json.dumps(job, ensure_ascii=False) + "\n", encoding="utf-8")
	manifest = {
		"protocol": "wise_verified_binary_transformers_direct_v1",
		"model": str(args.model),
		"dtype": "bfloat16",
		"quantization": "none",
		"do_sample": args.do_sample,
		"temperature": args.temperature if args.do_sample else None,
		"top_p": args.top_p if args.do_sample else None,
		"top_k": args.top_k if args.do_sample else None,
		"max_new_tokens": 8,
		"image_dir": str(image_dir),
		"num_shards": args.num_shards,
		"batch_size": args.batch_size,
		"seed": args.seed,
		"prompt_builder": "third_party/WISE/vllm_eval.py:build_evaluation_messages",
	}
	(out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	print(json.dumps({"prepared": 1000, "output": str(out)}))


def extract_score(raw: str) -> float | None:
	match = re.search(r"\*{0,2}Score\*{0,2}\s*[::]?\s*([01])\b", raw, re.IGNORECASE)
	if match:
		return float(match.group(1))
	bare = re.findall(r"(?m)^\s*([01])\s*$", raw)
	return float(bare[0]) if len(bare) == 1 else None


def category_for(prompt_id: int) -> str:
	if prompt_id <= 400:
		return "CULTURE"
	if prompt_id <= 520:
		return "TIME"
	if prompt_id <= 640:
		return "SPACE"
	if prompt_id <= 760:
		return "BIOLOGY"
	if prompt_id <= 880:
		return "PHYSICS"
	return "CHEMISTRY"


def finalize(args: argparse.Namespace) -> None:
	run = args.run.resolve()
	out = (args.output or run / "wise_eval_direct_qwen35").resolve()
	prompts = load_prompts()
	raw_by_id: dict[int, str] = {}
	for path in sorted(out.glob("output_*.json")):
		payload = json.loads(path.read_text(encoding="utf-8"))
		if not payload.get("complete"):
			raise ValueError(f"incomplete output: {path}")
		for row in payload["outputs"]:
			raw_by_id[int(row["id"])] = str(row["raw"])
	if set(raw_by_id) != set(range(1, 1001)):
		raise ValueError(f"worker output set incomplete: {len(raw_by_id)}")

	rows = []
	invalid = []
	by_category: dict[str, list[float]] = defaultdict(list)
	for prompt_id in range(1, 1001):
		score = extract_score(raw_by_id[prompt_id])
		if score is None:
			invalid.append(prompt_id)
			continue
		category = category_for(prompt_id)
		by_category[category].append(score)
		rows.append(
			{
				"prompt_id": prompt_id,
				"Subcategory": prompts[prompt_id]["Subcategory"],
				"score": score,
				"raw": raw_by_id[prompt_id],
			}
		)
	if invalid:
		(out / "invalid_prompt_ids.json").write_text(json.dumps(invalid) + "\n", encoding="utf-8")
		raise ValueError(f"unparseable WISE outputs: {invalid}")
	(out / "scores_results.jsonl").write_text(
		"".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
	)
	means = {category: sum(by_category[category]) / len(by_category[category]) for category in CATEGORIES}
	official = 0.4 * means["CULTURE"] + 0.12 * sum(means[category] for category in CATEGORIES[1:])
	legacy = (
		0.4 * means["CULTURE"] + 0.167 * means["TIME"] + 0.133 * means["SPACE"]
		+ 0.1 * sum(means[category] for category in CATEGORIES[3:])
	)
	result = {
		"num_samples": 1000,
		"categories": means,
		"official_calculate_verified_overall": official,
		"legacy_report_weighted_overall": legacy,
		"raw_1000_sample_mean": sum(row["score"] for row in rows) / 1000,
	}
	(out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("command", choices=("prepare", "finalize"))
	parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
	parser.add_argument("--image-dir", type=Path)
	parser.add_argument("--output", type=Path)
	parser.add_argument("--model", type=Path, default=Path(MODEL))
	parser.add_argument("--num-shards", type=int, default=4)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--do-sample", action="store_true")
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--top-k", type=int, default=50)
	parser.add_argument("--max-memory-per-gpu", default="74GiB")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	if args.command == "prepare":
		prepare(args)
	else:
		finalize(args)
