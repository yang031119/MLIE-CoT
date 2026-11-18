#!/usr/bin/env python3
"""Prepare MLIE-CoT outputs in the sample layout expected by T2I-CompBench."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
from typing import Dict, Iterable, List, Tuple


DEFAULT_IMAGE_SPECS = ["initial=round_00_bagel_generate.png", "final=final.png"]


def read_jsonl(path: Path) -> List[Dict[str, object]]:
	rows: List[Dict[str, object]] = []
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if line:
			rows.append(json.loads(line))
	return rows


def parse_specs(values: Iterable[str]) -> List[Tuple[str, str]]:
	specs: List[Tuple[str, str]] = []
	for value in values:
		if "=" not in value:
			raise ValueError(f"image spec must be label=image_name, got {value!r}")
		label, image_name = [item.strip() for item in value.split("=", 1)]
		if not label or not image_name:
			raise ValueError(f"image spec must be label=image_name, got {value!r}")
		specs.append((label, image_name))
	return specs


def sample_dir_name(index: int, prompt: str) -> str:
	slug = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt.strip()).strip("_")[:120]
	return f"{index:03d}_{slug or 'prompt'}"


def sanitize_prompt(prompt: str) -> str:
	text = prompt.strip().replace("_", " ")
	text = re.sub(r"[\\/:*?\"<>|]+", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text or "prompt"


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
	if dst.exists() or dst.is_symlink():
		dst.unlink()
	if mode == "copy":
		shutil.copy2(src, dst)
	elif mode == "symlink":
		dst.symlink_to(src.resolve())
	else:
		try:
			os.link(src, dst)
		except OSError:
			shutil.copy2(src, dst)


def write_dataset_files(output_dir: Path, rows: List[Dict[str, object]]) -> None:
	dataset_dir = output_dir / "dataset"
	dataset_dir.mkdir(parents=True, exist_ok=True)
	(dataset_dir / "all.txt").write_text("\n".join(str(row["prompt"]) for row in rows) + "\n", encoding="utf-8")
	by_category: Dict[str, List[str]] = {}
	for row in rows:
		by_category.setdefault(str(row["category"]), []).append(str(row["prompt"]))
	for category, prompts in sorted(by_category.items()):
		(dataset_dir / f"{category}.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")


def prepare_label(
	results_dir: Path,
	metadata_rows: List[Dict[str, object]],
	output_root: Path,
	label: str,
	image_name: str,
	mode: str,
	include_categories: set[str] | None,
	overwrite: bool,
) -> Dict[str, object]:
	output_dir = output_root / label
	if output_dir.exists() and overwrite:
		shutil.rmtree(output_dir)
	samples_dir = output_dir / "samples"
	samples_dir.mkdir(parents=True, exist_ok=True)

	rows: List[Dict[str, object]] = []
	missing: List[str] = []
	for row in metadata_rows:
		index = int(row["index"])
		prompt = str(row["prompt"])
		category = str(row["category"])
		if include_categories is not None and category not in include_categories:
			continue
		sample_dir = results_dir / sample_dir_name(index, prompt)
		image_path = sample_dir / image_name
		if not image_path.exists():
			missing.append(f"{index:04d}: {image_path}")
			continue
		clean_prompt = sanitize_prompt(prompt)
		dst = samples_dir / f"{clean_prompt}_{index:06d}{image_path.suffix.lower() or '.png'}"
		link_or_copy(image_path, dst, mode)
		rows.append(
			{
				"index": index,
				"label": label,
				"category": category,
				"prompt": clean_prompt,
				"original_prompt": prompt,
				"source_sample_dir": str(sample_dir),
				"source_image": str(image_path),
				"output_image": str(dst),
				"question_id": len(rows),
			}
		)

	with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
		for item in rows:
			handle.write(json.dumps(item, ensure_ascii=False) + "\n")
	write_dataset_files(output_dir, rows)
	summary = {
		"label": label,
		"image_name": image_name,
		"results_dir": str(results_dir),
		"output_dir": str(output_dir),
		"prepared": len(rows),
		"metadata_count": len(metadata_rows),
		"include_categories": sorted(include_categories) if include_categories is not None else None,
		"category_counts": dict(Counter(str(item["category"]) for item in rows)),
		"missing_count": len(missing),
		"missing": missing[:100],
	}
	(output_dir / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return summary


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--results-dir", required=True, type=Path)
	parser.add_argument("--metadata-jsonl", required=True, type=Path)
	parser.add_argument("--output-root", type=Path)
	parser.add_argument("--image-spec", action="append", default=None)
	parser.add_argument("--include-category", action="append", default=None)
	parser.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
	parser.add_argument("--overwrite", action="store_true")
	args = parser.parse_args()

	metadata_rows = read_jsonl(args.metadata_jsonl)
	output_root = args.output_root or (args.results_dir / "t2i_compbench_inputs")
	output_root.mkdir(parents=True, exist_ok=True)
	specs = parse_specs(args.image_spec or DEFAULT_IMAGE_SPECS)
	include_categories = set(args.include_category) if args.include_category else None
	summaries = [
		prepare_label(args.results_dir, metadata_rows, output_root, label, image_name, args.mode, include_categories, args.overwrite)
		for label, image_name in specs
	]
	payload = {"output_root": str(output_root), "runs": summaries}
	(output_root / "prepare_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
	return 2 if any(int(summary["missing_count"]) for summary in summaries) else 0


if __name__ == "__main__":
	raise SystemExit(main())
