#!/usr/bin/env python3
"""Prepare an MLIE-CoT result directory for the official GenEval evaluator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Dict, List


def read_metadata(path: Path) -> List[Dict[str, object]]:
	with path.open("r", encoding="utf-8") as handle:
		return [json.loads(line) for line in handle if line.strip()]


def index_sample_dirs(results_dir: Path) -> Dict[int, Path]:
	mapping: Dict[int, Path] = {}
	for item in results_dir.iterdir():
		if not item.is_dir():
			continue
		prefix = item.name.split("_", 1)[0]
		if not prefix.isdigit():
			continue
		mapping[int(prefix)] = item
	return mapping


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
	if dst.exists() or dst.is_symlink():
		dst.unlink()
	if mode == "copy":
		shutil.copy2(src, dst)
		return
	if mode == "symlink":
		dst.symlink_to(src.resolve())
		return
	try:
		os.link(src, dst)
	except OSError:
		shutil.copy2(src, dst)


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--results-dir", required=True, type=Path)
	parser.add_argument("--metadata-jsonl", default=Path("third_party/geneval/prompts/evaluation_metadata.jsonl"), type=Path)
	parser.add_argument("--image-name", default="final.png")
	parser.add_argument("--output-dir", type=Path)
	parser.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
	parser.add_argument("--overwrite", action="store_true")
	args = parser.parse_args()

	results_dir = args.results_dir
	if not results_dir.exists():
		raise FileNotFoundError(f"results dir not found: {results_dir}")
	output_dir = args.output_dir or (results_dir / f"geneval_eval_input_{Path(args.image_name).stem}")
	if output_dir.exists() and args.overwrite:
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	metadata_rows = read_metadata(args.metadata_jsonl)
	sample_dirs = index_sample_dirs(results_dir)
	missing: List[str] = []
	prepared = 0
	for index, metadata in enumerate(metadata_rows):
		sample_dir = sample_dirs.get(index)
		if sample_dir is None:
			missing.append(f"{index:03d}: sample dir")
			continue
		image_path = sample_dir / args.image_name
		if not image_path.exists():
			missing.append(f"{index:03d}: {args.image_name}")
			continue

		eval_dir = output_dir / f"{index:05d}"
		samples_dir = eval_dir / "samples"
		samples_dir.mkdir(parents=True, exist_ok=True)
		(eval_dir / "metadata.jsonl").write_text(
			json.dumps(metadata, ensure_ascii=False) + "\n",
			encoding="utf-8",
		)
		link_or_copy(image_path, samples_dir / "0.png", args.mode)
		prepared += 1

	payload = {
		"results_dir": str(results_dir),
		"output_dir": str(output_dir),
		"image_name": args.image_name,
		"metadata_count": len(metadata_rows),
		"prepared": prepared,
		"missing_count": len(missing),
		"missing": missing[:50],
	}
	(output_dir / "prepare_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
	if missing:
		return 2
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
