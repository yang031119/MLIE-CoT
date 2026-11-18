#!/usr/bin/env python3
"""Stable Diffusion 3.5 Turbo subprocess server for MLIE-CoT.

Reads JSON lines from stdin and writes __SD35_RESULT__ lines to stdout.
Only text-to-image generation is used by V8: SD3.5 Turbo is an initial and
regenerate backend, while image repair still goes through the MLIE-CoT tools.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import torch
from PIL import Image


def emit(obj: dict) -> None:
	"""Write one prefixed response on the service protocol channel."""
	print("__SD35_RESULT__" + json.dumps(obj, ensure_ascii=False), flush=True)


def parse_dtype(value: str) -> torch.dtype:
	"""Normalize service dtype aliases, defaulting unknown values to float16."""
	text = str(value or "").strip().lower()
	if text in {"bf16", "bfloat16"}:
		return torch.bfloat16
	if text in {"fp32", "float32"}:
		return torch.float32
	return torch.float16


def resolve_sd35_model(path_or_repo: str) -> str:
	"""Resolve a local SD3.5 directory or cached snapshot without downloading."""
	p = Path(path_or_repo)
	if p.exists():
		return str(p)
	roots = [
		Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub",
	]
	candidates = [
		Path(path_or_repo),
		Path("stabilityai") / "stable-diffusion-3___5-large-turbo",
		Path("models--" + path_or_repo.replace("/", "--")),
	]
	for root in roots:
		for candidate in candidates:
			base = root / candidate
			if (base / "model_index.json").exists():
				return str(base)
			snaps = base / "snapshots"
			if (snaps / "main" / "model_index.json").exists():
				return str(snaps / "main")
			refs = base / "refs" / "main"
			if refs.exists():
				sha = refs.read_text(encoding="utf-8").strip()
				cand = snaps / sha
				if (cand / "model_index.json").exists():
					return str(cand)
			if snaps.exists():
				for item in reversed(sorted(x for x in snaps.iterdir() if x.is_dir())):
					if (item / "model_index.json").exists():
						return str(item)
	raise FileNotFoundError(f"Cannot resolve SD3.5 Turbo model: {path_or_repo}")


class SD35TurboService:
	"""Own one local SD3.5 pipeline for repeated prompt-only requests."""
	def __init__(self, model_path: str, device: str, dtype: torch.dtype):
		"""Load the pipeline and configure model/sequential CPU offload."""
		from diffusers import StableDiffusion3Pipeline

		self.device = device
		self.pipe = StableDiffusion3Pipeline.from_pretrained(
			model_path,
			torch_dtype=dtype,
			local_files_only=True,
		)
		if str(device).startswith("cuda"):
			offload_mode = os.environ.get("VCOT_SD35_OFFLOAD", "model").strip().lower()
			if offload_mode == "sequential":
				self.pipe.enable_sequential_cpu_offload(device=device)
			else:
				self.pipe.enable_model_cpu_offload(device=device)
		else:
			self.pipe.to(device)
		self.pipe.set_progress_bar_config(disable=True)

	def _generator(self, seed: int) -> torch.Generator:
		"""Create an isolated device-local random generator for one request."""
		return torch.Generator(device=self.device).manual_seed(int(seed))

	def _save_result(self, image: Image.Image, output_image: str, resolution: int) -> Path:
		"""Resize if necessary and persist the square output artifact."""
		out = Path(output_image)
		out.parent.mkdir(parents=True, exist_ok=True)
		if image.size != (resolution, resolution):
			image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
		image.save(out)
		return out

	@torch.inference_mode()
	def generate(self, req: dict) -> Path:
		"""Generate one deterministic SD3.5 candidate from request settings."""
		resolution = int(req.get("resolution", 1024))
		if resolution <= 0:
			raise ValueError(f"resolution must be positive, got {resolution}")
		result = self.pipe(
			prompt=str(req["prompt"]),
			height=resolution,
			width=resolution,
			num_inference_steps=int(req.get("steps", 4)),
			guidance_scale=float(req.get("guidance_scale", 0.0)),
			generator=self._generator(int(req.get("seed", 0))),
			max_sequence_length=int(req.get("max_sequence_length", 512)),
		)
		return self._save_result(result.images[0], str(req["output_image"]), resolution)


def main() -> None:
	"""Load once, announce readiness, and serve JSON-lines requests."""
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-large-turbo")
	parser.add_argument("--device", default="cuda:0")
	parser.add_argument("--dtype", default="bfloat16")
	args = parser.parse_args()

	model_path = resolve_sd35_model(args.model)
	service = SD35TurboService(model_path, args.device, parse_dtype(args.dtype))
	emit({"status": "ready"})

	for line in sys.stdin:
		line = line.strip()
		if not line:
			continue
		try:
			req = json.loads(line)
			op = req.get("op")
			if op == "quit":
				emit({"status": "bye"})
				return
			if op != "generate":
				emit({"status": "error", "error": f"unknown op: {op}"})
				continue
			path = service.generate(req)
			emit({"status": "ok", "output_image": str(path)})
		except Exception as exc:
			traceback.print_exc(file=sys.stderr)
			emit({"status": "error", "error": repr(exc), "type": type(exc).__name__})


if __name__ == "__main__":
	main()
