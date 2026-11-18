#!/usr/bin/env python3
"""FLUX.2 subprocess server for queue scheduler.

Reads JSON lines from stdin, writes __FLUX_RESULT__ lines to stdout.
stderr is reserved for logs/progress.

Example request:
{"op":"edit","prompt":"...","input_image":"/path/in.png","output_image":"/path/out.png",
 "steps":28,"guidance_scale":4.0,"seed":42}
{"op":"generate","prompt":"...","output_image":"/path/out.png",
 "steps":28,"guidance_scale":3.5,"seed":42}
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
    """Write one prefixed JSON response on protocol stdout."""
    print("__FLUX_RESULT__" + json.dumps(obj, ensure_ascii=False), flush=True)


def model_class_name(model_path: str) -> str:
    """Inspect ``model_index.json`` without importing Diffusers."""
    model_index = Path(model_path) / "model_index.json"
    if not model_index.exists():
        return ""
    try:
        payload = json.loads(model_index.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(payload.get("_class_name") or "")


def resolve_hf_snapshot(path_or_repo: str) -> str:
    """Resolve explicit or cached FLUX.2 bytes without network access."""
    p = Path(path_or_repo)
    if p.exists():
        return str(p)
    roots = [
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub",
    ]
    cache_name = "models--" + path_or_repo.replace("/", "--")
    for root in roots:
        base = root / cache_name
        if not base.exists():
            continue
        snaps = base / "snapshots"
        if (snaps / "main").exists():
            return str(snaps / "main")
        refs = base / "refs" / "main"
        if refs.exists():
            sha = refs.read_text(encoding="utf-8").strip()
            cand = snaps / sha
            if cand.exists():
                return str(cand)
        if snaps.exists():
            dirs = sorted([x for x in snaps.iterdir() if x.is_dir()])
            if dirs:

                def ok(d: Path) -> bool:
                    """Return whether a snapshot appears to contain model weights."""
                    names = {x.name for x in d.iterdir()}
                    return "model_index.json" in names or any(
                        n.endswith(".safetensors") for n in names
                    )

                for d in reversed(dirs):
                    if ok(d):
                        return str(d)
                return str(dirs[-1])
    raise FileNotFoundError(f"Cannot resolve Flux model: {path_or_repo}")


class FluxService:
    """Own one local FLUX.2 pipeline and expose generation/edit operations."""

    def __init__(self, model_path: str, device: str, dtype: torch.dtype):
        """Load only ``Flux2Pipeline`` and configure CPU offload for the device."""
        self.device = device
        self.pipeline_class = model_class_name(model_path)
        if self.pipeline_class != "Flux2Pipeline":
            raise RuntimeError(f"Only Flux2Pipeline is supported, got {self.pipeline_class or 'unknown'}")
        from diffusers import Flux2Pipeline

        self.pipe = Flux2Pipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        self._patch_apply_chat_template()
        if str(device).startswith("cuda"):
            offload_mode = os.environ.get("VCOT_FLUX_OFFLOAD", "model").strip().lower()
            if offload_mode == "sequential":
                self.pipe.enable_sequential_cpu_offload(device=device)
            else:
                self.pipe.enable_model_cpu_offload(device=device)
        else:
            self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def _patch_apply_chat_template(self) -> None:
        """Bridge Transformers chat-template keyword migration across versions."""
        processor = getattr(self.pipe, "tokenizer", None)
        original = getattr(processor, "apply_chat_template", None)
        if processor is None or not callable(original):
            return
        migrated_keys = {
            "padding",
            "truncation",
            "max_length",
            "add_special_tokens",
            "return_offsets_mapping",
            "do_sample_frames",
            "fps",
            "num_frames",
        }

        def apply_chat_template_compat(*args, **kwargs):
            """Move tokenizer kwargs into the newer ``processor_kwargs`` slot."""
            processor_kwargs = dict(kwargs.pop("processor_kwargs", None) or {})
            for key in tuple(kwargs.keys()):
                if key in migrated_keys and key not in processor_kwargs:
                    processor_kwargs[key] = kwargs.pop(key)
            if processor_kwargs:
                kwargs["processor_kwargs"] = processor_kwargs
            return original(*args, **kwargs)

        setattr(processor, "apply_chat_template", apply_chat_template_compat)

    def _generator(self, seed: int) -> torch.Generator:
        """Create a device-local generator so each request has an isolated seed."""
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def _save_result(self, image: Image.Image, output_image: str, resolution: int) -> Path:
        """Enforce the square artifact contract and save the decoded image."""
        target_size = (resolution, resolution)
        out = Path(output_image)
        out.parent.mkdir(parents=True, exist_ok=True)
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        image.save(out)
        return out

    @torch.inference_mode()
    def generate(self, req: dict) -> Path:
        """Execute one seeded prompt-only FLUX.2 generation request."""
        resolution = int(req.get("resolution", 1024))
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        result = self.pipe(
            prompt=str(req["prompt"]),
            height=resolution,
            width=resolution,
            num_inference_steps=int(req.get("steps", 28)),
            guidance_scale=float(req.get("guidance_scale", 3.5)),
            generator=self._generator(int(req.get("seed", 0))),
            max_sequence_length=int(req.get("max_sequence_length", 512)),
        )
        return self._save_result(result.images[0], str(req["output_image"]), resolution)

    @torch.inference_mode()
    def generate_batch(self, req: dict) -> list[Path]:
        """Generate a small same-resolution batch in one FLUX.2 forward pass.

        FLUX.2-dev nearly fills an 80GB card with its transformer weights.  A
        batch of two still fits on A100-80GB and amortizes text-encoder/model
        offload over two WISE samples.  The caller falls back to scalar
        requests if a particular device cannot fit the requested batch.
        """
        items = req.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("generate_batch requires a non-empty items list")
        resolution = int(req.get("resolution", 1024))
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        prompts = [str(item["prompt"]) for item in items]
        generators = [self._generator(int(item.get("seed", 0))) for item in items]
        result = self.pipe(
            prompt=prompts,
            height=resolution,
            width=resolution,
            num_inference_steps=int(req.get("steps", 28)),
            guidance_scale=float(req.get("guidance_scale", 3.5)),
            generator=generators,
            max_sequence_length=int(req.get("max_sequence_length", 512)),
        )
        if len(result.images) != len(items):
            raise RuntimeError(f"FLUX.2 returned {len(result.images)} images for {len(items)} requests")
        return [
            self._save_result(image, str(item["output_image"]), resolution)
            for image, item in zip(result.images, items)
        ]

    @torch.inference_mode()
    def edit(self, req: dict) -> Path:
        """Execute one seeded single-reference whole-image FLUX.2 edit."""
        resolution = int(req.get("resolution", 1024))
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        target_size = (resolution, resolution)
        image = Image.open(req["input_image"]).convert("RGB")
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        seed = int(req.get("seed", 0))
        pipe_kwargs = {
            "image": image,
            "prompt": str(req["prompt"]),
            "num_inference_steps": int(req.get("steps", 28)),
            "guidance_scale": float(req.get("guidance_scale", 3.5)),
            "generator": self._generator(seed),
            "max_sequence_length": int(req.get("max_sequence_length", 512)),
        }
        pipe_kwargs.update({"height": resolution, "width": resolution})
        result = self.pipe(**pipe_kwargs)
        return self._save_result(result.images[0], str(req["output_image"]), resolution)


def main() -> None:
    """Load once, announce readiness, and process the JSON-lines protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-dev")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype.lower() in {"bf16", "bfloat16"} else torch.float16
    model_path = resolve_hf_snapshot(args.model)
    service = FluxService(model_path, args.device, dtype)
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
            if op == "generate":
                path = service.generate(req)
                emit({"status": "ok", "output_image": str(path)})
                continue
            if op == "generate_batch":
                paths = service.generate_batch(req)
                emit({"status": "ok", "output_images": [str(path) for path in paths]})
                continue
            if op != "edit":
                emit({"status": "error", "error": f"unknown op: {op}"})
                continue
            path = service.edit(req)
            emit({"status": "ok", "output_image": str(path)})
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            emit({"status": "error", "error": repr(e), "type": type(e).__name__})


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
