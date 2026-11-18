#!/usr/bin/env python3
"""Long-lived BAGEL text-to-image subprocess for initial generation and L3.

The parent communicates through newline-delimited JSON on stdin/stdout.  Model
and UniCoT progress is redirected to stderr so only ``__BAGEL_RESULT__`` lines
are interpreted as protocol responses.  The service loads local model bytes
once and handles multiple deterministic generation requests on one GPU.
"""

import argparse
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


def request_resolution(req):
    """Read and validate the square output resolution in a request."""
    resolution = int(req.get("resolution", 1024))
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    return resolution


def load_unicot(root: Path):
    """Import the pinned UniCoT inference script from an explicit checkout."""
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("unicot_inference", str(root / "inference_unicot_v0.2.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load UniCoT inference script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BagelService:
    """Own the loaded BAGEL/UniCoT model stack for one subprocess lifetime."""

    def __init__(self, args):
        """Import UniCoT and eagerly load all generation components."""
        self.args = args
        self.u = load_unicot(Path(args.unicot_root))
        self._load_model()

    def _load_model(self):
        """Reconstruct the upstream architecture and dispatch checkpoint weights."""
        u = self.u
        args = self.args
        llm_config = u.Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = u.SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
        vae_model, vae_config = u.load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))
        config = u.BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=args.max_latent_size,
        )
        language_model = u.Qwen2ForCausalLM(llm_config)
        vit_model = u.SiglipVisionModel(vit_config)
        model = u.Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
        tokenizer = u.Qwen2Tokenizer.from_pretrained(args.model_path)
        tokenizer, new_token_ids, _ = u.add_special_tokens(tokenizer)
        model = u.load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(args.fine_tuned_path, "ema.safetensors"),
            device_map={"": args.device},
            offload_buffers=False,
            dtype=torch.bfloat16,
            force_hooks=True,
            offload_folder="/tmp/offload_bagel_service",
        )
        u.tokenizer = tokenizer
        u.new_token_ids = new_token_ids
        u.model = model.eval()
        u.gen_model = u.model
        u.vae_model = vae_model.to(args.device).eval()

    @torch.inference_mode()
    def generate(self, req):
        """Generate and save one seeded text-to-image result."""
        seed = req.get("seed")
        if seed is not None:
            seed = int(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        img = self.u.generate_image(
            prompt=req["prompt"],
            resolution=request_resolution(req),
            device=self.args.device,
            cfg_scale=float(req.get("cfg_scale", 4.0)),
            cfg_interval=[0.4, 1.0],
            timestep_shift=3.0,
            num_timesteps=int(req.get("steps", 50)),
            cfg_renorm_min=0.0,
        )
        out = Path(req["output_image"])
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        return {"status": "ok", "output_image": str(out)}


def build_parser():
    """Define model locations and device settings fixed at service startup."""
    p = argparse.ArgumentParser()
    p.add_argument("--unicot-root", default="third_party/UniCoT")
    p.add_argument("--model-path", default="models/BAGEL-7B-MoT")
    p.add_argument("--fine-tuned-path", default="models/BAGEL-7B-MoT")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-latent-size", type=int, default=64)
    return p


def emit(obj):
    """Write one prefixed protocol response and flush it immediately."""
    print("__BAGEL_RESULT__" + json.dumps(obj, ensure_ascii=False), flush=True)


def main():
    """Load the service, announce readiness, and serve requests until quit."""
    args = build_parser().parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with contextlib.redirect_stdout(sys.stderr):
        service = BagelService(args)
    emit({"status": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            with contextlib.redirect_stdout(sys.stderr):
                if op == "generate":
                    res = service.generate(req)
                elif op == "quit":
                    emit({"status": "bye"})
                    return
                else:
                    res = {"status": "error", "error": f"unknown op: {op}"}
        except Exception as e:
            res = {"status": "error", "error": repr(e)}
        emit(res)


if __name__ == "__main__":
    main()
