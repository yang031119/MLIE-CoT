#!/usr/bin/env python3
"""Long-lived BAGEL text-to-image subprocess for initial generation and L3.

The parent communicates through newline-delimited JSON on stdin/stdout.  Model
and UniCoT progress is redirected to stderr so only ``__BAGEL_RESULT__`` lines
are interpreted as protocol responses.  The service loads local model bytes
once and handles multiple deterministic generation requests on one GPU.
"""
import argparse
import contextlib
import copy
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image


def request_resolution(req):
    """Read and validate the square output resolution in a request."""
    resolution = int(req.get("resolution", 1024))
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    return resolution


def resize_to_request_resolution(image, req):
    """Resize generated output to the request contract using Lanczos resampling."""
    resolution = request_resolution(req)
    target_size = (resolution, resolution)
    if image.size == target_size:
        return image
    return image.resize(target_size, Image.Resampling.LANCZOS)


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

    def _decode_text(self, token_tensor):
        raw = self.u.tokenizer.decode(token_tensor[:, 0])
        candidates = []
        for segment in reversed(raw.split("<|im_start|>")):
            segment = segment.split("<|im_end|>", 1)[0]
            cleaned = self._clean_reasoning_text(segment)
            if cleaned:
                candidates.append(cleaned)
        candidates.append(self._clean_reasoning_text(raw))
        for candidate in candidates:
            fragment = self._last_json_fragment(candidate)
            if fragment:
                return fragment
        for candidate in candidates:
            if candidate and candidate.upper() not in {"[ASSISTANT]", "ASSISTANT"}:
                return candidate
        return raw.strip()

    def _clean_reasoning_text(self, text):
        text = re.sub(r"<\|[^>]+?\|>", "", str(text or ""))
        text = re.sub(r"^\s*(?:assistant|user|system)\s*:?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*\[(?:ASSISTANT|USER|SYSTEM)\]\s*", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _last_json_fragment(self, text):
        starts = [index for index, char in enumerate(text) if char in "{["]
        for start in reversed(starts):
            fragment = self._balanced_json_from(text, start)
            if fragment is None:
                continue
            try:
                json.loads(fragment)
            except Exception:
                continue
            return fragment
        return None

    def _balanced_json_from(self, text, start):
        """Extract one balanced JSON object or array while respecting strings."""
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        matching_open = opener
        matching_close = closer
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == matching_open:
                depth += 1
            elif char == matching_close:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1].strip()
        return None

    def _update_cache_with_text(self, past_key_values, newlens, new_rope, text):
        if not str(text).strip():
            return past_key_values, newlens, new_rope
        u = self.u
        generation_input, newlens, new_rope = u.gen_model.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[str(text)],
            tokenizer=u.tokenizer,
            new_token_ids=u.new_token_ids,
        )
        generation_input = u.move_generation_input_to_device(generation_input, self.args.device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = u.gen_model.forward_cache_update_text(past_key_values, **generation_input)
        return past_key_values, newlens, new_rope

    def _update_cache_with_image(self, past_key_values, newlens, new_rope, image_path):
        u = self.u
        image = u.pil_img2rgb(Image.open(image_path).convert("RGB"))
        generation_input, newlens, new_rope = u.gen_model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=[image],
            transforms=u.vit_transform_und,
            new_token_ids=u.new_token_ids,
        )
        generation_input = u.move_generation_input_to_device(generation_input, self.args.device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = u.gen_model.forward_cache_update_vit(past_key_values, **generation_input)
        return past_key_values, newlens, new_rope

    @torch.inference_mode()
    def reason(self, req):
        """Run multimodal BAGEL reasoning and decode the final textual response."""
        u = self.u
        max_new_tokens = int(req.get("max_new_tokens", 1024))
        do_sample = bool(req.get("do_sample", False))
        temperature = float(req.get("temperature", 0.3))
        past_key_values = u.NaiveCache(u.gen_model.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]
        for block in req.get("content") or []:
            block_type = block.get("type")
            if block_type == "text":
                past_key_values, newlens, new_rope = self._update_cache_with_text(
                    past_key_values,
                    newlens,
                    new_rope,
                    block.get("text", ""),
                )
            elif block_type == "image":
                past_key_values, newlens, new_rope = self._update_cache_with_image(
                    past_key_values,
                    newlens,
                    new_rope,
                    block["path"],
                )
        generation_input = u.gen_model.prepare_start_tokens(newlens, new_rope, u.new_token_ids)
        generation_input = u.move_generation_input_to_device(generation_input, self.args.device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            output_tokens = u.gen_model.generate_text(
                past_key_values=copy.deepcopy(past_key_values),
                max_length=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=u.new_token_ids["eos_token_id"],
                **generation_input,
            )
        return {"status": "ok", "text": self._decode_text(output_tokens)}

    @torch.inference_mode()
    def generate(self, req):
        """Generate and save one seeded text-to-image result."""
        seed = req.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
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

    @torch.inference_mode()
    def edit(self, req):
        """Execute one seeded image-conditioned edit and persist the result."""
        seed = req.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        image = resize_to_request_resolution(Image.open(req["input_image"]).convert("RGB"), req)
        img = self.u.editing_image(
            image=self.u.pil_img2rgb(image),
            prompt=req["prompt"],
            device=self.args.device,
            cfg_text_scale=float(req.get("cfg_text_scale", 4.0)),
            cfg_img_scale=float(req.get("cfg_img_scale", 2.0)),
            cfg_interval=[0.0, 1.0],
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
                elif op == "edit":
                    res = service.edit(req)
                elif op == "reason":
                    res = service.reason(req)
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
