"""Shared CLIP, PickScore, and HPS reward runners."""
from __future__ import annotations
import gc
import os
from pathlib import Path
from typing import Any, Dict, List
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, AutoTokenizer
from evaluation.clip import CLIPLoss
from evaluation.hps import HPSLoss
from evaluation.pickscore import PickScoreLoss

DEFAULT_CACHE_DIR = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
CLIP_MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_MODEL_ID = "yuvalkirstain/PickScore_v1"
OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def parse_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_hps_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=OPENCLIP_MEAN, std=OPENCLIP_STD),
        ]
    )


class RewardRunner:
    def __init__(self, name: str, model: Any, preprocess, score_offset: float, device: torch.device, dtype: torch.dtype):
        self.name = name
        self.model = model
        self.preprocess = preprocess
        self.score_offset = score_offset
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def score_sample(self, image_path: Path, prompt: str) -> float:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)
        image_features = self.model.get_image_features(image_tensor)
        text_features = self.model.get_text_features(prompt)
        loss = self.model.compute_loss(image_features, text_features)
        score = self.score_offset - float(loss.detach().float().item())
        return score


def make_clip_runner(device: torch.device, dtype: torch.dtype, cache_dir: str) -> RewardRunner:
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir, local_files_only=True)
    reward = CLIPLoss(1.0, dtype=dtype, device=device, cache_dir=cache_dir, tokenizer=processor.tokenizer)
    preprocess = lambda image: processor(images=image, return_tensors="pt")["pixel_values"]
    return RewardRunner("CLIP", reward, preprocess, score_offset=100.0, device=device, dtype=dtype)


def make_pickscore_runner(device: torch.device, dtype: torch.dtype, cache_dir: str) -> RewardRunner:
    image_processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir, local_files_only=True)
    reward = PickScoreLoss(1.0, dtype=dtype, device=device, cache_dir=cache_dir, tokenizer=tokenizer)
    preprocess = lambda image: image_processor(images=image, return_tensors="pt")["pixel_values"]
    return RewardRunner("PickScore", reward, preprocess, score_offset=30.0, device=device, dtype=dtype)


def make_hps_runner(device: torch.device, dtype: torch.dtype, cache_dir: str) -> RewardRunner:
    reward = HPSLoss(1.0, dtype=dtype, device=device, cache_dir=cache_dir)
    preprocess = build_hps_transform()
    return RewardRunner("HPS", reward, preprocess, score_offset=1.0, device=device, dtype=dtype)


def release_runner(runner: RewardRunner) -> None:
    model = getattr(runner, "model", None)
    if model is not None:
        del model
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


