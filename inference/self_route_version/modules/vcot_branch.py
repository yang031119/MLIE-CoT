#!/usr/bin/env python3
"""Runtime helpers for language models, image generators, and ruler overlays."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
import torch

from .contracts import QWEN_DECODING_CONTRACT_V10, QWEN_RETRY_DECODING_CONTRACT_V10


DEFAULT_QWEN9B_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_QWEN9B_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
DEFAULT_QWEN9B_MODEL = str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / DEFAULT_QWEN9B_REVISION)
DEFAULT_QWEN35B_MODEL = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_QWEN122B_MODEL = "Qwen/Qwen3.5-122B-A10B"
DEFAULT_QWEN_MODEL = DEFAULT_QWEN9B_MODEL
DEFAULT_VCOT_PYTHON = sys.executable
DEFAULT_BAGEL_PYTHON = sys.executable
DEFAULT_FLUX_PYTHON = sys.executable


def annotate_image_with_ticks(image: Image.Image, step: int = 128) -> Image.Image:
    """Add an external pixel-coordinate ruler without shifting image coordinates."""
    base = image.copy().convert("RGB")
    width, height = base.size
    margin = 72
    annotated = Image.new("RGB", (width + margin * 2, height + margin * 2), (255, 255, 255))
    annotated.paste(base, (margin, margin))
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    tick_color = (255, 64, 64)
    label_bg = (255, 255, 255)
    label_fg = (180, 0, 0)
    major_len = 18
    minor_len = 10

    left = margin
    top = margin
    right = margin + width - 1
    bottom = margin + height - 1

    def draw_label_in_rect(cx: int, cy: int, text: str, rect: Tuple[int, int, int, int]) -> None:
        """Draw a ruler label centered at a point but clipped to its margin."""
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        pad_x = 3
        pad_y = 2
        x0 = int(round(cx - text_width / 2 - pad_x))
        y0 = int(round(cy - text_height / 2 - pad_y))
        x1 = x0 + text_width + pad_x * 2
        y1 = y0 + text_height + pad_y * 2
        rx0, ry0, rx1, ry1 = rect
        if x0 < rx0:
            x1 += rx0 - x0
            x0 = rx0
        if x1 > rx1:
            x0 -= x1 - rx1
            x1 = rx1
        if y0 < ry0:
            y1 += ry0 - y0
            y0 = ry0
        if y1 > ry1:
            y0 -= y1 - ry1
            y1 = ry1
        x0 = max(rx0, x0)
        y0 = max(ry0, y0)
        x1 = min(rx1, x1)
        y1 = min(ry1, y1)
        draw.rectangle((x0, y0, x1, y1), fill=label_bg)
        draw.text((x0 + pad_x, y0 + pad_y), text, fill=label_fg, font=font)

    for x in range(0, width, step):
        major = x % (step * 2) == 0
        tick_len = major_len if major else minor_len
        draw.line([(left + x, top - tick_len), (left + x, top - 1)], fill=tick_color, width=3)
        draw.line([(left + x, bottom + 1), (left + x, bottom + tick_len)], fill=tick_color, width=3)
        draw_label_in_rect(left + x, top // 2, str(x), (2, 2, annotated.size[0] - 3, top - major_len - 4))
        draw_label_in_rect(left + x, bottom + margin // 2, str(x), (2, bottom + major_len + 4, annotated.size[0] - 3, annotated.size[1] - 3))

    for y in range(0, height, step):
        major = y % (step * 2) == 0
        tick_len = major_len if major else minor_len
        draw.line([(left - tick_len, top + y), (left - 1, top + y)], fill=tick_color, width=3)
        draw.line([(right + 1, top + y), (right + tick_len, top + y)], fill=tick_color, width=3)
        draw_label_in_rect(left // 2, top + y, str(y), (2, 2, left - major_len - 4, annotated.size[1] - 3))
        draw_label_in_rect(right + margin // 2, top + y, str(y), (right + major_len + 4, 2, annotated.size[0] - 3, annotated.size[1] - 3))

    draw.rectangle((left, top, right, bottom), outline=(220, 220, 220), width=1)

    return annotated


def parse_dtype(name: str) -> torch.dtype:
    """Translate supported CLI dtype aliases to PyTorch dtypes."""
    n = name.lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    if n in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


# --- Qwen3.5-122B-A10B in-process planner / comparator -----------------------------------

def query_gpu_free_gib() -> Dict[int, float]:
    """Return per-physical-GPU free memory in GiB. Empty dict on failure."""

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except Exception:
        return {}
    result: Dict[int, float] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            result[int(parts[0])] = float(parts[1]) / 1024.0
        except Exception:
            continue
    return result


def build_qwen_max_memory(
    num_visible_gpus: int,
    free_by_physical_gpu: Dict[int, float],
    max_memory_per_gpu: Optional[str],
    safety_gib: float,
    min_free_gib: float,
    per_gpu_extra_reserve: Optional[Dict[int, float]],
    is_quantized: bool,
) -> Tuple[Dict[Any, str], int]:
    """Build Accelerate device budgets from live free-memory observations.

    Physical GPU ids are mapped through ``CUDA_VISIBLE_DEVICES`` while returned
    keys remain process-visible ids.  Quantized loads ignore editor-specific
    reserves but retain the generic safety margin.
    """
    max_memory: Dict[Any, str] = {}
    eligible_gpu_count = 0
    for visible_idx in range(num_visible_gpus):
        physical = _physical_gpu_arg(visible_idx)
        try:
            free_gib = free_by_physical_gpu.get(int(physical), 0.0)
        except Exception:
            free_gib = 0.0
        # For quantized loads, reserve only the generic safety margin so weight
        # planning is not blocked by editor-specific transient reservations.
        extra = 0.0 if is_quantized else (per_gpu_extra_reserve or {}).get(visible_idx, 0.0)
        usable_free = free_gib - safety_gib - extra
        if usable_free < min_free_gib:
            continue
        budget = max(min_free_gib, usable_free)
        if max_memory_per_gpu:
            cap = float(re.sub(r"[^0-9.]", "", max_memory_per_gpu) or "0")
            if cap > 0:
                budget = min(budget, cap)
        max_memory[visible_idx] = f"{int(budget)}GiB"
        eligible_gpu_count += 1
    return max_memory, eligible_gpu_count


def flatten_qwen_image_batches(images_batch: List[List[Image.Image]]) -> List[Image.Image]:
    """Flatten per-conversation images in chat-placeholder order.

    Qwen3VLProcessor consumes a flat image list and walks image placeholders
    across each text in batch order.  Keeping this explicit prevents the old
    ``images[0]`` behavior from silently dropping paired-review candidates.
    """
    return [image for sample_images in images_batch for image in sample_images]


class QwenPlanner:
    """Frozen Qwen runtime for language generation and repair-route features.

    Model loading supports dense and MoE Qwen3.5 variants plus none/int8/NF4
    quantization.  If a route head is configured, its cheap CPU metadata checks
    run before backbone allocation and its live processor/pooling checks run
    immediately afterward.
    """
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype,
        max_memory_per_gpu: Optional[str],
        safety_gib: float = 3.0,
        min_free_gib: float = 16.0,
        per_gpu_extra_reserve: Optional[Dict[int, float]] = None,
        quantization: str = "8bit",
        allow_cpu_offload: bool = False,
        route_head_path: Optional[str] = None,
    ):
        """Load Qwen across eligible GPUs and optionally attach a validated head."""
        preloaded_route_head = None
        preloaded_route_config: Dict[str, Any] = {}
        if route_head_path:
            # Validate the V10 schema before allocating the multi-GPU language
            # backbone; a copied V9 checkpoint must fail cheaply and explicitly.
            from .route_head import load_route_head, validate_route_model_identity

            preloaded_route_head, preloaded_route_config = load_route_head(route_head_path, "cpu")
            validate_route_model_identity(preloaded_route_config, model_path)
        from transformers import AutoConfig, AutoProcessor

        model_cls = None
        errors: List[str] = []
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model_type = str(getattr(config, "model_type", "") or "")
        architectures = [str(item) for item in (getattr(config, "architectures", None) or [])]
        is_moe = model_type.endswith("_moe") or any("Moe" in item for item in architectures)
        if is_moe:
            candidate_classes = [
                "Qwen3_5MoeForConditionalGeneration",
                "AutoModelForImageTextToText",
                "AutoModelForCausalLM",
            ]
        else:
            candidate_classes = [
                "Qwen3_5ForConditionalGeneration",
                "AutoModelForImageTextToText",
                "AutoModelForCausalLM",
            ]
        for class_name in candidate_classes:
            try:
                module = __import__("transformers", fromlist=[class_name])
                model_cls = getattr(module, class_name)
                break
            except Exception as exc:  # pragma: no cover - environment specific
                errors.append(f"{class_name}: {exc}")
        if model_cls is None:
            raise RuntimeError("Cannot locate a compatible Qwen model class. " + " | ".join(errors))

        max_memory: Optional[Dict[Any, str]] = None
        from .contracts import canonical_qwen_quantization, qwen_effective_quantization

        self.allow_cpu_offload = bool(allow_cpu_offload)
        self.requested_quantization = canonical_qwen_quantization(quantization)
        self.effective_quantization = qwen_effective_quantization(
            self.requested_quantization,
            self.allow_cpu_offload,
        )
        quant_mode = self.requested_quantization
        is_quantized = quant_mode not in {"none", ""}
        if torch.cuda.is_available():
            num = torch.cuda.device_count()
            free = query_gpu_free_gib()
            max_memory, eligible_gpu_count = build_qwen_max_memory(
                num_visible_gpus=num,
                free_by_physical_gpu=free,
                max_memory_per_gpu=max_memory_per_gpu,
                safety_gib=safety_gib,
                min_free_gib=min_free_gib,
                per_gpu_extra_reserve=per_gpu_extra_reserve,
                is_quantized=is_quantized,
            )
            if eligible_gpu_count == 0:
                raise RuntimeError(
                    "No GPU has enough free memory for Qwen. "
                    f"Require at least {min_free_gib:.1f} GiB usable free memory after safety reserve."
                )
            # For quantized models (int8/int4) we want to fail loudly rather than
            # silently overflow to CPU (which yields gibberish or unsupported errors).
            max_memory["cpu"] = "300GiB" if not is_quantized else "0GiB"
            print(
                json.dumps(
                    {
                        "event": "qwen_max_memory",
                        "quant": quant_mode,
                        "max_memory": max_memory,
                        "min_free_gib": min_free_gib,
                        "eligible_gpu_count": eligible_gpu_count,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        quant_config = None
        load_dtype = dtype
        quant = self.requested_quantization
        cpu_offload_mode = self.allow_cpu_offload and quant in {"8bit", "int8", "4bit", "int4", "nf4"}
        if cpu_offload_mode:
            from transformers import BitsAndBytesConfig

            _patch_bnb_int8params_new()
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            load_dtype = None  # bnb handles dtype internally
            if max_memory is not None:
                max_memory["cpu"] = "300GiB"
            print(
                json.dumps({"event": "qwen_quantization", "kind": "int8_cpu_offload"}, ensure_ascii=False),
                flush=True,
            )
        elif quant in {"8bit", "int8"}:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            load_dtype = None  # bnb handles dtype internally
            print(
                json.dumps({"event": "qwen_quantization", "kind": "int8"}, ensure_ascii=False),
                flush=True,
            )
        elif quant in {"4bit", "int4", "nf4"}:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_dtype = None
            print(
                json.dumps({"event": "qwen_quantization", "kind": "nf4"}, ensure_ascii=False),
                flush=True,
            )

        # Use "balanced" instead of "auto" so accelerate evenly distributes the
        # (small after nf4/int8 quantization) Qwen weights across all GPUs.
        # With "auto" the entire ~61 GiB nf4 model can fit on a single GPU,
        # which collides with the BAGEL/Flux subprocesses that share GPU 3 / 2.
        device_map_strategy = "auto" if cpu_offload_mode else ("balanced" if is_quantized else "auto")
        load_kwargs: Dict[str, Any] = dict(
            device_map=device_map_strategy,
            max_memory=max_memory,
            trust_remote_code=True,
            local_files_only=True,
        )
        if load_dtype is not None:
            load_kwargs["torch_dtype"] = load_dtype
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        try:
            self.model = model_cls.from_pretrained(model_path, **load_kwargs).eval()
        except ValueError as exc:
            if not self.allow_cpu_offload or "Some modules are dispatched on the CPU or the disk" not in str(exc):
                raise
            release_cuda_memory()
            from transformers import BitsAndBytesConfig

            _patch_bnb_int8params_new()
            retry_max_memory = dict(max_memory or {})
            retry_max_memory["cpu"] = "300GiB"
            retry_kwargs: Dict[str, Any] = dict(
                device_map="auto",
                max_memory=retry_max_memory,
                trust_remote_code=True,
                local_files_only=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                ),
            )
            print(
                json.dumps(
                    {
                        "event": "qwen_cpu_offload_retry",
                        "from_quant": quant_mode,
                        "retry_quant": "int8_cpu_offload",
                        "max_memory": retry_max_memory,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            self.model = model_cls.from_pretrained(model_path, **retry_kwargs).eval()
            self.effective_quantization = "int8_cpu_offload"
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"
        if preloaded_route_head is not None:
            from .route_head import validate_route_runtime_contract

            validate_route_runtime_contract(
                preloaded_route_config,
                processor=self.processor,
                encoder_callable=self.__class__.encode_route_features,
                model_path=model_path,
                dtype=dtype,
                quantization=self.requested_quantization,
                allow_cpu_offload=self.allow_cpu_offload,
                effective_quantization=self.effective_quantization,
            )
        self.input_device = next(self.model.parameters()).device
        self.route_head = preloaded_route_head
        self.route_head_config: Dict[str, Any] = preloaded_route_config
        if self.route_head is not None:
            self.route_head.to(device=self.input_device, dtype=torch.float32).eval()

    @torch.inference_mode()
    def encode_route_features(
        self,
        prompts: List[str],
        diagnoses: List[str],
        images: List[Image.Image],
    ) -> torch.Tensor:
        if not (len(prompts) == len(diagnoses) == len(images)):
            raise ValueError("route-head prompt/diagnosis/image batch lengths differ")
        from .route_head import build_route_encoder_messages

        texts: List[str] = []
        for prompt, diagnosis, image in zip(prompts, diagnoses, images):
            messages = build_route_encoder_messages(prompt, diagnosis, image)
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        inputs = {key: (value.to(self.input_device) if hasattr(value, "to") else value) for key, value in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Qwen route head requires output_hidden_states from the language backbone")
        return hidden_states[-1][:, -1, :].detach().float().cpu()

    @torch.inference_mode()
    def predict_routes(
        self,
        prompts: List[str],
        diagnoses: List[str],
        images: List[Image.Image],
    ) -> List[Dict[str, Any]]:
        """Encode inputs, append signed auxiliary features, and classify L1-L3."""
        if self.route_head is None:
            return []
        from .route_head import append_diagnosis_aux_features, route_prediction

        pooled = self.encode_route_features(prompts, diagnoses, images)
        head_device = next(self.route_head.parameters()).device
        pooled = append_diagnosis_aux_features(
            pooled.to(head_device),
            diagnoses,
            self.route_head_config,
        )
        logits = self.route_head(pooled)
        return route_prediction(logits, self.route_head_config)

    @torch.inference_mode()
    def _generate(
        self,
        messages: List[Dict[str, Any]],
        images: List[Image.Image],
        max_new_tokens: int,
        _decoding_contract: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate one response with the frozen greedy decoding contract."""
        if hasattr(self.processor, "apply_chat_template"):
            try:
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            processor_kwargs: Dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
            if images:
                processor_kwargs["images"] = images
            inputs = self.processor(**processor_kwargs)
        else:
            joined = " ".join(
                part.get("text", "") for msg in messages for part in msg.get("content", []) if isinstance(part, dict)
            )
            processor_kwargs = {"text": [joined], "return_tensors": "pt"}
            if images:
                processor_kwargs["images"] = images
            inputs = self.processor(**processor_kwargs)
        input_ids = inputs.get("input_ids")
        inputs = {k: (v.to(self.input_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **dict(_decoding_contract or QWEN_DECODING_CONTRACT_V10),
        )
        if input_ids is not None:
            generated = generated[:, input_ids.shape[-1] :]
        raw = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return raw

    def _generate_retry(self, messages: List[Dict[str, Any]], images: List[Image.Image], max_new_tokens: int) -> str:
        """Generate a correction response with the repetition-resistant contract."""
        return self._generate(messages, images, max_new_tokens, dict(QWEN_RETRY_DECODING_CONTRACT_V10))

    @torch.inference_mode()
    def _generate_batch(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        images_batch: List[List[Image.Image]],
        max_new_tokens: int,
        _decoding_contract: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Generate a padded batch while preserving all image-placeholder order."""
        if not messages_batch:
            return []

        if hasattr(self.processor, "apply_chat_template"):
            texts: List[str] = []
            for messages in messages_batch:
                try:
                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                texts.append(text)
            processor_kwargs = {"text": texts, "padding": True, "return_tensors": "pt"}
            flat_images = flatten_qwen_image_batches(images_batch)
            if flat_images:
                processor_kwargs["images"] = flat_images
            inputs = self.processor(**processor_kwargs)
        else:
            texts = []
            images = []
            for messages, msg_images in zip(messages_batch, images_batch):
                joined = " ".join(
                    part.get("text", "") for msg in messages for part in msg.get("content", []) if isinstance(part, dict)
                )
                texts.append(joined)
                images.extend(msg_images)
            processor_kwargs = {"text": texts, "padding": True, "return_tensors": "pt"}
            if images:
                processor_kwargs["images"] = images
            inputs = self.processor(**processor_kwargs)

        input_ids = inputs.get("input_ids")
        inputs = {k: (v.to(self.input_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **dict(_decoding_contract or QWEN_DECODING_CONTRACT_V10),
        )
        if input_ids is not None:
            generated = generated[:, input_ids.shape[-1] :]
        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [item.strip() for item in decoded]

    def _generate_batch_retry(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        images_batch: List[List[Image.Image]],
        max_new_tokens: int,
    ) -> List[str]:
        """Batch variant of the structured-output correction decoding pass."""
        return self._generate_batch(
            messages_batch,
            images_batch,
            max_new_tokens,
            dict(QWEN_RETRY_DECODING_CONTRACT_V10),
        )

    def _generate_with_single_retry(
        self,
        messages: List[Dict[str, Any]],
        images: List[Image.Image],
        max_new_tokens: int,
        validator,
        retry_instruction: str,
    ) -> Dict[str, Any]:
        """Validate one response and allow exactly one JSON correction turn."""
        current_messages = list(messages)
        last_result: Optional[Dict[str, Any]] = None
        for attempt_idx in range(2):
            generate = self._generate if attempt_idx == 0 else self._generate_retry
            raw = generate(current_messages, images, max_new_tokens)
            result = validator(raw)
            if result.get("valid_json"):
                if attempt_idx:
                    result["json_retry_count"] = attempt_idx
                return result
            last_result = result
            if attempt_idx == 0:
                current_messages = current_messages + [
                    {"role": "assistant", "content": [{"type": "text", "text": raw[:1200]}]},
                    {"role": "user", "content": [{"type": "text", "text": retry_instruction}]},
                ]
        assert last_result is not None
        last_result["json_retry_count"] = 1
        return last_result



# --- Subprocess service clients (BAGEL, Flux) --------------------------------------------

class _ServiceClient:
    """Common stdin/stdout-line protocol used by bagel_service / flux_service."""

    result_prefix = "__RESULT__"

    def __init__(self, cmd: List[str], env: Dict[str, str], result_prefix: str, ready_timeout_s: int = 900):
        """Start a service, forward diagnostic stderr, and await readiness."""
        self.result_prefix = result_prefix
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._stderr_thread = None
        if self.proc.stderr is not None:
            self._stderr_thread = threading.Thread(target=self._forward_stderr, daemon=True)
            self._stderr_thread.start()
        ready = self._read_response(ready_timeout_s)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Service did not become ready: {ready}")

    def _forward_stderr(self) -> None:
        """Continuously relay child diagnostics without polluting protocol stdout."""
        assert self.proc.stderr is not None
        try:
            while True:
                chunk = self.proc.stderr.read(4096)
                if not chunk:
                    break
                try:
                    sys.stderr.write(chunk)
                    sys.stderr.flush()
                except Exception:
                    pass
        except Exception:
            pass

    def _read_response(self, timeout_s: int = 3600) -> Dict[str, Any]:
        """Read until the next prefixed JSON response or service failure."""
        assert self.proc.stdout is not None
        start = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Service exited with code {self.proc.returncode}")
            line = self.proc.stdout.readline()
            if not line:
                if time.time() - start > timeout_s:
                    raise TimeoutError("Timed out waiting for service line")
                continue
            if line.startswith(self.result_prefix):
                return json.loads(line[len(self.result_prefix) :])
            if time.time() - start > timeout_s:
                raise TimeoutError("Timed out waiting for service result")

    def request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Send one JSON request and synchronously return its matching response."""
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return self._read_response()

    def close(self) -> None:
        """Request graceful shutdown, then terminate/kill on bounded timeouts."""
        if getattr(self, "proc", None) is None or self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
        for stream_name in ("stdin", "stdout"):
            stream = getattr(self.proc, stream_name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass
        stderr_stream = getattr(self.proc, "stderr", None)
        if stderr_stream is not None:
            try:
                stderr_stream.close()
            except Exception:
                pass
        if getattr(self, "_stderr_thread", None) is not None:
            try:
                self._stderr_thread.join(timeout=1)
            except Exception:
                pass


def _physical_gpu_arg(visible_index: int) -> str:
    """Convert a visible cuda:N index (relative to current process) to a physical GPU id."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(visible_index)
    parts = [x.strip() for x in visible.split(",") if x.strip()]
    if visible_index < len(parts):
        return parts[visible_index]
    return str(visible_index)


class BagelClient:
    """Parent-side typed wrapper around one GPU-isolated BAGEL service."""
    def __init__(self, args: argparse.Namespace):
        """Start BAGEL in its configured Python environment and physical GPU."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = _physical_gpu_arg(args.bagel_gpu_index)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cmd = [
            args.bagel_python,
            str(args.bagel_service_script),
            "--unicot-root",
            args.unicot_root,
            "--model-path",
            args.bagel_model_path,
            "--fine-tuned-path",
            args.bagel_fine_tuned_path,
            "--device",
            "cuda:0",
            "--max-latent-size",
            str(args.max_latent_size),
        ]
        self._inner = _ServiceClient(cmd, env, "__BAGEL_RESULT__", ready_timeout_s=1800)

    def generate(
        self,
        prompt: str,
        out_path: Path,
        args: argparse.Namespace,
        *,
        seed: int | None = None,
    ) -> Path:
        """Request one prompt-only BAGEL image and validate service status."""
        res = self._inner.request(
            {
                "op": "generate",
                "prompt": prompt,
                "output_image": str(out_path),
                "resolution": args.resolution,
                "steps": args.bagel_steps,
                "cfg_scale": args.bagel_cfg_scale,
                "seed": seed,
            }
        )
        if res.get("status") != "ok":
            raise RuntimeError(f"BAGEL generate error: {res}")
        return Path(res["output_image"])

    def edit(
        self,
        image_path: Path,
        prompt: str,
        out_path: Path,
        seed: int,
        args: argparse.Namespace,
    ) -> Path:
        """Request one BAGEL edit and return the service-validated output path."""
        res = self._inner.request(
            {
                "op": "edit",
                "prompt": prompt,
                "input_image": str(image_path),
                "output_image": str(out_path),
                "resolution": args.resolution,
                "steps": args.bagel_steps,
                "cfg_text_scale": args.bagel_cfg_text_scale,
                "cfg_img_scale": args.bagel_cfg_img_scale,
                "seed": seed,
            }
        )
        if res.get("status") != "ok":
            raise RuntimeError(f"BAGEL edit error: {res}")
        return Path(res["output_image"])

    def reason(
        self,
        content: List[Dict[str, Any]],
        *,
        max_new_tokens: int = 512,
    ) -> str:
        """Request deterministic multimodal reasoning from the BAGEL service."""
        res = self._inner.request(
            {
                "op": "reason",
                "content": content,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
                "temperature": 0.0,
            }
        )
        if res.get("status") != "ok":
            raise RuntimeError(f"BAGEL reason error: {res}")
        return str(res.get("text") or "")

    def close(self) -> None:
        """Shut down the underlying BAGEL service process."""
        self._inner.close()


class FluxClient:
    """Parent-side wrapper for FLUX.2 generate/edit service operations."""
    def __init__(self, args: argparse.Namespace):
        """Start FLUX.2 in an isolated process bound to one physical GPU."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = _physical_gpu_arg(args.flux_gpu_index)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        cmd = [
            args.flux_python,
            str(args.flux_service_script),
            "--model",
            args.flux_model,
            "--device",
            "cuda:0",
            "--dtype",
            str(getattr(args, "flux_dtype", "bfloat16")),
        ]
        self._inner = _ServiceClient(cmd, env, "__FLUX_RESULT__", ready_timeout_s=900)

    def edit(self, image_path: Path, prompt: str, out_path: Path, seed: int, args: argparse.Namespace) -> Path:
        """Request one policy-configured image edit."""
        res = self._inner.request(
            {
                "op": "edit",
                "prompt": prompt,
                "input_image": str(image_path),
                "output_image": str(out_path),
                "steps": args.flux_steps,
                "guidance_scale": args.flux_guidance_scale,
                "true_cfg_scale": args.flux_true_cfg_scale,
                "seed": seed,
                "resolution": args.resolution,
                "max_sequence_length": args.flux_max_sequence_length,
            }
        )
        if res.get("status") != "ok":
            raise RuntimeError(f"Flux edit error: {res}")
        return Path(res["output_image"])

    def generate(self, prompt: str, out_path: Path, seed: int, args: argparse.Namespace) -> Path:
        """Request one policy-configured prompt-only image."""
        res = self._inner.request(
            {
                "op": "generate",
                "prompt": prompt,
                "output_image": str(out_path),
                "steps": args.flux_steps,
                "guidance_scale": args.flux_guidance_scale,
                "seed": seed,
                "resolution": args.resolution,
                "max_sequence_length": args.flux_max_sequence_length,
            }
        )
        if res.get("status") != "ok":
            raise RuntimeError(f"Flux generate error: {res}")
        return Path(res["output_image"])

    def generate_batch(
        self,
        items: Sequence[tuple[str, Path, int]],
        args: argparse.Namespace,
    ) -> List[Path]:
        """Request a same-settings text-to-image batch and preserve item order."""
        if not items:
            return []
        res = self._inner.request(
            {
                "op": "generate_batch",
                "items": [
                    {"prompt": prompt, "output_image": str(out_path), "seed": int(seed)}
                    for prompt, out_path, seed in items
                ],
                "steps": args.flux_steps,
                "guidance_scale": args.flux_guidance_scale,
                "resolution": args.resolution,
                "max_sequence_length": args.flux_max_sequence_length,
            }
        )
        if res.get("status") != "ok":
            raise RuntimeError(f"Flux batch generate error: {res}")
        paths = [Path(value) for value in res.get("output_images") or []]
        if len(paths) != len(items):
            raise RuntimeError(f"Flux batch generate returned {len(paths)} paths for {len(items)} items")
        return paths

    def close(self) -> None:
        """Shut down the underlying FLUX.2 service process."""
        self._inner.close()


class SD35TurboClient:
    """Parent-side wrapper for the prompt-only SD3.5 service."""
    def __init__(self, args: argparse.Namespace):
        """Start SD3.5 in an isolated process bound to one physical GPU."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = _physical_gpu_arg(args.sd35_gpu_index)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        cmd = [
            args.vcot_python,
            str(args.sd35_service_script),
            "--model",
            args.sd35_model,
            "--device",
            "cuda:0",
            "--dtype",
            args.sd35_dtype,
        ]
        self._inner = _ServiceClient(cmd, env, "__SD35_RESULT__", ready_timeout_s=900)

    def generate(self, prompt: str, out_path: Path, seed: int, args: argparse.Namespace) -> Path:
        """Request one policy-configured SD3.5 text-to-image result."""
        res = self._inner.request(
            {
                "op": "generate",
                "prompt": prompt,
                "output_image": str(out_path),
                "steps": args.sd35_steps,
                "guidance_scale": args.sd35_guidance_scale,
                "seed": seed,
                "resolution": args.resolution,
                "max_sequence_length": args.sd35_max_sequence_length,
            }
        )
        if res.get("status") != "ok":
            raise RuntimeError(f"SD3.5 Turbo generate error: {res}")
        return Path(res["output_image"])

    def close(self) -> None:
        """Shut down the underlying SD3.5 service process."""
        self._inner.close()


def release_cuda_memory() -> None:
    """Best-effort release of in-process PyTorch CUDA caches."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


class HeavyRuntimeManager:
    """Lazily own Qwen and editor clients with explicit release boundaries."""
    def __init__(self, args: argparse.Namespace, per_gpu_extra: Dict[int, float]):
        """Store runtime policy without allocating any heavyweight model."""
        self.args = args
        self.per_gpu_extra = per_gpu_extra
        self.keep_models_loaded = bool(getattr(args, "keep_models_loaded", False))
        self._qwen: Optional[QwenPlanner] = None
        self._bagel: Optional[BagelClient] = None
        self._flux: Optional[FluxClient] = None

    def _build_qwen(self) -> QwenPlanner:
        """Instantiate Qwen from the validated CLI/runtime settings."""
        return QwenPlanner(
            model_path=self.args.qwen_model,
            dtype=parse_dtype(self.args.qwen_dtype),
            max_memory_per_gpu=self.args.qwen_max_memory_per_gpu,
            min_free_gib=float(getattr(self.args, "qwen_min_free_gib", 16.0)),
            per_gpu_extra_reserve=self.per_gpu_extra,
            quantization=self.args.qwen_quantization,
            allow_cpu_offload=bool(getattr(self.args, "qwen_allow_cpu_offload", False)),
            route_head_path=str(getattr(self.args, "qwen_route_head", "") or "") or None,
        )

    def _release_editors(self) -> None:
        """Close all generator/editor subprocess clients."""
        self._release_bagel()
        self._release_flux()

    def qwen(self) -> QwenPlanner:
        """Return the shared Qwen instance, releasing editors when required."""
        if not self.keep_models_loaded:
            self._release_editors()
        if self._qwen is None:
            self._qwen = self._build_qwen()
        return self._qwen

    def bagel(self) -> BagelClient:
        """Return a BAGEL client, releasing conflicting runtimes when required."""
        if not self.keep_models_loaded:
            self._release_qwen()
            self._release_flux()
        if self._bagel is None:
            self._bagel = BagelClient(self.args)
        return self._bagel

    def flux(self) -> FluxClient:
        """Return a FLUX.2 client, releasing conflicting runtimes when required."""
        if not self.keep_models_loaded:
            self._release_qwen()
            self._release_bagel()
        if self._flux is None:
            self._flux = FluxClient(self.args)
        return self._flux

    def _release_qwen(self) -> None:
        """Drop Qwen/head references and return their cached CUDA memory."""
        if self._qwen is None:
            return
        planner = self._qwen
        model = getattr(planner, "model", None)
        processor = getattr(planner, "processor", None)
        input_device = getattr(planner, "input_device", None)
        if hasattr(planner, "model"):
            planner.model = None
        if hasattr(planner, "processor"):
            planner.processor = None
        if hasattr(planner, "input_device"):
            planner.input_device = None
        self._qwen = None
        if model is not None:
            del model
        if processor is not None:
            del processor
        if input_device is not None:
            del input_device
        del planner
        release_cuda_memory()

    def _release_bagel(self) -> None:
        """Close BAGEL if loaded and collect local CUDA caches."""
        if self._bagel is None:
            return
        self._bagel.close()
        self._bagel = None
        release_cuda_memory()

    def _release_flux(self) -> None:
        """Close FLUX.2 if loaded and collect local CUDA caches."""
        if self._flux is None:
            return
        self._flux.close()
        self._flux = None
        release_cuda_memory()

    def bagel_generate(self, prompt: str, out_path: Path) -> Path:
        """Generate through the lazy BAGEL client and release it per policy."""
        client = self.bagel()
        try:
            return client.generate(prompt, out_path, self.args)
        finally:
            if not self.keep_models_loaded:
                self._release_bagel()

    def flux_edit(self, image_path: Path, prompt: str, out_path: Path, seed: int) -> Path:
        """Edit through the lazy FLUX.2 client and release it per policy."""
        client = self.flux()
        try:
            return client.edit(image_path, prompt, out_path, seed, self.args)
        finally:
            if not self.keep_models_loaded:
                self._release_flux()

    def flux_generate(self, prompt: str, out_path: Path, seed: int) -> Path:
        """Generate through the lazy FLUX.2 client and release it per policy."""
        client = self.flux()
        try:
            return client.generate(prompt, out_path, seed, self.args)
        finally:
            if not self.keep_models_loaded:
                self._release_flux()

    def close(self) -> None:
        """Release every in-process model and generator subprocess."""
        self._release_qwen()
        self._release_bagel()
        self._release_flux()

    def release_qwen(self) -> None:
        """Public idempotent Qwen release hook used by stage code."""
        self._release_qwen()

    def release_bagel(self) -> None:
        """Public idempotent BAGEL release hook used by stage code."""
        self._release_bagel()

    def release_flux(self) -> None:
        """Public idempotent FLUX.2 release hook used by stage code."""
        self._release_flux()


def _patch_bnb_int8params_new() -> None:
    """Ignore a newer Transformers-only kwarg in older bitsandbytes releases."""
    try:
        from bitsandbytes.nn.modules import Int8Params
    except Exception:
        return
    if getattr(Int8Params, "_copilot_patch_drop_hf_init", False):
        return
    original_new = Int8Params.__new__

    def patched_new(cls, *args: Any, **kwargs: Any) -> Any:
        """Drop the unsupported initialization marker before construction."""
        kwargs.pop("_is_hf_initialized", None)
        return original_new(cls, *args, **kwargs)

    Int8Params.__new__ = patched_new
    Int8Params._copilot_patch_drop_hf_init = True
