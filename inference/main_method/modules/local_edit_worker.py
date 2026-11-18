#!/usr/bin/env python3
"""Execute the isolated one-call L1 local-edit workload.

The local executor enables FLUX.2 only. Its action contract is ``l1_joint_multicrop_v1``: one editable full image
followed by one to three ordered localization crops, one joint semantic prompt,
one deterministic seed, and exactly one successful generation call.

The ``run`` command canonicalizes input assets, writes a manifest, and launches
model workers across GPU shards subject to per-model memory gates.  The
``worker`` command owns one model runtime, resumes completed per-sample outputs,
and writes ``final.png`` plus ``result.json`` in the order expected by the
parent pipeline's partial-batch recovery logic.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "local_edit_outputs"

try:
    from .contracts import (
        L1_CROP_MIN_SPAN,
        L1_CROP_PAD_RATIO,
        L1_MAX_CROPS,
        L1_QUAD_MIN_SPAN,
        build_l1_joint_prompt,
        l1_crop_roi,
        l1_crop_rois,
    )
except ImportError:  # Direct script execution: expose inference/main_method as an import root.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from modules.contracts import (
        L1_CROP_MIN_SPAN,
        L1_CROP_PAD_RATIO,
        L1_MAX_CROPS,
        L1_QUAD_MIN_SPAN,
        build_l1_joint_prompt,
        l1_crop_roi,
        l1_crop_rois,
    )

DEFAULT_VCOT_PYTHON = sys.executable

DEFAULT_FLUX2_MODEL = "black-forest-labs/FLUX.2-dev"

DEFAULT_MODELS = [
    "flux2",
]
DEFAULT_MODEL_MIN_FREE_GIB = {
    "flux2": 55.0,
}
DEFAULT_MODEL_MAX_PARALLEL = {
    "flux2": 4,
}

PNG_CANDIDATES = [
    "input.png",
    "round_00_input.png",
    "final.png",
    "round_03_chosen.png",
]
JSON_CANDIDATES = [
    "edit.json",
    "manual_sd_job_no_prompt.json",
    "round_00_qwen_analysis.json",
    "round_03_qwen_analysis.json",
    "round_00_source_analysis.json",
]


Point = Tuple[int, int]
Box = Tuple[int, int, int, int]
MIN_QUAD_SPAN = L1_QUAD_MIN_SPAN
L1_JOINT_ACTION_SCHEMA = "l1_joint_multicrop_v1"
MAX_JOINT_LOCAL_EDITS = L1_MAX_CROPS


def l1_reference_roles(edit_count: int) -> List[str]:
    """Return the exact ordered reference roles required by joint FLUX.2 L1."""
    if not 1 <= int(edit_count) <= MAX_JOINT_LOCAL_EDITS:
        raise ValueError(f"Joint L1 reference roles require 1-{MAX_JOINT_LOCAL_EDITS} edits, got {edit_count}")
    return ["editable_full_image", *[f"localization_crop_{index:02d}" for index in range(1, edit_count + 1)]]


@dataclass
class LocalEditSpec:
    """Canonical description of one ordered localized repair region."""
    id: str
    target_object: str
    issue: str
    edit_prompt: str
    whole_edit: str
    quad_points: List[Point]


@dataclass
class SampleSpec:
    """All durable inputs and action metadata required for one L1 sample."""
    sample_id: str
    sample_dir: Path
    input_image: Path
    edit_json: Path
    instruction: str
    whole_edit: str
    target_object: str
    quad_points: List[Point]
    edits: List[LocalEditSpec] = field(default_factory=list)
    action_schema: str = L1_JOINT_ACTION_SCHEMA
    action_seed: Optional[int] = None
    round_index: int = 0
    joint_prompt: str = ""


def local_edit_payload(edit: LocalEditSpec) -> Dict[str, Any]:
    """Serialize a local edit with integer coordinates for result manifests."""
    return {
        "id": edit.id,
        "target_object": edit.target_object,
        "issue": edit.issue,
        "edit_prompt": edit.edit_prompt,
        "whole_edit": edit.whole_edit,
        "quad_points": [[int(x), int(y)] for x, y in edit.quad_points],
    }


def ordered_local_edits(sample: SampleSpec) -> List[LocalEditSpec]:
    """Return 1-3 ordered edits, adapting the legacy single-edit fields."""
    edits = list(sample.edits)
    if not edits:
        edits = [
            LocalEditSpec(
                id="edit_01",
                target_object=sample.target_object,
                issue="",
                edit_prompt=sample.instruction,
                whole_edit=sample.whole_edit or sample.instruction,
                quad_points=list(sample.quad_points),
            )
        ]
    if not 1 <= len(edits) <= MAX_JOINT_LOCAL_EDITS:
        raise ValueError(
            f"{L1_JOINT_ACTION_SCHEMA} requires 1-{MAX_JOINT_LOCAL_EDITS} edits, got {len(edits)}"
        )
    return edits


def stable_l1_action_seed(
    base_seed: int,
    sample_id: str,
    edits: Sequence[LocalEditSpec | Dict[str, Any]],
    round_index: int = 0,
) -> int:
    """Derive a stable 31-bit seed from the complete semantic action identity."""
    canonical_edits: List[Dict[str, Any]] = []
    for index, edit in enumerate(edits, start=1):
        if isinstance(edit, LocalEditSpec):
            payload = local_edit_payload(edit)
        else:
            quad = normalize_quad(edit.get("quad_points") or box_to_quad(edit.get("box_xyxy")))
            edit_prompt = str(edit.get("edit_prompt") or edit.get("whole_edit") or edit.get("issue") or "").strip()
            payload = {
                "id": str(edit.get("id") or f"edit_{index:02d}").strip(),
                "target_object": str(edit.get("target_object") or "").strip(),
                "issue": str(edit.get("issue") or "").strip(),
                "edit_prompt": edit_prompt,
                "whole_edit": str(edit.get("whole_edit") or edit_prompt).strip(),
                "quad_points": [[int(x), int(y)] for x, y in quad],
            }
        canonical_edits.append(payload)
    material = json.dumps(
        {
            "action_schema": L1_JOINT_ACTION_SCHEMA,
            "base_seed": int(base_seed),
            "sample_id": str(sample_id),
            "round_index": int(round_index),
            "edits": canonical_edits,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31 - 1)


def parse_gpu_indices(text: str) -> List[int]:
    """Parse a non-empty comma-separated physical GPU list."""
    items = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        items.append(int(chunk))
    if not items:
        raise ValueError("gpu list is empty")
    return items


def parse_model_float_map(text: str, defaults: Dict[str, float]) -> Dict[str, float]:
    """Parse per-model floating resource overrides on top of defaults."""
    values = dict(defaults)
    for chunk in str(text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Expected model=value entry, got {chunk!r}")
        key, value = chunk.split("=", 1)
        key = key.strip()
        if key not in DEFAULT_MODELS:
            raise ValueError(f"Unknown model name in resource map: {key!r}")
        values[key] = float(value.strip())
    return values


def parse_model_int_map(text: str, defaults: Dict[str, int]) -> Dict[str, int]:
    """Parse positive per-model integer concurrency overrides."""
    values = dict(defaults)
    for chunk in str(text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Expected model=value entry, got {chunk!r}")
        key, value = chunk.split("=", 1)
        key = key.strip()
        if key not in DEFAULT_MODELS:
            raise ValueError(f"Unknown model name in resource map: {key!r}")
        values[key] = max(1, int(value.strip()))
    return values


def query_gpu_free_gib() -> Dict[int, float]:
    """Query physical GPU free memory in GiB, returning empty on failure."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    values: Dict[int, float] = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            values[int(parts[0])] = float(parts[1]) / 1024.0
        except ValueError:
            continue
    return values


def is_memory_related_text(text: str) -> bool:
    """Recognize GPU allocation/address failures eligible for worker retry."""
    value = str(text or "").lower()
    return any(
        pattern in value
        for pattern in (
            "outofmemoryerror",
            "cuda out of memory",
            "torch.outofmemoryerror",
            "cublas_status_alloc_failed",
            "cuda error: an illegal memory access was encountered",
            "cudaerrorillegaladdress",
        )
    )


def read_text_if_exists(path: Path) -> str:
    """Read a diagnostic text artifact, returning empty when absent."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def mkdir(path: Path) -> None:
    """Create a directory and all missing parents idempotently."""
    path.mkdir(parents=True, exist_ok=True)


def generation_resolution(args: argparse.Namespace) -> int:
    """Validate and return the square generator resolution."""
    resolution = int(getattr(args, "resolution", 1024))
    if resolution <= 0:
        raise ValueError(f"--resolution must be positive, got {resolution}")
    return resolution


def resize_to_generation_resolution(image: Any, args: argparse.Namespace) -> Any:
    """Resize a PIL-like image to the square generation contract."""
    from PIL import Image

    resolution = generation_resolution(args)
    target_size = (resolution, resolution)
    if image.size == target_size:
        return image
    return image.resize(target_size, Image.Resampling.LANCZOS)


def write_json(path: Path, payload: Any) -> None:
    """Write indented UTF-8 JSON with a trailing newline."""
    mkdir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append one compact record to a JSONL progress stream."""
    mkdir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sample_model_output_path(sample: SampleSpec, model_name: str) -> Path:
    """Return the canonical materialized model image beside sample inputs."""
    return sample.sample_dir / f"{model_name}.png"


def model_class_name(model_path: str) -> str:
    """Read a Diffusers pipeline class without importing the model."""
    model_index = Path(model_path) / "model_index.json"
    if not model_index.exists():
        return ""
    try:
        payload = json.loads(model_index.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(payload.get("_class_name") or "")


def resolve_hf_snapshot(path_or_repo: str) -> str:
    """Resolve local model bytes from an explicit path or HF cache only."""
    candidate = Path(path_or_repo)
    if candidate.exists():
        return str(candidate)

    roots = [
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    cache_name = "models--" + path_or_repo.replace("/", "--")
    for root in roots:
        base = root / cache_name
        if not base.exists():
            continue
        snaps = base / "snapshots"
        refs = base / "refs" / "main"
        if refs.exists() and snaps.exists():
            sha = refs.read_text(encoding="utf-8").strip()
            resolved = snaps / sha
            if resolved.exists():
                return str(resolved)
        if (snaps / "main").exists():
            return str(snaps / "main")
        if snaps.exists():
            dirs = sorted([item for item in snaps.iterdir() if item.is_dir()])
            if dirs:
                return str(dirs[-1])
    raise FileNotFoundError(f"Cannot resolve local Hugging Face snapshot: {path_or_repo}")


def pick_top_level_file(sample_dir: Path, candidates: Sequence[str], suffix: str) -> Path:
    """Select a named asset or an unambiguous sole file with the suffix."""
    for name in candidates:
        path = sample_dir / name
        if path.is_file():
            return path
    matches = sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() == suffix.lower()
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No {suffix} file found in {sample_dir}")
    raise RuntimeError(f"Multiple candidate {suffix} files found in {sample_dir}: {matches}")


def canonicalize_sample_assets(data_root: Path) -> List[SampleSpec]:
    """Normalize sample filenames and build a validated execution manifest.

    Legacy input names are migrated to ``input.png``/``edit.json`` where safe.
    Every accepted task is converted to the joint L1 schema before workers are
    launched.
    """
    manifest: List[SampleSpec] = []
    for sample_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        image_src = pick_top_level_file(sample_dir, PNG_CANDIDATES, ".png")
        json_src = pick_top_level_file(sample_dir, JSON_CANDIDATES, ".json")

        image_dst = sample_dir / "input.png"
        json_dst = sample_dir / "edit.json"

        if image_src != image_dst and not image_dst.exists():
            image_src.rename(image_dst)
        if json_src != json_dst and not json_dst.exists():
            json_src.rename(json_dst)

        if not image_dst.exists():
            image_dst = image_src
        if not json_dst.exists():
            json_dst = json_src

        payload = json.loads(json_dst.read_text(encoding="utf-8"))
        edits = normalize_local_edit_specs(payload)
        if not edits:
            continue
        instruction = normalize_instruction(payload)
        whole_edit = normalize_whole_edit(payload, instruction)
        first_edit = edits[0]
        action_seed_value = payload.get("action_seed")
        action_seed = int(action_seed_value) if action_seed_value is not None else None
        action_schema = str(payload.get("action_schema") or L1_JOINT_ACTION_SCHEMA)
        if action_schema != L1_JOINT_ACTION_SCHEMA:
            raise ValueError(f"Unsupported local action schema: {action_schema!r}")
        nested_action_plan = payload.get("action_plan") if isinstance(payload.get("action_plan"), dict) else {}
        has_joint_contract = (
            payload.get("operation") == "edit_multi_reference"
            or payload.get("action") == "joint_local_edit"
            or nested_action_plan.get("action") == "joint_local_edit"
        )
        joint_prompt = ""
        if has_joint_contract:
            joint_prompt = str(
                payload.get("joint_prompt")
                or payload.get("instruction")
                or nested_action_plan.get("instruction")
                or ""
            ).strip()
        manifest.append(
            SampleSpec(
                sample_id=sample_dir.name,
                sample_dir=sample_dir,
                input_image=image_dst,
                edit_json=json_dst,
                instruction=instruction,
                whole_edit=whole_edit,
                target_object=first_edit.target_object,
                quad_points=list(first_edit.quad_points),
                edits=edits,
                action_schema=action_schema,
                action_seed=action_seed,
                round_index=int(payload.get("round_index", 0) or 0),
                joint_prompt=joint_prompt,
            )
        )
    return manifest


def normalize_instruction(payload: Dict[str, Any]) -> str:
    """Recover a non-empty aggregate instruction from legacy/new payloads."""
    value = payload.get("edits")
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                candidate = str(
                    item.get("edit_prompt")
                    or item.get("instruction")
                    or item.get("fix")
                    or item.get("issue")
                    or ""
                ).strip()
            else:
                candidate = str(item).strip()
            if candidate:
                parts.append(candidate)
        text = " ".join(parts)
    else:
        text = ""
    if not text:
        text = str(payload.get("fix") or payload.get("issue") or payload.get("edit_prompt") or "").strip()
    if not text:
        raise ValueError("Instruction is empty")
    return text


def normalize_whole_edit(payload: Dict[str, Any], fallback_instruction: str) -> str:
    """Return the global semantic instruction with a safe local fallback."""
    value = payload.get("whole_edit")
    if value is None:
        text = str(payload.get("fix") or "").strip()
        return text or fallback_instruction
    text = str(value).strip()
    if not text:
        text = str(payload.get("fix") or "").strip()
        return text or fallback_instruction
    return text


def normalize_local_edit_specs(payload: Dict[str, Any]) -> List[LocalEditSpec]:
    """Normalize one to three edit objects and reject ambiguous geometry."""
    raw_edits = payload.get("edits")
    candidates: List[Dict[str, Any]] = []
    if isinstance(raw_edits, list):
        candidates = [item for item in raw_edits if isinstance(item, dict)]
        if len(candidates) != len(raw_edits):
            raise ValueError("Every joint L1 edit must be a JSON object")
    if not candidates:
        quad_source = resolve_quad_source(payload)
        if quad_source is None:
            return []
        candidates = [payload]
    if not 1 <= len(candidates) <= MAX_JOINT_LOCAL_EDITS:
        raise ValueError(
            f"{L1_JOINT_ACTION_SCHEMA} requires 1-{MAX_JOINT_LOCAL_EDITS} edits, got {len(candidates)}"
        )

    edits: List[LocalEditSpec] = []
    for index, item in enumerate(candidates, start=1):
        quad_source = item.get("quad_points")
        if quad_source is None and item.get("box_xyxy") is not None:
            quad_source = box_to_quad(item.get("box_xyxy"))
        if quad_source is None:
            raise ValueError(f"Joint L1 edit {index} has no quad_points/box_xyxy")
        edit_prompt = str(
            item.get("edit_prompt")
            or item.get("instruction")
            or item.get("whole_edit")
            or item.get("fix")
            or item.get("issue")
            or ""
        ).strip()
        if not edit_prompt:
            raise ValueError(f"Joint L1 edit {index} has an empty instruction")
        edits.append(
            LocalEditSpec(
                id=str(item.get("id") or f"edit_{index:02d}").strip(),
                target_object=str(item.get("target_object") or item.get("target") or "").strip(),
                issue=str(item.get("issue") or "").strip(),
                edit_prompt=edit_prompt,
                whole_edit=str(item.get("whole_edit") or edit_prompt).strip(),
                quad_points=normalize_quad(quad_source),
            )
        )
    return edits


def resolve_quad_source(payload: Dict[str, Any]) -> Optional[Any]:
    """Find the first explicit quad or unambiguous legacy box."""
    quad = payload.get("quad_points")
    if quad is not None:
        return quad

    box = payload.get("box_xyxy")
    if box is not None:
        return box_to_quad(box)

    edits = payload.get("edits")
    if isinstance(edits, list):
        for item in edits:
            if not isinstance(item, dict):
                continue
            if item.get("quad_points") is not None:
                return item.get("quad_points")
            if item.get("box_xyxy") is not None:
                return box_to_quad(item.get("box_xyxy"))

    return None


def box_to_quad(value: Any) -> List[List[int]]:
    """Convert an ``xyxy`` box to four axis-aligned corner points."""
    if not isinstance(value, Iterable):
        raise ValueError(f"box_xyxy is invalid: {value!r}")
    coords = list(value)
    if len(coords) < 4:
        raise ValueError(f"box_xyxy must contain 4 numbers: {value!r}")
    x0 = int(round(float(coords[0])))
    y0 = int(round(float(coords[1])))
    x1 = int(round(float(coords[2])))
    y1 = int(round(float(coords[3])))
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def normalize_quad(value: Any) -> List[Point]:
    """Validate four numeric points and round them to integer coordinates."""
    if not isinstance(value, Iterable):
        raise ValueError(f"quad_points is invalid: {value!r}")
    raw_points = list(value)
    if len(raw_points) != 4:
        raise ValueError(f"quad_points must contain 4 points: {value!r}")
    quad: List[Point] = []
    for item in raw_points:
        if not isinstance(item, Iterable):
            raise ValueError(f"quad point is invalid: {item!r}")
        coords = list(item)
        if len(coords) < 2:
            raise ValueError(f"quad point is invalid: {item!r}")
        quad.append((int(round(float(coords[0]))), int(round(float(coords[1])))))
    return quad


def clamp_quad(points: Sequence[Point], width: int, height: int) -> List[Point]:
    """Clamp and expand a quad to the shared minimum localization span."""
    quad: List[Point] = []
    for x, y in points:
        quad.append((max(0, min(width - 1, x)), max(0, min(height - 1, y))))
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    x0, x1 = expand_axis_span(min(xs), max(xs), width, MIN_QUAD_SPAN)
    y0, y1 = expand_axis_span(min(ys), max(ys), height, MIN_QUAD_SPAN)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def expand_axis_span(start: int, end: int, limit: int, min_span: int) -> Tuple[int, int]:
    """Expand an inclusive bounded span around its center when undersized."""
    if limit <= 1:
        return 0, 0
    lo = max(0, min(limit - 1, min(start, end)))
    hi = max(0, min(limit - 1, max(start, end)))
    target_span = min(min_span, limit - 1)
    current_span = hi - lo
    if current_span >= target_span:
        return lo, hi
    need = target_span - current_span
    pad_lo = need // 2
    pad_hi = need - pad_lo
    lo -= pad_lo
    hi += pad_hi
    if lo < 0:
        hi = min(limit - 1, hi - lo)
        lo = 0
    if hi > limit - 1:
        lo = max(0, lo - (hi - (limit - 1)))
        hi = limit - 1
    return lo, hi


def quad_to_box(quad: Sequence[Point], width: int, height: int) -> Box:
    """Convert inclusive points to a clipped half-open crop box."""
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return max(0, min(xs)), max(0, min(ys)), min(width, max(xs) + 1), min(height, max(ys) + 1)


def square_roi_for_box(box_xyxy: Box, image_size: Tuple[int, int], pad_ratio: float) -> Box:
    """Compatibility wrapper around the shared V10 L1 crop contract."""

    width, height = image_size
    x0, y0, x1, y1 = box_xyxy
    quad = [
        (x0, y0),
        (max(x0, x1 - 1), y0),
        (max(x0, x1 - 1), max(y0, y1 - 1)),
        (x0, max(y0, y1 - 1)),
    ]
    return tuple(
        l1_crop_roi(
            quad,
            width,
            height,
            pad_ratio=pad_ratio,
            min_span=L1_CROP_MIN_SPAN,
        )
    )


def compose_local_edit_prompt(instruction: str) -> str:
    """Wrap a local instruction with preservation and artifact constraints."""
    return (
        f"{instruction.strip()}. "
        "Apply only this local edit and satisfy any requested count exactly. "
        "If the target is clothing or an accessory, show the standalone item itself, not a new person, body part, mannequin, or wearer. "
        "Do not introduce unrelated people, text, labels, or artifacts. "
        "Keep all other objects, composition, perspective, lighting, and background unchanged."
    )


def compose_semantic_global_prompt(instruction: str) -> str:
    """Wrap a whole-image instruction with conservative preservation guidance."""
    return (
        f"{instruction.strip()}. "
        "Use the full image as the edit target and the cropped local reference image to identify the exact region. "
        "Edit only that intended local region and satisfy any requested count exactly. "
        "If the target is clothing or an accessory, show the standalone item itself, not a new person, body part, mannequin, or wearer. "
        "Do not introduce unrelated people, text, labels, or artifacts. "
        "Keep everything else unchanged."
    )


def compose_joint_multicrop_prompt(
    edits: Sequence[LocalEditSpec],
    *,
    target_prompt: str = "",
    diagnosis: Optional[Dict[str, Any]] = None,
    width: int = 1024,
    height: int = 1024,
) -> str:
    """Compile the shared V10 joint L1 prompt from ordered edit facts."""
    if not 1 <= len(edits) <= MAX_JOINT_LOCAL_EDITS:
        raise ValueError(
            f"{L1_JOINT_ACTION_SCHEMA} requires 1-{MAX_JOINT_LOCAL_EDITS} edits, got {len(edits)}"
        )
    if diagnosis is None:
        diagnosis = {
            "fully_satisfied": False,
            "reason": "Localized mismatches require a joint L1 edit.",
            "issue": " | ".join(edit.issue or "diagnosed mismatch" for edit in edits),
            "fix": " | ".join(edit.edit_prompt for edit in edits),
        }
    return build_l1_joint_prompt(
        target_prompt,
        diagnosis,
        [local_edit_payload(edit) for edit in edits],
        int(width),
        int(height),
    )


def local_crop_for_sample(sample: SampleSpec, image: Any, out_dir: Path, args: argparse.Namespace) -> Tuple[Any, Path, Box]:
    """Legacy one-crop helper retained for non-FLUX external callers."""
    mkdir(out_dir)
    first_edit = ordered_local_edits(sample)[0]
    roi = tuple(l1_crop_roi(first_edit.quad_points, image.width, image.height))
    crop = image.crop(roi)
    crop_path = out_dir / "local_crop.png"
    crop.save(crop_path)
    return resize_to_generation_resolution(crop, args), crop_path, roi


def local_crops_for_sample(
    sample: SampleSpec,
    image: Any,
    out_dir: Path,
    args: argparse.Namespace,
    edits: Optional[Sequence[LocalEditSpec]] = None,
) -> List[Tuple[Any, Path, Box]]:
    """Crop 1-3 padded ROIs in planner order and resize reference copies."""
    mkdir(out_dir)
    crops: List[Tuple[Any, Path, Box]] = []
    selected_edits = list(edits) if edits is not None else ordered_local_edits(sample)
    if len(selected_edits) > int(getattr(args, "l1_max_crops", L1_MAX_CROPS)):
        raise ValueError(f"L1 edit count exceeds locked max_crops={args.l1_max_crops}")
    rois = l1_crop_rois(
        [edit.quad_points for edit in selected_edits],
        image.width,
        image.height,
        pad_ratio=float(getattr(args, "l1_roi_pad_ratio", L1_CROP_PAD_RATIO)),
        min_span=int(getattr(args, "l1_min_span", L1_CROP_MIN_SPAN)),
    )
    for index, raw_roi in enumerate(rois, start=1):
        roi = tuple(raw_roi)
        crop = image.crop(roi)
        crop_path = out_dir / f"local_crop_{index:02d}.png"
        crop.save(crop_path)
        crops.append((resize_to_generation_resolution(crop, args), crop_path, roi))
    return crops


def scale_local_edits(
    edits: Sequence[LocalEditSpec],
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> List[LocalEditSpec]:
    """Map edit coordinates between source and generator image sizes."""
    if source_size == target_size:
        return [
            LocalEditSpec(
                id=edit.id,
                target_object=edit.target_object,
                issue=edit.issue,
                edit_prompt=edit.edit_prompt,
                whole_edit=edit.whole_edit,
                quad_points=list(edit.quad_points),
            )
            for edit in edits
        ]
    scale_x = target_size[0] / float(source_size[0])
    scale_y = target_size[1] / float(source_size[1])
    return [
        LocalEditSpec(
            id=edit.id,
            target_object=edit.target_object,
            issue=edit.issue,
            edit_prompt=edit.edit_prompt,
            whole_edit=edit.whole_edit,
            quad_points=[
                (int(round(x * scale_x)), int(round(y * scale_y)))
                for x, y in edit.quad_points
            ],
        )
        for edit in edits
    ]


def save_prepare_manifest(output_root: Path, samples: Sequence[SampleSpec]) -> None:
    """Persist canonical tasks so every GPU worker sees identical inputs."""
    payload = []
    for sample in samples:
        payload.append(
            {
                "sample_id": sample.sample_id,
                "sample_dir": str(sample.sample_dir),
                "input_image": str(sample.input_image),
                "edit_json": str(sample.edit_json),
                "instruction": sample.instruction,
                "whole_edit": sample.whole_edit,
                "target_object": sample.target_object,
                "quad_points": [[x, y] for x, y in sample.quad_points],
                "action_schema": sample.action_schema,
                "operation": "edit_multi_reference",
                "reference_roles": l1_reference_roles(len(ordered_local_edits(sample))),
                "action_seed": sample.action_seed,
                "round_index": sample.round_index,
                "joint_prompt": sample.joint_prompt,
                "edits": [local_edit_payload(edit) for edit in ordered_local_edits(sample)],
            }
        )
    write_json(output_root / "prepared_manifest.json", payload)


def save_final_image_outputs(final_image: Any, sample: SampleSpec, model_name: str, out_dir: Path) -> Tuple[Path, Path]:
    """Save both worker-local and parent-consumed copies of a result image."""
    worker_final_path = out_dir / "final.png"
    sample_final_path = sample_model_output_path(sample, model_name)
    final_image.save(worker_final_path)
    final_image.save(sample_final_path)
    return sample_final_path, worker_final_path


def load_prepare_manifest(output_root: Path) -> List[SampleSpec]:
    """Reconstruct and revalidate worker sample specs from the shared manifest."""
    payload = json.loads((output_root / "prepared_manifest.json").read_text(encoding="utf-8"))
    samples: List[SampleSpec] = []
    for item in payload:
        edits = normalize_local_edit_specs(item)
        first_edit = edits[0]
        action_schema = str(item.get("action_schema") or L1_JOINT_ACTION_SCHEMA)
        if action_schema != L1_JOINT_ACTION_SCHEMA:
            raise ValueError(f"Unsupported local action schema in prepared manifest: {action_schema!r}")
        samples.append(
            SampleSpec(
                sample_id=str(item["sample_id"]),
                sample_dir=Path(item["sample_dir"]),
                input_image=Path(item["input_image"]),
                edit_json=Path(item["edit_json"]),
                instruction=str(item["instruction"]),
                whole_edit=str(item.get("whole_edit") or item["instruction"]),
                target_object=str(item.get("target_object") or first_edit.target_object),
                quad_points=[(int(x), int(y)) for x, y in item.get("quad_points", first_edit.quad_points)],
                edits=edits,
                action_schema=action_schema,
                action_seed=int(item["action_seed"]) if item.get("action_seed") is not None else None,
                round_index=int(item.get("round_index", 0) or 0),
                joint_prompt=str(item.get("joint_prompt") or "").strip(),
            )
        )
    return samples


def choose_worker_python(model_name: str, args: argparse.Namespace) -> str:
    """Select the environment whose dependencies match the requested backend."""
    return args.vcot_python


def build_worker_command(model_name: str, gpu_index: int, args: argparse.Namespace) -> List[str]:
    """Build a self-contained worker CLI preserving action/model settings."""
    cmd = [
        choose_worker_python(model_name, args),
        str(SCRIPT_PATH),
        "worker",
        "--model",
        model_name,
        "--gpu-index",
        str(gpu_index),
        "--resolution",
        str(args.resolution),
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(args.output_root),
        "--flux2-model",
        args.flux2_model,
        "--flux2-steps",
        str(args.flux2_steps),
        "--flux2-guidance-scale",
        str(args.flux2_guidance_scale),
        "--flux2-max-sequence-length",
        str(args.flux2_max_sequence_length),
        "--seed",
        str(args.seed),
    ]
    if args.force:
        cmd.append("--force")
    return cmd


def shard_suffix(shard_index: int, shard_count: int) -> str:
    """Format a stable human-readable shard identifier."""
    return f"shard_{shard_index + 1:02d}_of_{shard_count:02d}"


def select_worker_samples(samples: Sequence[SampleSpec], shard_index: int, shard_count: int) -> List[Tuple[int, SampleSpec]]:
    """Select a deterministic modulo shard while retaining global indices."""
    selected: List[Tuple[int, SampleSpec]] = []
    for global_index, sample in enumerate(samples, start=1):
        if (global_index - 1) % max(1, shard_count) != shard_index:
            continue
        selected.append((global_index, sample))
    return selected


def run_orchestrator(args: argparse.Namespace) -> int:
    """Prepare tasks and schedule retryable worker shards across available GPUs.

    Each model is processed separately.  New workers start only when their
    physical GPU meets the model-specific free-memory gate; memory-related
    worker exits may be retried, while permanent failures are summarized
    without discarding other completed shards.
    """
    mkdir(args.output_root)
    samples = canonicalize_sample_assets(args.data_root)
    save_prepare_manifest(args.output_root, samples)

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    gpu_indices = parse_gpu_indices(args.gpu_indices)
    workers_per_model = max(1, int(args.workers_per_model))
    model_min_free_gib = parse_model_float_map(args.model_min_free_gib, DEFAULT_MODEL_MIN_FREE_GIB)
    model_max_parallel = parse_model_int_map(args.model_max_parallel, DEFAULT_MODEL_MAX_PARALLEL)
    max_memory_retries = max(0, int(args.max_memory_retries))
    memory_wait_sec = max(1.0, float(args.memory_wait_sec))
    summary_path = args.output_root / "summary.jsonl"
    if summary_path.exists() and args.force:
        summary_path.unlink()

    for model_name in models:
        shard_count = min(len(samples), len(gpu_indices), workers_per_model)
        running: Dict[int, Dict[str, Any]] = {}
        pending_shards = list(range(shard_count))
        retry_counts = {shard_index: 0 for shard_index in pending_shards}
        permanent_failures = 0
        max_parallel = min(shard_count, len(gpu_indices), model_max_parallel.get(model_name, workers_per_model), workers_per_model)
        min_free_gib = model_min_free_gib.get(model_name, 0.0)
        last_wait_log = 0.0

        while pending_shards or running:
            finished: List[int] = []
            for shard_index, state in list(running.items()):
                proc = state["proc"]
                exit_code = proc.poll()
                if exit_code is None:
                    continue
                finished.append(shard_index)
                state["log_handle"].close()
                model_summary = args.output_root / model_name / f"model_summary.{state['suffix']}.json"
                log_text = read_text_if_exists(state["log_path"])
                memory_related = exit_code != 0 and is_memory_related_text(log_text)
                can_retry = memory_related and retry_counts[shard_index] < max_memory_retries
                payload: Dict[str, Any] = {
                    "model": model_name,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "gpu_slot": state.get("gpu_slot"),
                    "gpu_index": state["gpu_index"],
                    "log_path": str(state["log_path"]),
                    "exit_code": exit_code,
                    "duration_sec": round(time.time() - state["started_at"], 3),
                    "attempt": retry_counts[shard_index] + 1,
                    "max_memory_retries": max_memory_retries,
                    "memory_related": memory_related,
                    "memory_retry": can_retry,
                }
                if model_summary.exists():
                    try:
                        payload["worker_summary"] = json.loads(model_summary.read_text(encoding="utf-8"))
                    except Exception as exc:
                        payload["worker_summary_error"] = repr(exc)
                append_jsonl(summary_path, payload)
                if can_retry:
                    retry_counts[shard_index] += 1
                    pending_shards.append(shard_index)
                elif exit_code != 0:
                    permanent_failures += 1
            for shard_index in finished:
                del running[shard_index]

            active_slots = {state["gpu_slot"] for state in running.values()}
            while pending_shards and len(running) < max_parallel:
                free_gib = query_gpu_free_gib()
                candidate_slots = [
                    (slot_index, gpu_index)
                    for slot_index, gpu_index in enumerate(gpu_indices)
                    if slot_index not in active_slots and free_gib.get(gpu_index, 0.0) >= min_free_gib
                ]
                if not candidate_slots:
                    now = time.time()
                    if now - last_wait_log >= max(10.0, memory_wait_sec):
                        append_jsonl(
                            summary_path,
                            {
                                "event": "memory_wait",
                                "model": model_name,
                                "pending_shards": list(pending_shards),
                                "running_shards": sorted(running),
                                "min_free_gib": min_free_gib,
                                "free_gib": free_gib,
                                "wait_sec": memory_wait_sec,
                                "timestamp": now,
                            },
                        )
                        last_wait_log = now
                    break
                active_gpu_indices = {state["gpu_index"] for state in running.values()}
                spread_slots = [
                    (slot_index, gpu_index)
                    for slot_index, gpu_index in candidate_slots
                    if gpu_index not in active_gpu_indices
                ]
                usable_slots = spread_slots or candidate_slots
                slot_index, gpu_index = max(usable_slots, key=lambda item: (free_gib.get(item[1], 0.0), -item[0]))
                shard_index = pending_shards.pop(0)
                active_slots.add(slot_index)
                suffix = shard_suffix(shard_index, shard_count)
                attempt = retry_counts[shard_index] + 1
                attempt_suffix = suffix if attempt == 1 else f"{suffix}.try_{attempt:02d}"
                log_path = args.output_root / f"{model_name}.{attempt_suffix}.log"
                append_jsonl(
                    summary_path,
                    {
                        "event": "worker_start",
                        "model": model_name,
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                        "gpu_slot": slot_index,
                        "gpu_index": gpu_index,
                        "attempt": attempt,
                        "min_free_gib": min_free_gib,
                        "free_gib": free_gib,
                        "max_parallel": max_parallel,
                        "timestamp": time.time(),
                    },
                )
                log_handle = log_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                cmd = build_worker_command(model_name, gpu_index, args)
                cmd.extend(["--shard-index", str(shard_index), "--shard-count", str(shard_count)])
                proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env, text=True)
                running[shard_index] = {
                    "proc": proc,
                    "gpu_slot": slot_index,
                    "gpu_index": gpu_index,
                    "log_handle": log_handle,
                    "log_path": log_path,
                    "started_at": time.time(),
                    "suffix": suffix,
                }

            if pending_shards or running:
                time.sleep(memory_wait_sec if pending_shards and not running else 2.0)

        if permanent_failures:
            append_jsonl(
                summary_path,
                {
                    "event": "model_permanent_failures",
                    "model": model_name,
                    "failure_count": permanent_failures,
                    "timestamp": time.time(),
                },
            )

    final_summary = {
        "models": models,
        "gpu_indices": gpu_indices,
        "workers_per_model": workers_per_model,
        "model_min_free_gib": model_min_free_gib,
        "model_max_parallel": model_max_parallel,
        "max_memory_retries": max_memory_retries,
        "memory_wait_sec": memory_wait_sec,
        "resolution": generation_resolution(args),
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "sample_count": len(samples),
    }
    write_json(args.output_root / "summary.json", final_summary)
    return 0


def run_worker(args: argparse.Namespace) -> int:
    """Load one model once and process the worker's deterministic sample shard."""
    all_samples = load_prepare_manifest(args.output_root)
    selected_samples = select_worker_samples(all_samples, args.shard_index, args.shard_count)
    model_out_root = args.output_root / args.model
    mkdir(model_out_root)
    start_time = time.time()
    records = []
    failures = 0
    runtime = initialize_model_runtime(args, model_out_root)
    try:
        for sample_index, sample in selected_samples:
            try:
                record = run_single_model_sample(args, sample, sample_index, runtime)
                records.append(record)
            except Exception as exc:
                failures += 1
                record = {
                    "model": args.model,
                    "sample_id": sample.sample_id,
                    "status": "error",
                    "error": repr(exc),
                }
                records.append(record)
                write_json(model_out_root / sample.sample_id / "error.json", record)
    finally:
        finalize_model_runtime(runtime)

    summary = {
        "model": args.model,
        "gpu_index": args.gpu_index,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "resolution": generation_resolution(args),
        "sample_count": len(selected_samples),
        "total_sample_count": len(all_samples),
        "failure_count": failures,
        "duration_sec": round(time.time() - start_time, 3),
        "records": records,
    }
    summary_file = model_out_root / f"model_summary.{shard_suffix(args.shard_index, args.shard_count)}.json"
    write_json(summary_file, summary)
    if args.shard_count == 1:
        write_json(model_out_root / "model_summary.json", summary)
    return 1 if failures else 0


def run_single_model_sample(
    args: argparse.Namespace,
    sample: SampleSpec,
    sample_index: int,
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    """Resume or execute one sample and commit result metadata after the image."""
    out_dir = args.output_root / args.model / sample.sample_id
    summary_path = out_dir / "result.json"
    worker_final_path = out_dir / "final.png"
    sample_final_path = sample_model_output_path(sample, args.model)

    if not sample_final_path.exists() and worker_final_path.exists():
        mkdir(sample_final_path.parent)
        shutil.copy2(worker_final_path, sample_final_path)

    can_skip = sample_final_path.exists()

    if can_skip and not args.force and summary_path.exists():
        record = json.loads(summary_path.read_text(encoding="utf-8"))
        needs_rewrite = record.get("final_image") != str(sample_final_path)
        if needs_rewrite:
            record["final_image"] = str(sample_final_path)
            if worker_final_path.exists():
                record["worker_final_image"] = str(worker_final_path)
            write_json(summary_path, record)
        return record

    mkdir(out_dir)
    if not (out_dir / "input.png").exists():
        shutil.copy2(sample.input_image, out_dir / "input.png")
    write_json(
        out_dir / "task.json",
        {
            "action_schema": sample.action_schema,
            "operation": "edit_multi_reference",
            "reference_roles": l1_reference_roles(len(ordered_local_edits(sample))),
            "sample_id": sample.sample_id,
            "instruction": sample.joint_prompt or sample.instruction,
            "edit_instruction_summary": sample.instruction,
            "whole_edit": sample.whole_edit,
            "quad_points": [[x, y] for x, y in sample.quad_points],
            "edits": [local_edit_payload(edit) for edit in ordered_local_edits(sample)],
            "action_seed": sample.action_seed,
            "round_index": sample.round_index,
            "input_image": str(sample.input_image),
        },
    )

    generation_started = time.time()
    if args.model == "flux2":
        result = run_flux2_edit(args, sample, out_dir, sample_index, runtime)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    result["duration_sec"] = round(time.time() - generation_started, 3)
    result["gpu_index"] = args.gpu_index

    with contextlib.suppress(FileNotFoundError):
        (out_dir / "error.json").unlink()
    write_json(out_dir / "result.json", result)
    return result


def initialize_model_runtime(args: argparse.Namespace, model_out_root: Path) -> Dict[str, Any]:
    """Load the selected backend once inside its GPU-isolated worker process."""
    runtime: Dict[str, Any] = {"model": args.model}

    if args.model == "flux2":
        import torch
        from diffusers import Flux2Pipeline

        model_path = resolve_hf_snapshot(args.flux2_model)
        dtype_name = str(getattr(args, "flux2_dtype", "bfloat16")).strip().lower()
        dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float16
        pipe = Flux2Pipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipe.enable_model_cpu_offload(device="cuda:0")
        pipe.set_progress_bar_config(disable=True)
        runtime["pipe"] = pipe
        runtime["resolved_model"] = model_path
        return runtime


    raise ValueError(f"Unsupported model: {args.model}")


def finalize_model_runtime(runtime: Dict[str, Any]) -> None:
    """Drop model references and best-effort release the worker CUDA cache."""
    with contextlib.suppress(Exception):
        pipe = runtime.get("pipe")
        if pipe is not None:
            del pipe
    with contextlib.suppress(Exception):
        u = runtime.get("u")
        if u is not None and hasattr(u, "model"):
            del u.model
    with contextlib.suppress(Exception):
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()






def run_flux2_edit(
    args: argparse.Namespace,
    sample: SampleSpec,
    out_dir: Path,
    _sample_index: int,
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the production joint L1 FLUX.2 action exactly once.

    The editable full image is reference zero and padded crops follow planner
    order.  All regions share one compiled prompt and one stable action seed;
    the returned record explicitly audits reference roles and call count.
    """
    import torch
    from PIL import Image

    source_image = Image.open(sample.input_image).convert("RGB")
    image = resize_to_generation_resolution(source_image, args)
    edits = ordered_local_edits(sample)
    scaled_edits = scale_local_edits(edits, source_image.size, image.size)
    scaled_edits = [
        LocalEditSpec(
            id=edit.id,
            target_object=edit.target_object,
            issue=edit.issue,
            edit_prompt=edit.edit_prompt,
            whole_edit=edit.whole_edit,
            quad_points=clamp_quad(edit.quad_points, image.width, image.height),
        )
        for edit in scaled_edits
    ]
    crop_records = local_crops_for_sample(sample, image, out_dir, args, edits=scaled_edits)
    crop_images = [record[0] for record in crop_records]
    crop_paths = [record[1] for record in crop_records]
    rois = [record[2] for record in crop_records]
    pipe = runtime["pipe"]
    model_path = runtime["resolved_model"]
    prompt = sample.joint_prompt.strip() or compose_joint_multicrop_prompt(
        scaled_edits,
        width=image.width,
        height=image.height,
    )
    action_seed = (
        int(sample.action_seed)
        if sample.action_seed is not None
        else stable_l1_action_seed(args.seed, sample.sample_id, edits, sample.round_index)
    )
    generator = torch.Generator(device="cuda:0").manual_seed(action_seed)
    result = pipe(
        image=[image, *crop_images],
        prompt=prompt,
        height=args.resolution,
        width=args.resolution,
        num_inference_steps=args.flux2_steps,
        guidance_scale=args.flux2_guidance_scale,
        generator=generator,
        max_sequence_length=args.flux2_max_sequence_length,
    )
    final_path, worker_final_path = save_final_image_outputs(result.images[0], sample, args.model, out_dir)
    return {
        "action_schema": L1_JOINT_ACTION_SCHEMA,
        "model": args.model,
        "sample_id": sample.sample_id,
        "status": "ok",
        "input_image": str(sample.input_image),
        "reference_images": [str(sample.input_image), *[str(path) for path in crop_paths]],
        "reference_roles": l1_reference_roles(len(scaled_edits)),
        "rois_xyxy": [list(roi) for roi in rois],
        "roi_xyxy": list(rois[0]),
        "edits": [local_edit_payload(edit) for edit in scaled_edits],
        "instruction": sample.instruction,
        "resolved_model": model_path,
        "semantic_prompt": prompt,
        "seed": action_seed,
        "flux_call_count": 1,
        "call_count": 1,
        "final_image": str(final_path),
        "worker_final_image": str(worker_final_path),
    }








def build_parser() -> argparse.ArgumentParser:
    """Build public orchestrator and internal worker subcommands."""
    parser = argparse.ArgumentParser(description="Run FLUX.2 full-image + bbox-crop local editing on MLIE-CoT samples.")
    subparsers = parser.add_subparsers(dest="command")

    def add_shared_arguments(p: argparse.ArgumentParser) -> None:
        """Register settings that must be identical in parent and worker."""
        p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        p.add_argument("--flux2-model", default=DEFAULT_FLUX2_MODEL)
        p.add_argument("--flux2-steps", type=int, default=50)
        p.add_argument("--flux2-guidance-scale", type=float, default=4.0)
        p.add_argument("--flux2-max-sequence-length", type=int, default=512)
        p.add_argument("--flux2-dtype", choices=["bfloat16", "bf16", "float16", "fp16"], default="bfloat16")
        p.add_argument("--l1-roi-pad-ratio", type=float, default=L1_CROP_PAD_RATIO)
        p.add_argument("--l1-min-span", type=int, default=L1_CROP_MIN_SPAN)
        p.add_argument("--l1-max-crops", type=int, default=L1_MAX_CROPS)
        p.add_argument("--seed", type=int, default=1234)
        p.add_argument("--resolution", type=int, default=1024)
        p.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Prepare data and dispatch model workers across GPUs.")
    add_shared_arguments(run_parser)
    run_parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    run_parser.add_argument("--gpu-indices", default="0,1,2,3")
    run_parser.add_argument("--workers-per-model", type=int, default=4)
    run_parser.add_argument(
        "--model-min-free-gib",
        default=",".join(f"{model}={value:g}" for model, value in DEFAULT_MODEL_MIN_FREE_GIB.items()),
        help="Per-model free-memory gate before launching a worker, e.g. flux2=55.",
    )
    run_parser.add_argument(
        "--model-max-parallel",
        default=",".join(f"{model}={value}" for model, value in DEFAULT_MODEL_MAX_PARALLEL.items()),
        help="Per-model maximum concurrent workers. Workers still wait for the memory gate.",
    )
    run_parser.add_argument("--max-memory-retries", type=int, default=8)
    run_parser.add_argument("--memory-wait-sec", type=float, default=20.0)
    run_parser.add_argument("--vcot-python", default=DEFAULT_VCOT_PYTHON)

    worker_parser = subparsers.add_parser("worker", help="Internal worker entrypoint.")
    add_shared_arguments(worker_parser)
    worker_parser.add_argument("--model", required=True, choices=DEFAULT_MODELS)
    worker_parser.add_argument("--gpu-index", type=int, required=True)
    worker_parser.add_argument("--shard-index", type=int, default=0)
    worker_parser.add_argument("--shard-count", type=int, default=1)

    return parser


def main() -> int:
    """Dispatch the requested orchestrator/worker command."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["run"])

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "run":
        return run_orchestrator(args)
    if args.command == "worker":
        return run_worker(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
