from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.reward_utils import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    make_clip_runner,
    make_hps_runner,
    make_pickscore_runner,
    parse_dtype,
    release_runner,
)


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "vcot"
DEFAULT_CACHE_JSONL = DEFAULT_RESULTS_DIR / "image_reward_scores.jsonl"
DEFAULT_SUMMARY_JSON = DEFAULT_RESULTS_DIR / "image_reward_summary.json"
LEADING_INDEX_RE = re.compile(r"^(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score every image in a simple result directory with CLIP/PickScore/HPS")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--cache-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=999999)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--final-only", action="store_true", help="Only score final.png in each sample directory")
    parser.add_argument("--image-name", default=None, help="Only score images with this exact file name in each sample directory")
    args = parser.parse_args()
    args.cache_jsonl = args.cache_jsonl or str(Path(args.results_dir) / "image_reward_scores.jsonl")
    args.summary_json = args.summary_json or str(Path(args.results_dir) / "image_reward_summary.json")
    return args


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_existing_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["image_path"])] = row
    return rows


def extract_sample_index(sample_dir: Path) -> int | None:
    if sample_dir.name.isdigit():
        return int(sample_dir.name)
    match = LEADING_INDEX_RE.match(sample_dir.name)
    if match is None:
        return None
    return int(match.group(1))


def list_sample_dirs(results_dir: Path, start_index: int, end_index: int, sample_limit: int | None) -> List[Path]:
    sample_dirs: List[Path] = []
    for path in sorted(results_dir.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "prompt.txt").exists():
            continue
        index = extract_sample_index(path)
        if index is None:
            continue
        if start_index <= index <= end_index:
            sample_dirs.append(path)
    if sample_limit is not None:
        sample_dirs = sample_dirs[:sample_limit]
    return sample_dirs


def collect_records(sample_dirs: List[Path], final_only: bool, image_name: str | None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    summary = {
        "sample_count": len(sample_dirs),
        "samples_with_prompt": 0,
        "samples_missing_prompt": 0,
        "image_count": 0,
        "samples": [],
    }

    for sample_dir in sample_dirs:
        prompt_path = sample_dir / "prompt.txt"
        sample_info: Dict[str, Any] = {
            "sample": sample_dir.name,
            "sample_dir": str(sample_dir),
        }
        if not prompt_path.exists():
            sample_info["error"] = "missing_prompt"
            summary["samples_missing_prompt"] += 1
            summary["samples"].append(sample_info)
            continue

        prompt = read_text(prompt_path)
        summary["samples_with_prompt"] += 1
        if final_only and image_name is not None:
            raise ValueError("--final-only and --image-name are mutually exclusive")
        if image_name is not None:
            png_paths = [sample_dir / image_name] if (sample_dir / image_name).exists() else []
        elif final_only:
            png_paths = [sample_dir / "final.png"] if (sample_dir / "final.png").exists() else []
        else:
            png_paths = sorted(path for path in sample_dir.glob("*.png"))
        sample_info["image_count"] = len(png_paths)
        sample_info["image_paths"] = [display_path(path) for path in png_paths]
        sample_info["image_names"] = [path.name for path in png_paths]
        summary["samples"].append(sample_info)

        for image_path in png_paths:
            records.append(
                {
                    "sample": sample_dir.name,
                    "sample_dir": str(sample_dir),
                    "prompt": prompt,
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "clip_score": None,
                    "pickscore_score": None,
                    "hps_score": None,
                }
            )

    summary["image_count"] = len(records)
    return records, summary


def unwrap_feature_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
        candidate = getattr(value, attr, None)
        if torch.is_tensor(candidate):
            if candidate.ndim > 2:
                return candidate[:, 0]
            return candidate
    if hasattr(value, "to_tuple"):
        for item in value.to_tuple():
            if torch.is_tensor(item):
                if item.ndim > 2:
                    return item[:, 0]
                return item
    raise TypeError(f"Unsupported feature type: {type(value)!r}")


@torch.inference_mode()
def score_record(runner: Any, image_path: Path, prompt: str) -> float:
    image = Image.open(image_path).convert("RGB")
    image_tensor = runner.preprocess(image)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(runner.device, dtype=runner.dtype)
    image_features = unwrap_feature_tensor(runner.model.get_image_features(image_tensor))
    text_features = unwrap_feature_tensor(runner.model.get_text_features(prompt))
    loss = runner.model.compute_loss(image_features, text_features)
    return runner.score_offset - float(loss.detach().float().item())


def score_missing_records(
    records: List[Dict[str, Any]],
    existing_cache: Dict[str, Dict[str, Any]],
    device: str,
    dtype_name: str,
    cache_dir: str,
    limit_images: int | None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    dtype = parse_dtype(dtype_name)
    missing_paths = [row["image_path"] for row in records if row["image_path"] not in existing_cache]
    if limit_images is not None:
        missing_paths = missing_paths[:limit_images]

    work_keys = set(missing_paths)
    working_cache = dict(existing_cache)
    counters = {
        "existing_records": len(existing_cache),
        "new_records": 0,
        "scored_clip": 0,
        "scored_pickscore": 0,
        "scored_hps": 0,
        "missing_files": 0,
        "failed_clip": 0,
        "failed_pickscore": 0,
        "failed_hps": 0,
    }

    for row in records:
        key = row["image_path"]
        if key not in work_keys:
            continue
        if not Path(key).exists():
            payload = dict(row)
            payload.update({"clip_score": None, "pickscore_score": None, "hps_score": None, "error": "missing_file"})
            working_cache[key] = payload
            counters["missing_files"] += 1
        else:
            working_cache[key] = dict(row)

    metric_specs = [
        ("clip_score", make_clip_runner, "scored_clip", "failed_clip"),
        ("pickscore_score", make_pickscore_runner, "scored_pickscore", "failed_pickscore"),
        ("hps_score", make_hps_runner, "scored_hps", "failed_hps"),
    ]
    for field, factory, counter_key, fail_key in metric_specs:
        runner = factory(device=device, dtype=dtype, cache_dir=cache_dir)
        try:
            for image_path in missing_paths:
                record = working_cache.get(image_path)
                if record is None or record.get("error") == "missing_file":
                    continue
                if record.get(field) is not None:
                    continue
                try:
                    record[field] = score_record(runner, Path(image_path), str(record["prompt"]))
                    counters[counter_key] += 1
                except Exception as exc:
                    record.setdefault("errors", {})[field] = str(exc)
                    counters[fail_key] += 1
        finally:
            release_runner(runner)

    counters["new_records"] = len([path for path in missing_paths if path in working_cache])
    return working_cache, counters


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    cache_jsonl = Path(args.cache_jsonl)
    summary_json = Path(args.summary_json)

    sample_dirs = list_sample_dirs(results_dir, args.start_index, args.end_index, args.sample_limit)
    records, summary = collect_records(sample_dirs, final_only=args.final_only, image_name=args.image_name)
    existing_cache = load_existing_cache(cache_jsonl)
    updated_cache, counters = score_missing_records(
        records,
        existing_cache,
        device=args.device,
        dtype_name=args.dtype,
        cache_dir=args.cache_dir,
        limit_images=args.limit_images,
    )

    output_rows = [updated_cache[key] for key in sorted(updated_cache)]
    write_jsonl(cache_jsonl, output_rows)
    summary["reward_cache"] = {
        "cached_record_count": len(output_rows),
        "cache_jsonl": str(cache_jsonl),
        "final_only": args.final_only,
        "image_name": args.image_name,
        **counters,
    }
    write_json(summary_json, summary)
    print(json.dumps({"status": "ok", "summary_json": str(summary_json), "cache_jsonl": str(cache_jsonl), **summary["reward_cache"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
