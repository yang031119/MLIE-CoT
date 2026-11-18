"""Command-line interface for Qwen self-routing with Flux.2 execution."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

from .config import (
	DEFAULT_BAGEL_FINETUNED_PATH,
	DEFAULT_BAGEL_MODEL_PATH,
	DEFAULT_EDITOR_GPU_INDICES,
	DEFAULT_EDITOR_PARALLEL_WORKERS,
	DEFAULT_FLUX_MODEL,
	DEFAULT_FLUX_MODEL_REVISION,
	DEFAULT_OUTPUT_ROOT,
	DEFAULT_PROMPTS_PATH,
	DEFAULT_QWEN_ROUTE_HEAD,
	DEFAULT_RETRY_MIN_SIDE,
	DEFAULT_RETRY_SCALE,
	DEFAULT_SD35_TURBO_MODEL,
	DEFAULT_UNICOT_ROOT,
	MODULE_DIR,
	QWEN_INPUT_MAX_SIDE,
	QWEN_MULTI_IMAGE_INPUT_MAX_SIDE,
	QWEN_REVIEW_INPUT_MAX_SIDE,
)
from .vcot_branch import (
	DEFAULT_BAGEL_PYTHON,
	DEFAULT_FLUX_PYTHON,
	DEFAULT_VCOT_PYTHON,
	DEFAULT_QWEN9B_MODEL_ID,
	DEFAULT_QWEN9B_REVISION,
	DEFAULT_QWEN_MODEL,
)
from .contracts import (
	DEFAULT_MAX_REPAIR_ACTIONS_V10,
	L1_CROP_MIN_SPAN,
	L1_CROP_PAD_RATIO,
	L1_MAX_CROPS,
	QWEN_ANALYZER_L1_MAX_NEW_TOKENS_V10,
	QWEN_ANALYZER_L1_QUANTIZATION_V10,
)

from .logging_utils import enable_terminal_logging
from .pipeline import run_pipeline

def build_parser() -> argparse.ArgumentParser:
	"""Construct the complete MLIE-CoT argument parser without parsing ``sys.argv``.

	Keeping construction side-effect free lets ``main.py``, ``single_infer.py``,
	and the route-ablation entry point apply narrow default overrides while
	sharing every option and help string.
	"""
	parser = argparse.ArgumentParser(
		description=(
			"MLIE-CoT self-route: Qwen diagnosis and semantic L1/L2/L3 routing, "
			"with Flux.2 image generation and repair"
		)
	)
	parser.add_argument("--data-path", type=Path, default=DEFAULT_PROMPTS_PATH)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / time.strftime("run_%Y%m%d_%H%M%S"))
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--start-index", type=int, default=0)
	parser.add_argument("--limit", type=int)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--resolution", type=int, default=1024)
	parser.add_argument("--initial-generator", default="flux", choices=["bagel", "flux", "sd35"])
	parser.add_argument(
		"--language-backend",
		default="qwen",
		choices=["qwen"],
		help="MLIE-CoT uses Qwen for diagnosis/planning so the same frozen backbone can encode router features.",
	)
	parser.add_argument("--skip-prompt-reasoning", action="store_true")
	parser.add_argument("--stage-mode", default="full", choices=["full", "qwen", "edit"])
	parser.add_argument(
		"--max-global-rounds",
		type=int,
		default=DEFAULT_MAX_REPAIR_ACTIONS_V10,
		help=(
			"Maximum generated repair actions per sample (default: 2). Diagnosis, JSON retries, "
			"bbox review, and candidate review do not consume this budget."
		),
	)
	parser.add_argument("--max-local-attempts", type=int, default=1, help=argparse.SUPPRESS)
	parser.add_argument("--qwen-box-review-max-rounds", type=int, default=0, help=argparse.SUPPRESS)
	parser.add_argument(
		"--enable-regenerate-level",
		dest="enable_regenerate_level",
		action="store_true",
		default=True,
		help="Allow the route head to select L3 prompt-only regeneration (enabled by default in MLIE-CoT).",
	)
	parser.add_argument(
		"--disable-regenerate-level",
		dest="enable_regenerate_level",
		action="store_false",
		help=argparse.SUPPRESS,
	)
	parser.add_argument(
		"--fixed-repair-level",
		type=int,
		choices=[1, 2, 3],
		help="Ablation override: for every non-satisfied global state, force L1, L2, or L3.",
	)
	parser.add_argument("--retry-min-side", type=int, default=DEFAULT_RETRY_MIN_SIDE)
	parser.add_argument("--retry-scale", type=float, default=DEFAULT_RETRY_SCALE)
	parser.add_argument("--local-edit-gpu-indices", default=DEFAULT_EDITOR_GPU_INDICES)
	parser.add_argument("--local-edit-workers-per-model", type=int, default=DEFAULT_EDITOR_PARALLEL_WORKERS)
	parser.add_argument("--local-edit-min-free-gib", type=float, default=24.0)
	parser.add_argument(
		"--local-edit-model-min-free-gib",
		default="bagel=55",
	)
	parser.add_argument(
		"--local-edit-model-max-parallel",
		default="bagel=4",
	)
	parser.add_argument("--local-edit-max-memory-retries", type=int, default=50)
	parser.add_argument("--local-edit-outer-max-memory-retries", type=int, default=0)
	parser.add_argument("--local-edit-memory-wait-sec", type=float, default=20.0)
	parser.add_argument("--edit-service-min-free-gib", type=float, default=24.0)
	parser.add_argument("--edit-service-memory-wait-sec", type=float, default=20.0)
	parser.add_argument("--edit-service-max-memory-retries", type=int, default=0)
	parser.add_argument("--edit-service-max-wait-sec", type=float, default=0.0)
	parser.add_argument("--vcot-python", default=DEFAULT_VCOT_PYTHON)

	# Default Qwen/runtime preset targets Qwen3.5-9B on four A100-80G GPUs:
	# keep BAGEL/Flux/local editors at 4-way parallelism on GPUs 0,1,2,3,
	# use 8-bit Qwen, batch Qwen prompts in chunks of 12, and keep the
	# default 1024px generated images uncompressed for Qwen review/compare.
	# For Qwen3.5-35B: override --qwen-model to DEFAULT_QWEN35B_MODEL,
	# keep --qwen-quantization 8bit, reduce --qwen-batch-size to 1 or 2 if
	# memory is tight, raise --qwen-min-free-gib to about 30, and keep image
	# shrink retry disabled unless the run is actually hitting Qwen OOM.
	# For Qwen3.5-122B: override --qwen-model to DEFAULT_QWEN122B_MODEL,
	# use --qwen-quantization nf4/4bit, keep --qwen-batch-size 1, consider
	# --qwen-enable-image-shrink-retry, and use a conservative single-job or
	# Compatibility resource knobs retained for older launch scripts.
	parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
	parser.add_argument(
		"--qwen-model-id",
		default=DEFAULT_QWEN9B_MODEL_ID,
		help="Logical Qwen repository identity used by the analyzer/L1 distribution contract.",
	)
	parser.add_argument(
		"--qwen-model-revision",
		default=DEFAULT_QWEN9B_REVISION,
		help="Immutable Qwen commit shared by collection, feature extraction, and online inference.",
	)
	parser.add_argument(
		"--qwen-route-head",
		type=Path,
		default=DEFAULT_QWEN_ROUTE_HEAD,
		help="Compatible three-class L1-L3 MLP checkpoint; native Qwen owns L0.",
	)
	parser.add_argument(
		"--allow-cross-executor-route-head-transfer",
		action="store_true",
		help=(
			"Explicit ablation-only override: apply a route head whose candidate-quality labels were "
			"trained with another repair executor to the selected initial-generator family. This keeps "
			"the frozen Qwen feature/head contract strict but intentionally does not claim action-policy "
			"compatibility. Disabled by default."
		),
	)
	parser.add_argument("--qwen-dtype", default="bfloat16")
	parser.add_argument("--qwen-quantization", default=QWEN_ANALYZER_L1_QUANTIZATION_V10, choices=["none", "8bit", "int8", "4bit", "int4", "nf4"])
	parser.add_argument("--qwen-max-memory-per-gpu", default="80GiB")
	parser.add_argument("--qwen-min-free-gib", type=float, default=18.0)
	parser.add_argument("--qwen-memory-wait-sec", type=float, default=30.0)
	parser.add_argument("--qwen-batch-size", type=int, default=12)
	parser.add_argument("--qwen-input-max-side", type=int, default=QWEN_INPUT_MAX_SIDE)
	parser.add_argument("--qwen-prompt-reasoning-max-new-tokens", type=int, default=1024)
	parser.add_argument("--qwen-max-new-tokens", type=int, default=QWEN_ANALYZER_L1_MAX_NEW_TOKENS_V10)
	parser.add_argument("--qwen-compare-max-new-tokens", type=int, default=1024)
	parser.add_argument("--qwen-route-max-new-tokens", type=int, default=512)
	parser.add_argument(
		"--semantic-router",
		choices=["bagel", "qwen", "head"],
		default="bagel",
		help=(
			"Semantic L1/L2/L3 controller. qwen and bagel generate a route decision; "
			"head applies --qwen-route-head to frozen Qwen diagnosis features."
		),
	)
	parser.add_argument(
		"--qwen-semantic-route-cache",
		type=Path,
		help=(
			"Optional predictions.jsonl produced by research_qwen_reroute_distribution.py. "
			"A cached route is reused only when sample, round, prompt, and image bytes match; "
			"exact diagnosis equality is recorded for auditing."
		),
	)
	parser.add_argument("--qwen-review-input-max-side", "--micro-review-input-max-side", dest="qwen_review_input_max_side", type=int, default=QWEN_REVIEW_INPUT_MAX_SIDE)
	parser.add_argument("--qwen-compare-input-max-side", type=int, default=QWEN_MULTI_IMAGE_INPUT_MAX_SIDE)
	parser.add_argument("--qwen-keep-job-files", action="store_true")
	parser.add_argument(
		"--qwen-worker-gpu-groups",
		default="",
		help=(
			"Semicolon-separated CUDA_VISIBLE_DEVICES groups for parallel Qwen workers, "
			"for example '0;1;2;3' for one worker per GPU."
		),
	)
	parser.add_argument(
		"--qwen-worker-fallback-gpu-groups",
		default="",
		help=(
			"Fallback semicolon-separated CUDA_VISIBLE_DEVICES groups for Qwen workers, "
			"for example '0,1;2,3'. Used for remaining items if the primary groups fail."
		),
	)
	parser.add_argument(
		"--qwen-disable-image-shrink-retry",
		dest="qwen_disable_image_shrink_retry",
		action="store_true",
		default=True,
		help="Disable OOM retry downscaling for Qwen images. Default for 9B/35B keeps 1024px review/compare images uncompressed.",
	)
	parser.add_argument(
		"--qwen-enable-image-shrink-retry",
		dest="qwen_disable_image_shrink_retry",
		action="store_false",
		help="Allow Qwen OOM retries to downscale images; useful for 122B or tight-memory runs.",
	)
	parser.add_argument("--qwen-allow-cpu-offload", action="store_true")
	parser.add_argument("--qwen-inprocess", action="store_true", help="Run Qwen in the main process so consecutive Qwen stages can reuse the loaded model.")
	parser.add_argument("--keep-models-loaded", action="store_true")

	parser.add_argument("--bagel-python", default=DEFAULT_BAGEL_PYTHON)
	parser.add_argument("--bagel-service-script", default=str(MODULE_DIR / "bagel_service.py"))
	parser.add_argument("--unicot-root", default=DEFAULT_UNICOT_ROOT)
	parser.add_argument("--bagel-model-path", default=DEFAULT_BAGEL_MODEL_PATH)
	parser.add_argument("--bagel-fine-tuned-path", default=DEFAULT_BAGEL_FINETUNED_PATH)
	parser.add_argument("--bagel-model-id", default="", help="Stable BAGEL base-model identity for L3 policy locking.")
	parser.add_argument("--bagel-model-revision", default="", help="Immutable BAGEL base-model commit required by a routed BAGEL run.")
	parser.add_argument("--bagel-fine-tuned-model-id", default="", help="Stable BAGEL fine-tuned-model identity for L3 policy locking.")
	parser.add_argument("--bagel-fine-tuned-model-revision", default="", help="Immutable BAGEL fine-tuned commit required by a routed BAGEL run.")
	parser.add_argument("--bagel-gpu-index", type=int, default=1)
	parser.add_argument("--bagel-gpu-indices", default=DEFAULT_EDITOR_GPU_INDICES)
	parser.add_argument("--bagel-parallel-workers", type=int, default=DEFAULT_EDITOR_PARALLEL_WORKERS)
	parser.add_argument("--max-latent-size", type=int, default=64)
	parser.add_argument("--bagel-steps", type=int, default=50)
	parser.add_argument("--bagel-cfg-scale", type=float, default=4.0)
	parser.add_argument("--bagel-cfg-text-scale", type=float, default=4.0)
	parser.add_argument("--bagel-cfg-img-scale", type=float, default=2.0)
	parser.add_argument("--bagel-route-max-new-tokens", type=int, default=512)
	parser.add_argument(
		"--initial-source-run",
		type=Path,
		help="Reuse exact initial.png files from a run with a matching immutable initial-generator policy.",
	)
	parser.add_argument(
		"--initial-source-image-name",
		default="initial.png",
		help="Per-sample image filename to reuse from --initial-source-run.",
	)
	parser.add_argument(
		"--initial-source-policy-mode",
		choices=["strict", "legacy-attested"],
		default="strict",
		help=(
			"Strict requires an exact immutable source policy. legacy-attested is only for a historical "
			"run that records the same initial-generator family but predates policy contracts."
		),
	)

	parser.add_argument("--flux-python", default=DEFAULT_FLUX_PYTHON)
	parser.add_argument("--flux-service-script", default=str(MODULE_DIR / "flux_service.py"))
	parser.add_argument("--flux-model", default=DEFAULT_FLUX_MODEL)
	parser.add_argument(
		"--flux-model-id",
		default="",
		help="Stable Hugging Face repo id when --flux-model is a custom local snapshots/<commit> path.",
	)
	parser.add_argument(
		"--flux-model-revision",
		default=DEFAULT_FLUX_MODEL_REVISION,
		help="Immutable 40-64 hex Flux.2 snapshot; required for formal route-head deployment unless --flux-model is already that exact snapshot directory.",
	)
	parser.add_argument("--flux-dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16"])
	parser.add_argument("--flux-gpu-index", type=int, default=2)
	parser.add_argument("--flux-gpu-indices", default=DEFAULT_EDITOR_GPU_INDICES)
	parser.add_argument("--flux-parallel-workers", type=int, default=DEFAULT_EDITOR_PARALLEL_WORKERS)
	parser.add_argument(
		"--flux-batch-size",
		type=int,
		default=1,
		help="Per-GPU FLUX.2 text-generation batch. Batch 2 is intended for A100-80GB; OOM falls back to scalar generation.",
	)
	parser.add_argument("--flux-steps", type=int, default=28)
	parser.add_argument("--flux-guidance-scale", type=float, default=3.5)
	parser.add_argument("--flux-max-sequence-length", type=int, default=512)
	parser.add_argument("--flux-l1-roi-pad-ratio", type=float, default=L1_CROP_PAD_RATIO)
	parser.add_argument("--flux-l1-min-span", type=int, default=L1_CROP_MIN_SPAN)
	parser.add_argument("--flux-l1-max-crops", type=int, default=L1_MAX_CROPS)
	parser.add_argument("--flux-prompt-version", default="vcot_v10_flux2_joint_atomic_v2", help=argparse.SUPPRESS)
	parser.add_argument("--flux-true-cfg-scale", type=float, default=1.0)

	parser.add_argument("--sd35-service-script", default=str(MODULE_DIR / "sd35_service.py"))
	parser.add_argument("--sd35-model", default=DEFAULT_SD35_TURBO_MODEL)
	parser.add_argument("--sd35-model-id", default="", help="Stable SD3.5 model identity for L3 policy locking.")
	parser.add_argument("--sd35-model-revision", default="", help="Immutable SD3.5 commit required by a routed SD3.5 run.")
	parser.add_argument("--sd35-dtype", default="bfloat16")
	parser.add_argument("--sd35-gpu-index", type=int, default=0)
	parser.add_argument("--sd35-gpu-indices", default=DEFAULT_EDITOR_GPU_INDICES)
	parser.add_argument("--sd35-parallel-workers", type=int, default=DEFAULT_EDITOR_PARALLEL_WORKERS)
	parser.add_argument("--sd35-steps", type=int, default=4)
	parser.add_argument("--sd35-guidance-scale", type=float, default=0.0)
	parser.add_argument("--sd35-max-sequence-length", type=int, default=512)

	return parser


def main() -> int:
	"""Standalone module entry point with durable terminal and exception logs."""
	parser = build_parser()
	args = parser.parse_args()
	log_path = enable_terminal_logging(Path(args.output_dir))
	print(f"terminal_log={log_path}", flush=True)
	try:
		return run_pipeline(args)
	except Exception:
		output_dir = Path(args.output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		(output_dir / "runner_exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
		raise
