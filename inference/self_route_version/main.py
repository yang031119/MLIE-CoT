#!/usr/bin/env python3
"""MLIE-CoT semantic self-routing baseline with Qwen and FLUX.2 repair actions."""

from __future__ import annotations

from pathlib import Path
import sys
import traceback

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from modules.cli import build_parser  # noqa: E402
from modules.logging_utils import enable_terminal_logging  # noqa: E402
from modules.pipeline import run_pipeline  # noqa: E402


def main() -> int:
	"""Parse production defaults, enable durable logging, and run inference.

	Returns:
	        The pipeline's process exit code.  Unexpected exceptions are deliberately
	        re-raised after their traceback is saved so schedulers still mark the job
	        as failed.
	"""
	parser = build_parser()
	# MLIE-CoT defaults are tuned for four-way Qwen/editor execution on A100-80G.
	# --qwen-quantization 8bit, --qwen-batch-size 4, and 4-way editor
	# parallelism on GPUs 0,1,2,3. 9B/35B defaults keep 1024px Qwen
	# review/compare images uncompressed and disable shrink retry. For 35B
	# runs, override --qwen-model, usually keep 8bit, use --qwen-batch-size 1
	# or 2 if memory is tight, and raise --qwen-min-free-gib to about 30. For
	# 122B runs, override --qwen-model, use nf4/4bit, keep --qwen-batch-size 1,
	# consider --qwen-enable-image-shrink-retry, and launch with conservative
	# job/editor parallelism.
	parser.set_defaults(
		enable_regenerate_level=True,
		initial_generator="flux",
		qwen_route_head="",
		semantic_router="qwen",
		local_edit_model_min_free_gib="flux2=55",
		local_edit_model_max_parallel="flux2=4",
	)
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


if __name__ == "__main__":
	raise SystemExit(main())
