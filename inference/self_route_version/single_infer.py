#!/usr/bin/env python3
"""Convenience entry point for MLIE-CoT self-routing with BAGEL initialization."""

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
	"""Run the shared pipeline with the BAGEL-initialization convenience defaults."""
	parser = build_parser()
	parser.set_defaults(
		enable_regenerate_level=True,
		initial_generator="bagel",
		qwen_route_head="",
		single_model="off",
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
