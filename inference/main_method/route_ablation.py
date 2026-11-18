#!/usr/bin/env python3
"""Run a controlled learned-route versus fixed-level MLIE-CoT ablation.

The ablation changes only the L1/L2/L3 selection policy.  Diagnosis prompts,
action executors, retain review, action budgets, and terminal diagnosis remain
the same as normal MLIE-CoT inference.  Before any GPU model is loaded, this entry
point locks the Qwen, FLUX.2, and initial/L3 model identities and writes a
manifest describing the exact comparison protocol.

When ``--initial-source-run`` is supplied, initial images are materialized
from a previously validated run and inference resumes after initial
generation.  A post-run audit then checks that the requested strategy and
same-initial guarantees were actually respected.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import traceback


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


from inference.main_method.modules.cli import build_parser  # noqa: E402
from inference.main_method.modules.logging_utils import enable_terminal_logging  # noqa: E402
from inference.main_method.modules.pipeline import run_pipeline  # noqa: E402
from inference.router_ablation.audit import audit_ablation_run  # noqa: E402
from inference.router_ablation.initials import seed_v10_initials  # noqa: E402
from inference.router_ablation.protocol import (  # noqa: E402
	STRATEGY_ALIASES,
	apply_ablation_strategy,
	build_ablation_manifest,
	lock_ablation_flux2_policy,
	lock_ablation_l3_policy,
	lock_ablation_qwen_model,
	write_ablation_manifest,
)


def build_ablation_parser():
	"""Extend the canonical MLIE-CoT parser with ablation-specific safeguards."""
	parser = build_parser()
	parser.description = (
		"MLIE-CoT route ablation: compare the native learned route with fixed "
		"L1/L2/L3 while preserving MLIE-CoT diagnosis, action, retain-review, and "
		"terminal-diagnosis semantics."
	)
	parser.add_argument(
		"--ablation-strategy",
		required=True,
		choices=sorted(STRATEGY_ALIASES),
		help="One run executes exactly one strategy: route, fixed-l1, fixed-l2, or fixed-l3.",
	)
	parser.add_argument(
		"--initial-source-run",
		type=Path,
		help=(
			"Optional MLIE-CoT run whose initial.png files are reused exactly. The source must "
			"use a compatible state schema and an immutable initial/L3 policy."
		),
	)
	parser.add_argument(
		"--initial-copy-mode",
		choices=["hardlink", "copy"],
		default="hardlink",
		help="How to materialize source initial images in this run (default: hardlink).",
	)
	parser.add_argument(
		"--allow-cross-policy-initial-assets",
		action="store_true",
		help=(
			"Import exact initial images from a MLIE-CoT source run even when its historical "
			"FLUX/L3 policy labels differ. Source and target digests plus every image SHA-256 "
			"are recorded; the target repair/L3 runtime remains bound to the route checkpoint."
		),
	)
	parser.set_defaults(
		data_path=PROJECT_ROOT / "data" / "prompts.txt",
		output_dir=None,
		enable_regenerate_level=True,
	)
	return parser


def _default_output_dir(strategy: str) -> Path:
	"""Build a timestamped result path whose suffix identifies the strategy."""
	stamp = time.strftime("%Y%m%d_%H%M%S")
	return PROJECT_ROOT / "results" / "route_ablation" / f"run_{stamp}_{strategy.replace('-', '_')}"


def main() -> int:
	"""Lock the experiment contract, execute it, and reject an invalid run."""
	parser = build_ablation_parser()
	args = parser.parse_args()
	try:
		strategy = apply_ablation_strategy(args)
	except ValueError as exc:
		parser.error(str(exc))
	try:
		lock_ablation_qwen_model(args)
	except (OSError, ValueError) as exc:
		parser.error(f"cannot lock the shared MLIE-CoT Qwen analyzer/planner: {exc}")
	try:
		flux2_policy, flux2_digest = lock_ablation_flux2_policy(args)
	except (OSError, ValueError) as exc:
		parser.error(f"cannot lock the shared MLIE-CoT FLUX.2 L1/L2 policy: {exc}")
	try:
		l3_policy, l3_digest = lock_ablation_l3_policy(args, flux2_policy)
	except (OSError, ValueError) as exc:
		parser.error(f"cannot lock the shared MLIE-CoT initial/L3 policy: {exc}")
	if args.output_dir is None:
		args.output_dir = _default_output_dir(strategy)
	args.output_dir = Path(args.output_dir).resolve()
	args.data_path = Path(args.data_path).resolve()

	seed_report = None
	if args.initial_source_run is not None:
		try:
			seed_report = seed_v10_initials(
				source_run=Path(args.initial_source_run).resolve(),
				target_run=args.output_dir,
				data_path=args.data_path,
				start_index=int(args.start_index),
				limit=args.limit,
				initial_generator=args.initial_generator,
				copy_mode=args.initial_copy_mode,
				flux2_policy_digest=flux2_digest,
				requested_l3_policy_digest=l3_digest,
				allow_cross_policy_initial_assets=bool(args.allow_cross_policy_initial_assets),
			)
		except (OSError, ValueError) as exc:
			parser.error(f"cannot seed same-initial ablation: {exc}")
		# The MLIE-CoT loader must restore the freshly seeded state instead of
		# scheduling a second initial generation.
		args.resume = True

	manifest = build_ablation_manifest(
		args,
		strategy,
		seed_report,
		flux2_policy,
		flux2_digest,
		l3_policy,
		l3_digest,
	)
	try:
		write_ablation_manifest(args.output_dir, manifest, resume=bool(args.resume))
	except ValueError as exc:
		parser.error(str(exc))

	log_path = enable_terminal_logging(args.output_dir)
	print(f"terminal_log={log_path}", flush=True)
	print(
		f"route_ablation strategy={strategy} fixed_level={args.fixed_repair_level} "
		f"same_initial_source={args.initial_source_run or 'none'}",
		flush=True,
	)
	try:
		code = run_pipeline(args)
		audit = audit_ablation_run(
			output_dir=args.output_dir,
			strategy=strategy,
			data_path=args.data_path,
			start_index=int(args.start_index),
			limit=args.limit,
			seed_report=seed_report,
		)
		if not audit["valid"]:
			raise RuntimeError(
				"MLIE-CoT route-ablation integrity audit failed; see "
				f"{args.output_dir / 'route_ablation_audit.json'}"
			)
		return code
	except Exception:
		args.output_dir.mkdir(parents=True, exist_ok=True)
		(args.output_dir / "runner_exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
		raise


if __name__ == "__main__":
	raise SystemExit(main())
