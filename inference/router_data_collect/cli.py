"""CPU-only command line entry point for Flux.2 router-data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference.main_method.modules.contracts import L1_CROP_MIN_SPAN, L1_CROP_PAD_RATIO, L1_MAX_CROPS

from .aggregate import aggregate_dataset
from .bootstrap import (
	build_input_states,
	collect_l1_action_plans,
	plan_initial_jobs,
	plan_l1_action_jobs,
	plan_reflection_jobs,
)
from .canonical import write_if_identical_or_absent
from .existing_initial_assets import collect_existing_states, plan_existing_reflections
from .online_states import export_online_states, verify_online_states
from .planner import plan_dataset
from .schema import PROFILE_NAME, SCHEMA_VERSION
from .sharding import (
	merge_completed_datasets,
	merge_completed_shards,
	merge_training_bundles,
	partition_file,
	validate_partition,
)
from .validate import validate_dataset


def example_config() -> dict:
	"""Construct the documented example configuration with resolved local paths."""
	return {
		"schema_version": SCHEMA_VERSION,
		"profile": PROFILE_NAME,
		"dataset_name": "vcot_flux2_router_data",
		"dataset_seed": 20260715,
		"rollouts_per_action": 3,
		"require_flux2_initial_state": True,
		"require_verified_initial_record": True,
		"flux2": {
			"model_id": "black-forest-labs/FLUX.2-dev",
			"model_revision": "replace_with_exact_snapshot_commit",
			"resolution": 1024,
			"steps": 28,
			"guidance_scale": 3.5,
			"max_sequence_length": 512,
			"dtype": "bfloat16",
			"l1_roi_pad_ratio": L1_CROP_PAD_RATIO,
			"l1_min_span": L1_CROP_MIN_SPAN,
			"l1_max_edits": L1_MAX_CROPS,
			"prompt_version": "vcot_v10_flux2_joint_atomic_v2",
		},
		"split": {"train": 0.8, "validation": 0.1, "test": 0.1},
		"utility": {
			"target_gain": 3.0,
			"targeted_fix": 2.0,
			"preservation": 1.0,
			"introduced_new_error": 3.0,
			"artifact_severity": 1.0,
			"success": 0.0,
			"cost_per_call": 0.15,
		},
		"soft_target_temperature": 0.5,
		"acceptable_epsilon": 0.25,
		"training_balance": {"enabled": True, "max_class_weight": 4.0},
	}


def build_parser() -> argparse.ArgumentParser:
	"""Construct the top-level router-data workflow command parser."""
	parser = argparse.ArgumentParser(
		description="Build Flux.2 L1/L2 plus initial-base-bound L3 supervision for the MLIE-CoT router"
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	init = subparsers.add_parser("init-config", help="write a configuration template")
	init.add_argument("--output", type=Path, required=True)

	initial = subparsers.add_parser("plan-initial", help="freeze Flux.2 initial-image jobs from prompt rows")
	initial.add_argument("--config", type=Path, required=True)
	initial.add_argument("--prompts", type=Path, required=True)
	initial.add_argument("--bootstrap-root", type=Path, required=True)
	initial.add_argument("--dry-run", action="store_true")

	reflections = subparsers.add_parser("plan-reflections", help="freeze route-free Qwen reflection jobs")
	reflections.add_argument("--bootstrap-root", type=Path, required=True)
	reflections.add_argument("--reflector-config", type=Path, required=True)
	reflections.add_argument("--dry-run", action="store_true")

	existing_reflections = subparsers.add_parser(
		"plan-existing-reflections",
		help="freeze route-free Qwen jobs over hash-locked existing initial images",
	)
	existing_reflections.add_argument("--assets-root", type=Path, required=True)
	existing_reflections.add_argument("--work-root", type=Path, required=True)
	existing_reflections.add_argument("--reflector-config", type=Path, required=True)
	existing_reflections.add_argument("--shard-id", type=int, default=0)
	existing_reflections.add_argument("--num-shards", type=int, default=1)
	existing_reflections.add_argument("--dry-run", action="store_true")

	collect_existing = subparsers.add_parser(
		"collect-existing-states",
		help="validate existing-image reflections, exclude L0, and emit planner states",
	)
	collect_existing.add_argument("--work-root", type=Path, required=True)
	collect_existing.add_argument("--output", type=Path, required=True)
	collect_existing.add_argument("--config", type=Path, required=True)
	collect_existing.add_argument("--project-root", type=Path, default=Path.cwd())
	collect_existing.add_argument("--dry-run", action="store_true")

	build_states = subparsers.add_parser("build-states", help="verify initial/reflection records and emit planner states")
	build_states.add_argument("--bootstrap-root", type=Path, required=True)
	build_states.add_argument("--output", type=Path, required=True)
	build_states.add_argument("--project-root", type=Path, default=Path.cwd())
	build_states.add_argument("--dry-run", action="store_true")

	l1_plans = subparsers.add_parser("plan-l1-actions", help="freeze post-route Qwen L1 localization jobs")
	l1_plans.add_argument("--states", type=Path, required=True)
	l1_plans.add_argument("--planner-config", type=Path, required=True)
	l1_plans.add_argument("--plan-root", type=Path, required=True)
	l1_plans.add_argument("--project-root", type=Path, default=Path.cwd())
	l1_plans.add_argument("--dry-run", action="store_true")

	collect_plans = subparsers.add_parser("collect-l1-actions", help="collect verified Qwen L1 plans")
	collect_plans.add_argument("--plan-root", type=Path, required=True)
	collect_plans.add_argument("--output", type=Path, required=True)
	collect_plans.add_argument("--dry-run", action="store_true")

	plan = subparsers.add_parser("plan", help="freeze states and create exactly 3 x K action jobs")
	plan.add_argument("--config", type=Path, required=True)
	plan.add_argument("--states", type=Path, required=True)
	plan.add_argument("--output-dir", type=Path, required=True)
	plan.add_argument("--project-root", type=Path, default=Path.cwd())
	plan.add_argument("--l1-plans", type=Path)
	plan.add_argument("--dry-run", action="store_true")

	aggregate = subparsers.add_parser("aggregate", help="compute rollout utilities and split-aware soft targets")
	aggregate.add_argument("--dataset-root", type=Path, required=True)
	aggregate.add_argument("--dry-run", action="store_true")
	aggregate.add_argument("--artifact-workers", type=int, default=1)

	validate = subparsers.add_parser("validate", help="check immutable contracts and optional completion stages")
	validate.add_argument("--dataset-root", type=Path, required=True)
	validate.add_argument("--require-results", action="store_true")
	validate.add_argument("--require-evaluations", action="store_true")
	validate.add_argument("--require-targets", action="store_true")
	validate.add_argument(
		"--structure-only",
		action="store_true",
		help="validate frozen state/job/feature and paired-seed contracts without scanning result artifacts",
	)
	validate.add_argument(
		"--artifact-workers",
		type=int,
		default=1,
		help="bounded threads for strict result/evaluation image and SHA-256 validation",
	)

	partition = subparsers.add_parser("partition-shards", help="freeze stable independent prompt-family shards")
	partition.add_argument("--input", type=Path, required=True)
	partition.add_argument("--output-root", type=Path, required=True)
	partition.add_argument("--num-shards", type=int, default=4)
	partition.add_argument("--dry-run", action="store_true")

	validate_shards = subparsers.add_parser("validate-shards", help="validate independent shard ownership and coverage")
	validate_shards.add_argument("--root", type=Path, required=True)

	merge_shards = subparsers.add_parser("merge-shards", help="merge completed shard targets into one training bundle")
	merge_shards.add_argument("--root", type=Path, required=True)
	merge_shards.add_argument("--output", type=Path, required=True)
	merge_shards.add_argument("--dry-run", action="store_true")

	merge_bundles = subparsers.add_parser("merge-bundles", help="merge disjoint completed training bundles")
	merge_bundles.add_argument("--input", type=Path, action="append", required=True)
	merge_bundles.add_argument("--output", type=Path, required=True)
	merge_bundles.add_argument("--dry-run", action="store_true")

	merge_datasets = subparsers.add_parser(
		"merge-datasets",
		help="merge disjoint completed datasets with symlinked state artifacts",
	)
	merge_datasets.add_argument("--input", type=Path, action="append", required=True)
	merge_datasets.add_argument("--output", type=Path, required=True)
	merge_datasets.add_argument("--dry-run", action="store_true")

	online_states = subparsers.add_parser(
		"export-online-states",
		help="ingest canonical unsatisfied states visited by MLIE-CoT without computing trajectory returns",
	)
	online_states.add_argument(
		"--input",
		type=Path,
		action="append",
		default=[],
		help="MLIE-CoT output root, sample directory, round directory, or error_analysis.json; repeatable",
	)
	online_states.add_argument("--output-dir", type=Path, required=True)
	online_states.add_argument("--project-root", type=Path, default=Path.cwd())
	online_states.add_argument(
		"--verify-only",
		action="store_true",
		help="verify the frozen JSONL, source hashes and family isolation without rewriting",
	)
	return parser


def main(argv: list[str] | None = None) -> None:
	"""Run this module's command-line workflow and return its process status."""
	args = build_parser().parse_args(argv)
	if args.command == "init-config":
		text = json.dumps(example_config(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
		write_if_identical_or_absent(args.output.resolve(), text)
		result = {"status": "created", "path": str(args.output.resolve())}
	elif args.command == "plan-initial":
		result = plan_initial_jobs(
			args.config, args.prompts, args.bootstrap_root, dry_run=args.dry_run
		)
	elif args.command == "plan-reflections":
		result = plan_reflection_jobs(
			args.bootstrap_root, args.reflector_config, dry_run=args.dry_run
		)
	elif args.command == "plan-existing-reflections":
		result = plan_existing_reflections(
			args.assets_root,
			args.work_root,
			args.reflector_config,
			shard_id=args.shard_id,
			num_shards=args.num_shards,
			dry_run=args.dry_run,
		)
	elif args.command == "collect-existing-states":
		result = collect_existing_states(
			args.work_root,
			args.output,
			args.config,
			project_root=args.project_root,
			dry_run=args.dry_run,
		)
	elif args.command == "build-states":
		result = build_input_states(
			args.bootstrap_root,
			args.output,
			project_root=args.project_root,
			dry_run=args.dry_run,
		)
	elif args.command == "plan-l1-actions":
		result = plan_l1_action_jobs(
			args.states,
			args.planner_config,
			args.plan_root,
			project_root=args.project_root,
			dry_run=args.dry_run,
		)
	elif args.command == "collect-l1-actions":
		result = collect_l1_action_plans(args.plan_root, args.output, dry_run=args.dry_run)
	elif args.command == "plan":
		result = plan_dataset(
			args.config,
			args.states,
			args.output_dir,
			args.project_root,
			l1_plans_path=args.l1_plans,
			dry_run=args.dry_run,
		)
	elif args.command == "aggregate":
		validate_dataset(args.dataset_root, scan_artifacts=False)
		result = aggregate_dataset(
			args.dataset_root,
			dry_run=args.dry_run,
			artifact_workers=args.artifact_workers,
		)
	elif args.command == "validate":
		result = validate_dataset(
			args.dataset_root,
			require_results=args.require_results,
			require_evaluations=args.require_evaluations,
			require_targets=args.require_targets,
			scan_artifacts=not args.structure_only,
			artifact_workers=args.artifact_workers,
		)
	elif args.command == "partition-shards":
		result = partition_file(args.input, args.output_root, args.num_shards, dry_run=args.dry_run)
	elif args.command == "validate-shards":
		result = validate_partition(args.root)
	elif args.command == "merge-shards":
		result = merge_completed_shards(args.root, args.output, dry_run=args.dry_run)
	elif args.command == "merge-bundles":
		result = merge_training_bundles(args.input, args.output, dry_run=args.dry_run)
	elif args.command == "merge-datasets":
		result = merge_completed_datasets(args.input, args.output, dry_run=args.dry_run)
	elif args.command == "export-online-states":
		if args.verify_only:
			result = verify_online_states(args.output_dir, inputs=args.input)
		else:
			if not args.input:
				raise ValueError("export-online-states requires at least one --input")
			result = export_online_states(
				args.input,
				args.output_dir,
				project_root=args.project_root,
			)
	else:  # pragma: no cover - argparse enforces this
		raise AssertionError(args.command)
	print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
	main()
