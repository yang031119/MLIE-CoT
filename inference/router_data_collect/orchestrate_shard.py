"""Run one independently owned router-data shard as materialized stages.

Each GPU process receives a different shard root.  Stages run in subprocesses so
Flux and Qwen never need to coexist in memory.  Conditional Flux editing remains
scalar unless a verified nested-reference batch adapter is available; this is an
intentional safety property preventing references from different samples from
being concatenated and shared across a prompt batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

from .canonical import read_json, write_json_atomic
from .schema import DataValidationError
from .sharding import validate_partition


STAGES = (
	"plan_initial",
	"initial_images",
	"plan_diagnoses",
	"diagnoses",
	"build_states",
	"plan_l1",
	"l1_plans",
	"collect_l1",
	"plan_actions",
	"actions_flux2",
	"actions_l3_initial_base",
	"evaluations",
	"aggregate",
	"validate",
)


def build_parser() -> argparse.ArgumentParser:
	"""Construct the CLI parser for resumable per-shard orchestration."""
	parser = argparse.ArgumentParser(description="Run one stable V10 router-data shard stage by stage")
	parser.add_argument("--root", type=Path, required=True, help="Root containing shard_manifest.json and shards/")
	parser.add_argument("--shard-id", type=int, required=True)
	parser.add_argument("--config", type=Path, required=True)
	parser.add_argument("--reflector-config", type=Path, required=True)
	parser.add_argument("--evaluator-model", required=True)
	parser.add_argument("--evaluator-model-revision", required=True)
	parser.add_argument("--project-root", type=Path, default=Path.cwd())
	parser.add_argument("--device", default="cuda:0")
	parser.add_argument("--flux-batch-size", type=int, default=2)
	parser.add_argument("--qwen-batch-size", type=int, default=8)
	parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
	parser.add_argument("--through-stage", choices=STAGES, default=STAGES[-1])
	parser.add_argument("--force", action="store_true")
	parser.add_argument("--print-only", action="store_true")
	return parser


def _commands(args: argparse.Namespace, shard_root: Path) -> list[tuple[str, list[str]]]:
	"""Build the ordered stage commands for one explicitly owned shard."""
	py = sys.executable
	module = [py, "-m"]
	bootstrap = shard_root / "bootstrap"
	states = shard_root / "states.jsonl"
	l1_root = shard_root / "l1_plans"
	l1_plans = shard_root / "l1_action_plans.jsonl"
	dataset = shard_root / "dataset"
	prompts = shard_root / "input_prompts.jsonl"
	return [
		("plan_initial", module + ["inference.router_data_collect.cli", "plan-initial", "--config", str(args.config), "--prompts", str(prompts), "--bootstrap-root", str(bootstrap)]),
		("initial_images", module + ["inference.router_data_collect.execute_initial_flux2", "--bootstrap-root", str(bootstrap), "--device", args.device, "--batch-size", str(args.flux_batch_size)]),
		("plan_diagnoses", module + ["inference.router_data_collect.cli", "plan-reflections", "--bootstrap-root", str(bootstrap), "--reflector-config", str(args.reflector_config)]),
		("diagnoses", module + ["inference.router_data_collect.execute_reflections_qwen", "--bootstrap-root", str(bootstrap), "--batch-size", str(args.qwen_batch_size)]),
		("build_states", module + ["inference.router_data_collect.cli", "build-states", "--bootstrap-root", str(bootstrap), "--output", str(states), "--project-root", str(args.project_root)]),
		("plan_l1", module + ["inference.router_data_collect.cli", "plan-l1-actions", "--states", str(states), "--planner-config", str(args.reflector_config), "--plan-root", str(l1_root), "--project-root", str(args.project_root)]),
		("l1_plans", module + ["inference.router_data_collect.execute_l1_plans_qwen", "--plan-root", str(l1_root), "--batch-size", str(args.qwen_batch_size)]),
		("collect_l1", module + ["inference.router_data_collect.cli", "collect-l1-actions", "--plan-root", str(l1_root), "--output", str(l1_plans)]),
		("plan_actions", module + ["inference.router_data_collect.cli", "plan", "--config", str(args.config), "--states", str(states), "--l1-plans", str(l1_plans), "--output-dir", str(dataset), "--project-root", str(args.project_root)]),
		("actions_flux2", module + ["inference.router_data_collect.execute_flux2", "--dataset-root", str(dataset), "--device", args.device, "--batch-size", str(args.flux_batch_size)]),
		("actions_l3_initial_base", module + ["inference.router_data_collect.execute_l3_initial_base", "--dataset-root", str(dataset), "--gpu-index", "0"]),
		("evaluations", module + ["inference.router_data_collect.evaluate_qwen", "--dataset-root", str(dataset), "--model", args.evaluator_model, "--model-revision", args.evaluator_model_revision, "--batch-size", str(args.qwen_batch_size)]),
		("aggregate", module + ["inference.router_data_collect.cli", "aggregate", "--dataset-root", str(dataset)]),
		("validate", module + ["inference.router_data_collect.cli", "validate", "--dataset-root", str(dataset), "--require-results", "--require-evaluations", "--require-targets"]),
	]


def main(argv: Sequence[str] | None = None) -> None:
	"""Run this module's command-line workflow and return its process status."""
	args = build_parser().parse_args(argv)
	root = args.root.resolve()
	partition = validate_partition(root)
	if not 0 <= int(args.shard_id) < int(partition["num_shards"]):
		raise DataValidationError("shard_id is outside the frozen partition")
	shard_root = root / "shards" / f"shard_{args.shard_id:02d}"
	owner = read_json(shard_root / "owner.json")
	if owner.get("shard_id") != args.shard_id:
		raise DataValidationError("shard owner mismatch")
	start = STAGES.index(args.from_stage)
	stop = STAGES.index(args.through_stage)
	if stop < start:
		raise DataValidationError("through-stage precedes from-stage")
	commands = [(stage, command) for stage, command in _commands(args, shard_root) if start <= STAGES.index(stage) <= stop]
	print(json.dumps({"shard_id": args.shard_id, "stages": [stage for stage, _ in commands], "commands": [command for _, command in commands]}, ensure_ascii=False, indent=2))
	if args.print_only:
		return
	status_root = shard_root / "stage_status"
	for stage, command in commands:
		marker = status_root / f"{stage}.success.json"
		if marker.is_file() and not args.force:
			continue
		started = time.time()
		completed = subprocess.run(command, cwd=args.project_root.resolve(), check=False)
		if completed.returncode != 0:
			raise RuntimeError(f"shard {args.shard_id} stage {stage} failed with exit code {completed.returncode}")
		write_json_atomic(
			marker,
			{
				"status": "success",
				"stage": stage,
				"shard_id": args.shard_id,
				"command": command,
				"started_unix": started,
				"completed_unix": time.time(),
			},
		)


if __name__ == "__main__":
	main()
