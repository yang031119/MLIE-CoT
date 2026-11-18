"""Strategy selection and provenance for isolated V10 route ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from inference.main_method.modules.flux_policy import build_online_flux2_policy, flux2_policy_digest
from inference.main_method.modules.l3_policy import l3_policy_digest, resolve_online_l3_policy
from inference.main_method.modules.memory import resolve_qwen_model_path


PROTOCOL_VERSION = "route_ablation_v1"
STRATEGY_ALIASES = {
	"route": "route",
	"router": "route",
	"joint-route": "route",
	"joint_route": "route",
	"fixed-l1": "fixed-l1",
	"fixed_l1": "fixed-l1",
	"fixed-l2": "fixed-l2",
	"fixed_l2": "fixed-l2",
	"fixed-l3": "fixed-l3",
	"fixed_l3": "fixed-l3",
}
FIXED_LEVELS = {"fixed-l1": 1, "fixed-l2": 2, "fixed-l3": 3}
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$", flags=re.IGNORECASE)


def normalize_strategy(value: str) -> str:
	"""Normalize and validate a router-ablation strategy name."""
	key = str(value or "").strip().lower()
	try:
		return STRATEGY_ALIASES[key]
	except KeyError as exc:
		raise ValueError(f"unknown V10 route-ablation strategy: {value!r}") from exc


def apply_ablation_strategy(args: argparse.Namespace) -> str:
	"""Make the strategy authoritative without changing V10 runtime code."""
	strategy = normalize_strategy(args.ablation_strategy)
	expected_level = FIXED_LEVELS.get(strategy)
	requested_level = getattr(args, "fixed_repair_level", None)
	reference_route_head_raw = str(getattr(args, "qwen_route_head", "") or "").strip()
	reference_route_head = str(Path(reference_route_head_raw).expanduser().resolve()) if reference_route_head_raw else ""
	args.ablation_reference_route_head = reference_route_head
	if strategy == "route":
		if requested_level is not None:
			raise ValueError("route strategy cannot be combined with --fixed-repair-level")
		route_head = Path(reference_route_head) if reference_route_head else Path()
		if not reference_route_head or not route_head.is_file():
			raise ValueError(
				"route strategy requires an existing V10 --qwen-route-head checkpoint; "
				"V9 route-head checkpoints are incompatible"
			)
		args.fixed_repair_level = None
		args.qwen_route_head = route_head
	else:
		if requested_level is not None and int(requested_level) != int(expected_level):
			raise ValueError(
				f"{strategy} conflicts with --fixed-repair-level {requested_level}; "
				f"expected {expected_level}"
			)
		args.fixed_repair_level = int(expected_level)
		# Fixed-level collection deliberately bypasses the learned head. V10's
		# native L0 satisfied gate remains active inside apply_fixed_repair_level.
		args.qwen_route_head = ""
	args.ablation_strategy = strategy
	return strategy


def lock_ablation_flux2_policy(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
	"""Freeze the L1/L2 editor for every arm, including headless fixed runs."""
	original_model = str(getattr(args, "flux_model", "") or "")
	resolved_model, policy = build_online_flux2_policy(args, require_immutable=True)
	# Keep the commit-visible path/repo id for the pipeline's second contract
	# check. It will replace this with the physical service path only after the
	# same immutable policy has been validated again.
	args.flux_model = original_model or str(resolved_model)
	args.flux_model_id = policy["executor"]["model_id"]
	args.flux_model_revision = policy["executor"]["model_revision"]
	return policy, flux2_policy_digest(policy)


def lock_ablation_qwen_model(args: argparse.Namespace) -> Path:
	"""Require one immutable Qwen snapshot for routed and fixed arms alike."""
	# Preserve the caller-visible commit component. Hugging Face cache entries
	# may legitimately be commit-named symlinks to a materialized local folder
	# such as ``main``; resolving the symlink before checking its name would
	# incorrectly reject the pinned entry.
	resolved = Path(resolve_qwen_model_path(str(args.qwen_model))).expanduser().absolute()
	if not resolved.is_dir() or IMMUTABLE_REVISION_RE.fullmatch(resolved.name) is None:
		raise ValueError(
			"--qwen-model must resolve directly to an immutable snapshots/<40-64 hex commit> directory"
		)
	args.qwen_model = str(resolved)
	return resolved


def lock_ablation_l3_policy(
	args: argparse.Namespace,
	flux2_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
	"""Freeze the initial generator and same-base L3 executor for every arm."""
	resolved_paths, policy = resolve_online_l3_policy(args, flux2_policy=flux2_policy)
	executor = policy["executor"]
	family = executor["family"]
	if family == "bagel":
		args.bagel_model_path = str(resolved_paths["model"])
		args.bagel_fine_tuned_path = str(resolved_paths["auxiliary_model"])
		args.bagel_model_id = executor["model_id"]
		args.bagel_model_revision = executor["model_revision"]
		args.bagel_fine_tuned_model_id = executor["auxiliary_model_id"]
		args.bagel_fine_tuned_model_revision = executor["auxiliary_model_revision"]
	elif family == "sd35":
		args.sd35_model = str(resolved_paths["model"])
		args.sd35_model_id = executor["model_id"]
		args.sd35_model_revision = executor["model_revision"]
	return policy, l3_policy_digest(policy)


def _file_sha256(path_value: str) -> str | None:
	if not path_value:
		return None
	path = Path(path_value)
	if not path.is_file():
		return None
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def build_ablation_manifest(
	args: argparse.Namespace,
	strategy: str,
	seed_report: Mapping[str, Any] | None,
	flux2_policy: Mapping[str, Any],
	flux2_digest: str,
	l3_policy: Mapping[str, Any],
	l3_digest: str,
) -> dict[str, Any]:
	"""Record the exact route strategy, same-initial provenance, and locked runtime policies."""
	reference_route_head = str(getattr(args, "ablation_reference_route_head", "") or "")
	return {
		"protocol_version": PROTOCOL_VERSION,
		"strategy": normalize_strategy(strategy),
		"fixed_repair_level": getattr(args, "fixed_repair_level", None),
		"v10_runtime_modified": False,
		"native_l0_gate_preserved": True,
		"one_candidate_per_action": True,
		"candidate_retain_review_preserved": True,
		"fresh_diagnosis_after_retain_review": True,
		"l3_semantics": "initial-base prompt-only candidate followed by source-blind original/candidate retain review",
		"data_path": str(Path(args.data_path).resolve()),
		"start_index": int(args.start_index),
		"limit": args.limit,
		"initial_generator": str(args.initial_generator),
		"max_global_rounds": int(args.max_global_rounds),
		"qwen_route_head": str(getattr(args, "qwen_route_head", "") or ""),
		"reference_route_head": reference_route_head,
		"reference_route_head_sha256": _file_sha256(reference_route_head),
		"qwen_runtime": {
			"model": str(args.qwen_model),
			"dtype": str(args.qwen_dtype),
			"quantization": str(args.qwen_quantization),
			"input_max_side": int(args.qwen_input_max_side),
			"max_new_tokens": int(args.qwen_max_new_tokens),
			"image_shrink_retry": not bool(args.qwen_disable_image_shrink_retry),
		},
		"flux2_policy_contract": dict(flux2_policy),
		"flux2_policy_digest": str(flux2_digest),
		"l3_policy_contract": dict(l3_policy),
		"l3_policy_digest": str(l3_digest),
		"initial_source_run": str(Path(args.initial_source_run).resolve()) if args.initial_source_run else None,
		"allow_cross_policy_initial_assets": bool(
			getattr(args, "allow_cross_policy_initial_assets", False)
		),
		"initial_seed_report": dict(seed_report) if seed_report is not None else None,
	}


def _compatibility_view(payload: Mapping[str, Any]) -> dict[str, Any]:
	keys = (
		"protocol_version",
		"strategy",
		"fixed_repair_level",
		"data_path",
		"start_index",
		"limit",
		"initial_generator",
		"max_global_rounds",
		"qwen_route_head",
		"reference_route_head",
		"reference_route_head_sha256",
		"qwen_runtime",
		"flux2_policy_digest",
		"l3_policy_digest",
		"initial_source_run",
		"allow_cross_policy_initial_assets",
	)
	return {key: payload.get(key) for key in keys}


def write_ablation_manifest(output_dir: Path, payload: Mapping[str, Any], *, resume: bool) -> Path:
	"""Create or resume the immutable manifest that defines an ablation run."""
	output_dir = Path(output_dir)
	path = output_dir / "route_ablation_config.json"
	if path.is_file():
		try:
			existing = json.loads(path.read_text(encoding="utf-8"))
		except Exception as exc:
			raise ValueError(f"cannot resume: invalid existing {path}: {exc}") from exc
		if not resume:
			raise ValueError(f"output already contains {path.name}; use --resume or a new output directory")
		if _compatibility_view(existing) != _compatibility_view(payload):
			raise ValueError("cannot resume: route-ablation protocol/configuration changed")
		return path
	if output_dir.is_dir() and any(output_dir.iterdir()):
		seed_manifest = output_dir / "route_ablation_initials.json"
		if not (resume and seed_manifest.is_file()):
			raise ValueError(
				"refusing to attach a route-ablation protocol to a non-empty untracked output directory"
			)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
	return path
