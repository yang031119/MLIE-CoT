from __future__ import annotations

import copy
from pathlib import Path

import pytest

from inference.main_method.modules.cli import build_parser
from inference.main_method.modules.config import DEFAULT_FLUX_MODEL_REVISION
from inference.main_method.modules.flux_policy import (
	build_flux2_policy_contract,
	build_online_flux2_policy,
	flux2_policy_digest,
	resolve_online_flux2_snapshot,
	validate_flux2_policy_contract,
)
from inference.main_method.modules.route_head import validate_route_flux2_runtime_policy


def _policy(**overrides):
	values = {
		"model_id": "black-forest-labs/FLUX.2-dev",
		"model_revision": "a" * 40,
		"resolution": 1024,
		"steps": 28,
		"guidance_scale": 3.5,
		"max_sequence_length": 512,
		"dtype": "bfloat16",
		"l1_roi_pad_ratio": 1.35,
		"l1_min_span": 64,
		"l1_max_crops": 3,
		"prompt_version": "vcot_v10_flux2_joint_atomic_v2",
	}
	values.update(overrides)
	return build_flux2_policy_contract(**values)


def test_policy_locks_executor_generation_geometry_and_atomic_semantics() -> None:
	policy = _policy()
	assert policy["executor"] == {
		"family": "flux2",
		"implementation": "diffusers.Flux2Pipeline",
		"loader": "from_pretrained(local_files_only=True)",
		"model_id": "black-forest-labs/FLUX.2-dev",
		"model_revision": "a" * 40,
		"dtype": "bfloat16",
	}
	assert policy["generation"] == {
		"resolution": 1024,
		"steps": 28,
		"guidance_scale": 3.5,
		"max_sequence_length": 512,
	}
	assert policy["l1_crop"]["roi_pad_ratio"] == 1.35
	assert policy["l1_crop"]["min_span"] == 64
	assert policy["l1_crop"]["max_crops"] == 3
	assert set(policy["actions"]) == {"L1", "L2"}
	for level in ("L1", "L2"):
		assert policy["actions"][level]["candidate_count"] == 1
		assert policy["actions"][level]["successful_flux_call_count"] == 1
		assert policy["actions"][level]["semantic_retry_count"] == 0
	assert len(flux2_policy_digest(policy)) == 64

	tampered = copy.deepcopy(policy)
	tampered["actions"]["L1"]["candidate_count"] = 5
	with pytest.raises(ValueError, match="one-call/candidate"):
		validate_flux2_policy_contract(tampered)


def test_formal_online_policy_requires_and_resolves_an_immutable_snapshot(require_symlink, tmp_path: Path) -> None:
	revision = "b" * 40
	snapshot = tmp_path / "hub" / "models--black-forest-labs--FLUX.2-dev" / "snapshots" / revision
	snapshot.mkdir(parents=True)
	resolved, model_id, actual_revision = resolve_online_flux2_snapshot(str(snapshot))
	assert resolved == snapshot.resolve()
	assert model_id == "black-forest-labs/FLUX.2-dev"
	assert actual_revision == revision

	materialized = tmp_path / "materialized" / "modelscope-local"
	materialized.mkdir(parents=True)
	linked_snapshot = tmp_path / "linked" / "snapshots" / ("c" * 40)
	linked_snapshot.parent.mkdir(parents=True)
	linked_snapshot.symlink_to(materialized, target_is_directory=True)
	resolved, _model_id, actual_revision = resolve_online_flux2_snapshot(
		str(linked_snapshot),
		model_id="black-forest-labs/FLUX.2-dev",
		model_revision="c" * 40,
	)
	assert resolved == materialized.resolve()
	assert actual_revision == "c" * 40

	mutable = tmp_path / "local-flux-main"
	mutable.mkdir()
	with pytest.raises(ValueError, match="requires --flux-model-revision"):
		resolve_online_flux2_snapshot(str(mutable), require_immutable=True)


def test_online_cli_defaults_build_the_same_policy_and_reject_drift(tmp_path: Path) -> None:
	revision = DEFAULT_FLUX_MODEL_REVISION
	snapshot = tmp_path / "hub" / "models--black-forest-labs--FLUX.2-dev" / "snapshots" / revision
	snapshot.mkdir(parents=True)
	args = build_parser().parse_args(["--flux-model", str(snapshot)])
	resolved, actual = build_online_flux2_policy(args)
	assert resolved == snapshot.resolve()
	expected = _policy(model_revision=revision)
	config = {
		"flux2_policy_contract": expected,
		"flux2_policy_digest": flux2_policy_digest(expected),
	}
	validate_route_flux2_runtime_policy(config, actual)

	drifted = copy.deepcopy(actual)
	drifted["generation"]["steps"] = 29
	with pytest.raises(ValueError, match="differs from the route-head collector policy"):
		validate_route_flux2_runtime_policy(config, drifted)


def test_repo_id_without_revision_is_rejected_for_formal_route_deployment() -> None:
	args = build_parser().parse_args(
		["--flux-model", "black-forest-labs/FLUX.2-dev", "--flux-model-revision", ""]
	)
	with pytest.raises(ValueError, match="requires --flux-model-revision"):
		build_online_flux2_policy(args, require_immutable=True)
