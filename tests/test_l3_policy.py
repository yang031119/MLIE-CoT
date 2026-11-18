from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from inference.main_method.modules.contracts import build_level_action_plan, validate_distribution_contract
from inference.main_method.modules.l3_policy import (
	L3_PROMPT_VERSION,
	build_l3_policy_contract,
	l3_policy_digest,
	resolve_initial_base_snapshot,
	resolve_online_l3_policy,
	normalize_initial_generator_family,
	validate_l3_policy_contract,
	validate_matching_l3_policy,
)


def _diagnosis() -> dict:
	return {
		"fully_satisfied": False,
		"reason": "the requested cup is missing",
		"constraint_audit": {
			"objects": "no cup is visible",
			"count": "zero cups",
			"attributes": "not observable",
			"relations": "not observable",
			"action_state": "not applicable",
			"scene": "a table is visible",
		},
		"issues": [{
			"constraint_type": "object_identity",
			"target_entity": "cup",
			"observed": "absent",
			"expected": "one cup",
			"evidence": "only the table is visible",
			"confidence": 0.9,
		}],
		"preservation_constraints": ["preserve the table"],
	}


@pytest.mark.parametrize(
	("event_model", "family"),
	[("BAGEL", "bagel"), ("FLUX2", "flux2"), ("FLUX2-dev", "flux2"), ("SD35_TURBO", "sd35"), ("Stable-Diffusion-3.5-Turbo", "sd35")],
)
def test_initial_generation_timeline_labels_normalize_for_resume(event_model: str, family: str) -> None:
	assert normalize_initial_generator_family(event_model) == family


def test_pre_family_field_sd35_status_resumes_from_timeline(tmp_path) -> None:
	from inference.main_method.modules.pipeline import (
		_explicit_initial_base_policy_requested,
		_persisted_initial_generator_family,
	)
	from inference.main_method.modules.state import SampleState

	state = SampleState(index=0, prompt="a cup", sample_dir=tmp_path, sample_id="sample")
	state.current_image = tmp_path / "initial.png"
	state.history = [{"event": "initial_generate", "model": "SD35_TURBO"}]
	assert _persisted_initial_generator_family(state) == "sd35"
	assert _explicit_initial_base_policy_requested(
		SimpleNamespace(initial_generator="sd35", sd35_model_revision="a" * 40)
	) is True


def test_status_roundtrips_exact_initial_generator_policy_binding(tmp_path) -> None:
	from inference.main_method.modules.io_utils import load_states, save_status
	from inference.main_method.modules.state import SampleState, safe_prompt_id

	prompt = "a cup"
	sample_id = safe_prompt_id(0, prompt)
	sample_dir = tmp_path / sample_id
	sample_dir.mkdir()
	image_path = sample_dir / "initial.png"
	image_path.write_bytes(b"frozen-image-placeholder")
	state = SampleState(index=0, prompt=prompt, sample_dir=sample_dir, sample_id=sample_id)
	state.current_image = image_path
	state.initial_generator_family = "bagel"
	state.initial_generator_policy_digest = "e" * 64
	save_status(state)
	loaded = load_states([(0, prompt)], tmp_path, resume=True, max_actions=2)[0]
	assert loaded.initial_generator_family == "bagel"
	assert loaded.initial_generator_policy_digest == "e" * 64


@pytest.mark.parametrize(
	("family", "kwargs"),
	[
		("flux2", {"max_sequence_length": 512}),
		("sd35", {"max_sequence_length": 512}),
		(
			"bagel",
			{
				"max_sequence_length": None,
				"max_latent_size": 64,
				"auxiliary_model_id": "vcot/bagel-ft",
				"auxiliary_model_revision": "b" * 40,
			},
		),
	],
)
def test_l3_policy_canonically_locks_each_supported_initial_base(family: str, kwargs: dict) -> None:
	policy = build_l3_policy_contract(
		family=family,
		model_id=f"vcot/{family}",
		model_revision="a" * 40,
		resolution=1024,
		steps=4,
		guidance_scale=1.0,
		dtype="bfloat16",
		prompt_version="test-v1",
		**kwargs,
	)
	assert validate_l3_policy_contract(policy) == policy
	assert policy["executor"]["family"] == family
	assert policy["action"]["candidate_count"] == 1
	assert len(l3_policy_digest(policy)) == 64


def test_l3_plan_requires_and_preserves_initial_generator_binding() -> None:
	with pytest.raises(ValueError, match="initial_generator_family"):
		build_level_action_plan(3, "one cup", _diagnosis())
	plan = build_level_action_plan(
		3,
		"one cup",
		_diagnosis(),
		initial_generator_family="sd35",
	)
	validate_distribution_contract(plan)
	assert plan["executor_family"] == plan["initial_generator_family"] == "sd35"
	assert "Construct the scene afresh" in plan["instruction"]
	assert "Do not" not in plan["instruction"] and "Avoid" not in plan["instruction"]
	tampered = dict(plan, executor_family="flux2")
	with pytest.raises(ValueError, match="exactly follow"):
		validate_distribution_contract(tampered)


def test_l3_runtime_policy_rejects_a_different_initial_base_or_model() -> None:
	expected = build_l3_policy_contract(
		family="flux2",
		model_id="vcot/flux2",
		model_revision="a" * 40,
		resolution=1024,
		steps=28,
		guidance_scale=3.5,
		max_sequence_length=512,
		dtype="bfloat16",
		prompt_version="test-v1",
	)
	tampered = copy.deepcopy(expected)
	tampered["executor"]["model_revision"] = "c" * 40
	with pytest.raises(ValueError, match="differs"):
		validate_matching_l3_policy(expected, tampered)


def test_online_bagel_l3_resolves_the_same_immutable_bytes_used_for_initial_generation(tmp_path) -> None:
	base_revision = "a" * 40
	finetuned_revision = "b" * 40
	base = tmp_path / "models--vcot--bagel-base" / "snapshots" / base_revision
	finetuned = tmp_path / "models--vcot--bagel-ft" / "snapshots" / finetuned_revision
	base.mkdir(parents=True)
	finetuned.mkdir(parents=True)
	args = SimpleNamespace(
		initial_generator="bagel",
		bagel_model_path=str(base),
		bagel_model_id="vcot/bagel-base",
		bagel_model_revision=base_revision,
		bagel_fine_tuned_path=str(finetuned),
		bagel_fine_tuned_model_id="vcot/bagel-ft",
		bagel_fine_tuned_model_revision=finetuned_revision,
		resolution=1024,
		bagel_steps=50,
		bagel_cfg_scale=4.0,
		max_latent_size=64,
	)
	paths, policy = resolve_online_l3_policy(args)
	assert paths == {"model": base.resolve(), "auxiliary_model": finetuned.resolve()}
	assert policy["executor"]["family"] == "bagel"
	assert policy["executor"]["model_id"] == "vcot/bagel-base"
	assert policy["executor"]["model_revision"] == base_revision
	assert policy["executor"]["auxiliary_model_revision"] == finetuned_revision


def test_initial_base_resolution_rejects_a_declared_revision_on_mutable_or_mismatched_bytes(tmp_path) -> None:
	revision = "c" * 40
	mutable = tmp_path / revision
	mutable.mkdir()
	with pytest.raises(ValueError, match="snapshots/<revision>"):
		resolve_initial_base_snapshot(str(mutable), model_revision=revision)

	snapshot = tmp_path / "models--vcot--actual" / "snapshots" / revision
	snapshot.mkdir(parents=True)
	with pytest.raises(ValueError, match="model_id differs"):
		resolve_initial_base_snapshot(
			str(snapshot),
			model_id="vcot/declared",
			model_revision=revision,
		)


def test_flux_l3_contract_is_identical_offline_through_checkpoint_and_online() -> None:
	from inference.main_method.modules.route_head import validate_route_l3_runtime_policy
	from inference.router_data_collect.schema import DataValidationError, DatasetConfig, PROFILE_NAME, SCHEMA_VERSION

	revision = "d" * 40
	config = DatasetConfig.from_mapping(
		{
			"schema_version": SCHEMA_VERSION,
			"profile": PROFILE_NAME,
			"flux2": {
				"model_id": "vcot/flux2",
				"model_revision": revision,
				"resolution": 1024,
				"steps": 28,
				"guidance_scale": 3.5,
				"max_sequence_length": 512,
				"dtype": "bfloat16",
				"l1_roi_pad_ratio": 1.35,
				"l1_min_span": 64,
				"l1_max_edits": 3,
				"prompt_version": "vcot_v10_flux2_joint_atomic_v2",
			},
		},
		strict_revision=True,
	)
	offline = config.l3_policy.contract
	assert offline["prompt_version"] == L3_PROMPT_VERSION
	assert offline["prompt_version"] != config.policy.collector_contract["prompt_version"]
	_paths, online = resolve_online_l3_policy(
		SimpleNamespace(initial_generator="flux", resolution=1024),
		flux2_policy=config.policy.collector_contract,
	)
	assert online == offline
	checkpoint_config = {
		"l3_policy_contract": offline,
		"l3_policy_digest": l3_policy_digest(offline),
	}
	validate_route_l3_runtime_policy(checkpoint_config, online)
	drifted = copy.deepcopy(config.as_dict())
	drifted["l3"]["generation"]["steps"] += 1
	with pytest.raises(DataValidationError, match="generation settings"):
		DatasetConfig.from_mapping(drifted, strict_revision=True)
