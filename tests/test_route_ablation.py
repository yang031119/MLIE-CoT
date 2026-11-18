from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest


def _flux2_policy() -> dict:
	from inference.main_method.modules.flux_policy import build_flux2_policy_contract

	return build_flux2_policy_contract(
		model_id="black-forest-labs/FLUX.2-dev",
		model_revision="a" * 40,
		resolution=1024,
		steps=28,
		guidance_scale=3.5,
		max_sequence_length=512,
		dtype="bfloat16",
		l1_roi_pad_ratio=1.15,
		l1_min_span=32,
		l1_max_crops=3,
		prompt_version="vcot_v10_flux2_joint_atomic_v2",
	)


def _l3_policy() -> dict:
	from inference.main_method.modules.l3_policy import L3_PROMPT_VERSION, build_l3_policy_contract

	return build_l3_policy_contract(
		family="flux2",
		model_id="black-forest-labs/FLUX.2-dev",
		model_revision="a" * 40,
		resolution=1024,
		steps=28,
		guidance_scale=3.5,
		max_sequence_length=512,
		dtype="bfloat16",
		prompt_version=L3_PROMPT_VERSION,
	)


def _make_source_run(tmp_path: Path, prompts: list[str]) -> tuple[Path, Path, str, str]:
	from inference.main_method.modules.flux_policy import flux2_policy_digest
	from inference.main_method.modules.io_utils import save_status
	from inference.main_method.modules.l3_policy import l3_policy_digest
	from inference.main_method.modules.state import SampleState, safe_prompt_id

	source = tmp_path / "source"
	data_path = tmp_path / "prompts.txt"
	data_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
	flux_policy = _flux2_policy()
	l3_policy = _l3_policy()
	l3_digest = l3_policy_digest(l3_policy)
	(source / "run_config.json").parent.mkdir(parents=True, exist_ok=True)
	(source / "run_config.json").write_text(
		json.dumps(
			{
				"initial_generator": "flux",
				"l3_policy_contract": l3_policy,
				"l3_policy_digest": l3_digest,
				"flux2_collector_policy": flux_policy,
				"flux2_policy_digest": flux2_policy_digest(flux_policy),
			}
		),
		encoding="utf-8",
	)
	for index, prompt in enumerate(prompts):
		sample_id = safe_prompt_id(index, prompt)
		sample_dir = source / sample_id
		sample_dir.mkdir(parents=True)
		image_path = sample_dir / "initial.png"
		Image.new("RGB", (16, 16), (index * 40, 20, 30)).save(image_path)
		state = SampleState(index=index, prompt=prompt, sample_dir=sample_dir, sample_id=sample_id)
		state.initial_generator_family = "flux2"
		state.initial_generator_policy_digest = l3_digest
		state.generation_prompt = f"generation {index}"
		state.current_image = image_path
		state.needs_global_analysis = True
		save_status(state)
	return source, data_path, flux2_policy_digest(flux_policy), l3_digest


def test_fixed_repair_level_keeps_native_l0_and_forces_only_unsatisfied_states() -> None:
	from inference.main_method.modules.qwen_stages import apply_fixed_repair_level

	state = SimpleNamespace(phase="global")
	for level, mode in ((1, "micro"), (2, "global"), (3, "regenerate")):
		analysis = {"fully_satisfied": False, "recommended_mode": "global", "category": "L2"}
		decision = apply_fixed_repair_level(SimpleNamespace(fixed_repair_level=level), state, analysis)
		assert decision["level"] == level
		assert decision["mode"] == mode
		assert decision["source"] == "fixed_repair_level_ablation"

	analysis = {"fully_satisfied": True, "recommended_mode": "done", "category": "L0"}
	decision = apply_fixed_repair_level(SimpleNamespace(fixed_repair_level=3), state, analysis)
	assert decision["level"] == 0
	assert decision["mode"] == "done"


def test_ablation_strategy_is_authoritative_and_rejects_missing_route_head(require_symlink, tmp_path: Path) -> None:
	from inference.router_ablation.protocol import apply_ablation_strategy, lock_ablation_qwen_model

	fixed = SimpleNamespace(
		ablation_strategy="fixed_l2",
		fixed_repair_level=None,
		qwen_route_head=tmp_path / "future-head.pt",
	)
	assert apply_ablation_strategy(fixed) == "fixed-l2"
	assert fixed.fixed_repair_level == 2
	assert fixed.qwen_route_head == ""
	assert fixed.ablation_reference_route_head.endswith("future-head.pt")

	missing = SimpleNamespace(
		ablation_strategy="route",
		fixed_repair_level=None,
		qwen_route_head=tmp_path / "missing.pt",
	)
	with pytest.raises(ValueError, match="existing V10"):
		apply_ablation_strategy(missing)

	head = tmp_path / "head.pt"
	head.write_bytes(b"v10 checkpoint placeholder")
	routed = SimpleNamespace(ablation_strategy="router", fixed_repair_level=None, qwen_route_head=head)
	assert apply_ablation_strategy(routed) == "route"
	assert routed.qwen_route_head == head

	mutable_qwen = tmp_path / "snapshots" / "main"
	mutable_qwen.mkdir(parents=True)
	with pytest.raises(ValueError, match="immutable snapshots"):
		lock_ablation_qwen_model(SimpleNamespace(qwen_model=str(mutable_qwen)))
	exact_qwen = tmp_path / "snapshots" / ("b" * 40)
	exact_qwen.mkdir()
	qwen_args = SimpleNamespace(qwen_model=str(exact_qwen))
	assert lock_ablation_qwen_model(qwen_args) == exact_qwen.absolute()
	assert qwen_args.qwen_model == str(exact_qwen.absolute())
	materialized = tmp_path / "materialized" / "main"
	materialized.mkdir(parents=True)
	linked_qwen = tmp_path / "snapshots" / ("c" * 40)
	linked_qwen.symlink_to(materialized, target_is_directory=True)
	linked_args = SimpleNamespace(qwen_model=str(linked_qwen))
	assert lock_ablation_qwen_model(linked_args) == linked_qwen.absolute()
	assert linked_args.qwen_model == str(linked_qwen.absolute())


def test_v10_same_initial_seeder_preserves_bytes_and_policy_provenance(tmp_path: Path) -> None:
	from inference.router_ablation.initials import seed_v10_initials

	source, data_path, flux_digest, l3_digest = _make_source_run(tmp_path, ["a red car", "a blue boat"])
	target = tmp_path / "target"
	report = seed_v10_initials(
		source_run=source,
		target_run=target,
		data_path=data_path,
		start_index=0,
		limit=None,
		initial_generator="flux",
		flux2_policy_digest=flux_digest,
		requested_l3_policy_digest=l3_digest,
		copy_mode="copy",
	)
	assert report["seeded"] == 2
	assert report["already_seeded"] == 0
	for sample_id, expected_hash in report["initial_hashes"].items():
		status = json.loads((target / sample_id / "status.json").read_text(encoding="utf-8"))
		assert status["state_schema_version"] == 10
		assert status["initial_generator_family"] == "flux2"
		assert status["initial_generator_policy_digest"] == report["initial_generator_policy_digest"]
		assert status["needs_global_analysis"] is True
		assert status["current_image"] == str(target / sample_id / "initial.png")
		assert (target / sample_id / "round0" / "initial.png").is_file()
		assert len(expected_hash) == 64

	second = seed_v10_initials(
		source_run=source,
		target_run=target,
		data_path=data_path,
		start_index=0,
		limit=None,
		initial_generator="flux",
		flux2_policy_digest=flux_digest,
		requested_l3_policy_digest=l3_digest,
		copy_mode="copy",
	)
	assert second["seeded"] == 0
	assert second["already_seeded"] == 2


def test_seeder_rejects_legacy_state_and_flux_policy_drift(tmp_path: Path) -> None:
	from inference.router_ablation.initials import seed_v10_initials

	source, data_path, flux_digest, l3_digest = _make_source_run(tmp_path, ["a prompt"])
	with pytest.raises(ValueError, match="FLUX.2 L1/L2 policy digests differ"):
		seed_v10_initials(
			source_run=source,
			target_run=tmp_path / "wrong-policy",
			data_path=data_path,
			start_index=0,
			limit=None,
			initial_generator="flux",
			flux2_policy_digest="f" * 64,
			requested_l3_policy_digest=l3_digest,
		)

	status_path = next(source.glob("*/status.json"))
	status = json.loads(status_path.read_text(encoding="utf-8"))
	status["state_schema_version"] = 9
	status_path.write_text(json.dumps(status), encoding="utf-8")
	with pytest.raises(ValueError, match="expected V10 schema 10"):
		seed_v10_initials(
			source_run=source,
			target_run=tmp_path / "legacy",
			data_path=data_path,
			start_index=0,
			limit=None,
			initial_generator="flux",
			flux2_policy_digest=flux_digest,
			requested_l3_policy_digest=l3_digest,
		)


def test_seeder_can_explicitly_import_cross_policy_initial_assets(tmp_path: Path) -> None:
	from inference.router_ablation.initials import seed_v10_initials

	source, data_path, flux_digest, l3_digest = _make_source_run(tmp_path, ["a prompt"])
	target = tmp_path / "target"
	report = seed_v10_initials(
		source_run=source,
		target_run=target,
		data_path=data_path,
		start_index=0,
		limit=None,
		initial_generator="flux",
		flux2_policy_digest="f" * 64,
		requested_l3_policy_digest="e" * 64,
		allow_cross_policy_initial_assets=True,
	)
	assert report["cross_policy_initial_asset_import"] is True
	assert report["source_flux2_policy_digest"] == flux_digest
	assert report["source_initial_generator_policy_digest"] == l3_digest
	assert report["initial_generator_policy_digest"] == "e" * 64
	status_path = next(target.glob("*/status.json"))
	status = json.loads(status_path.read_text(encoding="utf-8"))
	assert status["initial_generator_policy_digest"] == "e" * 64
	timeline_path = next(target.glob("*/timeline.jsonl"))
	event = json.loads(timeline_path.read_text(encoding="utf-8").splitlines()[0])
	assert event["initial_asset_imported"] is True
	assert event["source_initial_policy_digest"] == l3_digest

def test_post_run_audit_rejects_fixed_l1_to_l2_fallback(tmp_path: Path) -> None:
	from inference.router_ablation.audit import audit_ablation_run
	from inference.router_ablation.initials import seed_v10_initials

	source, data_path, flux_digest, l3_digest = _make_source_run(tmp_path, ["a red car"])
	target = tmp_path / "target"
	seed_report = seed_v10_initials(
		source_run=source,
		target_run=target,
		data_path=data_path,
		start_index=0,
		limit=None,
		initial_generator="flux",
		flux2_policy_digest=flux_digest,
		requested_l3_policy_digest=l3_digest,
	)
	sample_dir = next(path for path in target.iterdir() if path.is_dir())
	status_path = sample_dir / "status.json"
	status = json.loads(status_path.read_text(encoding="utf-8"))
	status.update({"done": True, "status": "ok", "error": None})
	status_path.write_text(json.dumps(status), encoding="utf-8")
	(sample_dir / "summary.json").write_text("{}\n", encoding="utf-8")
	Image.open(sample_dir / "initial.png").save(sample_dir / "final.png")
	round_dir = sample_dir / "round0"
	classification = {
		"fully_satisfied": False,
		"decision": {"source": "fixed_repair_level_ablation", "level": 1},
	}
	(round_dir / "level_classification.json").write_text(json.dumps(classification), encoding="utf-8")
	(round_dir / "repair_plan.json").write_text(json.dumps({"action_plan": {"level": 1}}), encoding="utf-8")
	valid = audit_ablation_run(
		output_dir=target,
		strategy="fixed-l1",
		data_path=data_path,
		start_index=0,
		limit=None,
		seed_report=seed_report,
	)
	assert valid["valid"] is True

	classification["decision"].update(
		{"level": 2, "original_level": 1, "planning_failure": "invalid_l1_plan"}
	)
	(round_dir / "level_classification.json").write_text(json.dumps(classification), encoding="utf-8")
	(round_dir / "repair_plan.json").write_text(json.dumps({"action_plan": {"level": 2}}), encoding="utf-8")
	invalid = audit_ablation_run(
		output_dir=target,
		strategy="fixed-l1",
		data_path=data_path,
		start_index=0,
		limit=None,
		seed_report=seed_report,
	)
	assert invalid["valid"] is False
	assert invalid["cross_level_violations"]


def test_ablation_manifest_refuses_untracked_output_and_strategy_drift(tmp_path: Path) -> None:
	from inference.router_ablation.protocol import write_ablation_manifest

	payload = {
		"protocol_version": "route_ablation_v1",
		"strategy": "fixed-l1",
		"fixed_repair_level": 1,
		"data_path": "/prompts.txt",
		"start_index": 0,
		"limit": 1,
		"initial_generator": "flux",
		"max_global_rounds": 2,
		"qwen_route_head": "",
		"reference_route_head": "/head.pt",
		"reference_route_head_sha256": None,
		"qwen_runtime": {},
		"flux2_policy_digest": "a" * 64,
		"l3_policy_digest": "b" * 64,
		"initial_source_run": None,
	}
	untracked = tmp_path / "untracked"
	untracked.mkdir()
	(untracked / "status.json").write_text("{}", encoding="utf-8")
	with pytest.raises(ValueError, match="non-empty untracked"):
		write_ablation_manifest(untracked, payload, resume=True)

	tracked = tmp_path / "tracked"
	write_ablation_manifest(tracked, payload, resume=False)
	write_ablation_manifest(tracked, payload, resume=True)
	drifted = {**payload, "strategy": "fixed-l2", "fixed_repair_level": 2}
	with pytest.raises(ValueError, match="configuration changed"):
		write_ablation_manifest(tracked, drifted, resume=True)
