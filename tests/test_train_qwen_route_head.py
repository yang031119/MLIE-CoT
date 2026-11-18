from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

import scripts.train_route_head as trainer
from inference.main_method.modules.contracts import (
	DIAGNOSIS_SCHEMA,
	ROUTER_ACTION_POLICY_VERSION,
	ROUTER_REPAIR_EXECUTOR_FAMILY,
	build_qwen_analyzer_l1_distribution_contract,
	canonical_diagnosis_text,
)
from inference.main_method.modules.route_head import (
	DIAGNOSIS_AUX_FEATURE_NAMES,
	RouteMLP,
	append_diagnosis_aux_features,
	diagnosis_aux_vector,
)
from inference.main_method.modules.flux_policy import build_flux2_policy_contract, flux2_policy_digest
from inference.main_method.modules.l3_policy import L3_PROMPT_VERSION, build_l3_policy_contract, l3_policy_digest
from inference.router_data_collect.schema import SCHEMA_VERSION as DATA_SCHEMA_VERSION


def _sha256(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _qwen_distribution_signature() -> str:
	return build_qwen_analyzer_l1_distribution_contract(
		model="/test/qwen",
		model_revision="a" * 40,
		dtype="bfloat16",
		quantization="8bit",
		allow_cpu_offload=False,
	)["distribution_signature"]


def _flux2_policy() -> dict:
	return build_flux2_policy_contract(
		model_id="black-forest-labs/FLUX.2-dev",
		model_revision="c" * 40,
		resolution=1024,
		steps=28,
		guidance_scale=3.5,
		max_sequence_length=512,
		dtype="bfloat16",
		l1_roi_pad_ratio=1.35,
		l1_min_span=64,
		l1_max_crops=3,
		prompt_version="vcot_v10_flux2_joint_atomic_v2",
	)


def _l3_policy() -> dict:
	return build_l3_policy_contract(
		family="flux2",
		model_id="black-forest-labs/FLUX.2-dev",
		model_revision="c" * 40,
		resolution=1024,
		steps=28,
		guidance_scale=3.5,
		max_sequence_length=512,
		dtype="bfloat16",
		prompt_version=L3_PROMPT_VERSION,
	)


def _diagnosis() -> dict:
	return {
		"schema_version": DIAGNOSIS_SCHEMA,
		"fully_satisfied": False,
		"reason": "one requested object is missing",
		"constraint_audit": {
			"objects": "one cup is visible",
			"count": "one cup instead of two",
			"attributes": "the visible cup is red",
			"relations": "the cup is on the table",
			"action_state": "not applicable",
			"scene": "the table scene is otherwise correct",
		},
		"issues": [
			{
				"constraint_type": "count",
				"target_entity": "red cups",
				"observed": "one red cup",
				"expected": "two red cups",
				"evidence": "only one cup silhouette is visible",
				"confidence": 0.95,
			}
		],
		"preservation_constraints": ["preserve the table and existing cup"],
	}


def _pairwise(utility: list[float], epsilon: float) -> list[dict]:
	rows = []
	for left, right in ((1, 2), (1, 3), (2, 3)):
		delta = utility[left - 1] - utility[right - 1]
		rows.append(
			{
				"left_level": left,
				"right_level": right,
				"utility_delta": delta,
				"preference": 1 if delta > epsilon else (-1 if delta < -epsilon else 0),
			}
		)
	return rows


def _write_bundle(root: Path) -> Path:
	derived = root / "derived"
	derived.mkdir(parents=True, exist_ok=True)
	image_path = root / "current.png"
	Image.new("RGB", (19, 11), (7, 23, 101)).save(image_path)
	epsilon = 0.05
	qwen_distribution_signature = _qwen_distribution_signature()
	flux2_policy = _flux2_policy()
	flux2_digest = flux2_policy_digest(flux2_policy)
	l3_policy = _l3_policy()
	l3_digest = l3_policy_digest(l3_policy)
	counts = {"train": 4, "validation": 2, "test": 2}
	for split, count in counts.items():
		rows = []
		for index in range(count):
			utility = [0.25 + index * 0.01, 0.10, -0.10]
			center = sum(utility) / 3.0
			rows.append(
				{
					"schema_version": DATA_SCHEMA_VERSION,
					"state_id": f"{split}-{index}",
					"qwen_distribution_signature": qwen_distribution_signature,
					"flux2_policy_contract": flux2_policy,
					"flux2_policy_digest": flux2_digest,
					"l3_policy_contract": l3_policy,
					"l3_policy_digest": l3_digest,
					"initial_generator_family": "flux2",
					"l3_initial_identity_binding": "exact_model_identity",
					"group_id": f"{split}-group-{index}",
					"split": split,
					"image_path": str(image_path),
					"image_sha256": _sha256(image_path),
					"target_prompt": "two red cups on a wooden table",
					"diagnosis_text": canonical_diagnosis_text(_diagnosis()),
					"action_mask": [1, 1, 1],
					"utility_vector": utility,
					"centered_utility_vector": [value - center for value in utility],
					"soft_targets": [0.62, 0.25, 0.13],
					"pairwise_preferences": _pairwise(utility, epsilon),
				}
			)
		(derived / f"{split}.jsonl").write_text(
			"".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
		)
	(derived / "summary.json").write_text(
		json.dumps(
			{
				"schema_version": DATA_SCHEMA_VERSION,
				"qwen_distribution_signature": qwen_distribution_signature,
				"flux2_policy_contract": flux2_policy,
				"flux2_policy_digest": flux2_digest,
				"l3_policy_contract": l3_policy,
				"l3_policy_digest": l3_digest,
				"initial_generator_family": "flux2",
				"l3_initial_identity_binding_counts": {"exact_model_identity": sum(counts.values())},
				"aggregation": {"acceptable_epsilon": epsilon},
				"training_balance": {"class_factors": {"L1": 1.2, "L2": 0.8, "L3": 1.1}},
			}
		),
		encoding="utf-8",
	)
	return image_path


def _train_args(root: Path, output: Path, feature_cache: Path) -> argparse.Namespace:
	return argparse.Namespace(
		dataset_root=root,
		output_dir=output,
		feature_cache=feature_cache,
		skip_image_hash_check=False,
		export_checkpoint=None,
		train_device="cpu",
		hidden_dim=8,
		dropout=0.0,
		learning_rate=1e-2,
		weight_decay=0.0,
		epochs=3,
		patience=0,
		min_delta=0.0,
		mlp_batch_size=2,
		max_grad_norm=5.0,
		seed=17,
		class_factors=None,
		utility_temperature=0.5,
		pairwise_temperature=0.5,
		soft_ce_weight=1.0,
		utility_alignment_weight=0.15,
		pairwise_weight=0.20,
	)


def test_aggregate_contract_and_raw_route_image(tmp_path: Path) -> None:
	image_path = _write_bundle(tmp_path)
	bundle = trainer.load_dataset_bundle(tmp_path)
	assert tuple(bundle.class_factors) == (1.2, 0.8, 1.1)
	assert {name: len(rows) for name, rows in bundle.rows.items()} == {
		"train": 4,
		"validation": 2,
		"test": 2,
	}
	image = trainer.load_raw_route_image(image_path)
	try:
		assert image.size == (19, 11)
		assert image.getpixel((0, 0)) == (7, 23, 101)
	finally:
		image.close()


def test_candidate_quality_reconstruction_verifies_digest_and_paired_direction(tmp_path: Path) -> None:
	derived = tmp_path / "derived"
	derived.mkdir(parents=True)
	(tmp_path / "config.lock.json").write_text(
		json.dumps({"utility": {"target_gain": 3.0, "cost_per_call": 0.15}}),
		encoding="utf-8",
	)
	state_id = "candidate-state"
	qualities = {
		1: [4.0, 0.0, 0.0],
		2: [1.0, 1.0, 1.0],
		3: [0.0, 0.0, 0.0],
	}
	originals = {1: 0.9, 2: 0.1, 3: 0.5}
	rollouts = []
	for level in trainer.LABEL_LEVELS:
		for rollout_index, quality in enumerate(qualities[level]):
			rollouts.append(
				{
					"valid": True,
					"state_id": state_id,
					"action_level": level,
					"rollout_index": rollout_index,
					"seed": 100 + rollout_index,
					"utility": quality - 3.0 * originals[level] - 0.15,
					"call_count": 1,
					"evaluation_scores": {"original_target_satisfaction": originals[level]},
					"split": "train",
					"group_id": "candidate-group",
					"qwen_distribution_signature": "qwen-signature",
					"flux2_policy_digest": "flux-digest",
					"l3_policy_digest": "l3-digest",
				}
			)
	rollout_path = derived / "rollout_utilities.jsonl"
	rollout_path.write_text("".join(json.dumps(row) + "\n" for row in rollouts), encoding="utf-8")
	(derived / "summary.json").write_text(
		json.dumps({"derived_digests": {"rollout_utilities": trainer.sha256_json(rollouts)}}),
		encoding="utf-8",
	)
	rows = {
		"train": [
			{
				"state_id": state_id,
				"split": "train",
				"group_id": "candidate-group",
				"qwen_distribution_signature": "qwen-signature",
				"flux2_policy_digest": "flux-digest",
				"l3_policy_digest": "l3-digest",
				"utility_vector": [0.0, 0.0, 0.0],
				"action_statistics": {str(level): {"rollout_count": 3} for level in trainer.LABEL_LEVELS},
			}
		],
		"validation": [],
		"test": [],
	}
	processing = trainer._apply_candidate_quality_targets(
		tmp_path,
		rows,
		acceptable_epsilon=0.25,
		soft_target_temperature=0.5,
	)
	row = rows["train"][0]
	assert row["utility_vector"] == pytest.approx([4.0 / 3.0, 1.0, 0.0])
	# L1-L2 has a positive mean only because of one outlier; the paired majority
	# points in the opposite direction, so it must not enter pairwise training.
	assert row["quality_pairwise_mask"] == [False, False, True]
	assert processing["schema"] == "vcot_v10_candidate_quality_paired_seed_v3_quality_priority"

	rollouts[0]["utility"] += 0.5
	rollout_path.write_text("".join(json.dumps(item) + "\n" for item in rollouts), encoding="utf-8")
	with pytest.raises(trainer.TrainingDataError, match="rollout_utilities.jsonl digest mismatch"):
		trainer._apply_candidate_quality_targets(
			tmp_path,
			copy.deepcopy(rows),
			acceptable_epsilon=0.25,
			soft_target_temperature=0.5,
		)


def test_empirical_bayes_is_train_only_and_cannot_reenable_unreliable_pairs() -> None:
	def make_row(offset: float, mask: list[bool]) -> dict:
		vectors = [
			[2.0 + offset, 1.0, 0.0 - offset],
			[0.0 + offset, 1.0, 2.0 - offset],
			[2.0 + offset, 1.0, 0.0 - offset],
		]
		means = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(3)]
		return {
			"quality_rollout_vectors": vectors,
			"quality_pairwise_mask": mask,
			"utility_vector": means,
			"soft_targets": trainer._softmax_vector(means, 0.5),
		}

	base = {
		"train": [
			make_row(-0.4, [False, True, False]),
			make_row(0.0, [False, True, False]),
			make_row(0.4, [False, True, False]),
		],
		"validation": [make_row(20.0, [True, True, True])],
		"test": [make_row(-20.0, [True, True, True])],
	}
	rows_a = copy.deepcopy(base)
	validation_before = copy.deepcopy(rows_a["validation"])
	test_before = copy.deepcopy(rows_a["test"])
	parameters_a = trainer._apply_empirical_bayes_l2_shrinkage(
		rows_a,
		acceptable_epsilon=0.25,
		soft_target_temperature=0.5,
	)
	assert rows_a["validation"] == validation_before
	assert rows_a["test"] == test_before
	assert all(not row["quality_pairwise_mask"][0] for row in rows_a["train"])
	assert all(not row["quality_pairwise_mask"][2] for row in rows_a["train"])

	rows_b = copy.deepcopy(base)
	rows_b["validation"][0] = make_row(2000.0, [True, True, True])
	rows_b["test"][0] = make_row(-2000.0, [True, True, True])
	parameters_b = trainer._apply_empirical_bayes_l2_shrinkage(
		rows_b,
		acceptable_epsilon=0.25,
		soft_target_temperature=0.5,
	)
	assert parameters_a == parameters_b
	assert [row["utility_vector"] for row in rows_a["train"]] == [
		row["utility_vector"] for row in rows_b["train"]
	]


def test_exact_quality_ties_only_downweight_quality_not_base_reliability() -> None:
	rows = [
		{
			"utility_vector": [1.0, 1.0, 1.0],
			"unshrunk_quality_pairwise_mask": [False, True, False],
			"centered_utility_vector": [0.0, 0.0, 0.0],
			"soft_targets": [1.0 / 3.0] * 3,
			"confidence": 1.0,
			"sample_weight": 2.0,
		},
		{
			"utility_vector": [1.0, 2.0, 1.0],
			"centered_utility_vector": [-1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0],
			"soft_targets": [0.1, 0.8, 0.1],
			"confidence": 1.0,
			"sample_weight": 2.0,
		},
	]
	tensors = trainer._row_tensors(rows, exact_tie_quality_weight=0.1)
	assert tensors["sample_weight"].tolist() == pytest.approx([2.0, 2.0])
	assert tensors["quality_row_weight"].tolist() == pytest.approx([0.1, 1.0])
	assert tensors["safe_pairwise_mask"][0].tolist() == pytest.approx([0.0, 1.0, 0.0])
	assert torch.allclose(
		tensors["efficiency_regret_targets"],
		torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
	)


def test_diagnosis_aux_features_match_training_and_online_contract() -> None:
	count_diagnosis = _diagnosis()
	attribute_diagnosis = copy.deepcopy(count_diagnosis)
	attribute_diagnosis["issues"] = [
		{
			**attribute_diagnosis["issues"][0],
			"constraint_type": "attribute",
			"confidence": 0.75,
		},
		{
			**attribute_diagnosis["issues"][0],
			"constraint_type": "quality",
			"confidence": 0.25,
		},
	]
	attribute_diagnosis["preservation_constraints"] = []
	assert diagnosis_aux_vector(count_diagnosis) == pytest.approx(
		[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.95]
	)
	assert diagnosis_aux_vector(attribute_diagnosis) == pytest.approx(
		[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.5]
	)

	features = {
		"train": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
		"validation": torch.tensor([[5.0, 6.0]]),
		"test": torch.tensor([[7.0, 8.0]]),
	}
	rows = {
		"train": [
			{"diagnosis_text": canonical_diagnosis_text(count_diagnosis)},
			{"diagnosis_text": canonical_diagnosis_text(attribute_diagnosis)},
		],
		"validation": [{"diagnosis_text": canonical_diagnosis_text(attribute_diagnosis)}],
		"test": [{"diagnosis_text": canonical_diagnosis_text(count_diagnosis)}],
	}
	augmented, contract = trainer._append_training_diagnosis_aux_features(features, rows)
	assert augmented["train"].shape == (2, 2 + len(DIAGNOSIS_AUX_FEATURE_NAMES))
	assert augmented["validation"].shape == (1, 2 + len(DIAGNOSIS_AUX_FEATURE_NAMES))
	config = {
		"input_dim": 2 + len(DIAGNOSIS_AUX_FEATURE_NAMES),
		"qwen_feature_dim": 2,
		"diagnosis_aux_features_enabled": True,
		"diagnosis_aux_contract": contract,
		"diagnosis_aux_contract_digest": trainer._sha256_text(trainer._canonical_json(contract)),
	}
	for split in trainer.SPLITS:
		online = append_diagnosis_aux_features(
			features[split],
			[row["diagnosis_text"] for row in rows[split]],
			config,
		)
		assert torch.equal(online, augmented[split])

	tampered = copy.deepcopy(config)
	tampered["diagnosis_aux_contract"]["normalization"]["mean"][0] += 1.0
	with pytest.raises(ValueError, match="contract digest mismatch"):
		append_diagnosis_aux_features(
			features["validation"],
			[rows["validation"][0]["diagnosis_text"]],
			tampered,
		)

	string_stat = copy.deepcopy(config)
	string_stat["diagnosis_aux_contract"]["normalization"]["mean"][0] = "0.0"
	string_stat["diagnosis_aux_contract_digest"] = trainer._sha256_text(
		trainer._canonical_json(string_stat["diagnosis_aux_contract"])
	)
	with pytest.raises(ValueError, match="statistics must be numeric"):
		append_diagnosis_aux_features(
			features["validation"],
			[rows["validation"][0]["diagnosis_text"]],
			string_stat,
		)

	wrong_extractor = copy.deepcopy(config)
	wrong_extractor["diagnosis_aux_contract"]["extractor_source_sha256"] = "0" * 64
	wrong_extractor["diagnosis_aux_contract_digest"] = trainer._sha256_text(
		trainer._canonical_json(wrong_extractor["diagnosis_aux_contract"])
	)
	with pytest.raises(ValueError, match="extractor source mismatch"):
		append_diagnosis_aux_features(
			features["validation"],
			[rows["validation"][0]["diagnosis_text"]],
			wrong_extractor,
		)

	non_boolean_enabled = copy.deepcopy(config)
	non_boolean_enabled["diagnosis_aux_features_enabled"] = "false"
	with pytest.raises(ValueError, match="must be boolean"):
		append_diagnosis_aux_features(
			features["validation"],
			[rows["validation"][0]["diagnosis_text"]],
			non_boolean_enabled,
		)


def test_cli_defaults_and_pinned_qwen_parameters(tmp_path: Path) -> None:
	revision = "b" * 40
	model_path = tmp_path / "snapshots" / revision
	model_path.mkdir(parents=True)
	args = trainer.parse_args(
		[
			"extract",
			"--dataset-root", str(tmp_path),
			"--output-dir", str(tmp_path / "out"),
			"--qwen-model", str(model_path),
			"--qwen-revision", revision,
		]
	)
	assert args.qwen_dtype == "bfloat16"
	assert args.qwen_quantization == "8bit"
	assert args.gpu_indices == "0,1,2,3"
	assert args.allow_legacy_l3_identity_ablation is False
	trainer.validate_pinned_qwen_identity(str(model_path), revision)
	alias = tmp_path / "snapshots" / "main"
	alias.mkdir()
	with pytest.raises(trainer.TrainingDataError, match="snapshots/<qwen_revision>"):
		trainer.validate_pinned_qwen_identity(str(alias), revision)
	contract = trainer.build_feature_contract(
		qwen_model=str(model_path),
		qwen_revision=revision,
		processor_metadata={},
		encoder_source_digest="0" * 64,
		qwen_dtype="bf16",
		qwen_quantization="int8",
		qwen_distribution_signature=_qwen_distribution_signature(),
	)
	assert contract["qwen_dtype"] == "bfloat16"
	assert contract["qwen_quantization"] == "8bit"
	assert contract["qwen_allow_cpu_offload"] is False
	assert contract["qwen_effective_quantization"] == "int8"


def test_contract_rejects_route_leakage_and_bad_pairwise(tmp_path: Path) -> None:
	_write_bundle(tmp_path)
	train_path = tmp_path / "derived" / "train.jsonl"
	rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
	diagnosis = json.loads(rows[0]["diagnosis_text"])
	diagnosis["route"] = "L1"
	rows[0]["diagnosis_text"] = json.dumps(diagnosis, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
	train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
	with pytest.raises(trainer.TrainingDataError, match="diagnosis_text"):
		trainer.load_dataset_bundle(tmp_path)

	_write_bundle(tmp_path)
	rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
	rows[0]["pairwise_preferences"][0]["preference"] = -1
	train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
	with pytest.raises(trainer.TrainingDataError, match="pairwise_preferences"):
		trainer.load_dataset_bundle(tmp_path)


def test_cpu_cached_features_use_shared_loss_and_checkpoint_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	_write_bundle(tmp_path)
	bundle = trainer.load_dataset_bundle(tmp_path)
	feature_cache = tmp_path / "features.pt"
	generator = torch.Generator().manual_seed(9)
	features = {
		split: torch.randn((len(rows), 6), generator=generator)
		for split, rows in bundle.rows.items()
	}
	contract = trainer._static_test_feature_contract()
	trainer.save_feature_cache(feature_cache, bundle, features, contract)

	actual_loss = trainer.utility_aware_router_loss
	calls: list[list[float]] = []

	def recording_loss(*args, **kwargs):
		calls.append(kwargs["class_weight"].detach().cpu().tolist())
		return actual_loss(*args, **kwargs)

	monkeypatch.setattr(trainer, "utility_aware_router_loss", recording_loss)
	output = tmp_path / "training"
	train_args = _train_args(tmp_path, output, feature_cache)
	train_args.diagnosis_aux_features = True
	summary = trainer.train_head(train_args)
	assert summary["best_epoch"] >= 1
	assert summary["evaluated_splits"] == ["train", "validation"]
	assert "test" not in summary["metrics"]
	assert calls
	assert calls[0] == pytest.approx([1.2, 0.8, 1.1])

	checkpoint_path = output / "model" / "route_head.pt"
	payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	config = payload["config"]
	assert config["diagnosis_schema"] == DIAGNOSIS_SCHEMA
	assert config["action_policy_version"] == ROUTER_ACTION_POLICY_VERSION
	assert config["repair_executor_family"] == ROUTER_REPAIR_EXECUTOR_FAMILY
	assert config["flux2_policy_contract"] == bundle.flux2_policy_contract
	assert config["flux2_policy_digest"] == bundle.flux2_policy_digest
	assert config["l3_policy_contract"] == bundle.l3_policy_contract
	assert config["l3_policy_digest"] == bundle.l3_policy_digest
	assert config["allow_legacy_l3_identity_ablation"] is False
	assert config["l3_training_initial_identity_binding_counts"] == {"exact_model_identity": 8}
	assert config["feature_schema"] == trainer.FEATURE_SCHEMA
	assert config["feature_schema_version"] == trainer.FEATURE_SCHEMA_VERSION
	assert config["label_levels"] == [1, 2, 3]
	assert config["qwen_model"] == "/test/qwen"
	assert config["qwen_revision"] == "a" * 40
	assert config["qwen_dtype"] == "bfloat16"
	assert config["qwen_quantization"] == "8bit"
	assert config["qwen_feature_dim"] == 6
	assert config["input_dim"] == 6 + len(DIAGNOSIS_AUX_FEATURE_NAMES)
	assert config["diagnosis_aux_features_enabled"] is True
	assert config["diagnosis_aux_contract"]["extractor_source_sha256"]
	assert len(config["diagnosis_aux_contract_digest"]) == 64
	for key in (
		"feature_contract_digest",
		"processor_chat_template_preprocess_pooling_contract_digest",
		"processor_contract_digest",
		"chat_template_contract_digest",
		"preprocess_contract_digest",
		"pooling_contract_digest",
	):
		assert len(config[key]) == 64
	head = RouteMLP(config["input_dim"], config["hidden_dim"], 3, config["dropout"])
	head.load_state_dict(payload["state_dict"])
	assert isinstance(head, RouteMLP)
	online_features = append_diagnosis_aux_features(
		features["validation"][:1],
		[bundle.rows["validation"][0]["diagnosis_text"]],
		config,
	)
	assert online_features.shape == (1, config["input_dim"])
	assert head(online_features).shape == (1, 3)


def test_feature_cache_is_bound_to_aggregate_rows(tmp_path: Path) -> None:
	_write_bundle(tmp_path)
	bundle = trainer.load_dataset_bundle(tmp_path)
	cache = tmp_path / "features.pt"
	features = {split: torch.ones((len(rows), 4)) for split, rows in bundle.rows.items()}
	trainer.save_feature_cache(cache, bundle, features, trainer._static_test_feature_contract())
	payload = torch.load(cache, map_location="cpu", weights_only=False)
	payload["splits"]["train"]["input_digests"][0] = "0" * 64
	torch.save(payload, cache)
	with pytest.raises(trainer.TrainingDataError, match="row identities"):
		trainer.load_feature_cache(cache, bundle)


def test_legacy_l3_identity_is_rejected_by_default_and_requires_explicit_ablation(tmp_path: Path) -> None:
	_write_bundle(tmp_path)
	derived = tmp_path / "derived"
	legacy_l3_policy = build_l3_policy_contract(
		family="bagel",
		model_id="vcot/bagel-base",
		model_revision="d" * 40,
		auxiliary_model_id="vcot/bagel-ft",
		auxiliary_model_revision="e" * 40,
		resolution=1024,
		steps=50,
		guidance_scale=4.0,
		max_sequence_length=None,
		max_latent_size=64,
		dtype="bfloat16",
		prompt_version=L3_PROMPT_VERSION,
	)
	legacy_l3_digest = l3_policy_digest(legacy_l3_policy)
	row_count = 0
	for split in trainer.SPLITS:
		path = derived / f"{split}.jsonl"
		rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
		for row in rows:
			row["l3_policy_contract"] = legacy_l3_policy
			row["l3_policy_digest"] = legacy_l3_digest
			row["initial_generator_family"] = "bagel"
			row["l3_initial_identity_binding"] = "legacy_bagel_family_only_unverified_revision"
		row_count += len(rows)
		path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
	summary_path = derived / "summary.json"
	summary = json.loads(summary_path.read_text(encoding="utf-8"))
	summary["l3_policy_contract"] = legacy_l3_policy
	summary["l3_policy_digest"] = legacy_l3_digest
	summary["initial_generator_family"] = "bagel"
	summary["l3_initial_identity_binding_counts"] = {
		"legacy_bagel_family_only_unverified_revision": row_count,
	}
	summary_path.write_text(json.dumps(summary), encoding="utf-8")
	with pytest.raises(trainer.TrainingDataError, match="main training requires exact model identity"):
		trainer.load_dataset_bundle(tmp_path)
	bundle = trainer.load_dataset_bundle(tmp_path, allow_legacy_l3_identity_ablation=True)
	assert bundle.allow_legacy_l3_identity_ablation is True
	assert bundle.l3_initial_identity_binding_counts == {
		"legacy_bagel_family_only_unverified_revision": row_count,
	}
	args = trainer.parse_args(
		[
			"validate-data",
			"--dataset-root", str(tmp_path),
			"--output-dir", str(tmp_path / "out"),
			"--allow-legacy-l3-identity-ablation",
		]
	)
	assert args.allow_legacy_l3_identity_ablation is True
	feature_cache = tmp_path / "legacy_features.pt"
	features = {
		split: torch.ones((len(rows), 4), dtype=torch.float32)
		for split, rows in bundle.rows.items()
	}
	trainer.save_feature_cache(feature_cache, bundle, features, trainer._static_test_feature_contract())
	train_args = _train_args(tmp_path, tmp_path / "legacy_training", feature_cache)
	train_args.allow_legacy_l3_identity_ablation = True
	trainer.train_head(train_args)
	checkpoint = torch.load(
		tmp_path / "legacy_training" / "model" / "route_head.pt",
		map_location="cpu",
		weights_only=False,
	)["config"]
	assert checkpoint["allow_legacy_l3_identity_ablation"] is True
	assert checkpoint["l3_training_initial_identity_binding_counts"] == {
		"legacy_bagel_family_only_unverified_revision": row_count,
	}
	from inference.main_method.modules.route_head import load_route_head

	with pytest.raises(ValueError, match="offline-ablation-only"):
		load_route_head(tmp_path / "legacy_training" / "model" / "route_head.pt", "cpu")


def test_flux2_collector_policy_is_bound_to_summary_rows_and_checkpoint_input(tmp_path: Path) -> None:
	_write_bundle(tmp_path)
	train_path = tmp_path / "derived" / "train.jsonl"
	rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
	rows[0]["flux2_policy_digest"] = "f" * 64
	train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
	with pytest.raises(trainer.TrainingDataError, match="flux2_policy_digest"):
		trainer.load_dataset_bundle(tmp_path)

	_write_bundle(tmp_path)
	summary_path = tmp_path / "derived" / "summary.json"
	summary = json.loads(summary_path.read_text(encoding="utf-8"))
	summary["flux2_policy_contract"]["generation"]["steps"] = 29
	summary_path.write_text(json.dumps(summary), encoding="utf-8")
	with pytest.raises(trainer.TrainingDataError, match="collector-policy digest mismatch"):
		trainer.load_dataset_bundle(tmp_path)


def test_qwen_distribution_is_bound_to_rows_cache_and_effective_quantization(tmp_path: Path) -> None:
	_write_bundle(tmp_path)
	train_path = tmp_path / "derived" / "train.jsonl"
	rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
	rows[0]["qwen_distribution_signature"] = "f" * 64
	train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
	with pytest.raises(trainer.TrainingDataError, match="qwen_distribution_signature"):
		trainer.load_dataset_bundle(tmp_path)

	_write_bundle(tmp_path)
	bundle = trainer.load_dataset_bundle(tmp_path)
	cache = tmp_path / "qwen-bound.pt"
	features = {split: torch.ones((len(split_rows), 4)) for split, split_rows in bundle.rows.items()}
	contract = trainer._static_test_feature_contract(
		qwen_distribution_signature=bundle.qwen_distribution_signature
	)
	trainer.save_feature_cache(cache, bundle, features, contract)
	payload = torch.load(cache, map_location="cpu", weights_only=False)
	payload["qwen_distribution_signature"] = "e" * 64
	torch.save(payload, cache)
	with pytest.raises(trainer.TrainingDataError, match="Qwen distribution"):
		trainer.load_feature_cache(cache, bundle)

	offload_contract = trainer.build_feature_contract(
		qwen_model="/test/qwen",
		qwen_revision="a" * 40,
		processor_metadata={},
		encoder_source_digest="0" * 64,
		qwen_quantization="4bit",
		qwen_allow_cpu_offload=True,
		qwen_distribution_signature="d" * 64,
	)
	assert offload_contract["qwen_quantization"] == "nf4"
	assert offload_contract["qwen_effective_quantization"] == "int8_cpu_offload"
