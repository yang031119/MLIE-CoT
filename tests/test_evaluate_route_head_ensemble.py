from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from inference.main_method.modules.route_head import (
	DIAGNOSIS_AUX_FEATURE_NAMES,
	RouteMLP,
	build_diagnosis_aux_contract,
)
import scripts.evaluate_route_head_ensemble as evaluator


def _canonical_digest(value: object) -> str:
	text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _diagnosis(*, constraint_type: str = "count", confidence: float = 0.75) -> str:
	return json.dumps(
		{
			"issues": [
				{
					"constraint_type": constraint_type,
					"confidence": confidence,
				}
			],
			"preservation_constraints": ["preserve the table", "preserve the existing cup"],
		},
		ensure_ascii=False,
		sort_keys=True,
		separators=(",", ":"),
	)


def _aux_config(*, mean: float = 0.0) -> dict:
	feature_count = len(DIAGNOSIS_AUX_FEATURE_NAMES)
	contract = build_diagnosis_aux_contract(
		[mean] * feature_count,
		[1.0] * feature_count,
		std_floor=0.1,
	)
	return {
		"input_dim": 2 + feature_count,
		"qwen_feature_dim": 2,
		"hidden_dim": 4,
		"num_labels": 3,
		"dropout": 0.0,
		"label_levels": [1, 2, 3],
		"diagnosis_aux_features_enabled": True,
		"diagnosis_aux_contract": contract,
		"diagnosis_aux_contract_digest": _canonical_digest(contract),
	}


def _legacy_config() -> dict:
	return {
		"input_dim": 2,
		"hidden_dim": 4,
		"num_labels": 3,
		"dropout": 0.0,
		"label_levels": [1, 2, 3],
	}


def test_checkpoint_features_append_normalized_diagnosis_aux_values() -> None:
	config = _aux_config()
	pooled = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
	rows = [
		{"diagnosis_text": _diagnosis(constraint_type="count", confidence=0.75)},
		{"diagnosis_text": _diagnosis(constraint_type="quality", confidence=0.25)},
	]

	features = evaluator._checkpoint_features(pooled, rows, config)

	assert features.shape == (2, 2 + len(DIAGNOSIS_AUX_FEATURE_NAMES))
	assert features[:, :2] == pytest.approx(pooled)
	assert features[0, 2:].tolist() == pytest.approx(
		[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.75]
	)
	assert features[1, 2:].tolist() == pytest.approx(
		[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.25]
	)


def test_checkpoint_features_keep_legacy_non_aux_inputs_unchanged() -> None:
	pooled = torch.tensor([[1.0, 2.0]])
	result = evaluator._checkpoint_features(
		pooled,
		[{"diagnosis_text": _diagnosis()}],
		_legacy_config(),
	)
	assert torch.equal(result, pooled)


def test_ensemble_rejects_different_diagnosis_aux_normalization() -> None:
	first = _aux_config(mean=0.0)
	second = _aux_config(mean=0.5)
	with pytest.raises(ValueError, match="diagnosis_aux_contract"):
		evaluator._require_compatible([first, second])


def test_checkpoint_rejects_invalid_aux_digest_and_feature_cache_dimension() -> None:
	config = _aux_config()
	config["diagnosis_aux_contract_digest"] = "0" * 64
	with pytest.raises(ValueError, match="contract digest mismatch"):
		evaluator._checkpoint_qwen_feature_dim(config)

	config = _aux_config()
	with pytest.raises(ValueError, match="feature cache tensor differs"):
		evaluator._checkpoint_features(
			torch.zeros(1, 3),
			[{"diagnosis_text": _diagnosis()}],
			config,
		)


def test_load_model_constructs_aux_input_dimension(tmp_path: Path) -> None:
	config = _aux_config()
	model = RouteMLP(
		input_dim=config["input_dim"],
		hidden_dim=config["hidden_dim"],
		num_labels=config["num_labels"],
		dropout=config["dropout"],
	)
	checkpoint = tmp_path / "route_head.pt"
	torch.save({"config": copy.deepcopy(config), "state_dict": model.state_dict()}, checkpoint)

	loaded, loaded_config = evaluator._load_model(checkpoint)

	assert loaded_config == config
	assert loaded.net[0].normalized_shape == (config["input_dim"],)
	assert loaded(torch.zeros(2, config["input_dim"])).shape == (2, 3)
