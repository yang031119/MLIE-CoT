from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import scripts.train_traditional_route_baselines as traditional


def _diagnosis(constraint_type: str, *, expected: str = "two red cups") -> str:
	return json.dumps(
		{
			"reason": "one requested constraint is not satisfied",
			"constraint_audit": {
				"objects": "one cup is visible",
				"count": "one instead of two",
			},
			"issues": [
				{
					"constraint_type": constraint_type,
					"target_entity": "cups",
					"observed": "one blue cup",
					"expected": expected,
					"evidence": "one silhouette is visible",
					"confidence": 0.9,
				}
			],
			"preservation_constraints": ["preserve the table"],
		},
		ensure_ascii=False,
		sort_keys=True,
		separators=(",", ":"),
	)


def _feature_row(
	state_id: str,
	image_path: Path,
	*,
	constraint_type: str,
	prompt: str,
) -> dict:
	return {
		"state_id": state_id,
		"feature_input_digest": f"digest-{state_id}",
		"target_prompt": prompt,
		"diagnosis_text": _diagnosis(constraint_type),
		"image_path": str(image_path),
	}


def _feature_bundle(tmp_path: Path, *, changed_eval: bool = False) -> SimpleNamespace:
	tmp_path.mkdir(parents=True, exist_ok=True)
	image_a = tmp_path / "a.png"
	image_b = tmp_path / "b.png"
	Image.new("RGB", (20, 12), (220, 20, 40)).save(image_a)
	Image.new("RGB", (20, 12), (20, 40, 220) if changed_eval else (30, 180, 60)).save(image_b)
	rows = {
		"train": [
			_feature_row("train-0", image_a, constraint_type="count", prompt="two red cups on a table"),
			_feature_row("train-1", image_a, constraint_type="attribute", prompt="one blue cup on a table"),
		],
		"validation": [
			_feature_row(
				"validation-0",
				image_b,
				constraint_type="scene" if changed_eval else "count",
				prompt="a completely different evaluation prompt" if changed_eval else "three cups on a table",
			)
		],
		"test": [
			_feature_row(
				"test-0",
				image_b,
				constraint_type="relation",
				prompt="test text changes" if changed_eval else "a cup beside a plate",
			)
		],
	}
	return SimpleNamespace(rows=rows, qwen_distribution_signature="a" * 64)


def _metric_row(
	state_id: str,
	utility: list[float],
	*,
	diagnosis_type: str,
) -> dict:
	return {
		"state_id": state_id,
		"feature_input_digest": f"digest-{state_id}",
		"target_prompt": f"prompt for {state_id}",
		"diagnosis_text": _diagnosis(diagnosis_type),
		"image_path": "/not/read/by/synthetic/features.png",
		"utility_vector": utility,
		"unshrunk_candidate_quality_vector": list(utility),
		"confidence": 1.0,
		"sample_weight": 1.0,
		"action_statistics": {
			"1": {"success_rate": float(utility[0] > 0.5)},
			"2": {"success_rate": float(utility[1] > 0.5)},
			"3": {"success_rate": float(utility[2] > 0.5)},
		},
	}


def _search_bundle() -> SimpleNamespace:
	rows = {
		"train": [
			_metric_row("train-0", [1.2, 0.4, 0.1], diagnosis_type="count"),
			_metric_row("train-1", [0.2, 1.0, 0.3], diagnosis_type="attribute"),
			_metric_row("train-2", [0.1, 0.4, 1.1], diagnosis_type="scene"),
			_metric_row("train-3", [0.3, 1.1, 0.2], diagnosis_type="relation"),
		],
		"validation": [
			_metric_row("validation-0", [1.0, 0.2, 0.1], diagnosis_type="count"),
			_metric_row("validation-1", [0.1, 0.9, 0.2], diagnosis_type="attribute"),
		],
		"test": [
			_metric_row("test-0", [0.1, 0.2, 1.0], diagnosis_type="scene"),
		],
	}
	return SimpleNamespace(
		rows=rows,
		digest="dataset-digest",
		target_digest="target-digest",
		target_mode="candidate_quality",
		train_target_shrinkage="diagnosis_group_eb_l2",
		qwen_distribution_signature="a" * 64,
		acceptable_epsilon=0.25,
	)


def _synthetic_features(bundle: SimpleNamespace) -> traditional.FeatureBlocks:
	matrices: dict[str, dict[str, np.ndarray]] = {}
	for split, rows in bundle.rows.items():
		base = np.arange(len(rows) * 3, dtype=np.float64).reshape(len(rows), 3)
		matrices[split] = {
			"diagnosis_structured": base + 0.5,
			"prompt_tfidf": base + 1.5,
			"diagnosis_tfidf": base + 2.5,
			"image_handcrafted": base + 3.5,
		}
	contract = {
		"schema": traditional.FEATURE_CONTRACT_SCHEMA,
		"uses_qwen_route_features": False,
		"route_feature_backbone": "none",
		"fitted_on_split": "train",
		"blocks": {name: {"dimension": 3} for name in matrices["train"]},
	}
	contract["feature_contract_digest"] = traditional._sha256_json(contract)
	return traditional.FeatureBlocks(matrices=matrices, contract=contract, build_seconds=0.01)


def _search_args(output_dir: Path) -> argparse.Namespace:
	return argparse.Namespace(
		output_dir=output_dir,
		dataset_root=Path("/synthetic"),
		skip_image_hash_check=True,
		train_target_shrinkage="diagnosis_group_eb_l2",
		families="diagnosis_structured_ridge",
		max_tfidf_features=32,
		tfidf_min_df=1,
		ridge_alphas="1",
		lookup_priors="1",
		escape_thresholds="0,0.25",
		efficiency_costs=(3.0, 2.0, 1.0),
		bootstrap_seed=7,
		bootstrap_samples=20,
	)


def test_feature_contract_is_train_only_and_declares_no_qwen_route_features(tmp_path: Path) -> None:
	first_bundle = _feature_bundle(tmp_path / "first")
	second_bundle = _feature_bundle(tmp_path / "second", changed_eval=True)

	first = traditional.build_feature_blocks(first_bundle, max_tfidf_features=64, tfidf_min_df=1)
	second = traditional.build_feature_blocks(second_bundle, max_tfidf_features=64, tfidf_min_df=1)

	assert first.contract["uses_qwen_route_features"] is False
	assert first.contract["route_feature_backbone"] == "none"
	assert first.contract["fitted_on_split"] == "train"
	assert first.contract == second.contract
	for block in first.matrices["train"]:
		assert first.matrices["train"][block] == pytest.approx(second.matrices["train"][block])
	assert not np.array_equal(first.matrices["validation"]["image_handcrafted"], second.matrices["validation"]["image_handcrafted"])


def test_tfidf_ridge_and_safe_route_are_deterministic() -> None:
	vectorizer = traditional.TfidfLite.fit(
		["two red cups", "one blue cup", "red cup beside plate"], min_df=1, max_features=32
	)
	features = vectorizer.transform(["two red cups", "one blue cup", "red cup beside plate"])
	targets = np.asarray([[1.0, -0.8, -0.2], [-0.3, -0.5, 0.8], [0.5, -0.9, 0.4]])
	weights = np.asarray([1.0, 1.0, 0.5])
	first = traditional.fit_weighted_ridge(features, targets, weights, alpha=0.1)
	second = traditional.fit_weighted_ridge(features, targets, weights, alpha=0.1)

	assert first == second
	deltas = traditional.predict_weighted_ridge(first, features)
	assert traditional.route_from_deltas(
		np.asarray([[0.2, 0.1], [0.2, 0.4], [-0.1, -0.2]]),
		escape_threshold=0.25,
		efficiency_costs=(3.0, 2.0, 1.0),
	) == [1, 2, 1]
	assert np.isfinite(deltas).all()


def test_search_is_reproducible_and_does_not_evaluate_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
	bundle = _search_bundle()
	features = _synthetic_features(bundle)
	monkeypatch.setattr(traditional, "load_dataset_bundle", lambda *args, **kwargs: bundle)
	monkeypatch.setattr(traditional, "build_feature_blocks", lambda *args, **kwargs: features)

	first = traditional.search(_search_args(tmp_path / "run-a"))
	second = traditional.search(_search_args(tmp_path / "run-b"))
	first_manifest = json.loads((tmp_path / "run-a" / "selection_manifest.json").read_text(encoding="utf-8"))
	second_manifest = json.loads((tmp_path / "run-b" / "selection_manifest.json").read_text(encoding="utf-8"))
	first_model = json.loads((tmp_path / "run-a" / "selected_model.json").read_text(encoding="utf-8"))
	second_model = json.loads((tmp_path / "run-b" / "selected_model.json").read_text(encoding="utf-8"))

	assert first["selected"] == second["selected"]
	assert first_manifest == second_manifest
	assert first_model == second_model
	assert first_manifest["evaluated_splits"] == ["train", "validation"]
	assert first_manifest["test_gate"] is False
	assert first_manifest["test_labels_used_for_selection"] is False
	assert not (tmp_path / "run-a" / "evaluation_test.json").exists()
	assert not (tmp_path / "run-a" / "predictions_test.jsonl").exists()


def test_feature_cache_rejects_contract_and_array_tampering(tmp_path: Path) -> None:
	bundle = _search_bundle()
	features = _synthetic_features(bundle)
	run_dir = tmp_path / "run"
	traditional.save_feature_cache(run_dir, bundle, features)
	loaded = traditional.load_feature_cache(run_dir, bundle)
	assert loaded.contract == features.contract

	manifest_path = run_dir / "feature_manifest.json"
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	manifest["dataset_digest"] = "wrong"
	manifest_path.write_text(json.dumps(manifest))
	with pytest.raises(traditional.TraditionalRouteError, match="dataset digest"):
		traditional.load_feature_cache(run_dir, bundle)

	traditional.save_feature_cache(run_dir, bundle, features)
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	manifest["feature_contract"]["route_feature_backbone"] = "qwen"
	manifest_path.write_text(json.dumps(manifest))
	with pytest.raises(traditional.TraditionalRouteError, match="contract digest"):
		traditional.load_feature_cache(run_dir, bundle)


def test_test_metrics_require_explicit_gate(tmp_path: Path) -> None:
	args = argparse.Namespace(
		split="test",
		evaluate_test=False,
		run_dir=tmp_path,
		dataset_root=tmp_path,
		skip_image_hash_check=True,
		bootstrap_seed=1,
		bootstrap_samples=0,
		latency_repeats=1,
	)
	with pytest.raises(traditional.TraditionalRouteError, match="explicit --evaluate-test gate"):
		traditional.evaluate(args)


def test_family_best_refit_requires_locked_validation_prediction_digest() -> None:
	bundle = _search_bundle()
	features = _synthetic_features(bundle)
	targets, weights = traditional._training_targets(bundle.rows["train"])
	model = traditional.fit_weighted_ridge(
		traditional._family_matrix(features, "train", ("diagnosis_structured",)),
		targets,
		weights,
		alpha=1.0,
	)
	values = traditional.predict_weighted_ridge(
		model,
		traditional._family_matrix(features, "validation", ("diagnosis_structured",)),
	)
	deltas = np.column_stack((values[:, 0] - values[:, 1], values[:, 2] - values[:, 1]))
	predictions = traditional.route_from_deltas(
		deltas,
		escape_threshold=0.0,
		efficiency_costs=(3.0, 2.0, 1.0),
	)
	record = {
		"regularization": 1.0,
		"escape_threshold": 0.0,
		"prediction_digest": traditional._sha256_json(predictions),
	}

	locked = traditional._locked_family_model(
		"diagnosis_structured_ridge",
		record,
		bundle=bundle,
		features=features,
		efficiency_costs=(3.0, 2.0, 1.0),
	)
	assert locked["locked_validation_prediction_digest"] == record["prediction_digest"]

	bad_record = dict(record, prediction_digest="0" * 64)
	with pytest.raises(traditional.TraditionalRouteError, match="no longer reproduces"):
		traditional._locked_family_model(
			"diagnosis_structured_ridge",
			bad_record,
			bundle=bundle,
			features=features,
			efficiency_costs=(3.0, 2.0, 1.0),
		)


def test_diagnosis_signature_ignores_free_text_and_order() -> None:
	left = {"diagnosis_text": _diagnosis("count")}
	payload = json.loads(_diagnosis("attribute"))
	payload["reason"] = "completely different prose"
	payload["issues"].append(copy.deepcopy(json.loads(_diagnosis("count"))["issues"][0]))
	right = {"diagnosis_text": json.dumps(payload)}
	assert traditional.diagnosis_signature(left) == ("count",)
	assert traditional.diagnosis_signature(right) == ("attribute", "count")
