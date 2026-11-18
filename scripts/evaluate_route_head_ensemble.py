#!/usr/bin/env python3
"""Evaluate a frozen mean-logit V10 route-head ensemble without test leakage."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from inference.main_method.modules.route_head import (
	RouteMLP,
	append_diagnosis_aux_features,
	validate_diagnosis_aux_contract,
)
from scripts.train_route_head import load_dataset_bundle, load_feature_cache


def _sha256_file(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _checkpoint_qwen_feature_dim(config: Mapping[str, Any]) -> int:
	"""Validate the checkpoint's base/auxiliary feature dimensions."""

	try:
		input_dim = int(config["input_dim"])
	except (KeyError, TypeError, ValueError) as exc:
		raise ValueError("route checkpoint input_dim must be a positive integer") from exc
	if input_dim <= 0:
		raise ValueError("route checkpoint input_dim must be a positive integer")
	aux_contract = validate_diagnosis_aux_contract(config)
	if aux_contract is None:
		try:
			qwen_feature_dim = int(config.get("qwen_feature_dim", input_dim))
		except (TypeError, ValueError) as exc:
			raise ValueError("route checkpoint qwen_feature_dim must be a positive integer") from exc
		if qwen_feature_dim != input_dim:
			raise ValueError("route checkpoint without diagnosis auxiliary features must use the Qwen feature dimension")
		return qwen_feature_dim
	try:
		qwen_feature_dim = int(config["qwen_feature_dim"])
	except (KeyError, TypeError, ValueError) as exc:
		raise ValueError("route checkpoint with diagnosis auxiliary features is missing qwen_feature_dim") from exc
	aux_dim = len(aux_contract["feature_names"])
	if qwen_feature_dim <= 0 or qwen_feature_dim + aux_dim != input_dim:
		raise ValueError("route checkpoint diagnosis auxiliary input dimensions are inconsistent")
	return qwen_feature_dim


def _load_model(path: Path) -> tuple[RouteMLP, dict[str, Any]]:
	payload = torch.load(path, map_location="cpu", weights_only=False)
	if not isinstance(payload, Mapping):
		raise ValueError(f"route checkpoint payload must be an object: {path}")
	config_value = payload.get("config")
	if not isinstance(config_value, Mapping):
		raise ValueError(f"route checkpoint is missing config metadata: {path}")
	config = dict(config_value)
	try:
		hidden_dim = int(config["hidden_dim"])
		num_labels = int(config["num_labels"])
		dropout = float(config["dropout"])
	except (KeyError, TypeError, ValueError) as exc:
		raise ValueError(f"route checkpoint model dimensions are invalid: {path}") from exc
	if hidden_dim <= 0 or num_labels != 3 or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
		raise ValueError(f"route checkpoint model dimensions are invalid: {path}")
	if [int(value) for value in config.get("label_levels") or []] != [1, 2, 3]:
		raise ValueError(f"route checkpoint label_levels must be [1,2,3]: {path}")
	_checkpoint_qwen_feature_dim(config)
	input_dim = int(config["input_dim"])
	model = RouteMLP(
		input_dim=input_dim,
		hidden_dim=hidden_dim,
		num_labels=num_labels,
		dropout=dropout,
	)
	state_dict = payload.get("state_dict")
	if not isinstance(state_dict, Mapping):
		raise ValueError(f"route checkpoint is missing state_dict: {path}")
	model.load_state_dict(state_dict)
	model.eval()
	return model, config


def _require_compatible(configs: Sequence[Mapping[str, Any]]) -> None:
	if not configs:
		raise ValueError("at least one checkpoint is required")
	required_equal = (
		"version",
		"input_dim",
		"hidden_dim",
		"num_labels",
		"dropout",
		"label_levels",
		"dataset_digest",
		"target_mode",
		"train_target_shrinkage",
		"target_digest",
		"feature_contract_digest",
		"qwen_distribution_signature",
		"flux2_policy_digest",
		"l3_policy_digest",
		"class_factors",
		"loss",
		"exact_tie_quality_weight",
	)
	reference = configs[0]
	reference_qwen_feature_dim = _checkpoint_qwen_feature_dim(reference)
	reference_aux_enabled = bool(reference.get("diagnosis_aux_features_enabled", False))
	reference_aux_contract = validate_diagnosis_aux_contract(reference)
	for index, config in enumerate(configs[1:], start=1):
		for key in required_equal:
			if config.get(key) != reference.get(key):
				raise ValueError(f"checkpoint {index} is incompatible for mean-logit ensemble: {key}")
		if _checkpoint_qwen_feature_dim(config) != reference_qwen_feature_dim:
			raise ValueError(f"checkpoint {index} is incompatible for mean-logit ensemble: qwen_feature_dim")
		if bool(config.get("diagnosis_aux_features_enabled", False)) != reference_aux_enabled:
			raise ValueError(
				f"checkpoint {index} is incompatible for mean-logit ensemble: diagnosis_aux_features_enabled"
			)
		if validate_diagnosis_aux_contract(config) != reference_aux_contract:
			raise ValueError(f"checkpoint {index} is incompatible for mean-logit ensemble: diagnosis_aux_contract")


def _checkpoint_features(
	features: torch.Tensor,
	rows: Sequence[Mapping[str, Any]],
	config: Mapping[str, Any],
) -> torch.Tensor:
	"""Build the exact MLP inputs represented by one compatible checkpoint."""

	qwen_feature_dim = _checkpoint_qwen_feature_dim(config)
	if features.ndim != 2 or features.shape != (len(rows), qwen_feature_dim):
		raise ValueError(
			"feature cache tensor differs from checkpoint Qwen feature dimensions: "
			f"actual={tuple(features.shape)} expected={(len(rows), qwen_feature_dim)}"
		)
	diagnoses: list[str] = []
	for index, row in enumerate(rows):
		diagnosis = row.get("diagnosis_text")
		if not isinstance(diagnosis, str) or not diagnosis.strip():
			raise ValueError(f"aggregate row {index} is missing canonical diagnosis_text")
		diagnoses.append(diagnosis)
	result = append_diagnosis_aux_features(features, diagnoses, config)
	if result.ndim != 2 or result.shape != (len(rows), int(config["input_dim"])):
		raise ValueError("checkpoint feature augmentation produced an invalid MLP input shape")
	if not torch.isfinite(result.float()).all():
		raise ValueError("checkpoint feature augmentation produced non-finite values")
	return result


def _mean(values: Sequence[float]) -> float:
	return sum(values) / len(values)


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int, samples: int) -> list[float]:
	if not values or samples <= 0:
		return []
	rng = random.Random(seed)
	n = len(values)
	means = sorted(_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(samples))
	return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def _policy_metrics(
	rows: Sequence[Mapping[str, Any]],
	predictions: Sequence[int],
	*,
	acceptable_epsilon: float,
	efficiency_costs: Sequence[float],
	bootstrap_seed: int,
	bootstrap_samples: int,
) -> dict[str, Any]:
	selected_quality: list[float] = []
	regrets: list[float] = []
	gains_vs_l2: list[float] = []
	acceptable: list[bool] = []
	success_rates: list[float] = []
	for row, prediction in zip(rows, predictions):
		quality = [float(value) for value in row["utility_vector"]]
		selected_quality.append(quality[prediction])
		regrets.append(max(quality) - quality[prediction])
		gains_vs_l2.append(quality[prediction] - quality[1])
		acceptable.append(regrets[-1] <= acceptable_epsilon + 1e-12)
		success_rates.append(float(row["action_statistics"][str(prediction + 1)]["success_rate"]))
	return {
		"rows": len(rows),
		"mean_quality_regret": _mean(regrets),
		"mean_selected_quality": _mean(selected_quality),
		"acceptable_quality_rate": _mean([float(value) for value in acceptable]),
		"mean_success_rate": _mean(success_rates),
		"mean_efficiency_cost": _mean([float(efficiency_costs[index]) for index in predictions]),
		"prediction_distribution": dict(sorted(Counter(f"L{index + 1}" for index in predictions).items())),
		"gain_vs_fixed_l2": {
			"mean": _mean(gains_vs_l2),
			"bootstrap_95pct_ci": _bootstrap_mean_ci(
				gains_vs_l2,
				seed=bootstrap_seed,
				samples=bootstrap_samples,
			),
			"wins": sum(value > 1e-9 for value in gains_vs_l2),
			"ties": sum(abs(value) <= 1e-9 for value in gains_vs_l2),
			"losses": sum(value < -1e-9 for value in gains_vs_l2),
		},
	}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset-root", type=Path, required=True)
	parser.add_argument("--feature-cache", type=Path, required=True)
	parser.add_argument("--checkpoint", type=Path, action="append", required=True)
	parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
	parser.add_argument("--evaluate-test", action="store_true")
	parser.add_argument("--skip-image-hash-check", action="store_true")
	parser.add_argument("--bootstrap-seed", type=int, default=20260722)
	parser.add_argument("--bootstrap-samples", type=int, default=10000)
	parser.add_argument("--efficiency-costs", type=float, nargs=3, default=(204.5771, 154.4044, 151.5316))
	parser.add_argument("--output", type=Path)
	return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	if args.split == "test" and not args.evaluate_test:
		raise ValueError("test evaluation requires the explicit --evaluate-test gate")
	models_and_configs = [_load_model(path) for path in args.checkpoint]
	models = [value[0] for value in models_and_configs]
	configs = [value[1] for value in models_and_configs]
	_require_compatible(configs)
	config = configs[0]
	bundle = load_dataset_bundle(
		args.dataset_root,
		verify_image_hashes=not args.skip_image_hash_check,
		target_mode=str(config["target_mode"]),
		train_target_shrinkage=str(config["train_target_shrinkage"]),
	)
	if bundle.digest != config["dataset_digest"] or bundle.target_digest != config["target_digest"]:
		raise ValueError("checkpoint dataset/target contract differs from current aggregate bundle")
	features, feature_contract = load_feature_cache(args.feature_cache, bundle)
	if feature_contract["feature_contract_digest"] != config["feature_contract_digest"]:
		raise ValueError("checkpoint feature contract differs from the feature cache")
	model_features = _checkpoint_features(features[args.split], bundle.rows[args.split], config)
	with torch.inference_mode():
		logits = torch.stack([model(model_features) for model in models]).mean(dim=0)
	predictions = logits.argmax(dim=-1).tolist()
	metrics = _policy_metrics(
		bundle.rows[args.split],
		predictions,
		acceptable_epsilon=bundle.acceptable_epsilon,
		efficiency_costs=args.efficiency_costs,
		bootstrap_seed=args.bootstrap_seed,
		bootstrap_samples=args.bootstrap_samples,
	)
	baselines = {
		f"fixed_L{level}": _policy_metrics(
			bundle.rows[args.split],
			[level - 1] * len(bundle.rows[args.split]),
			acceptable_epsilon=bundle.acceptable_epsilon,
			efficiency_costs=args.efficiency_costs,
			bootstrap_seed=args.bootstrap_seed,
			bootstrap_samples=args.bootstrap_samples,
		)
		for level in (1, 2, 3)
	}
	result = {
		"protocol": "vcot_v10_mean_logit_ensemble_static_eval_v2_diagnosis_aux",
		"split": args.split,
		"test_gate": bool(args.evaluate_test),
		"dataset_digest": bundle.digest,
		"target_digest": bundle.target_digest,
		"feature_contract_digest": config["feature_contract_digest"],
		"qwen_feature_dim": _checkpoint_qwen_feature_dim(config),
		"mlp_input_dim": int(config["input_dim"]),
		"diagnosis_aux_features_enabled": bool(config.get("diagnosis_aux_features_enabled", False)),
		"diagnosis_aux_contract_digest": config.get("diagnosis_aux_contract_digest"),
		"checkpoints": [
			{"path": str(path), "sha256": _sha256_file(path), "seed": configs[index].get("seed")}
			for index, path in enumerate(args.checkpoint)
		],
		"ensemble": metrics,
		"baselines": baselines,
	}
	text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
	if args.output is not None:
		args.output.parent.mkdir(parents=True, exist_ok=True)
		args.output.write_text(text, encoding="utf-8")
	print(text, end="")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
