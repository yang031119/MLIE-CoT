#!/usr/bin/env python3
"""Train reproducible V10 route baselines without Qwen hidden features.

This is a static ablation trainer, not an online V10 replacement.  It keeps the
same candidate-quality targets, train/validation/test split and policy metrics
as the frozen-Qwen route head, but the route encoder itself is deliberately
traditional:

* a train-only diagnosis-signature empirical-Bayes lookup;
* weighted ridge regression over low-dimensional diagnosis statistics;
* word TF-IDF over the target prompt and/or canonical diagnosis;
* handcrafted colour, layout, edge and entropy image features.

Some families consume the canonical diagnosis emitted by the upstream
analyser.  Those are accurately described as "without Qwen route features",
not as a Qwen-free end-to-end system.  The prompt/image-only families are also
reported so this distinction is measurable rather than rhetorical.

``search`` never evaluates the test split.  It fits every extractor on train,
selects family/regularisation/L2-escape threshold on validation, and writes a
locked model plus feature contract.  ``evaluate --split test`` requires an
explicit ``--evaluate-test`` gate and cannot change the selected model.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from inference.main_method.modules.route_head import (  # noqa: E402
	DIAGNOSIS_AUX_FEATURE_NAMES,
	diagnosis_aux_vector,
)
from scripts.evaluate_route_head_ensemble import _policy_metrics  # noqa: E402
from scripts.train_route_head import (  # noqa: E402
	DatasetBundle,
	load_dataset_bundle,
	load_raw_route_image,
)


SPLITS = ("train", "validation", "test")
FEATURE_CACHE_SCHEMA = "vcot_v10_traditional_route_feature_cache_v1"
FEATURE_CONTRACT_SCHEMA = "vcot_v10_traditional_route_features_v1"
MODEL_SCHEMA = "vcot_v10_traditional_route_model_v1"
SELECTION_SCHEMA = "vcot_v10_traditional_route_selection_v1"
EVALUATION_SCHEMA = "vcot_v10_traditional_route_static_eval_v1"
DEFAULT_EFFICIENCY_COSTS = (204.5771, 154.4044, 151.5316)
WORD_RE = re.compile(r"[a-z0-9]+(?:['_-][a-z0-9]+)*", flags=re.IGNORECASE)

RIDGE_FAMILIES: dict[str, tuple[str, ...]] = {
	"diagnosis_structured_ridge": ("diagnosis_structured",),
	"prompt_tfidf_ridge": ("prompt_tfidf",),
	"diagnosis_tfidf_ridge": ("diagnosis_tfidf",),
	"prompt_diagnosis_ridge": ("prompt_tfidf", "diagnosis_tfidf", "diagnosis_structured"),
	"image_handcrafted_ridge": ("image_handcrafted",),
	"prompt_image_ridge": ("prompt_tfidf", "image_handcrafted"),
	"all_traditional_ridge": (
		"prompt_tfidf",
		"diagnosis_tfidf",
		"diagnosis_structured",
		"image_handcrafted",
	),
}
LOOKUP_FAMILY = "diagnosis_signature_lookup"
DEFAULT_FAMILIES = (LOOKUP_FAMILY, *RIDGE_FAMILIES)


class TraditionalRouteError(ValueError):
	"""Raised when a traditional baseline artifact violates its contract."""


def _canonical_json(value: Any) -> str:
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
	return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
	return _sha256_text(_canonical_json(value))


def _write_json(path: Path, value: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
	try:
		value = json.loads(path.read_text(encoding="utf-8"))
	except Exception as exc:
		raise TraditionalRouteError(f"Cannot read JSON {path}: {exc}") from exc
	if not isinstance(value, dict):
		raise TraditionalRouteError(f"Expected a JSON object: {path}")
	return value


def _array_digest(value: np.ndarray) -> str:
	array = np.ascontiguousarray(value)
	digest = hashlib.sha256()
	digest.update(str(array.dtype).encode("ascii"))
	digest.update(_canonical_json(list(array.shape)).encode("ascii"))
	digest.update(array.tobytes(order="C"))
	return digest.hexdigest()


def _finite_float_list(values: Iterable[Any], name: str) -> list[float]:
	try:
		result = [float(value) for value in values]
	except (TypeError, ValueError) as exc:
		raise TraditionalRouteError(f"{name} must contain numeric values") from exc
	if not result or not all(math.isfinite(value) for value in result):
		raise TraditionalRouteError(f"{name} must contain finite values")
	return result


def _parse_grid(value: str, name: str, *, positive: bool = False, non_negative: bool = False) -> list[float]:
	items = _finite_float_list((item.strip() for item in str(value).split(",") if item.strip()), name)
	if positive and any(item <= 0.0 for item in items):
		raise TraditionalRouteError(f"{name} must be positive")
	if non_negative and any(item < 0.0 for item in items):
		raise TraditionalRouteError(f"{name} must be non-negative")
	return sorted(set(items))


def _diagnosis_payload(row: Mapping[str, Any]) -> dict[str, Any]:
	try:
		payload = json.loads(str(row["diagnosis_text"]))
	except (KeyError, TypeError, json.JSONDecodeError) as exc:
		raise TraditionalRouteError("canonical diagnosis_text must be valid JSON") from exc
	if not isinstance(payload, dict):
		raise TraditionalRouteError("canonical diagnosis_text must decode to an object")
	return payload


def diagnosis_signature(row: Mapping[str, Any]) -> tuple[str, ...]:
	"""Return the route-free multiset of diagnosed constraint types."""

	payload = _diagnosis_payload(row)
	issues = payload.get("issues")
	if not isinstance(issues, list) or not issues:
		raise TraditionalRouteError("diagnosis signature requires at least one issue")
	types: list[str] = []
	for issue in issues:
		if not isinstance(issue, Mapping):
			raise TraditionalRouteError("diagnosis issue must be an object")
		constraint_type = str(issue.get("constraint_type") or "").strip().lower()
		if not constraint_type:
			raise TraditionalRouteError("diagnosis issue is missing constraint_type")
		types.append(constraint_type)
	return tuple(sorted(types))


def diagnosis_document(row: Mapping[str, Any]) -> str:
	"""Flatten factual diagnosis fields without adding a route or repair plan."""

	payload = _diagnosis_payload(row)
	parts: list[str] = []
	reason = payload.get("reason")
	if reason:
		parts.append(f"reason {reason}")
	audit = payload.get("constraint_audit")
	if isinstance(audit, Mapping):
		for key, value in sorted(audit.items()):
			parts.append(f"audit_{key} {value}")
	issues = payload.get("issues")
	if isinstance(issues, list):
		for issue in issues:
			if not isinstance(issue, Mapping):
				continue
			for key in ("constraint_type", "target_entity", "observed", "expected", "evidence"):
				value = issue.get(key)
				if value:
					parts.append(f"issue_{key} {value}")
	preservation = payload.get("preservation_constraints")
	if isinstance(preservation, list):
		parts.extend(f"preserve {value}" for value in preservation if value)
	return " ".join(parts)


def _word_terms(document: str) -> list[str]:
	tokens = [token.lower() for token in WORD_RE.findall(str(document))]
	terms = [f"u:{token}" for token in tokens]
	terms.extend(f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:]))
	return terms


@dataclass(frozen=True)
class TfidfLite:
	vocabulary: tuple[str, ...]
	idf: np.ndarray
	min_df: int
	max_features: int

	@classmethod
	def fit(cls, documents: Sequence[str], *, min_df: int, max_features: int) -> "TfidfLite":
		if not documents:
			raise TraditionalRouteError("TF-IDF fit requires train documents")
		if min_df <= 0 or max_features <= 0:
			raise TraditionalRouteError("TF-IDF min_df and max_features must be positive")
		document_frequency: Counter[str] = Counter()
		for document in documents:
			document_frequency.update(set(_word_terms(document)))
		terms = [term for term, count in document_frequency.items() if count >= min_df]
		terms.sort(key=lambda term: (-document_frequency[term], term))
		vocabulary = tuple(terms[:max_features])
		if not vocabulary:
			raise TraditionalRouteError("TF-IDF train vocabulary is empty")
		n = len(documents)
		idf = np.asarray(
			[math.log((1.0 + n) / (1.0 + document_frequency[term])) + 1.0 for term in vocabulary],
			dtype=np.float64,
		)
		return cls(vocabulary=vocabulary, idf=idf, min_df=min_df, max_features=max_features)

	def transform(self, documents: Sequence[str]) -> np.ndarray:
		index = {term: offset for offset, term in enumerate(self.vocabulary)}
		matrix = np.zeros((len(documents), len(self.vocabulary)), dtype=np.float64)
		for row_index, document in enumerate(documents):
			counts = Counter(_word_terms(document))
			for term, count in counts.items():
				column = index.get(term)
				if column is not None:
					matrix[row_index, column] = (1.0 + math.log(float(count))) * self.idf[column]
			norm = float(np.linalg.norm(matrix[row_index]))
			if norm > 0.0:
				matrix[row_index] /= norm
		return matrix

	def contract(self) -> dict[str, Any]:
		return {
			"tokenization": "lowercase_ascii_word_unigram_bigram_v1",
			"sublinear_tf": True,
			"smooth_idf": True,
			"l2_normalize": True,
			"min_df": self.min_df,
			"max_features": self.max_features,
			"dimension": len(self.vocabulary),
			"vocabulary": list(self.vocabulary),
			"idf": self.idf.tolist(),
		}


def _normalized_histogram(values: np.ndarray, bins: int) -> list[float]:
	counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
	total = int(counts.sum())
	return (counts.astype(np.float64) / max(total, 1)).tolist()


def image_feature_names() -> list[str]:
	names: list[str] = []
	for channel in range(3):
		names.extend(
			[
				f"rgb_c{channel}_mean",
				f"rgb_c{channel}_std",
				f"rgb_c{channel}_q10",
				f"rgb_c{channel}_q25",
				f"rgb_c{channel}_q50",
				f"rgb_c{channel}_q75",
				f"rgb_c{channel}_q90",
			]
		)
		names.extend(f"rgb_c{channel}_hist_{index:02d}" for index in range(16))
	names.extend(f"gray_hist_{index:02d}" for index in range(16))
	names.extend(["gray_mean", "gray_std", "gray_q10", "gray_q50", "gray_q90"])
	names.extend(f"saturation_hist_{index:02d}" for index in range(16))
	names.extend(
		["saturation_mean", "saturation_std", "saturation_q10", "saturation_q50", "saturation_q90"]
	)
	for row in range(4):
		for column in range(4):
			names.extend(f"grid_{row}_{column}_rgb_c{channel}_mean" for channel in range(3))
			names.extend(f"grid_{row}_{column}_rgb_c{channel}_std" for channel in range(3))
	names.extend(
		[
			"sobel_magnitude_mean",
			"sobel_magnitude_std",
			"sobel_magnitude_gt_025",
			"laplacian_variance",
			"gray_entropy",
			"log2_width",
			"log2_height",
			"aspect_ratio",
		]
	)
	return names


def image_feature_vector(path: Path) -> np.ndarray:
	"""Extract deterministic traditional visual statistics from one raw RGB image."""

	image = load_raw_route_image(path)
	width, height = image.size
	resampling = getattr(Image, "Resampling", Image).BILINEAR
	rgb_image = image.resize((128, 128), resampling)
	rgb = np.asarray(rgb_image, dtype=np.float64) / 255.0
	gray = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float64)
	maximum = rgb.max(axis=2)
	minimum = rgb.min(axis=2)
	saturation = np.divide(
		maximum - minimum,
		maximum,
		out=np.zeros_like(maximum),
		where=maximum > 0.0,
	)
	features: list[float] = []
	for channel in range(3):
		values = rgb[:, :, channel]
		features.extend(
			[
				float(values.mean()),
				float(values.std()),
				*map(float, np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])),
			]
		)
		features.extend(_normalized_histogram(values, 16))
	features.extend(_normalized_histogram(gray, 16))
	features.extend(
		[
			float(gray.mean()),
			float(gray.std()),
			*map(float, np.quantile(gray, [0.1, 0.5, 0.9])),
		]
	)
	features.extend(_normalized_histogram(saturation, 16))
	features.extend(
		[
			float(saturation.mean()),
			float(saturation.std()),
			*map(float, np.quantile(saturation, [0.1, 0.5, 0.9])),
		]
	)
	for row in range(4):
		for column in range(4):
			cell = rgb[row * 32 : (row + 1) * 32, column * 32 : (column + 1) * 32]
			features.extend(map(float, cell.mean(axis=(0, 1))))
			features.extend(map(float, cell.std(axis=(0, 1))))
	# scipy.ndimage uses a fixed reflect-padded Sobel/Laplacian operator; no
	# learned visual backbone or fitted filter is involved.
	sobel_x = ndimage.sobel(gray, axis=1)
	sobel_y = ndimage.sobel(gray, axis=0)
	sobel_magnitude = np.hypot(sobel_x, sobel_y)
	laplacian = ndimage.laplace(gray)
	gray_hist = np.asarray(_normalized_histogram(gray, 64), dtype=np.float64)
	nonzero = gray_hist[gray_hist > 0.0]
	entropy = float(-(nonzero * np.log2(nonzero)).sum())
	features.extend(
		[
			float(sobel_magnitude.mean()),
			float(sobel_magnitude.std()),
			float((sobel_magnitude > 0.25).mean()),
			float(laplacian.var()),
			entropy,
			math.log2(max(width, 1)),
			math.log2(max(height, 1)),
			float(width / max(height, 1)),
		]
	)
	result = np.asarray(features, dtype=np.float64)
	if result.shape != (len(image_feature_names()),) or not np.isfinite(result).all():
		raise TraditionalRouteError(f"invalid handcrafted image feature vector for {path}")
	return result


@dataclass(frozen=True)
class FeatureBlocks:
	matrices: dict[str, dict[str, np.ndarray]]
	contract: dict[str, Any]
	build_seconds: float


def _source_digest(*objects: Any) -> str:
	return _sha256_text("\n\n".join(inspect.getsource(value) for value in objects))


def build_feature_blocks(
	bundle: DatasetBundle,
	*,
	max_tfidf_features: int,
	tfidf_min_df: int,
) -> FeatureBlocks:
	"""Fit all learned transforms on train, then transform every split label-free."""

	started = time.perf_counter()
	prompt_documents = {
		split: [str(row["target_prompt"]) for row in bundle.rows[split]] for split in SPLITS
	}
	diagnosis_documents = {
		split: [diagnosis_document(row) for row in bundle.rows[split]] for split in SPLITS
	}
	prompt_tfidf = TfidfLite.fit(
		prompt_documents["train"], min_df=tfidf_min_df, max_features=max_tfidf_features
	)
	diagnosis_tfidf = TfidfLite.fit(
		diagnosis_documents["train"], min_df=tfidf_min_df, max_features=max_tfidf_features
	)
	matrices: dict[str, dict[str, np.ndarray]] = {}
	for split in SPLITS:
		structured = np.asarray(
			[diagnosis_aux_vector(row["diagnosis_text"]) for row in bundle.rows[split]],
			dtype=np.float64,
		)
		images = np.stack(
			[image_feature_vector(Path(str(row["image_path"]))) for row in bundle.rows[split]],
			axis=0,
		)
		matrices[split] = {
			"diagnosis_structured": structured,
			"prompt_tfidf": prompt_tfidf.transform(prompt_documents[split]),
			"diagnosis_tfidf": diagnosis_tfidf.transform(diagnosis_documents[split]),
			"image_handcrafted": images,
		}
		for block_name, matrix in matrices[split].items():
			if matrix.ndim != 2 or matrix.shape[0] != len(bundle.rows[split]) or not np.isfinite(matrix).all():
				raise TraditionalRouteError(f"invalid {split}/{block_name} feature matrix")
	structured_mean = matrices["train"]["diagnosis_structured"].mean(axis=0)
	structured_std = np.maximum(matrices["train"]["diagnosis_structured"].std(axis=0), 0.1)
	image_mean = matrices["train"]["image_handcrafted"].mean(axis=0)
	image_std = np.maximum(matrices["train"]["image_handcrafted"].std(axis=0), 0.05)
	for split in SPLITS:
		matrices[split]["diagnosis_structured"] = (
			matrices[split]["diagnosis_structured"] - structured_mean
		) / structured_std
		image = (matrices[split]["image_handcrafted"] - image_mean) / image_std
		row_norm = np.linalg.norm(image, axis=1, keepdims=True)
		matrices[split]["image_handcrafted"] = image / np.maximum(row_norm, 1e-12)
	block_contracts = {
		"diagnosis_structured": {
			"dimension": len(DIAGNOSIS_AUX_FEATURE_NAMES),
			"feature_names": list(DIAGNOSIS_AUX_FEATURE_NAMES),
			"extractor_source_sha256": _source_digest(diagnosis_aux_vector),
			"normalization": "train-only column z-score std_floor=0.1",
			"train_mean": structured_mean.tolist(),
			"train_std_with_floor": structured_std.tolist(),
		},
		"prompt_tfidf": prompt_tfidf.contract(),
		"diagnosis_tfidf": diagnosis_tfidf.contract(),
		"image_handcrafted": {
			"dimension": len(image_feature_names()),
			"feature_names": image_feature_names(),
			"extractor_source_sha256": _source_digest(
				_normalized_histogram,
				image_feature_names,
				image_feature_vector,
			),
			"resize": "PIL bilinear 128x128",
			"normalization": "train-only column z-score std_floor=0.05, then per-row L2",
			"train_mean": image_mean.tolist(),
			"train_std_with_floor": image_std.tolist(),
		},
	}
	contract: dict[str, Any] = {
		"schema": FEATURE_CONTRACT_SCHEMA,
		"uses_qwen_route_features": False,
		"route_feature_backbone": "none",
		"entire_system_qwen_free": False,
		"canonical_diagnosis_is_upstream_qwen_output": True,
		"qwen_distribution_signature": bundle.qwen_distribution_signature,
		"fitted_on_split": "train",
		"train_state_ids_digest": _sha256_json([row["state_id"] for row in bundle.rows["train"]]),
		"text_extractor_source_sha256": _source_digest(_word_terms, diagnosis_document, TfidfLite),
		"blocks": block_contracts,
	}
	contract["feature_contract_digest"] = _sha256_json(contract)
	return FeatureBlocks(
		matrices=matrices,
		contract=contract,
		build_seconds=time.perf_counter() - started,
	)


def save_feature_cache(output_dir: Path, bundle: DatasetBundle, features: FeatureBlocks) -> None:
	arrays: dict[str, np.ndarray] = {}
	array_contracts: dict[str, Any] = {}
	for split in SPLITS:
		for block_name, matrix in sorted(features.matrices[split].items()):
			key = f"{split}__{block_name}"
			arrays[key] = matrix
			array_contracts[key] = {
				"shape": list(matrix.shape),
				"dtype": str(matrix.dtype),
				"sha256": _array_digest(matrix),
			}
	cache_path = output_dir / "features.npz"
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(cache_path, **arrays)
	manifest = {
		"schema": FEATURE_CACHE_SCHEMA,
		"dataset_digest": bundle.digest,
		"feature_contract": features.contract,
		"feature_contract_digest": features.contract["feature_contract_digest"],
		"splits": {
			split: {
				"state_ids": [row["state_id"] for row in bundle.rows[split]],
				"input_digests": [row["feature_input_digest"] for row in bundle.rows[split]],
			}
			for split in SPLITS
		},
		"arrays": array_contracts,
	}
	_write_json(output_dir / "feature_manifest.json", manifest)


def load_feature_cache(run_dir: Path, bundle: DatasetBundle) -> FeatureBlocks:
	manifest = _read_json(run_dir / "feature_manifest.json")
	if manifest.get("schema") != FEATURE_CACHE_SCHEMA:
		raise TraditionalRouteError("traditional feature-cache schema mismatch")
	if manifest.get("dataset_digest") != bundle.digest:
		raise TraditionalRouteError("traditional feature cache dataset digest mismatch")
	contract = manifest.get("feature_contract")
	if not isinstance(contract, dict):
		raise TraditionalRouteError("traditional feature cache is missing its feature contract")
	declared_digest = str(manifest.get("feature_contract_digest") or "")
	contract_payload = dict(contract)
	embedded_digest = str(contract_payload.pop("feature_contract_digest", ""))
	actual_digest = _sha256_json(contract_payload)
	if declared_digest != embedded_digest or declared_digest != actual_digest:
		raise TraditionalRouteError("traditional feature contract digest mismatch")
	if contract.get("uses_qwen_route_features") is not False or contract.get("route_feature_backbone") != "none":
		raise TraditionalRouteError("traditional feature cache unexpectedly declares a route backbone")
	for split in SPLITS:
		expected = manifest.get("splits", {}).get(split)
		if not isinstance(expected, Mapping):
			raise TraditionalRouteError(f"traditional feature cache is missing split {split}")
		if expected.get("state_ids") != [row["state_id"] for row in bundle.rows[split]]:
			raise TraditionalRouteError(f"traditional feature cache {split} state order mismatch")
		if expected.get("input_digests") != [row["feature_input_digest"] for row in bundle.rows[split]]:
			raise TraditionalRouteError(f"traditional feature cache {split} inputs mismatch")
	array_contracts = manifest.get("arrays")
	if not isinstance(array_contracts, Mapping) or not array_contracts:
		raise TraditionalRouteError("traditional feature cache has no array contracts")
	matrices: dict[str, dict[str, np.ndarray]] = {split: {} for split in SPLITS}
	try:
		with np.load(run_dir / "features.npz", allow_pickle=False) as payload:
			if set(payload.files) != set(array_contracts):
				raise TraditionalRouteError("traditional feature cache array names mismatch")
			for key, expected in array_contracts.items():
				if not isinstance(expected, Mapping) or "__" not in key:
					raise TraditionalRouteError("traditional feature cache array contract is invalid")
				split, block_name = key.split("__", 1)
				if split not in matrices:
					raise TraditionalRouteError(f"traditional feature cache has unknown split: {split}")
				matrix = np.asarray(payload[key], dtype=np.float64)
				if list(matrix.shape) != expected.get("shape") or str(matrix.dtype) != expected.get("dtype"):
					raise TraditionalRouteError(f"traditional feature cache {key} shape/dtype mismatch")
				if not np.isfinite(matrix).all() or _array_digest(matrix) != expected.get("sha256"):
					raise TraditionalRouteError(f"traditional feature cache {key} digest/non-finite mismatch")
				matrices[split][block_name] = matrix
	except TraditionalRouteError:
		raise
	except Exception as exc:
		raise TraditionalRouteError(f"Cannot load traditional feature cache: {exc}") from exc
	return FeatureBlocks(matrices=matrices, contract=contract, build_seconds=0.0)


def _observed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
	result: list[dict[str, Any]] = []
	for source in rows:
		row = dict(source)
		observed = source.get("unshrunk_candidate_quality_vector")
		if isinstance(observed, list) and len(observed) == 3:
			row["utility_vector"] = [float(value) for value in observed]
		result.append(row)
	return result


def _training_targets(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
	targets: list[list[float]] = []
	weights: list[float] = []
	for row in rows:
		quality = [float(value) for value in row["utility_vector"]]
		center = sum(quality) / 3.0
		targets.append([value - center for value in quality])
		observed = [
			float(value)
			for value in row.get("unshrunk_candidate_quality_vector", row["utility_vector"])
		]
		exact_tie_weight = 0.0 if max(observed) - min(observed) <= 1e-8 else 1.0
		weights.append(exact_tie_weight)
	target_matrix = np.asarray(targets, dtype=np.float64)
	weight_vector = np.asarray(weights, dtype=np.float64)
	if target_matrix.shape != (len(rows), 3) or not np.isfinite(target_matrix).all():
		raise TraditionalRouteError("traditional training targets are invalid")
	if not np.isfinite(weight_vector).all() or (weight_vector < 0.0).any() or weight_vector.sum() <= 0.0:
		raise TraditionalRouteError("traditional training weights are invalid")
	return target_matrix, weight_vector


def _family_matrix(features: FeatureBlocks, split: str, blocks: Sequence[str]) -> np.ndarray:
	try:
		parts = [features.matrices[split][block] for block in blocks]
	except KeyError as exc:
		raise TraditionalRouteError(f"missing traditional feature block: {exc}") from exc
	result = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
	if result.ndim != 2 or not np.isfinite(result).all():
		raise TraditionalRouteError("traditional family feature matrix is invalid")
	return result


def fit_weighted_ridge(
	features: np.ndarray,
	targets: np.ndarray,
	weights: np.ndarray,
	*,
	alpha: float,
) -> dict[str, Any]:
	if alpha <= 0.0 or not math.isfinite(alpha):
		raise TraditionalRouteError("ridge alpha must be finite and positive")
	if features.ndim != 2 or targets.shape != (features.shape[0], 3) or weights.shape != (features.shape[0],):
		raise TraditionalRouteError("ridge fit shapes are inconsistent")
	positive = weights > 0.0
	fit_features = features[positive]
	fit_targets = targets[positive]
	fit_weights = weights[positive]
	weight_sum = float(fit_weights.sum())
	feature_mean = (fit_features * fit_weights[:, None]).sum(axis=0) / weight_sum
	target_mean = (fit_targets * fit_weights[:, None]).sum(axis=0) / weight_sum
	sqrt_weights = np.sqrt(fit_weights)[:, None]
	weighted_features = (fit_features - feature_mean) * sqrt_weights
	weighted_targets = (fit_targets - target_mean) * sqrt_weights
	if features.shape[1] <= features.shape[0]:
		gram = weighted_features.T @ weighted_features
		coefficient = np.linalg.solve(
			gram + (alpha + 1e-10) * np.eye(gram.shape[0], dtype=np.float64),
			weighted_features.T @ weighted_targets,
		)
	else:
		gram = weighted_features @ weighted_features.T
		dual = np.linalg.solve(
			gram + (alpha + 1e-10) * np.eye(gram.shape[0], dtype=np.float64),
			weighted_targets,
		)
		coefficient = weighted_features.T @ dual
	if not all(np.isfinite(value).all() for value in (feature_mean, target_mean, coefficient)):
		raise TraditionalRouteError("ridge fit produced non-finite parameters")
	return {
		"kind": "weighted_ridge_centered_quality",
		"alpha": float(alpha),
		"feature_mean": feature_mean.tolist(),
		"target_mean": target_mean.tolist(),
		"coefficient": coefficient.tolist(),
		"trainable_parameter_count": int(coefficient.size + target_mean.size),
	}


def predict_weighted_ridge(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
	mean = np.asarray(model["feature_mean"], dtype=np.float64)
	target_mean = np.asarray(model["target_mean"], dtype=np.float64)
	coefficient = np.asarray(model["coefficient"], dtype=np.float64)
	if features.shape[1] != mean.size or target_mean.shape != (3,) or coefficient.shape != (mean.size, 3):
		raise TraditionalRouteError("ridge model dimensions do not match features")
	result = (features - mean) @ coefficient + target_mean
	if result.shape != (features.shape[0], 3) or not np.isfinite(result).all():
		raise TraditionalRouteError("ridge prediction produced invalid deltas")
	return result


def fit_diagnosis_lookup(
	rows: Sequence[Mapping[str, Any]],
	targets: np.ndarray,
	weights: np.ndarray,
	*,
	prior_strength: float,
) -> dict[str, Any]:
	if prior_strength <= 0.0 or not math.isfinite(prior_strength):
		raise TraditionalRouteError("lookup prior strength must be finite and positive")
	weight_sum = float(weights.sum())
	global_delta = (targets * weights[:, None]).sum(axis=0) / weight_sum
	group_weight: defaultdict[tuple[str, ...], float] = defaultdict(float)
	group_sum: defaultdict[tuple[str, ...], np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.float64))
	group_count: Counter[tuple[str, ...]] = Counter()
	for row, target, weight in zip(rows, targets, weights):
		signature = diagnosis_signature(row)
		group_count[signature] += 1
		group_weight[signature] += float(weight)
		group_sum[signature] += float(weight) * target
	groups: dict[str, Any] = {}
	for signature in sorted(group_count):
		posterior = (
			group_sum[signature] + prior_strength * global_delta
		) / (group_weight[signature] + prior_strength)
		groups["|".join(signature)] = {
			"member_count": group_count[signature],
			"effective_weight": group_weight[signature],
			"posterior_delta": posterior.tolist(),
		}
	return {
		"kind": "diagnosis_signature_empirical_bayes_l2_delta",
		"prior_strength": float(prior_strength),
		"global_delta": global_delta.tolist(),
		"groups": groups,
		"group_count": len(groups),
		"trainable_parameter_count": int(2 * len(groups) + 2),
	}


def predict_diagnosis_lookup(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
	global_delta = np.asarray(model["global_delta"], dtype=np.float64)
	groups = model.get("groups")
	if global_delta.shape != (2,) or not isinstance(groups, Mapping):
		raise TraditionalRouteError("diagnosis lookup model is invalid")
	result: list[np.ndarray] = []
	for row in rows:
		group = groups.get("|".join(diagnosis_signature(row)))
		delta = global_delta if not isinstance(group, Mapping) else np.asarray(group["posterior_delta"], dtype=np.float64)
		result.append(delta)
	matrix = np.stack(result, axis=0)
	if matrix.shape != (len(rows), 2) or not np.isfinite(matrix).all():
		raise TraditionalRouteError("diagnosis lookup prediction produced invalid deltas")
	return matrix


def route_from_deltas(
	deltas: np.ndarray,
	*,
	escape_threshold: float,
	efficiency_costs: Sequence[float],
) -> list[int]:
	"""Choose L2 unless predicted L1/L3 gain clears the locked threshold."""

	if deltas.ndim != 2 or deltas.shape[1] != 2 or not np.isfinite(deltas).all():
		raise TraditionalRouteError("route deltas must have shape [N,2]")
	if escape_threshold < 0.0 or not math.isfinite(escape_threshold):
		raise TraditionalRouteError("escape threshold must be finite and non-negative")
	costs = _finite_float_list(efficiency_costs, "efficiency_costs")
	if len(costs) != 3:
		raise TraditionalRouteError("efficiency_costs must contain L1/L2/L3")
	predictions: list[int] = []
	for l1_delta, l3_delta in deltas:
		best_gain = max(float(l1_delta), float(l3_delta))
		if best_gain <= escape_threshold + 1e-12:
			predictions.append(1)
		elif l1_delta > l3_delta + 1e-12:
			predictions.append(0)
		elif l3_delta > l1_delta + 1e-12:
			predictions.append(2)
		else:
			predictions.append(0 if costs[0] <= costs[2] else 2)
	return predictions


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
	return {
		"rows": int(metrics["rows"]),
		"mean_quality_regret": float(metrics["mean_quality_regret"]),
		"mean_selected_quality": float(metrics["mean_selected_quality"]),
		"acceptable_quality_rate": float(metrics["acceptable_quality_rate"]),
		"mean_success_rate": float(metrics["mean_success_rate"]),
		"mean_efficiency_cost": float(metrics["mean_efficiency_cost"]),
		"prediction_distribution": dict(metrics["prediction_distribution"]),
	}


def _selection_key(record: Mapping[str, Any], family_order: Mapping[str, int]) -> tuple[Any, ...]:
	metrics = record["validation"]
	return (
		round(float(metrics["mean_quality_regret"]), 12),
		round(float(metrics["mean_efficiency_cost"]), 12),
		-round(float(metrics["acceptable_quality_rate"]), 12),
		family_order[str(record["family"])],
		float(record["regularization"]),
		float(record["escape_threshold"]),
	)


def _model_deltas(
	model: Mapping[str, Any],
	features: FeatureBlocks,
	rows: Sequence[Mapping[str, Any]],
	*,
	split: str,
) -> np.ndarray:
	if model.get("kind") == "weighted_ridge_centered_quality":
		blocks = model.get("feature_blocks")
		if not isinstance(blocks, list) or not blocks:
			raise TraditionalRouteError("ridge model is missing feature blocks")
		values = predict_weighted_ridge(model, _family_matrix(features, split, blocks))
		return np.column_stack((values[:, 0] - values[:, 1], values[:, 2] - values[:, 1]))
	if model.get("kind") == "diagnosis_signature_empirical_bayes_l2_delta":
		return predict_diagnosis_lookup(model, rows)
	raise TraditionalRouteError(f"unsupported traditional model kind: {model.get('kind')}")


def _metric_bundle(
	rows: Sequence[Mapping[str, Any]],
	predictions: Sequence[int],
	*,
	acceptable_epsilon: float,
	efficiency_costs: Sequence[float],
	bootstrap_seed: int,
	bootstrap_samples: int,
) -> dict[str, Any]:
	return _policy_metrics(
		rows,
		predictions,
		acceptable_epsilon=acceptable_epsilon,
		efficiency_costs=efficiency_costs,
		bootstrap_seed=bootstrap_seed,
		bootstrap_samples=bootstrap_samples,
	)


def _fixed_metrics(
	rows: Sequence[Mapping[str, Any]],
	*,
	acceptable_epsilon: float,
	efficiency_costs: Sequence[float],
	bootstrap_seed: int,
	bootstrap_samples: int,
) -> dict[str, Any]:
	return {
		f"fixed_L{level}": _metric_bundle(
			rows,
			[level - 1] * len(rows),
			acceptable_epsilon=acceptable_epsilon,
			efficiency_costs=efficiency_costs,
			bootstrap_seed=bootstrap_seed,
			bootstrap_samples=bootstrap_samples,
		)
		for level in (1, 2, 3)
	}


def _prediction_rows(
	rows: Sequence[Mapping[str, Any]], predictions: Sequence[int], deltas: np.ndarray
) -> list[dict[str, Any]]:
	return [
		{
			"state_id": row["state_id"],
			"predicted_level": int(prediction + 1),
			"predicted_delta_L1_minus_L2": float(delta[0]),
			"predicted_delta_L3_minus_L2": float(delta[1]),
		}
		for row, prediction, delta in zip(rows, predictions, deltas)
	]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(
		"".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
		encoding="utf-8",
	)


def _family_uses_diagnosis(family: str) -> bool:
	return family == LOOKUP_FAMILY or any(
		block.startswith("diagnosis_") for block in RIDGE_FAMILIES.get(family, ())
	)


def search(args: argparse.Namespace) -> dict[str, Any]:
	if args.output_dir.exists() and any(args.output_dir.iterdir()):
		raise TraditionalRouteError(f"output directory must be absent or empty: {args.output_dir}")
	families = tuple(item.strip() for item in args.families.split(",") if item.strip())
	unknown = sorted(set(families) - set(DEFAULT_FAMILIES))
	if not families or unknown:
		raise TraditionalRouteError(f"invalid traditional families: {unknown or families}")
	if len(set(families)) != len(families):
		raise TraditionalRouteError("traditional families must not be repeated")
	ridge_alphas = _parse_grid(args.ridge_alphas, "ridge_alphas", positive=True)
	lookup_priors = _parse_grid(args.lookup_priors, "lookup_priors", positive=True)
	escape_thresholds = _parse_grid(args.escape_thresholds, "escape_thresholds", non_negative=True)
	efficiency_costs = _finite_float_list(args.efficiency_costs, "efficiency_costs")
	if len(efficiency_costs) != 3:
		raise TraditionalRouteError("efficiency_costs must contain three values")
	started = time.perf_counter()
	bundle = load_dataset_bundle(
		args.dataset_root,
		verify_image_hashes=not args.skip_image_hash_check,
		target_mode="candidate_quality",
		train_target_shrinkage=args.train_target_shrinkage,
	)
	feature_cache_dir = getattr(args, "feature_cache_dir", None)
	if feature_cache_dir is None:
		features = build_feature_blocks(
			bundle,
			max_tfidf_features=args.max_tfidf_features,
			tfidf_min_df=args.tfidf_min_df,
		)
	else:
		features = load_feature_cache(feature_cache_dir, bundle)
	save_feature_cache(args.output_dir, bundle, features)
	train_rows = bundle.rows["train"]
	validation_rows = bundle.rows["validation"]
	observed_train = _observed_rows(train_rows)
	targets, weights = _training_targets(train_rows)
	delta_targets = np.column_stack((targets[:, 0] - targets[:, 1], targets[:, 2] - targets[:, 1]))
	family_order = {family: index for index, family in enumerate(families)}
	search_records: list[dict[str, Any]] = []
	models: dict[tuple[str, float], dict[str, Any]] = {}
	for family in families:
		if family == LOOKUP_FAMILY:
			regularizations = lookup_priors
			for prior in regularizations:
				model = fit_diagnosis_lookup(train_rows, delta_targets, weights, prior_strength=prior)
				model.update(
					{
						"schema": MODEL_SCHEMA,
						"family": family,
						"feature_blocks": [],
						"regularization": prior,
					}
				)
				models[(family, prior)] = model
				validation_deltas = predict_diagnosis_lookup(model, validation_rows)
				for threshold in escape_thresholds:
					predictions = route_from_deltas(
						validation_deltas,
						escape_threshold=threshold,
						efficiency_costs=efficiency_costs,
					)
					metrics = _metric_bundle(
						validation_rows,
						predictions,
						acceptable_epsilon=bundle.acceptable_epsilon,
						efficiency_costs=efficiency_costs,
						bootstrap_seed=args.bootstrap_seed,
						bootstrap_samples=0,
					)
					search_records.append(
						{
							"family": family,
							"regularization_name": "prior_strength",
							"regularization": prior,
							"escape_threshold": threshold,
							"validation": _compact_metrics(metrics),
							"prediction_digest": _sha256_json(predictions),
						}
					)
			continue
		blocks = RIDGE_FAMILIES[family]
		train_features = _family_matrix(features, "train", blocks)
		validation_features = _family_matrix(features, "validation", blocks)
		for alpha in ridge_alphas:
			model = fit_weighted_ridge(train_features, targets, weights, alpha=alpha)
			model.update(
				{
					"schema": MODEL_SCHEMA,
					"family": family,
					"feature_blocks": list(blocks),
					"regularization": alpha,
				}
			)
			models[(family, alpha)] = model
			validation_values = predict_weighted_ridge(model, validation_features)
			validation_deltas = np.column_stack(
				(
					validation_values[:, 0] - validation_values[:, 1],
					validation_values[:, 2] - validation_values[:, 1],
				)
			)
			for threshold in escape_thresholds:
				predictions = route_from_deltas(
					validation_deltas,
					escape_threshold=threshold,
					efficiency_costs=efficiency_costs,
				)
				metrics = _metric_bundle(
					validation_rows,
					predictions,
					acceptable_epsilon=bundle.acceptable_epsilon,
					efficiency_costs=efficiency_costs,
					bootstrap_seed=args.bootstrap_seed,
					bootstrap_samples=0,
				)
				search_records.append(
					{
						"family": family,
						"regularization_name": "ridge_alpha",
						"regularization": alpha,
						"escape_threshold": threshold,
						"validation": _compact_metrics(metrics),
						"prediction_digest": _sha256_json(predictions),
					}
				)
	selected_record = min(search_records, key=lambda record: _selection_key(record, family_order))
	family_best = {
		family: min(
			(record for record in search_records if record["family"] == family),
			key=lambda record: _selection_key(record, family_order),
		)
		for family in families
	}
	selected_model = dict(models[(str(selected_record["family"]), float(selected_record["regularization"]))])
	selected_model.update(
		{
			"escape_threshold": float(selected_record["escape_threshold"]),
			"dataset_digest": bundle.digest,
			"target_digest": bundle.target_digest,
			"target_mode": bundle.target_mode,
			"train_target_shrinkage": bundle.train_target_shrinkage,
			"feature_contract_digest": features.contract["feature_contract_digest"],
			"uses_qwen_route_features": False,
			"route_feature_backbone": "none",
			"uses_canonical_diagnosis": _family_uses_diagnosis(str(selected_record["family"])),
			"fitted_on_split": "train",
			"selection_split": "validation",
			"efficiency_costs": efficiency_costs,
		}
	)
	selected_model["model_digest"] = _sha256_json(selected_model)
	validation_deltas = _model_deltas(
		selected_model, features, validation_rows, split="validation"
	)
	validation_predictions = route_from_deltas(
		validation_deltas,
		escape_threshold=float(selected_model["escape_threshold"]),
		efficiency_costs=efficiency_costs,
	)
	train_deltas = _model_deltas(selected_model, features, train_rows, split="train")
	train_predictions = route_from_deltas(
		train_deltas,
		escape_threshold=float(selected_model["escape_threshold"]),
		efficiency_costs=efficiency_costs,
	)
	validation_metrics = _metric_bundle(
		validation_rows,
		validation_predictions,
		acceptable_epsilon=bundle.acceptable_epsilon,
		efficiency_costs=efficiency_costs,
		bootstrap_seed=args.bootstrap_seed,
		bootstrap_samples=args.bootstrap_samples,
	)
	train_metrics = _metric_bundle(
		observed_train,
		train_predictions,
		acceptable_epsilon=bundle.acceptable_epsilon,
		efficiency_costs=efficiency_costs,
		bootstrap_seed=args.bootstrap_seed,
		bootstrap_samples=args.bootstrap_samples,
	)
	baselines = {
		"train": _fixed_metrics(
			observed_train,
			acceptable_epsilon=bundle.acceptable_epsilon,
			efficiency_costs=efficiency_costs,
			bootstrap_seed=args.bootstrap_seed,
			bootstrap_samples=args.bootstrap_samples,
		),
		"validation": _fixed_metrics(
			validation_rows,
			acceptable_epsilon=bundle.acceptable_epsilon,
			efficiency_costs=efficiency_costs,
			bootstrap_seed=args.bootstrap_seed,
			bootstrap_samples=args.bootstrap_samples,
		),
	}
	selection_manifest = {
		"schema": SELECTION_SCHEMA,
		"protocol": "train_fit_validation_select_test_locked_v1",
		"dataset_digest": bundle.digest,
		"target_digest": bundle.target_digest,
		"target_mode": bundle.target_mode,
		"train_target_shrinkage": bundle.train_target_shrinkage,
		"qwen_distribution_signature": bundle.qwen_distribution_signature,
		"feature_contract_digest": features.contract["feature_contract_digest"],
		"uses_qwen_route_features": False,
		"route_feature_backbone": "none",
		"entire_system_qwen_free": False,
		"candidate_families": list(families),
		"hyperparameter_grid": {
			"ridge_alphas": ridge_alphas,
			"lookup_priors": lookup_priors,
			"escape_thresholds": escape_thresholds,
			"max_tfidf_features": args.max_tfidf_features,
			"tfidf_min_df": args.tfidf_min_df,
		},
		"selection_key": [
			"validation mean candidate-quality regret ascending",
			"validation mean efficiency cost ascending",
			"validation acceptable-quality rate descending",
			"deterministic family/hyperparameter order",
		],
		"selected": selected_record,
		"selected_model_digest": selected_model["model_digest"],
		"selection_split": "validation",
		"evaluated_splits": ["train", "validation"],
		"test_gate": False,
		"test_labels_used_for_selection": False,
		"efficiency_costs": efficiency_costs,
		"bootstrap_seed": args.bootstrap_seed,
		"bootstrap_samples": args.bootstrap_samples,
	}
	_write_json(args.output_dir / "selected_model.json", selected_model)
	_write_json(
		args.output_dir / "development_search.json",
		{
			"schema": SELECTION_SCHEMA,
			"family_best": family_best,
			"all_candidates": search_records,
		},
	)
	_write_json(args.output_dir / "selection_manifest.json", selection_manifest)
	_write_json(
		args.output_dir / "validation_metrics.json",
		{
			"schema": EVALUATION_SCHEMA,
			"split": "validation",
			"test_gate": False,
			"selected_model_digest": selected_model["model_digest"],
			"traditional_router": validation_metrics,
			"baselines": baselines["validation"],
		},
	)
	_write_json(
		args.output_dir / "train_metrics.json",
		{
			"schema": EVALUATION_SCHEMA,
			"split": "train",
			"test_gate": False,
			"selected_model_digest": selected_model["model_digest"],
			"traditional_router": train_metrics,
			"baselines": baselines["train"],
		},
	)
	_write_jsonl(
		args.output_dir / "predictions_validation.jsonl",
		_prediction_rows(validation_rows, validation_predictions, validation_deltas),
	)
	_write_jsonl(
		args.output_dir / "predictions_train.jsonl",
		_prediction_rows(observed_train, train_predictions, train_deltas),
	)
	result = {
		"status": "selected_on_validation",
		"output_dir": str(args.output_dir.resolve()),
		"rows": {split: len(bundle.rows[split]) for split in SPLITS},
		"feature_build_seconds": features.build_seconds,
		"total_search_seconds": time.perf_counter() - started,
		"selected": selected_record,
		"validation": validation_metrics,
		"fixed_validation": baselines["validation"],
		"model_digest": selected_model["model_digest"],
		"test_evaluated": False,
	}
	_write_json(args.output_dir / "search_summary.json", result)
	write_report(args.output_dir)
	return result


def _validated_model(run_dir: Path, bundle: DatasetBundle, features: FeatureBlocks) -> dict[str, Any]:
	model = _read_json(run_dir / "selected_model.json")
	if model.get("schema") != MODEL_SCHEMA:
		raise TraditionalRouteError("traditional selected-model schema mismatch")
	payload = dict(model)
	declared_digest = str(payload.pop("model_digest", ""))
	if declared_digest != _sha256_json(payload):
		raise TraditionalRouteError("traditional selected-model digest mismatch")
	if model.get("dataset_digest") != bundle.digest or model.get("target_digest") != bundle.target_digest:
		raise TraditionalRouteError("traditional selected model dataset/target digest mismatch")
	if model.get("feature_contract_digest") != features.contract.get("feature_contract_digest"):
		raise TraditionalRouteError("traditional selected model feature contract mismatch")
	if model.get("uses_qwen_route_features") is not False or model.get("route_feature_backbone") != "none":
		raise TraditionalRouteError("traditional selected model unexpectedly uses a route backbone")
	if model.get("fitted_on_split") != "train" or model.get("selection_split") != "validation":
		raise TraditionalRouteError("traditional selected model fit/selection split mismatch")
	return model


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
	if args.split == "test" and not args.evaluate_test:
		raise TraditionalRouteError("test evaluation requires the explicit --evaluate-test gate")
	manifest = _read_json(args.run_dir / "selection_manifest.json")
	if manifest.get("schema") != SELECTION_SCHEMA or manifest.get("selection_split") != "validation":
		raise TraditionalRouteError("traditional selection manifest is invalid")
	bundle = load_dataset_bundle(
		args.dataset_root,
		verify_image_hashes=not args.skip_image_hash_check,
		target_mode="candidate_quality",
		train_target_shrinkage=str(manifest["train_target_shrinkage"]),
	)
	features = load_feature_cache(args.run_dir, bundle)
	model = _validated_model(args.run_dir, bundle, features)
	rows = _observed_rows(bundle.rows[args.split])
	deltas = _model_deltas(model, features, bundle.rows[args.split], split=args.split)
	efficiency_costs = _finite_float_list(model["efficiency_costs"], "efficiency_costs")
	predictions = route_from_deltas(
		deltas,
		escape_threshold=float(model["escape_threshold"]),
		efficiency_costs=efficiency_costs,
	)
	started = time.perf_counter()
	for _ in range(max(1, args.latency_repeats)):
		_model_deltas(model, features, bundle.rows[args.split], split=args.split)
	model_latency_ms = (
		(time.perf_counter() - started) * 1000.0 / (max(1, args.latency_repeats) * max(len(rows), 1))
	)
	metrics = _metric_bundle(
		rows,
		predictions,
		acceptable_epsilon=bundle.acceptable_epsilon,
		efficiency_costs=efficiency_costs,
		bootstrap_seed=args.bootstrap_seed,
		bootstrap_samples=args.bootstrap_samples,
	)
	result = {
		"schema": EVALUATION_SCHEMA,
		"split": args.split,
		"test_gate": bool(args.evaluate_test),
		"selection_is_locked": True,
		"selection_split": "validation",
		"dataset_digest": bundle.digest,
		"target_digest": bundle.target_digest,
		"feature_contract_digest": features.contract["feature_contract_digest"],
		"selected_model_digest": model["model_digest"],
		"uses_qwen_route_features": False,
		"uses_canonical_diagnosis": bool(model["uses_canonical_diagnosis"]),
		"traditional_router": metrics,
		"baselines": _fixed_metrics(
			rows,
			acceptable_epsilon=bundle.acceptable_epsilon,
			efficiency_costs=efficiency_costs,
			bootstrap_seed=args.bootstrap_seed,
			bootstrap_samples=args.bootstrap_samples,
		),
		"model_prediction_latency_ms_per_row": model_latency_ms,
	}
	_write_json(args.run_dir / f"evaluation_{args.split}.json", result)
	_write_jsonl(
		args.run_dir / f"predictions_{args.split}.jsonl",
		_prediction_rows(rows, predictions, deltas),
	)
	write_report(args.run_dir)
	return result


def _locked_family_model(
	family: str,
	record: Mapping[str, Any],
	*,
	bundle: DatasetBundle,
	features: FeatureBlocks,
	efficiency_costs: Sequence[float],
) -> dict[str, Any]:
	"""Refit one validation-locked family and prove its prediction identity."""

	train_rows = bundle.rows["train"]
	targets, weights = _training_targets(train_rows)
	delta_targets = np.column_stack((targets[:, 0] - targets[:, 1], targets[:, 2] - targets[:, 1]))
	regularization = float(record["regularization"])
	if family == LOOKUP_FAMILY:
		model = fit_diagnosis_lookup(
			train_rows,
			delta_targets,
			weights,
			prior_strength=regularization,
		)
		blocks: tuple[str, ...] = ()
	else:
		if family not in RIDGE_FAMILIES:
			raise TraditionalRouteError(f"locked search contains unknown family: {family}")
		blocks = RIDGE_FAMILIES[family]
		model = fit_weighted_ridge(
			_family_matrix(features, "train", blocks),
			targets,
			weights,
			alpha=regularization,
		)
	model.update(
		{
			"schema": MODEL_SCHEMA,
			"family": family,
			"feature_blocks": list(blocks),
			"regularization": regularization,
			"escape_threshold": float(record["escape_threshold"]),
			"dataset_digest": bundle.digest,
			"target_digest": bundle.target_digest,
			"target_mode": bundle.target_mode,
			"train_target_shrinkage": bundle.train_target_shrinkage,
			"feature_contract_digest": features.contract["feature_contract_digest"],
			"uses_qwen_route_features": False,
			"route_feature_backbone": "none",
			"uses_canonical_diagnosis": _family_uses_diagnosis(family),
			"fitted_on_split": "train",
			"selection_split": "validation",
			"selection_scope": "family_best",
			"efficiency_costs": list(efficiency_costs),
		}
	)
	validation_deltas = _model_deltas(
		model,
		features,
		bundle.rows["validation"],
		split="validation",
	)
	validation_predictions = route_from_deltas(
		validation_deltas,
		escape_threshold=float(model["escape_threshold"]),
		efficiency_costs=efficiency_costs,
	)
	if _sha256_json(validation_predictions) != record.get("prediction_digest"):
		raise TraditionalRouteError(
			f"locked {family} validation prediction digest no longer reproduces"
		)
	model["locked_validation_prediction_digest"] = record["prediction_digest"]
	model["model_digest"] = _sha256_json(model)
	return model


def evaluate_family_best(args: argparse.Namespace) -> dict[str, Any]:
	"""Evaluate every pre-test validation-locked family without reselection."""

	if args.split == "test" and not args.evaluate_test:
		raise TraditionalRouteError("test evaluation requires the explicit --evaluate-test gate")
	selection_manifest = _read_json(args.run_dir / "selection_manifest.json")
	search_result = _read_json(args.run_dir / "development_search.json")
	if selection_manifest.get("schema") != SELECTION_SCHEMA:
		raise TraditionalRouteError("traditional selection manifest is invalid")
	family_best = search_result.get("family_best")
	if not isinstance(family_best, Mapping) or set(family_best) != set(selection_manifest["candidate_families"]):
		raise TraditionalRouteError("locked family-best search records are incomplete")
	bundle = load_dataset_bundle(
		args.dataset_root,
		verify_image_hashes=not args.skip_image_hash_check,
		target_mode="candidate_quality",
		train_target_shrinkage=str(selection_manifest["train_target_shrinkage"]),
	)
	features = load_feature_cache(args.run_dir, bundle)
	efficiency_costs = _finite_float_list(selection_manifest["efficiency_costs"], "efficiency_costs")
	rows = _observed_rows(bundle.rows[args.split])
	metrics_by_family: dict[str, Any] = {}
	model_dir = args.run_dir / "family_best_models"
	for family in selection_manifest["candidate_families"]:
		record = family_best[family]
		if not isinstance(record, Mapping):
			raise TraditionalRouteError(f"locked family-best record is invalid: {family}")
		model = _locked_family_model(
			family,
			record,
			bundle=bundle,
			features=features,
			efficiency_costs=efficiency_costs,
		)
		deltas = _model_deltas(model, features, bundle.rows[args.split], split=args.split)
		predictions = route_from_deltas(
			deltas,
			escape_threshold=float(model["escape_threshold"]),
			efficiency_costs=efficiency_costs,
		)
		metrics_by_family[family] = {
			"locked_validation_config": {
				"regularization": record["regularization"],
				"escape_threshold": record["escape_threshold"],
				"validation_prediction_digest": record["prediction_digest"],
			},
			"uses_canonical_diagnosis": bool(model["uses_canonical_diagnosis"]),
			"trainable_parameter_count": int(model["trainable_parameter_count"]),
			"model_digest": model["model_digest"],
			"metrics": _metric_bundle(
				rows,
				predictions,
				acceptable_epsilon=bundle.acceptable_epsilon,
				efficiency_costs=efficiency_costs,
				bootstrap_seed=args.bootstrap_seed,
				bootstrap_samples=args.bootstrap_samples,
			),
		}
		_write_json(model_dir / f"{family}.json", model)
		_write_jsonl(
			args.run_dir / f"predictions_family_{family}_{args.split}.jsonl",
			_prediction_rows(rows, predictions, deltas),
		)
	result = {
		"schema": EVALUATION_SCHEMA,
		"protocol": "all_validation_locked_family_best_no_test_reselection_v1",
		"split": args.split,
		"test_gate": bool(args.evaluate_test),
		"selection_is_locked": True,
		"selection_split": "validation",
		"dataset_digest": bundle.digest,
		"target_digest": bundle.target_digest,
		"feature_contract_digest": features.contract["feature_contract_digest"],
		"uses_qwen_route_features": False,
		"family_best": metrics_by_family,
		"baselines": _fixed_metrics(
			rows,
			acceptable_epsilon=bundle.acceptable_epsilon,
			efficiency_costs=efficiency_costs,
			bootstrap_seed=args.bootstrap_seed,
			bootstrap_samples=args.bootstrap_samples,
		),
	}
	_write_json(args.run_dir / f"evaluation_family_best_{args.split}.json", result)
	write_report(args.run_dir)
	return result


def _format_metric_row(label: str, metrics: Mapping[str, Any]) -> str:
	distribution = metrics.get("prediction_distribution", {})
	dist = ", ".join(f"{key}={value}" for key, value in distribution.items())
	return (
		f"| {label} | {float(metrics['mean_quality_regret']):.6f} | "
		f"{float(metrics['mean_selected_quality']):.6f} | "
		f"{float(metrics['acceptable_quality_rate']):.4f} | "
		f"{float(metrics['mean_success_rate']):.4f} | "
		f"{float(metrics['mean_efficiency_cost']):.3f} | {dist} |"
	)


def write_report(run_dir: Path) -> None:
	manifest_path = run_dir / "selection_manifest.json"
	validation_path = run_dir / "validation_metrics.json"
	if not manifest_path.is_file() or not validation_path.is_file():
		return
	manifest = _read_json(manifest_path)
	validation = _read_json(validation_path)
	selected = manifest["selected"]
	lines = [
		"# MLIE-CoT traditional router ablation on the development set",
		"",
		"> These experiments do not use Qwen hidden routing features or `features.pt`. Models that consume canonical diagnoses still depend on the upstream Qwen analyzer. Set-A test results were exposed during development and provide development evidence only.",
		"",
		"## Selection protocol",
		"",
		f"- Training target: `{manifest['target_mode']}`; training-only shrinkage: `{manifest['train_target_shrinkage']}`.",
		"- Vocabulary, IDF, normalization, lookup statistics, and regression parameters are fitted only on training data.",
		"- Validation quality regret determines the family, regularization, and L2 escape threshold. Efficiency breaks ties.",
		"- Selection excludes the test split. Test evaluation requires a separate explicit opt-in.",
		"",
		"## Validation selection",
		"",
		f"- Family: `{selected['family']}`",
		f"- Regularization: `{selected['regularization']}`",
		f"- L2 escape threshold: `{selected['escape_threshold']}`",
		"",
		"| Policy | Regret ↓ | Selected quality ↑ | Acceptable ↑ | Success ↑ | Cost ↓ | Routes |",
		"|---|---:|---:|---:|---:|---:|---|",
		_format_metric_row("Traditional selected", validation["traditional_router"]),
	]
	for key in ("fixed_L1", "fixed_L2", "fixed_L3"):
		lines.append(_format_metric_row(key, validation["baselines"][key]))
	search_path = run_dir / "development_search.json"
	if search_path.is_file():
		search_result = _read_json(search_path)
		family_best = search_result.get("family_best", {})
		if isinstance(family_best, Mapping):
			lines.extend(
				[
					"",
					"## Best validation configuration per traditional router family",
					"",
					"| Policy | Regret ↓ | Selected quality ↑ | Acceptable ↑ | Success ↑ | Cost ↓ | Routes |",
					"|---|---:|---:|---:|---:|---:|---|",
				]
			)
			for family in manifest["candidate_families"]:
				lines.append(_format_metric_row(family, family_best[family]["validation"]))
	test_path = run_dir / "evaluation_test.json"
	if test_path.is_file():
		test = _read_json(test_path)
		lines.extend(
			[
				"",
				"## Exposed Set-A test (development evidence only)",
				"",
				"| Policy | Regret ↓ | Selected quality ↑ | Acceptable ↑ | Success ↑ | Cost ↓ | Routes |",
				"|---|---:|---:|---:|---:|---:|---|",
				_format_metric_row("Traditional selected", test["traditional_router"]),
			]
		)
		for key in ("fixed_L1", "fixed_L2", "fixed_L3"):
			lines.append(_format_metric_row(key, test["baselines"][key]))
	family_test_path = run_dir / "evaluation_family_best_test.json"
	if family_test_path.is_file():
		family_test = _read_json(family_test_path)
		lines.extend(
			[
				"",
				"## Exposed test results by family with settings fixed on validation",
				"",
				"| Policy | Regret ↓ | Selected quality ↑ | Acceptable ↑ | Success ↑ | Cost ↓ | Routes |",
				"|---|---:|---:|---:|---:|---:|---|",
			]
		)
		for family in manifest["candidate_families"]:
			lines.append(_format_metric_row(family, family_test["family_best"][family]["metrics"]))
	lines.extend(
		[
			"",
			"## Interpretation",
			"",
			"- `prompt_tfidf_ridge`, `image_handcrafted_ridge`, and `prompt_image_ridge` do not read diagnoses and serve as negative controls without Qwen routing input.",
			"- The diagnosis lookup, structured, and text families use canonical diagnoses but do not use Qwen hidden states.",
			"- This ablation estimates action value for individual static states. It does not establish end-to-end performance with the Qwen analyzer, L1 planner, and reviewer replaced.",
		]
	)
	(run_dir / "DEVELOPMENT_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	subparsers = parser.add_subparsers(dest="command", required=True)
	search_parser = subparsers.add_parser("search", help="Fit train-only features and select on validation")
	search_parser.add_argument("--dataset-root", type=Path, required=True)
	search_parser.add_argument("--output-dir", type=Path, required=True)
	search_parser.add_argument(
		"--feature-cache-dir",
		type=Path,
		default=None,
		help="Reuse a previously contracted label-free traditional feature cache.",
	)
	search_parser.add_argument("--skip-image-hash-check", action="store_true")
	search_parser.add_argument(
		"--train-target-shrinkage",
		choices=("none", "empirical_bayes_l2", "diagnosis_group_eb_l2"),
		default="diagnosis_group_eb_l2",
	)
	search_parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
	search_parser.add_argument("--max-tfidf-features", type=int, default=5000)
	search_parser.add_argument("--tfidf-min-df", type=int, default=2)
	search_parser.add_argument("--ridge-alphas", default="0.0001,0.001,0.01,0.1,1,10,100,1000")
	search_parser.add_argument("--lookup-priors", default="1,2,4,8,16,32,50,64")
	search_parser.add_argument("--escape-thresholds", default="0,0.1,0.25,0.5,0.75,1")
	search_parser.add_argument("--efficiency-costs", type=float, nargs=3, default=DEFAULT_EFFICIENCY_COSTS)
	search_parser.add_argument("--bootstrap-seed", type=int, default=20260722)
	search_parser.add_argument("--bootstrap-samples", type=int, default=10000)
	evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a locked selected model")
	evaluate_parser.add_argument("--dataset-root", type=Path, required=True)
	evaluate_parser.add_argument("--run-dir", type=Path, required=True)
	evaluate_parser.add_argument("--split", choices=SPLITS, default="validation")
	evaluate_parser.add_argument("--evaluate-test", action="store_true")
	evaluate_parser.add_argument("--skip-image-hash-check", action="store_true")
	evaluate_parser.add_argument("--bootstrap-seed", type=int, default=20260722)
	evaluate_parser.add_argument("--bootstrap-samples", type=int, default=10000)
	evaluate_parser.add_argument("--latency-repeats", type=int, default=100)
	family_parser = subparsers.add_parser(
		"evaluate-family-best",
		help="Evaluate every validation-locked family-best configuration without reselection",
	)
	family_parser.add_argument("--dataset-root", type=Path, required=True)
	family_parser.add_argument("--run-dir", type=Path, required=True)
	family_parser.add_argument("--split", choices=SPLITS, default="validation")
	family_parser.add_argument("--evaluate-test", action="store_true")
	family_parser.add_argument("--skip-image-hash-check", action="store_true")
	family_parser.add_argument("--bootstrap-seed", type=int, default=20260722)
	family_parser.add_argument("--bootstrap-samples", type=int, default=10000)
	return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	if args.command == "search":
		result = search(args)
	elif args.command == "evaluate-family-best":
		result = evaluate_family_best(args)
	else:
		result = evaluate(args)
	print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
