"""Frozen-Qwen MLP repair-route head used by MLIE-CoT.

Native Qwen exclusively decides whether an image has a problem.  The trained
head is only invoked for problematic images and chooses among L1 local edit,
L2 global edit, and L3 prompt-only regeneration.  L0 is therefore a Qwen gate,
not a learned MLP class.

A deployable checkpoint is more than MLP weights.  Its metadata fingerprints
the Qwen model revision, processor and chat template, image preprocessing,
last-token pooling code, canonical diagnosis schema, deterministic auxiliary
features, training objective, and the FLUX.2/L3 executor policies that define
the three labels.  Validation fails closed before large models are allocated
when any feature-producing or action-producing component differs.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import nn

from .contracts import (
	ROUTER_ACTION_POLICY_VERSION,
	ROUTER_REPAIR_EXECUTOR_FAMILY,
	diagnosis_payload_is_valid,
	model_visible_diagnosis_text,
	qwen_effective_quantization,
	validate_qwen_analyzer_l1_distribution_contract,
)
from .flux_policy import (
	flux2_policy_digest,
	validate_flux2_policy_contract,
	validate_matching_flux2_policy,
)
from .l3_policy import (
	l3_policy_digest,
	validate_l3_policy_contract,
	validate_matching_l3_policy,
)


ROUTE_MARKER = "__VCOT_ROUTE_HEAD_B64__"
ROUTE_DIAGNOSIS_SCHEMA = "vcot_v10_canonical_diagnosis_v1"
ROUTE_FEATURE_SCHEMA = "vcot_v10_qwen_route_feature"
ROUTE_FEATURE_SCHEMA_VERSION = 2
ROUTE_CHECKPOINT_VERSION = "vcot_v10_qwen_route_head_v5_quality_first_safe_pairs"
ROUTE_TARGET_MODE = "candidate_quality"
ROUTE_TRAIN_TARGET_SHRINKAGE = "diagnosis_group_eb_l2"
ROUTE_CANDIDATE_QUALITY_SCHEMA = "vcot_v10_candidate_quality_paired_seed_v3_quality_priority"
ROUTE_DIAGNOSIS_GROUP_EB_SCHEMA = "vcot_v10_train_only_diagnosis_group_eb_l2_v2_raw_safe_pairs"
ROUTE_ENCODER_SYSTEM_PROMPT_V10 = """Encode this image state and its canonical diagnosis as the representation used by a frozen repair-level classifier. The supplied assistant message is the factual diagnosis record to encode.
"""
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$", flags=re.IGNORECASE)
FEATURE_COMPONENTS = ("processor", "chat_template", "preprocess", "pooling")
DIAGNOSIS_AUX_SCHEMA = "vcot_v10_diagnosis_aux_features_v2"
DIAGNOSIS_AUX_FEATURE_NAMES = (
	"issue_count_object_identity",
	"issue_count_count",
	"issue_count_attribute",
	"issue_count_relation",
	"issue_count_action_state",
	"issue_count_scene",
	"issue_count_quality",
	"issue_count_other",
	"issue_count_total",
	"preservation_constraint_count",
	"mean_issue_confidence",
)
DIAGNOSIS_AUX_KNOWN_TYPES = (
	"object_identity",
	"count",
	"attribute",
	"relation",
	"action_state",
	"scene",
	"quality",
)
ROUTE_NAMES = {
	0: "L0_keep",
	1: "L1_local_flux2_full_plus_bbox_crop",
	2: "L2_global_flux2",
	3: "L3_regenerate",
}


def canonical_qwen_dtype_name(value: Any) -> str:
	"""Normalize Qwen dtype aliases used in the route-feature fingerprint."""
	name = str(value or "").replace("torch.", "").strip().lower()
	aliases = {
		"bf16": "bfloat16",
		"bfloat16": "bfloat16",
		"fp16": "float16",
		"float16": "float16",
		"half": "float16",
		"fp32": "float32",
		"float32": "float32",
	}
	if name not in aliases:
		raise ValueError(f"Unsupported Qwen dtype for the route feature contract: {value!r}")
	return aliases[name]


def canonical_qwen_quantization_name(value: Any) -> str:
	"""Normalize user quantization aliases to ``none``, ``8bit``, or ``nf4``."""
	name = str(value or "none").strip().lower()
	aliases = {
		"": "none",
		"none": "none",
		"8bit": "8bit",
		"int8": "8bit",
		"4bit": "nf4",
		"int4": "nf4",
		"nf4": "nf4",
	}
	if name not in aliases:
		raise ValueError(f"Unsupported Qwen quantization for the route feature contract: {value!r}")
	return aliases[name]


class RouteMLP(nn.Module):
	"""Small classifier over frozen Qwen features plus diagnosis statistics.

	Output index 0/1/2 maps to repair level L1/L2/L3; there is intentionally no
	L0 logit because native Qwen owns the satisfied gate.
	"""

	def __init__(self, input_dim: int, hidden_dim: int = 512, num_labels: int = 3, dropout: float = 0.15):
		"""Build the layer-normalized two-layer classification head."""
		super().__init__()
		self.net = nn.Sequential(
			nn.LayerNorm(input_dim),
			nn.Linear(input_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, num_labels),
		)

	def forward(self, hidden: torch.Tensor) -> torch.Tensor:
		"""Return float32 route logits for a batch of frozen feature vectors."""
		return self.net(hidden.float())


def _canonical_json(value: Any) -> str:
	"""Serialize feature-contract data deterministically."""
	return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
	"""Return a SHA-256 hex digest for UTF-8 contract material."""
	return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
	"""Recursively convert processor configuration into stable JSON values."""
	if value is None or isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, Mapping):
		return {
			str(key): _json_safe(child)
			for key, child in sorted(value.items(), key=lambda item: str(item[0]))
		}
	if isinstance(value, (list, tuple)):
		return [_json_safe(child) for child in value]
	return str(value)


def diagnosis_aux_vector(diagnosis: str | Mapping[str, Any]) -> List[float]:
	"""Extract model-independent structural statistics from canonical diagnosis."""

	if isinstance(diagnosis, Mapping):
		payload = dict(diagnosis)
	else:
		try:
			payload = json.loads(str(diagnosis))
		except json.JSONDecodeError as exc:
			raise ValueError("Diagnosis auxiliary features require canonical diagnosis JSON") from exc
	issues = payload.get("issues")
	if not isinstance(issues, list) or not issues:
		raise ValueError("Diagnosis auxiliary features require at least one issue")
	counts = {name: 0 for name in DIAGNOSIS_AUX_KNOWN_TYPES}
	other = 0
	confidences: List[float] = []
	for issue in issues:
		if not isinstance(issue, Mapping):
			raise ValueError("Diagnosis auxiliary feature issue must be an object")
		constraint_type = str(issue.get("constraint_type") or "").strip().lower()
		if constraint_type in counts:
			counts[constraint_type] += 1
		else:
			other += 1
		try:
			confidence = float(issue.get("confidence"))
		except (TypeError, ValueError) as exc:
			raise ValueError("Diagnosis auxiliary feature confidence must be numeric") from exc
		if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
			raise ValueError("Diagnosis auxiliary feature confidence must be in [0,1]")
		confidences.append(confidence)
	preservation = payload.get("preservation_constraints") or []
	if not isinstance(preservation, list):
		raise ValueError("Diagnosis preservation_constraints must be a list")
	return [
		*[float(counts[name]) for name in DIAGNOSIS_AUX_KNOWN_TYPES],
		float(other),
		float(len(issues)),
		float(len(preservation)),
		sum(confidences) / len(confidences),
	]


def diagnosis_aux_extractor_source_sha256() -> str:
	"""Bind checkpoints to the exact deterministic auxiliary extractor."""

	return _sha256_text(inspect.getsource(diagnosis_aux_vector))


def build_diagnosis_aux_contract(
	mean: Sequence[float],
	std: Sequence[float],
	*,
	std_floor: float,
) -> Dict[str, Any]:
	"""Build the immutable train-only normalization and extractor contract."""

	contract = {
		"schema": DIAGNOSIS_AUX_SCHEMA,
		"diagnosis_schema": ROUTE_DIAGNOSIS_SCHEMA,
		"extractor_source_sha256": diagnosis_aux_extractor_source_sha256(),
		"feature_names": list(DIAGNOSIS_AUX_FEATURE_NAMES),
		"known_constraint_types": list(DIAGNOSIS_AUX_KNOWN_TYPES),
		"unknown_constraint_type_policy": "count_as_other",
		"normalization": {
			"kind": "train_zscore",
			"mean": [float(value) for value in mean],
			"std": [float(value) for value in std],
			"std_floor": float(std_floor),
		},
	}
	# Reuse the online validator so training cannot emit a contract deployment
	# would later reject.
	return validate_diagnosis_aux_contract(
		{
			"diagnosis_aux_features_enabled": True,
			"diagnosis_aux_contract": contract,
			"diagnosis_aux_contract_digest": _sha256_text(_canonical_json(contract)),
		}
	) or {}


def validate_diagnosis_aux_contract(config: Mapping[str, Any]) -> Dict[str, Any] | None:
	"""Validate deterministic auxiliary features and train-only z-score stats."""
	enabled = config.get("diagnosis_aux_features_enabled", False)
	if not isinstance(enabled, bool):
		raise ValueError("Route checkpoint diagnosis_aux_features_enabled must be boolean")
	if not enabled:
		if config.get("diagnosis_aux_contract") is not None or config.get("diagnosis_aux_contract_digest") is not None:
			raise ValueError("Disabled diagnosis auxiliary features must not carry a contract")
		return None
	value = config.get("diagnosis_aux_contract")
	if not isinstance(value, Mapping):
		raise ValueError("Route checkpoint is missing diagnosis_aux_contract")
	contract = dict(value)
	expected_keys = {
		"schema",
		"diagnosis_schema",
		"extractor_source_sha256",
		"feature_names",
		"known_constraint_types",
		"unknown_constraint_type_policy",
		"normalization",
	}
	if set(contract) != expected_keys:
		raise ValueError("Route checkpoint diagnosis auxiliary contract fields mismatch")
	if contract.get("schema") != DIAGNOSIS_AUX_SCHEMA:
		raise ValueError("Route checkpoint diagnosis auxiliary schema mismatch")
	if contract.get("diagnosis_schema") != ROUTE_DIAGNOSIS_SCHEMA:
		raise ValueError("Route checkpoint diagnosis auxiliary diagnosis schema mismatch")
	if contract.get("extractor_source_sha256") != diagnosis_aux_extractor_source_sha256():
		raise ValueError("Route checkpoint diagnosis auxiliary extractor source mismatch")
	if contract.get("feature_names") != list(DIAGNOSIS_AUX_FEATURE_NAMES):
		raise ValueError("Route checkpoint diagnosis auxiliary feature names mismatch")
	if contract.get("known_constraint_types") != list(DIAGNOSIS_AUX_KNOWN_TYPES):
		raise ValueError("Route checkpoint diagnosis auxiliary constraint types mismatch")
	if contract.get("unknown_constraint_type_policy") != "count_as_other":
		raise ValueError("Route checkpoint diagnosis auxiliary unknown-type policy mismatch")
	normalization = contract.get("normalization")
	if (
		not isinstance(normalization, Mapping)
		or set(normalization) != {"kind", "mean", "std", "std_floor"}
		or normalization.get("kind") != "train_zscore"
	):
		raise ValueError("Route checkpoint diagnosis auxiliary normalization is invalid")
	try:
		raw_mean = list(normalization["mean"])
		raw_std = list(normalization["std"])
		raw_std_floor = normalization["std_floor"]
	except (KeyError, TypeError, ValueError) as exc:
		raise ValueError("Route checkpoint diagnosis auxiliary statistics are invalid") from exc
	if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_mean + raw_std):
		raise ValueError("Route checkpoint diagnosis auxiliary statistics must be numeric")
	if isinstance(raw_std_floor, bool) or not isinstance(raw_std_floor, (int, float)):
		raise ValueError("Route checkpoint diagnosis auxiliary std floor must be numeric")
	mean = [float(value) for value in raw_mean]
	std = [float(value) for value in raw_std]
	std_floor = float(raw_std_floor)
	feature_count = len(DIAGNOSIS_AUX_FEATURE_NAMES)
	if len(mean) != feature_count or len(std) != feature_count:
		raise ValueError("Route checkpoint diagnosis auxiliary statistic dimensions mismatch")
	if not all(math.isfinite(value) for value in mean + std) or any(value <= 0.0 for value in std):
		raise ValueError("Route checkpoint diagnosis auxiliary statistics must be finite")
	if not math.isfinite(std_floor) or std_floor <= 0.0 or any(value < std_floor for value in std):
		raise ValueError("Route checkpoint diagnosis auxiliary std floor is invalid")
	expected_digest = _sha256_text(_canonical_json(contract))
	if config.get("diagnosis_aux_contract_digest") != expected_digest:
		raise ValueError("Route checkpoint diagnosis auxiliary contract digest mismatch")
	return {
		**contract,
		"normalization": {
			"kind": "train_zscore",
			"mean": mean,
			"std": std,
			"std_floor": std_floor,
		},
	}


def validate_route_training_contract(config: Mapping[str, Any]) -> None:
	"""Reject online checkpoints outside the frozen quality-first training policy."""

	if config.get("version") != ROUTE_CHECKPOINT_VERSION:
		raise ValueError("Route checkpoint version is not the deployable quality-first V10 contract")
	if config.get("target_mode") != ROUTE_TARGET_MODE:
		raise ValueError("Route checkpoint target mode must be candidate_quality")
	if config.get("train_target_shrinkage") != ROUTE_TRAIN_TARGET_SHRINKAGE:
		raise ValueError("Route checkpoint train target shrinkage mismatch")
	processing = config.get("target_processing")
	if not isinstance(processing, Mapping):
		raise ValueError("Route checkpoint is missing target processing metadata")
	if processing.get("target_mode") != ROUTE_TARGET_MODE:
		raise ValueError("Route checkpoint target processing mode mismatch")
	if processing.get("train_target_shrinkage") != ROUTE_TRAIN_TARGET_SHRINKAGE:
		raise ValueError("Route checkpoint target processing shrinkage mismatch")
	candidate = processing.get("candidate_quality")
	if not isinstance(candidate, Mapping) or candidate.get("schema") != ROUTE_CANDIDATE_QUALITY_SCHEMA:
		raise ValueError("Route checkpoint candidate-quality target schema mismatch")
	group = processing.get("diagnosis_group_eb")
	if not isinstance(group, Mapping) or group.get("schema") != ROUTE_DIAGNOSIS_GROUP_EB_SCHEMA:
		raise ValueError("Route checkpoint diagnosis-group target schema mismatch")
	for key in ("dataset_digest", "target_digest"):
		if re.fullmatch(r"[0-9a-f]{64}", str(config.get(key) or "")) is None:
			raise ValueError(f"Route checkpoint is missing a valid {key}")
	if config.get("diagnosis_aux_features_enabled") is not True:
		raise ValueError("Deployable route checkpoint requires signed diagnosis auxiliary features")
	if float(config.get("exact_tie_quality_weight", -1.0)) != 0.0:
		raise ValueError("Deployable route checkpoint must mask exact-tie quality gradients")
	if config.get("class_factors") != [1.0, 1.0, 1.0]:
		raise ValueError("Deployable candidate-quality checkpoint must use neutral class factors")
	loss = config.get("loss")
	expected_loss = {
		"utility_temperature": 0.5,
		"pairwise_temperature": 0.5,
		"policy_temperature": 1.0,
		"quality_tie_epsilon": 0.0,
		"soft_ce_weight": 0.5,
		"utility_alignment_weight": 0.1,
		"pairwise_weight": 0.2,
		"expected_regret_weight": 0.0,
		"efficiency_weight": 0.001,
		"uncertain_safe_prior_weight": 0.0,
		"safe_action_index": 1,
	}
	if loss != expected_loss:
		raise ValueError("Route checkpoint loss contract is not the locked quality-first policy")
	if config.get("evaluated_splits") != ["train", "validation"]:
		raise ValueError("Deployable route checkpoint must not be selected using its test split")


def append_diagnosis_aux_features(
	pooled: torch.Tensor,
	diagnoses: Sequence[str | Mapping[str, Any]],
	config: Mapping[str, Any],
) -> torch.Tensor:
	"""Append train-normalized diagnosis statistics to frozen Qwen features."""

	contract = validate_diagnosis_aux_contract(config)
	if contract is None:
		return pooled
	if pooled.ndim != 2 or pooled.shape[0] != len(diagnoses):
		raise ValueError("Qwen route features and diagnoses must have the same batch size")
	qwen_feature_dim = int(config.get("qwen_feature_dim", -1))
	if pooled.shape[1] != qwen_feature_dim:
		raise ValueError("Qwen route feature dimension differs from checkpoint input contract")
	normalization = contract["normalization"]
	mean = torch.tensor(normalization["mean"], device=pooled.device, dtype=torch.float32)
	std = torch.tensor(normalization["std"], device=pooled.device, dtype=torch.float32)
	raw = torch.tensor(
		[diagnosis_aux_vector(diagnosis) for diagnosis in diagnoses],
		device=pooled.device,
		dtype=torch.float32,
	)
	result = torch.cat([pooled.float(), (raw - mean) / std], dim=-1)
	if result.shape[1] != int(config.get("input_dim", -1)):
		raise ValueError("Combined route-head input dimension differs from checkpoint")
	return result


def _safe_to_dict(value: Any) -> Dict[str, Any]:
	"""Call ``to_dict`` when available and require mapping output."""
	method = getattr(value, "to_dict", None)
	if not callable(method):
		return {}
	result = method()
	if not isinstance(result, Mapping):
		raise ValueError("Route feature processor to_dict() must return a mapping")
	return dict(result)


def _resolved_chat_template(processor: Any) -> Any:
	"""Resolve and canonicalize processor- or tokenizer-level chat templates."""
	value = getattr(processor, "chat_template", None)
	if value is None:
		value = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
	return _json_safe(value)


def _expected_component_digest(name: str, component: Mapping[str, Any]) -> str:
	"""Compute the canonical digest for one named feature component."""
	return _sha256_text(_canonical_json(dict(component)))


def validate_route_feature_config(config: Mapping[str, Any]) -> Dict[str, Any]:
	"""Reject checkpoints whose route-feature contract is absent or stale."""

	if config.get("feature_schema") != ROUTE_FEATURE_SCHEMA:
		raise ValueError(
			"Route checkpoint feature schema mismatch: "
			f"expected={ROUTE_FEATURE_SCHEMA!r} actual={config.get('feature_schema')!r}"
		)
	if int(config.get("feature_schema_version", -1)) != ROUTE_FEATURE_SCHEMA_VERSION:
		raise ValueError("Route checkpoint feature schema version mismatch")
	contract_value = config.get("feature_contract")
	if not isinstance(contract_value, Mapping):
		raise ValueError("Route checkpoint is missing its immutable feature_contract")
	contract = dict(contract_value)
	for key in (
		"feature_schema",
		"feature_schema_version",
		"diagnosis_schema",
		"label_levels",
		"qwen_model",
		"qwen_revision",
		"qwen_dtype",
		"qwen_quantization",
		"qwen_allow_cpu_offload",
		"qwen_effective_quantization",
		"qwen_distribution_signature",
	):
		if config.get(key) != contract.get(key):
			raise ValueError(f"Route checkpoint top-level/feature_contract mismatch for {key}")
	if contract.get("feature_schema") != ROUTE_FEATURE_SCHEMA:
		raise ValueError("Nested route feature schema mismatch")
	if int(contract.get("feature_schema_version", -1)) != ROUTE_FEATURE_SCHEMA_VERSION:
		raise ValueError("Nested route feature schema version mismatch")
	if contract.get("diagnosis_schema") != ROUTE_DIAGNOSIS_SCHEMA:
		raise ValueError("Nested route diagnosis schema mismatch")
	if contract.get("label_levels") != [1, 2, 3]:
		raise ValueError("Nested route label levels must be [1,2,3]")
	if IMMUTABLE_REVISION_RE.fullmatch(str(contract.get("qwen_revision") or "")) is None:
		raise ValueError("Route checkpoint Qwen revision is not immutable")
	if contract.get("qwen_dtype") != canonical_qwen_dtype_name(contract.get("qwen_dtype")):
		raise ValueError("Route checkpoint Qwen dtype is not canonical")
	if contract.get("qwen_quantization") != canonical_qwen_quantization_name(contract.get("qwen_quantization")):
		raise ValueError("Route checkpoint Qwen quantization is not canonical")
	if not isinstance(contract.get("qwen_allow_cpu_offload"), bool):
		raise ValueError("Route checkpoint Qwen CPU-offload policy is not boolean")
	expected_effective = qwen_effective_quantization(
		contract.get("qwen_quantization"),
		contract["qwen_allow_cpu_offload"],
	)
	if contract.get("qwen_effective_quantization") != expected_effective:
		raise ValueError("Route checkpoint Qwen effective quantization is inconsistent")
	if re.fullmatch(r"[0-9a-f]{64}", str(contract.get("qwen_distribution_signature") or "")) is None:
		raise ValueError("Route checkpoint is missing its Qwen distribution signature")
	components = contract.get("components")
	if not isinstance(components, Mapping) or set(components) != set(FEATURE_COMPONENTS):
		raise ValueError("Route checkpoint feature components are incomplete")
	for name in FEATURE_COMPONENTS:
		component = components.get(name)
		if not isinstance(component, Mapping):
			raise ValueError(f"Route checkpoint {name} feature component is invalid")
		expected = _expected_component_digest(name, component)
		if contract.get(f"{name}_contract_digest") != expected:
			raise ValueError(f"Route checkpoint {name} contract digest mismatch")
		if config.get(f"{name}_contract_digest") != expected:
			raise ValueError(f"Route checkpoint top-level {name} contract digest mismatch")
	combined = _sha256_text(_canonical_json(dict(components)))
	combined_key = "processor_chat_template_preprocess_pooling_contract_digest"
	if contract.get(combined_key) != combined or config.get(combined_key) != combined:
		raise ValueError("Route checkpoint combined feature-component digest mismatch")
	expected_contract_digest = _sha256_text(
		_canonical_json({key: value for key, value in contract.items() if key != "feature_contract_digest"})
	)
	if (
		contract.get("feature_contract_digest") != expected_contract_digest
		or config.get("feature_contract_digest") != expected_contract_digest
	):
		raise ValueError("Route checkpoint feature_contract digest mismatch")

	chat = components["chat_template"]
	if chat.get("messages_builder_source_sha256") != _sha256_text(inspect.getsource(build_route_encoder_messages)):
		raise ValueError("Route checkpoint was built with a different route-message encoder")
	preprocess = components["preprocess"]
	expected_preprocess = {
		"source": "aggregate.image_path/raw current image",
		"open": "PIL.Image.open",
		"color_mode": "RGB",
		"annotation": "none",
		"manual_resize_before_processor": "none",
		"processor_call": {
			"text": "batch chat strings",
			"images": "one raw image per row",
			"padding": True,
			"return_tensors": "pt",
		},
	}
	if preprocess != expected_preprocess:
		raise ValueError("Route checkpoint does not use the raw unannotated V10 route image contract")
	pooling = components["pooling"]
	if pooling.get("backbone_runtime") != {
		"dtype": contract["qwen_dtype"],
		"quantization": contract["qwen_quantization"],
		"allow_cpu_offload": contract["qwen_allow_cpu_offload"],
		"effective_quantization": contract["qwen_effective_quantization"],
	}:
		raise ValueError("Route checkpoint pooling Qwen runtime contract mismatch")
	if (
		pooling.get("hidden_layer") != -1
		or pooling.get("token_position") != -1
		or pooling.get("padding_precondition") != "tokenizer.padding_side=left"
		or pooling.get("expression") != "hidden_states[-1][:, -1, :].detach().float().cpu()"
		or pooling.get("output_dtype") != "float32"
	):
		raise ValueError("Route checkpoint pooling contract mismatch")
	return contract


def validate_route_qwen_distribution(
	config: Mapping[str, Any],
	current_distribution: Mapping[str, Any],
) -> Dict[str, Any]:
	"""Require label collection, feature extraction and online Qwen to match."""

	feature = validate_route_feature_config(config)
	try:
		current = validate_qwen_analyzer_l1_distribution_contract(current_distribution)
	except ValueError as exc:
		raise ValueError(f"Online Qwen analyzer/L1 distribution is invalid: {exc}") from exc
	if config.get("qwen_distribution_signature") != current["distribution_signature"]:
		raise ValueError("Online Qwen analyzer/L1 distribution differs from route-head training data")
	expected_feature_values = {
		"qwen_revision": current["model_revision"],
		"qwen_dtype": canonical_qwen_dtype_name(current["dtype"]),
		"qwen_quantization": canonical_qwen_quantization_name(current["quantization"]),
		"qwen_allow_cpu_offload": current["allow_cpu_offload"],
		"qwen_effective_quantization": current["effective_quantization"],
	}
	for key, expected in expected_feature_values.items():
		if feature.get(key) != expected:
			raise ValueError(f"Online Qwen distribution differs from route-feature extraction for {key}")
	return current


def validate_route_model_identity(config: Mapping[str, Any], model_path: str) -> None:
	"""Reject a mutable/aliased Qwen path before allocating the backbone."""

	contract = validate_route_feature_config(config)
	local_path = Path(model_path).expanduser()
	if not local_path.is_dir():
		raise ValueError(f"Online Qwen model path is not an existing snapshot directory: {model_path!r}")
	# Check the caller-visible snapshot component before resolving symlinks.  A
	# commit-named cache entry may legitimately link to its materialized local
	# directory, while passing that mutable directory (for example ``main``)
	# directly must remain forbidden.
	model_name = local_path.name.lower()
	revision = str(contract.get("qwen_revision") or "").lower()
	if model_name != revision:
		raise ValueError(
			"Online Qwen model path is not the checkpoint's pinned revision: "
			f"expected a snapshots/{revision} directory, got {model_path!r}"
		)


def validate_route_runtime_contract(
	config: Mapping[str, Any],
	*,
	processor: Any,
	encoder_callable: Any,
	model_path: str,
	dtype: Any,
	quantization: str,
	allow_cpu_offload: bool,
	effective_quantization: str,
) -> None:
	"""Match the loaded Qwen/processor runtime to the training feature cache."""

	contract = validate_route_feature_config(config)
	components = contract["components"]
	tokenizer = getattr(processor, "tokenizer", None)
	image_processor = getattr(processor, "image_processor", None)
	if tokenizer is None or str(getattr(tokenizer, "padding_side", "")) != "left":
		raise ValueError("V10 route runtime requires a left-padding tokenizer")
	actual_processor = {
		"loader": "transformers.AutoProcessor.from_pretrained(local_files_only=True,trust_remote_code=True)",
		"processor_class": f"{processor.__class__.__module__}.{processor.__class__.__qualname__}",
		"tokenizer_class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
		"tokenizer_padding_side": str(getattr(tokenizer, "padding_side", "")),
		"model_input_names": list(getattr(processor, "model_input_names", None) or []),
		"resolved_processor_config": _json_safe(_safe_to_dict(processor)),
		"resolved_image_processor_config": _json_safe(
			_safe_to_dict(image_processor) if image_processor is not None else {}
		),
		"resolved_tokenizer_init_kwargs": _json_safe(getattr(tokenizer, "init_kwargs", None) or {}),
	}
	if actual_processor != components["processor"]:
		raise ValueError("Online Qwen processor configuration differs from route-feature extraction")
	if _resolved_chat_template(processor) != components["chat_template"].get("resolved_template"):
		raise ValueError("Online Qwen chat template differs from route-feature extraction")
	if _sha256_text(inspect.getsource(encoder_callable)) != components["pooling"].get("encoder_source_sha256"):
		raise ValueError("Online route feature encoder source differs from training")
	dtype_name = canonical_qwen_dtype_name(dtype)
	if dtype_name != str(contract.get("qwen_dtype") or "").lower():
		raise ValueError("Online Qwen dtype differs from route-feature extraction")
	if canonical_qwen_quantization_name(quantization) != str(contract.get("qwen_quantization") or "none").lower():
		raise ValueError("Online Qwen quantization differs from route-feature extraction")
	if allow_cpu_offload is not contract.get("qwen_allow_cpu_offload"):
		raise ValueError("Online Qwen CPU-offload policy differs from route-feature extraction")
	if str(effective_quantization).strip().lower() != contract.get("qwen_effective_quantization"):
		raise ValueError("Online Qwen effective quantization differs from route-feature extraction")
	validate_route_model_identity(config, model_path)

# Route features use this exact multimodal conversation: the prompt is user
# context and the canonical diagnosis is an assistant record.  Keep commentary
# outside the function because checkpoints validate its source-code hash.
def build_route_encoder_messages(prompt: str, diagnosis: str, image: Any) -> List[Dict[str, Any]]:
	try:
		payload = json.loads(str(diagnosis))
	except json.JSONDecodeError as exc:
		raise ValueError("V10 route encoder requires canonical diagnosis JSON, not raw model text") from exc
	if not diagnosis_payload_is_valid(payload):
		# Canonical records carry schema_version, while strict native payloads do
		# not require it; both otherwise share the same allowed fields.
		if not (
			isinstance(payload, dict)
			and payload.get("schema_version") == ROUTE_DIAGNOSIS_SCHEMA
			and isinstance(payload.get("constraint_audit"), dict)
			and isinstance(payload.get("issues"), list)
		):
			raise ValueError("V10 route encoder received an invalid diagnosis contract")
	diagnosis = model_visible_diagnosis_text(payload)
	return [
		{
			"role": "system",
			"content": [
				{
					"type": "text",
					"text": ROUTE_ENCODER_SYSTEM_PROMPT_V10,
				}
			],
		},
		{
			"role": "user",
			"content": [
				{"type": "image", "image": image},
				{"type": "text", "text": f"Target prompt:\n{prompt.strip()}"},
			],
		},
		{
			"role": "assistant",
			"content": [{"type": "text", "text": diagnosis.strip()}],
		},
	]


def validate_route_flux2_policy_config(config: Mapping[str, Any]) -> Dict[str, Any]:
	"""Require the full Flux.2 collector policy, not only a family string."""

	try:
		contract = validate_flux2_policy_contract(config.get("flux2_policy_contract"))
	except (TypeError, ValueError) as exc:
		raise ValueError(f"Route checkpoint Flux.2 collector policy is invalid: {exc}") from exc
	digest = flux2_policy_digest(contract)
	if config.get("flux2_policy_digest") != digest:
		raise ValueError("Route checkpoint Flux.2 collector-policy digest mismatch")
	return contract


def validate_route_flux2_runtime_policy(config: Mapping[str, Any], actual_policy: Mapping[str, Any]) -> None:
	"""Reject online L1/L2 editor settings that differ from training."""

	expected = validate_route_flux2_policy_config(config)
	validate_matching_flux2_policy(expected, actual_policy)


def validate_route_l3_policy_config(config: Mapping[str, Any]) -> Dict[str, Any]:
	"""Require one exact, homogeneous initial-base L3 policy per head."""

	binding_counts = config.get("l3_training_initial_identity_binding_counts") or {}
	if not isinstance(binding_counts, Mapping):
		raise ValueError("Route checkpoint L3 identity-binding counts are invalid")
	try:
		legacy_count = int(binding_counts.get("legacy_bagel_family_only_unverified_revision", 0) or 0)
	except (TypeError, ValueError) as exc:
		raise ValueError("Route checkpoint legacy L3 identity count is invalid") from exc
	if legacy_count < 0:
		raise ValueError("Route checkpoint legacy L3 identity count is invalid")
	if bool(config.get("allow_legacy_l3_identity_ablation", False)) or legacy_count > 0:
		raise ValueError(
			"Route checkpoint was trained with legacy/unverified BAGEL identity and is offline-ablation-only; "
			"normal V10 deployment requires exact initial-model identity"
		)

	try:
		contract = validate_l3_policy_contract(config.get("l3_policy_contract"))
	except (TypeError, ValueError) as exc:
		raise ValueError(f"Route checkpoint initial-base L3 policy is invalid: {exc}") from exc
	digest = l3_policy_digest(contract)
	if config.get("l3_policy_digest") != digest:
		raise ValueError("Route checkpoint initial-base L3 policy digest mismatch")
	return contract


def validate_route_l3_runtime_policy(config: Mapping[str, Any], actual_policy: Mapping[str, Any]) -> None:
	"""Fail before model allocation if serving would use a different L3 base."""

	expected = validate_route_l3_policy_config(config)
	validate_matching_l3_policy(expected, actual_policy)


def peek_route_head_config(path: str | Path) -> Dict[str, Any]:
	"""Read checkpoint metadata for cheap pre-backbone startup validation."""

	checkpoint_path = Path(path)
	if checkpoint_path.is_dir():
		checkpoint_path = checkpoint_path / "route_head.pt"
	payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	config = payload.get("config")
	if not isinstance(config, Mapping):
		raise ValueError(f"Route checkpoint is missing config metadata: {checkpoint_path}")
	return dict(config)


def load_route_head(path: str | Path, device: torch.device | str) -> tuple[RouteMLP, Dict[str, Any]]:
	"""Validate checkpoint metadata, load weights, and return an eval-mode head."""
	checkpoint_path = Path(path)
	if checkpoint_path.is_dir():
		checkpoint_path = checkpoint_path / "route_head.pt"
	payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	config = dict(payload.get("config") or {})
	if int(config.get("num_labels", 0)) != 3:
		raise ValueError(f"V10 route head must have exactly 3 outputs (L1-L3): {checkpoint_path}")
	label_levels = [int(value) for value in config.get("label_levels") or [1, 2, 3]]
	if label_levels != [1, 2, 3]:
		raise ValueError(f"V10 route head label_levels must be [1, 2, 3]: {checkpoint_path}")
	if config.get("diagnosis_schema") != ROUTE_DIAGNOSIS_SCHEMA:
		raise ValueError(
			"Route checkpoint diagnosis schema mismatch: "
			f"expected={ROUTE_DIAGNOSIS_SCHEMA!r} actual={config.get('diagnosis_schema')!r}. "
			"V10 must not deploy a head trained on raw/V9 reflections."
		)
	if config.get("action_policy_version") != ROUTER_ACTION_POLICY_VERSION:
		raise ValueError("Route checkpoint action-policy version is not the V10 initial-base-L3 atomic policy")
	if config.get("repair_executor_family") != ROUTER_REPAIR_EXECUTOR_FAMILY:
		raise ValueError("Route checkpoint was not trained for Flux.2 L1/L2 plus initial-base L3 actions")
	validate_route_flux2_policy_config(config)
	validate_route_l3_policy_config(config)
	validate_route_feature_config(config)
	aux_contract = validate_diagnosis_aux_contract(config)
	if aux_contract is not None:
		qwen_feature_dim = int(config.get("qwen_feature_dim", -1))
		if qwen_feature_dim <= 0 or qwen_feature_dim + len(DIAGNOSIS_AUX_FEATURE_NAMES) != int(config["input_dim"]):
			raise ValueError("Route checkpoint diagnosis auxiliary input dimensions are inconsistent")
	validate_route_training_contract(config)
	head = RouteMLP(
		input_dim=int(config["input_dim"]),
		hidden_dim=int(config.get("hidden_dim", 512)),
		num_labels=3,
		dropout=float(config.get("dropout", 0.15)),
	)
	head.load_state_dict(payload["state_dict"])
	head.to(device=device, dtype=torch.float32).eval()
	return head, {**config, "checkpoint_path": str(checkpoint_path)}


def route_prediction(logits: torch.Tensor, config: Dict[str, Any]) -> List[Dict[str, Any]]:
	"""Convert three-class logits into auditable L1/L2/L3 predictions."""
	label_levels = [int(value) for value in config.get("label_levels") or [1, 2, 3]]
	if logits.shape[-1] != len(label_levels):
		raise ValueError(f"route-head logits/config mismatch: logits={logits.shape[-1]} levels={label_levels}")
	probabilities = torch.softmax(logits.float(), dim=-1)
	confidence, predicted = probabilities.max(dim=-1)
	results: List[Dict[str, Any]] = []
	for row_logits, row_probs, row_confidence, row_predicted in zip(
		logits.detach().cpu(), probabilities.detach().cpu(), confidence.detach().cpu(), predicted.detach().cpu()
	):
		level = int(label_levels[int(row_predicted.item())])
		results.append(
			{
				"level": level,
				"route": ROUTE_NAMES[level],
				"confidence": round(float(row_confidence.item()), 6),
				"probabilities": [round(float(value), 6) for value in row_probs.tolist()],
				"logits": [round(float(value), 6) for value in row_logits.tolist()],
				"checkpoint": config.get("checkpoint_path", ""),
			}
		)
	return results


def qwen_raw_reports_problem(raw: Any) -> bool:
	"""Return whether raw native-Qwen triage says the image needs repair.

	Invalid/partial JSON is treated as problematic so the normal JSON retry path
	can recover it without accidentally suppressing a needed route prediction.
	"""
	text = str(raw or "").strip()
	decoder = json.JSONDecoder()
	obj: Any = None
	for position, character in enumerate(text):
		if character != "{":
			continue
		try:
			obj, _ = decoder.raw_decode(text[position:])
			break
		except json.JSONDecodeError:
			continue
	if not isinstance(obj, dict):
		return True
	raw_satisfied = obj.get("fully_satisfied", False)
	fully_satisfied = raw_satisfied is True or str(raw_satisfied).strip().lower() == "true"
	return not fully_satisfied


def append_route_marker(raw: str, prediction: Dict[str, Any]) -> str:
	"""Attach a machine-readable prediction without modifying diagnosis JSON."""
	payload = json.dumps(prediction, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
	encoded = base64.urlsafe_b64encode(payload).decode("ascii")
	return f"{raw.rstrip()}\n{ROUTE_MARKER}{encoded}"


def extract_route_marker(raw: Any) -> Optional[Dict[str, Any]]:
	"""Decode the final embedded route marker, returning ``None`` if malformed."""
	text = str(raw or "")
	position = text.rfind(ROUTE_MARKER)
	if position < 0:
		return None
	encoded = text[position + len(ROUTE_MARKER) :].strip().split()[0]
	try:
		return json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
	except Exception:
		return None


def apply_route_head_decision(state: Any, analysis: Dict[str, Any]) -> Dict[str, Any]:
	"""Apply native L0 or an embedded MLP L1/L2/L3 decision to stage analysis.

	The MLP is enforced only during global triage.  This function mutates the
	state-machine-facing category/mode fields but leaves canonical diagnosis
	facts unchanged; L1 planning occurs in a separate post-route call.
	"""
	prediction = extract_route_marker(analysis.get("raw_output"))
	native_no_problem = bool(analysis.get("fully_satisfied"))
	decision: Dict[str, Any] = {
		"enabled": prediction is not None and not native_no_problem,
		"enforced": False,
		"source": "qwen_native_gate" if native_no_problem else "qwen35_mlp_head",
		"native_qwen_has_problem": not native_no_problem,
		"original_recommended_mode": analysis.get("recommended_mode"),
		"original_category": analysis.get("category"),
	}
	if native_no_problem:
		decision.update({"level": 0, "route": ROUTE_NAMES[0], "confidence": 1.0, "enforced": True})
		decision["enforced_mode"] = "done"
		analysis["route_head_decision"] = decision
		return decision
	if prediction is None:
		decision["fallback"] = "missing_route_head_prediction"
		analysis["route_head_decision"] = decision
		return decision
	decision.update(prediction)
	level = int(prediction.get("level", -1))
	if str(getattr(state, "phase", "global")) != "global":
		decision["fallback"] = "route_head_only_enforced_during_global_triage"
		analysis["route_head_decision"] = decision
		return decision
	if level not in {1, 2, 3}:
		decision["fallback"] = "invalid_route_head_level"
		analysis["route_head_decision"] = decision
		return decision
	if level == 1:
		analysis["fully_satisfied"] = False
		analysis["recommended_mode"] = "micro"
		analysis["category"] = "L1"
		analysis["edits"] = []
		decision["planning_required"] = "l1_localization_plan"
	elif level == 2:
		analysis["fully_satisfied"] = False
		analysis["recommended_mode"] = "global"
		analysis["category"] = "L2"
		analysis["edits"] = []
	elif level == 3:
		analysis["fully_satisfied"] = False
		analysis["recommended_mode"] = "regenerate"
		analysis["category"] = "L3"
		analysis["edits"] = []
	decision["enforced"] = True
	decision["enforced_mode"] = analysis.get("recommended_mode")
	analysis["route_head_decision"] = decision
	return decision


__all__ = [
	"ROUTE_MARKER",
	"ROUTE_DIAGNOSIS_SCHEMA",
	"ROUTE_FEATURE_SCHEMA",
	"ROUTE_FEATURE_SCHEMA_VERSION",
	"ROUTE_CHECKPOINT_VERSION",
	"ROUTE_TARGET_MODE",
	"ROUTE_TRAIN_TARGET_SHRINKAGE",
	"ROUTE_CANDIDATE_QUALITY_SCHEMA",
	"ROUTE_DIAGNOSIS_GROUP_EB_SCHEMA",
	"ROUTE_ENCODER_SYSTEM_PROMPT_V10",
	"ROUTE_NAMES",
	"DIAGNOSIS_AUX_FEATURE_NAMES",
	"DIAGNOSIS_AUX_SCHEMA",
	"RouteMLP",
	"append_diagnosis_aux_features",
	"append_route_marker",
	"apply_route_head_decision",
	"build_diagnosis_aux_contract",
	"build_route_encoder_messages",
	"canonical_qwen_dtype_name",
	"canonical_qwen_quantization_name",
	"diagnosis_aux_extractor_source_sha256",
	"diagnosis_aux_vector",
	"extract_route_marker",
	"load_route_head",
	"peek_route_head_config",
	"qwen_raw_reports_problem",
	"route_prediction",
	"validate_route_feature_config",
	"validate_route_flux2_policy_config",
	"validate_route_flux2_runtime_policy",
	"validate_route_model_identity",
	"validate_route_qwen_distribution",
	"validate_route_runtime_contract",
	"validate_diagnosis_aux_contract",
	"validate_route_training_contract",
]
