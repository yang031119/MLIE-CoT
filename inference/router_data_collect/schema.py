"""Schema for Flux.2 L1/L2 plus initial-base-bound L3 router data."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from inference.main_method.modules.contracts import (
	ACTION_PLAN_SCHEMA,
	DIAGNOSIS_SCHEMA,
	L1_CROP_MIN_SPAN,
	L1_CROP_PAD_RATIO,
	L1_MAX_CROPS,
	build_level_action_plan,
	canonical_diagnosis_text,
	canonicalize_diagnosis as canonicalize_v10_diagnosis,
	model_visible_diagnosis_text,
	validate_distribution_contract,
)
from inference.main_method.modules.flux_policy import build_flux2_policy_contract, flux2_policy_digest
from inference.main_method.modules.l3_policy import (
	L3_PROMPT_VERSION,
	build_l3_policy_contract,
	l3_policy_digest,
	normalize_initial_generator_family,
	validate_l3_policy_contract,
)

from .canonical import canonical_json, sha256_file, sha256_json, stable_id


SCHEMA_VERSION = "vcot_v10_router_data_initial_base_l3_v3"
PROFILE_NAME = "flux2_l1_l2_initial_base_l3_v10_joint_v3"
ACTION_LEVELS = (1, 2, 3)
ACTION_NAMES = {
	1: "flux2_joint_local_full_plus_bbox_crops",
	2: "flux2_full_image_edit",
	3: "initial_base_prompt_only_regenerate",
}
ACTION_OPERATIONS = {1: "edit_multi_reference", 2: "edit_single_reference", 3: "generate_text_only"}
FORBIDDEN_DIAGNOSIS_KEYS = {
	"action_level",
	"category",
	"expected_success",
	"expected_utility",
	"recommended_mode",
	"recommended_action",
	"recommendedmode",
	"route",
	"repair_level",
	"repairlevel",
	"selected_action",
	"mode",
	"level",
	"bbox",
	"bboxes",
	"box_xyxy",
	"crop",
	"edit_prompt",
	"edits",
	"fix",
	"global_edit_instruction",
	"instruction",
	"quad_points",
}
PIN_PLACEHOLDERS = {
	"",
	"main",
	"latest",
	"master",
	"replace_with_pinned_flux2_revision",
	"replace_with_exact_snapshot_commit",
}
EMPTY_TEXT_SENTINELS = {"", "none", "null", "n/a", "na", "unknown"}
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$", flags=re.IGNORECASE)


class DataValidationError(ValueError):
	"""Raised when an artifact violates the immutable dataset contract."""


def _require(condition: bool, message: str) -> None:
	if not condition:
		raise DataValidationError(message)


def _number(value: Any, name: str, lower: float | None = None, upper: float | None = None) -> float:
	try:
		result = float(value)
	except (TypeError, ValueError) as exc:
		raise DataValidationError(f"{name} must be numeric") from exc
	_require(math.isfinite(result), f"{name} must be finite")
	if lower is not None:
		_require(result >= lower, f"{name} must be >= {lower}")
	if upper is not None:
		_require(result <= upper, f"{name} must be <= {upper}")
	return result


def normalize_text(value: Any) -> str:
	"""Convert a scalar to stripped text, mapping null to an empty string."""
	return "" if value is None else " ".join(str(value).strip().split())


def meaningful_text(value: Any) -> str:
	"""Normalize optional text and reject empty or sentinel-like values."""
	text = normalize_text(value)
	return "" if text.casefold() in EMPTY_TEXT_SENTINELS else text


def strict_bool(value: Any, name: str) -> bool:
	"""Accept only real booleans rather than truthy strings or integers."""
	if isinstance(value, bool):
		return value
	if value in (0, 1):
		return bool(value)
	if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
		return value.strip().casefold() == "true"
	raise DataValidationError(f"{name} must be boolean")


def find_forbidden_key(value: Any) -> str | None:
	"""Recursively locate keys that would leak labels or future-state information."""
	if isinstance(value, Mapping):
		for key, child in value.items():
			normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).strip().casefold()).strip("_")
			if normalized_key in FORBIDDEN_DIAGNOSIS_KEYS:
				return str(key)
			found = find_forbidden_key(child)
			if found:
				return found
	elif isinstance(value, list):
		for child in value:
			found = find_forbidden_key(child)
			if found:
				return found
	return None


def normalize_prompt_for_group(value: Any) -> str:
	"""Canonicalize prompt spacing and case for duplicate-group comparison."""
	return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def immutable_revision(value: Any, name: str) -> str:
	"""Require a full immutable source revision instead of a moving branch name."""
	revision = normalize_text(value)
	_require(revision.casefold() not in PIN_PLACEHOLDERS, f"{name} must be pinned, not main/latest/placeholder")
	_require(
		IMMUTABLE_REVISION_RE.fullmatch(revision) is not None,
		f"{name} must be an immutable 40-64 character hexadecimal commit",
	)
	return revision.lower()


def canonical_quad(value: Any, name: str = "quad_points") -> list[list[int]]:
	"""Normalize a four-coordinate box and enforce positive axis-aligned area."""
	_require(isinstance(value, list) and len(value) == 4, f"{name} must contain exactly four points")
	points: list[list[int]] = []
	for index, point in enumerate(value):
		_require(isinstance(point, (list, tuple)) and len(point) == 2, f"{name}[{index}] must be [x,y]")
		coordinates: list[int] = []
		for axis, coordinate in zip(("x", "y"), point):
			_require(
				isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool),
				f"{name}[{index}].{axis} must be a JSON number",
			)
			numeric = float(coordinate)
			_require(math.isfinite(numeric) and numeric.is_integer(), f"{name}[{index}].{axis} must be an integer")
			coordinates.append(int(numeric))
		x, y = coordinates
		points.append([x, y])
	xs = [point[0] for point in points]
	ys = [point[1] for point in points]
	x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
	_require(x1 > x0 and y1 > y0, f"{name} must have non-zero area")
	ordered = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
	_require({tuple(point) for point in points} == {tuple(point) for point in ordered}, f"{name} must be an axis-aligned rectangle")
	return ordered


def canonical_edit(value: Mapping[str, Any], index: int, *, name: str = "l1_action_plan.edits") -> dict[str, Any]:
	"""Normalize a post-route L1 edit, including legacy V9 edit rows."""

	_require(isinstance(value, Mapping), f"{name}[{index}] must be an object")
	edit_prompt = meaningful_text(
		value.get("instruction") or value.get("edit_prompt") or value.get("whole_edit") or value.get("issue")
	)
	_require(bool(edit_prompt), f"{name}[{index}] needs instruction/edit_prompt/whole_edit")
	quad = canonical_quad(value.get("quad_points"), f"{name}[{index}].quad_points")
	xs = [point[0] for point in quad]
	ys = [point[1] for point in quad]
	return {
		"id": normalize_text(value.get("id")) or f"edit_{index + 1:02d}",
		"priority": index + 1,
		"target_entity": meaningful_text(value.get("target_entity") or value.get("target_object")) or "localized region",
		"target_object": meaningful_text(value.get("target_entity") or value.get("target_object")) or "localized region",
		"issue": meaningful_text(value.get("issue")),
		"instruction": edit_prompt,
		"edit_prompt": edit_prompt,
		"whole_edit": meaningful_text(value.get("whole_edit")) or edit_prompt,
		"quad_points": quad,
		"box_xyxy": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
	}


def canonicalize_diagnosis(value: Any) -> dict[str, Any]:
	"""Canonicalize new diagnoses while accepting legacy rows for migration.

	Legacy executable fields are consumed only by ``canonicalize_l1_action_plan``;
	they are never serialized into the route-visible diagnosis.
	"""

	_require(isinstance(value, Mapping), "diagnosis must be an object")
	leaked = find_forbidden_key({key: child for key, child in value.items() if key not in {
		"bbox", "bboxes", "box_xyxy", "crop", "edit_prompt", "edits", "fix",
		"global_edit_instruction", "instruction", "quad_points",
	}})
	_require(leaked is None, f"diagnosis contains legacy route leakage: {leaked}")
	for text_value in _iter_text_values(value):
		_require(
			re.search(r"\bL[0-3]\b", text_value, flags=re.IGNORECASE) is None,
			"diagnosis contains textual L0/L1/L2/L3 route leakage",
		)
	return canonicalize_v10_diagnosis(value)


def canonicalize_l1_action_plan(
	value: Any,
	*,
	target_prompt: str,
	diagnosis: Mapping[str, Any],
	width: int,
	height: int,
) -> dict[str, Any] | None:
	"""Return a canonical one-call L1 plan or ``None`` when no plan exists."""

	if not isinstance(value, Mapping):
		return None
	edits_value = value.get("edits") or []
	if not isinstance(edits_value, list) or not 1 <= len(edits_value) <= 3:
		return None
	edits = [canonical_edit(item, index) for index, item in enumerate(edits_value)]
	for edit_index, edit in enumerate(edits):
		xs = [point[0] for point in edit["quad_points"]]
		ys = [point[1] for point in edit["quad_points"]]
		_require(min(xs) >= 0 and max(xs) < width, f"l1_action_plan.edits[{edit_index}] bbox is outside image width")
		_require(min(ys) >= 0 and max(ys) < height, f"l1_action_plan.edits[{edit_index}] bbox is outside image height")
		area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
		_require(area < width * height * 0.95, f"l1_action_plan.edits[{edit_index}] is a whole-image pseudo-local box")
	plan = build_level_action_plan(
		1,
		target_prompt,
		diagnosis,
		edits=edits,
		width=width,
		height=height,
	)
	validate_distribution_contract(plan)
	return plan


def _iter_text_values(value: Any):
	if isinstance(value, Mapping):
		for child in value.values():
			yield from _iter_text_values(child)
	elif isinstance(value, list):
		for child in value:
			yield from _iter_text_values(child)
	elif isinstance(value, str):
		yield value


@dataclass(frozen=True)
class Flux2Policy:
	"""Define and validate the immutable FLUX.2 L1/L2 executor policy stored with every collection run."""
	model_id: str
	model_revision: str
	resolution: int
	steps: int
	guidance_scale: float
	max_sequence_length: int
	dtype: str
	l1_roi_pad_ratio: float
	l1_min_span: int
	l1_max_edits: int
	prompt_version: str

	@classmethod
	def from_mapping(cls, value: Mapping[str, Any], strict_revision: bool = True) -> "Flux2Policy":
		"""Construct and validate this contract from an untrusted mapping."""
		_require(isinstance(value, Mapping), "config.flux2 must be an object")
		model_id = normalize_text(value.get("model_id"))
		model_revision = normalize_text(value.get("model_revision"))
		_require(bool(model_id), "config.flux2.model_id is required")
		_require(bool(model_revision), "config.flux2.model_revision is required")
		if strict_revision:
			model_revision = immutable_revision(model_revision, "Flux.2 model_revision")
		return cls(
			model_id=model_id,
			model_revision=model_revision,
			resolution=int(value.get("resolution", 1024)),
			steps=int(value.get("steps", 28)),
			guidance_scale=_number(value.get("guidance_scale", 3.5), "guidance_scale", 0.0),
			max_sequence_length=int(value.get("max_sequence_length", 512)),
			dtype=normalize_text(value.get("dtype") or "bfloat16").lower(),
			l1_roi_pad_ratio=_number(
				value.get("l1_roi_pad_ratio", L1_CROP_PAD_RATIO),
				"l1_roi_pad_ratio",
				1.0,
			),
			l1_min_span=int(value.get("l1_min_span", L1_CROP_MIN_SPAN)),
			l1_max_edits=int(value.get("l1_max_edits", L1_MAX_CROPS)),
			prompt_version=normalize_text(value.get("prompt_version") or "vcot_v10_flux2_joint_atomic_v2"),
		)

	def validate(self) -> None:
		"""Reject invalid FLUX.2 dimensions, precision, or crop settings."""
		_require(self.resolution > 0, "resolution must be positive")
		_require(self.steps > 0, "steps must be positive")
		_require(self.max_sequence_length > 0, "max_sequence_length must be positive")
		_require(self.dtype in {"bfloat16", "float16"}, "dtype must be bfloat16 or float16")
		_require(self.l1_min_span >= 2, "l1_min_span must be >= 2")
		_require(
			1 <= self.l1_max_edits <= L1_MAX_CROPS,
			f"l1_max_edits must be between 1 and {L1_MAX_CROPS}",
		)

	def as_dict(self) -> dict[str, Any]:
		"""Serialize the validated FLUX.2 policy for the locked config."""
		return {
			"family": "flux2",
			"model_id": self.model_id,
			"model_revision": self.model_revision,
			"resolution": self.resolution,
			"steps": self.steps,
			"guidance_scale": self.guidance_scale,
			"max_sequence_length": self.max_sequence_length,
			"dtype": self.dtype,
			"l1_roi_pad_ratio": self.l1_roi_pad_ratio,
			"l1_min_span": self.l1_min_span,
			"l1_max_edits": self.l1_max_edits,
			"prompt_version": self.prompt_version,
		}

	@property
	def collector_contract(self) -> dict[str, Any]:
		"""Build the exact online-compatible FLUX.2 executor contract."""
		return build_flux2_policy_contract(
			model_id=self.model_id,
			model_revision=self.model_revision,
			resolution=self.resolution,
			steps=self.steps,
			guidance_scale=self.guidance_scale,
			max_sequence_length=self.max_sequence_length,
			dtype=self.dtype,
			l1_roi_pad_ratio=self.l1_roi_pad_ratio,
			l1_min_span=self.l1_min_span,
			l1_max_crops=self.l1_max_edits,
			prompt_version=self.prompt_version,
		)

	@property
	def collector_contract_digest(self) -> str:
		"""Hash the complete FLUX.2 collector contract."""
		return flux2_policy_digest(self.collector_contract)

	@property
	def signature(self) -> str:
		"""Return the short stable identity used to bind jobs to this policy."""
		return stable_id("policy", self.collector_contract, length=32)


@dataclass(frozen=True)
class InitialBaseL3Policy:
	"""One homogeneous L3 executor policy for a router dataset/head."""

	contract: dict[str, Any]

	@classmethod
	def from_mapping(
		cls,
		value: Mapping[str, Any] | None,
		*,
		flux2: Flux2Policy,
		strict_revision: bool,
	) -> "InitialBaseL3Policy":
		"""Construct and validate this contract from an untrusted mapping."""
		if not value:
			# Backward-compatible config parsing is safe only for Flux-initial
			# datasets: the derived policy is still explicit in config.lock.json.
			contract = build_l3_policy_contract(
				family="flux2",
				model_id=flux2.model_id,
				model_revision=flux2.model_revision,
				resolution=flux2.resolution,
				steps=flux2.steps,
				guidance_scale=flux2.guidance_scale,
				max_sequence_length=flux2.max_sequence_length,
				dtype=flux2.dtype,
				prompt_version=L3_PROMPT_VERSION,
				require_immutable_revision=strict_revision,
			)
			return cls(contract)
		_require(isinstance(value, Mapping), "config.l3 must be an object")
		if value.get("schema_version") is not None:
			try:
				return cls(
					validate_l3_policy_contract(
						value,
						require_immutable_revision=strict_revision,
					)
				)
			except ValueError as exc:
				raise DataValidationError(f"invalid locked config.l3 policy: {exc}") from exc
		try:
			contract = build_l3_policy_contract(
				family=normalize_text(value.get("family")),
				model_id=normalize_text(value.get("model_id")),
				model_revision=normalize_text(value.get("model_revision")),
				auxiliary_model_id=normalize_text(value.get("auxiliary_model_id")),
				auxiliary_model_revision=normalize_text(value.get("auxiliary_model_revision")),
				resolution=int(value.get("resolution", flux2.resolution)),
				steps=int(value.get("steps", flux2.steps)),
				guidance_scale=_number(value.get("guidance_scale", flux2.guidance_scale), "l3.guidance_scale", 0.0),
				max_sequence_length=(
					None if value.get("max_sequence_length") is None else int(value.get("max_sequence_length"))
				),
				max_latent_size=(None if value.get("max_latent_size") is None else int(value.get("max_latent_size"))),
				dtype=normalize_text(value.get("dtype") or "bfloat16"),
				prompt_version=normalize_text(value.get("prompt_version") or L3_PROMPT_VERSION),
				require_immutable_revision=strict_revision,
			)
		except (TypeError, ValueError) as exc:
			raise DataValidationError(f"invalid config.l3 policy: {exc}") from exc
		return cls(contract)

	@property
	def family(self) -> str:
		"""Return the normalized initial-image executor family."""
		return str(self.contract["executor"]["family"])

	@property
	def digest(self) -> str:
		"""Hash the complete L3 executor and generation contract."""
		return l3_policy_digest(self.contract)

	def as_dict(self) -> dict[str, Any]:
		"""Return a detached serializable copy of the L3 contract."""
		return dict(self.contract)


@dataclass(frozen=True)
class DatasetConfig:
	"""Represent the complete collection configuration after strict path, model, and policy validation."""
	dataset_name: str
	dataset_seed: int
	rollouts_per_action: int
	require_flux2_initial_state: bool
	require_verified_initial_record: bool
	allow_legacy_external_l3_identity: bool
	policy: Flux2Policy
	l3_policy: InitialBaseL3Policy
	split_fractions: dict[str, float]
	utility: dict[str, float]
	soft_target_temperature: float
	acceptable_epsilon: float
	balance_train_soft_mass: bool
	max_train_class_weight: float

	@classmethod
	def from_mapping(cls, value: Mapping[str, Any], strict_revision: bool = True) -> "DatasetConfig":
		"""Construct and validate this contract from an untrusted mapping."""
		_require(isinstance(value, Mapping), "dataset config must be an object")
		_require(value.get("schema_version", SCHEMA_VERSION) == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")
		_require(value.get("profile", PROFILE_NAME) == PROFILE_NAME, f"profile must be {PROFILE_NAME}")
		split_value = value.get("split") or {"train": 0.8, "validation": 0.1, "test": 0.1}
		_require(isinstance(split_value, Mapping), "config.split must be an object")
		splits = {name: _number(split_value.get(name), f"split.{name}", 0.0, 1.0) for name in ("train", "validation", "test")}
		_require(abs(sum(splits.values()) - 1.0) <= 1e-9, "split fractions must sum to 1")
		utility_value = value.get("utility") or {}
		_require(isinstance(utility_value, Mapping), "config.utility must be an object")
		balance_value = value.get("training_balance") or {}
		_require(isinstance(balance_value, Mapping), "config.training_balance must be an object")
		utility = {
			"target_gain": _number(utility_value.get("target_gain", 3.0), "utility.target_gain", 0.0),
			"targeted_fix": _number(utility_value.get("targeted_fix", 2.0), "utility.targeted_fix", 0.0),
			"preservation": _number(utility_value.get("preservation", 1.0), "utility.preservation", 0.0),
			"introduced_new_error": _number(utility_value.get("introduced_new_error", 3.0), "utility.introduced_new_error", 0.0),
			"artifact_severity": _number(utility_value.get("artifact_severity", 1.0), "utility.artifact_severity", 0.0),
			"success": _number(utility_value.get("success", 0.0), "utility.success", 0.0),
			"cost_per_call": _number(utility_value.get("cost_per_call", 0.15), "utility.cost_per_call", 0.0),
		}
		flux2_policy = Flux2Policy.from_mapping(value.get("flux2") or {}, strict_revision=strict_revision)
		config = cls(
			dataset_name=normalize_text(value.get("dataset_name") or "vcot_flux2_router_data"),
			dataset_seed=int(value.get("dataset_seed", 20260715)),
			rollouts_per_action=int(value.get("rollouts_per_action", 2)),
			require_flux2_initial_state=strict_bool(
				value.get("require_flux2_initial_state", True),
				"require_flux2_initial_state",
			),
			require_verified_initial_record=strict_bool(
				value.get("require_verified_initial_record", True),
				"require_verified_initial_record",
			),
			allow_legacy_external_l3_identity=strict_bool(
				value.get("allow_legacy_external_l3_identity", False),
				"allow_legacy_external_l3_identity",
			),
			policy=flux2_policy,
			l3_policy=InitialBaseL3Policy.from_mapping(
				value.get("l3"),
				flux2=flux2_policy,
				strict_revision=strict_revision,
			),
			split_fractions=splits,
			utility=utility,
			soft_target_temperature=_number(value.get("soft_target_temperature", 0.5), "soft_target_temperature", 1e-6),
			acceptable_epsilon=_number(value.get("acceptable_epsilon", 0.25), "acceptable_epsilon", 0.0),
			balance_train_soft_mass=strict_bool(
				balance_value.get("enabled", True),
				"training_balance.enabled",
			),
			max_train_class_weight=_number(
				balance_value.get("max_class_weight", 4.0),
				"training_balance.max_class_weight",
				1.0,
			),
		)
		config.validate()
		return config

	def validate(self) -> None:
		"""Validate cross-policy consistency and all dataset-wide invariants."""
		_require(bool(self.dataset_name), "dataset_name cannot be empty")
		_require(self.rollouts_per_action > 0, "rollouts_per_action must be positive")
		self.policy.validate()
		try:
			validate_l3_policy_contract(self.l3_policy.contract)
		except ValueError as exc:
			raise DataValidationError(f"invalid L3 policy: {exc}") from exc
		if self.l3_policy.family == "flux2":
			# Online V10 has one resolved Flux.2 service for initial generation,
			# L1/L2 and Flux-initial L3.  Keep the offline executor distribution
			# identical while retaining a distinct L3 prompt-version contract.
			flux_executor = self.policy.collector_contract["executor"]
			flux_generation = self.policy.collector_contract["generation"]
			l3_executor = self.l3_policy.contract["executor"]
			l3_generation = self.l3_policy.contract["generation"]
			_require(
				all(l3_executor[key] == flux_executor[key] for key in ("family", "implementation", "loader", "model_id", "model_revision", "dtype")),
				"Flux-initial L3 model identity/dtype must match the online Flux.2 L1/L2 service",
			)
			_require(
				all(l3_generation[key] == flux_generation[key] for key in ("resolution", "steps", "guidance_scale", "max_sequence_length")),
				"Flux-initial L3 generation settings must match the online Flux.2 service",
			)

	@property
	def action_policy_signature(self) -> str:
		"""Bind L1/L2 and L3 executor contracts into one stable policy identity."""
		return stable_id(
			"action_policy",
			{
				"flux2_l1_l2": self.policy.collector_contract,
				"initial_base_l3": self.l3_policy.contract,
			},
			length=32,
		)

	def as_dict(self) -> dict[str, Any]:
		"""Serialize the fully validated configuration for ``config.lock.json``."""
		return {
			"schema_version": SCHEMA_VERSION,
			"profile": PROFILE_NAME,
			"dataset_name": self.dataset_name,
			"dataset_seed": self.dataset_seed,
			"rollouts_per_action": self.rollouts_per_action,
			"require_flux2_initial_state": self.require_flux2_initial_state,
			"require_verified_initial_record": self.require_verified_initial_record,
			"allow_legacy_external_l3_identity": self.allow_legacy_external_l3_identity,
			"flux2": self.policy.as_dict(),
			"l3": self.l3_policy.as_dict(),
			"split": self.split_fractions,
			"utility": self.utility,
			"soft_target_temperature": self.soft_target_temperature,
			"acceptable_epsilon": self.acceptable_epsilon,
			"training_balance": {
				"enabled": self.balance_train_soft_mass,
				"max_class_weight": self.max_train_class_weight,
			},
			"policy_signature": self.action_policy_signature,
		}


def image_descriptor(path: Path) -> dict[str, Any]:
	"""Describe an image by resolved path, dimensions, mode, and content hash."""
	_require(path.is_file(), f"source image does not exist: {path}")
	try:
		from PIL import Image

		with Image.open(path) as image:
			width, height = image.size
	except Exception as exc:
		raise DataValidationError(f"Cannot read source image {path}: {exc}") from exc
	# Preserve the caller-visible absolute path instead of resolving symlinks.
	# Completed datasets may be moved behind an explicit compatibility symlink;
	# resolving it would make an unchanged, hash-identical image fail its frozen
	# descriptor solely because the storage location changed.
	return {
		"path": str(path.absolute()),
		"sha256": sha256_file(path),
		"width": int(width),
		"height": int(height),
	}


def canonicalize_state(raw: Mapping[str, Any], config: DatasetConfig, project_root: Path) -> dict[str, Any]:
	"""Convert a raw state row into the canonical, leakage-free diagnosis and image descriptor contract."""
	_require(isinstance(raw, Mapping), "state row must be an object")
	target_prompt = normalize_text(raw.get("target_prompt") or raw.get("prompt"))
	_require(bool(target_prompt), "state target_prompt is required")
	image_value = raw.get("current_image") or raw.get("selected_image") or raw.get("source_image")
	_require(bool(image_value), "state current_image/source_image is required")
	image_path = Path(str(image_value))
	if not image_path.is_absolute():
		image_path = project_root / image_path
	raw_diagnosis = raw.get("diagnosis") or raw.get("analysis") or raw
	diagnosis = canonicalize_diagnosis(raw_diagnosis)
	_require(not diagnosis["fully_satisfied"], "router data only accepts states that need repair; L0 belongs to the gate dataset")
	_require(bool(diagnosis["issues"]), "a problematic V10 diagnosis must contain at least one factual issue")
	initial_value = raw.get("initial_generator") or {}
	if isinstance(initial_value, str):
		initial_value = {"family": initial_value}
	_require(isinstance(initial_value, Mapping), "initial_generator must be an object or string")
	try:
		initial_family = normalize_initial_generator_family(
			initial_value.get("family") or raw.get("initial_backend") or "unknown"
		)
	except ValueError as exc:
		raise DataValidationError(str(exc)) from exc
	if config.require_flux2_initial_state:
		_require(initial_family == "flux2", f"profile requires Flux.2 initial states, got {initial_family!r}")
	_require(
		initial_family == config.l3_policy.family,
		f"state initial generator {initial_family!r} differs from the dataset L3 executor {config.l3_policy.family!r}; "
		"mixed initial bases require separate datasets/route heads",
	)
	lineage_id = normalize_text(raw.get("lineage_id") or raw.get("trajectory_id") or raw.get("sample_id"))
	_require(bool(lineage_id), "state lineage_id/trajectory_id/sample_id is required")
	prompt_family_id = normalize_text(raw.get("prompt_family_id") or raw.get("base_prompt_id"))
	group_payload = (
		{"prompt_family_id": prompt_family_id}
		if prompt_family_id
		else {"prompt_identity": normalize_prompt_for_group(target_prompt)}
	)
	# Split identity is derived here and cannot be overridden by a legacy row.
	# This conservatively keeps every repeat/trajectory of one prompt together.
	group_id = stable_id("group", group_payload, length=32)
	provided_group_id = normalize_text(raw.get("group_id"))
	source = image_descriptor(image_path)
	plan_value = raw.get("l1_action_plan") or raw.get("l1_plan")
	if not plan_value and isinstance(raw_diagnosis, Mapping) and raw_diagnosis.get("edits"):
		# Migration-only: legacy reflections carried executable edits in diagnosis.
		plan_value = {"edits": raw_diagnosis.get("edits")}
	l1_action_plan = canonicalize_l1_action_plan(
		plan_value,
		target_prompt=target_prompt,
		diagnosis=diagnosis,
		width=source["width"],
		height=source["height"],
	)
	initial_model_id = normalize_text(initial_value.get("model_id") or raw.get("initial_model"))
	initial_revision = normalize_text(initial_value.get("model_revision"))
	initial_auxiliary_model_id = normalize_text(initial_value.get("auxiliary_model_id"))
	initial_auxiliary_revision = normalize_text(initial_value.get("auxiliary_model_revision"))
	initial_seed = initial_value.get("seed")
	l3_executor = config.l3_policy.contract["executor"]
	exact_l3_identity = (
		initial_model_id == l3_executor["model_id"]
		and initial_revision == l3_executor["model_revision"]
		and initial_auxiliary_model_id == l3_executor["auxiliary_model_id"]
		and initial_auxiliary_revision == l3_executor["auxiliary_model_revision"]
	)
	if config.require_flux2_initial_state:
		_require(initial_model_id == config.policy.model_id, "initial Flux.2 model_id must match the locked action model_id")
		_require(initial_revision == config.policy.model_revision, "initial Flux.2 model_revision must match the locked action revision")
		_require(isinstance(initial_seed, int) and not isinstance(initial_seed, bool), "initial Flux.2 seed must be an integer")
	initial_record_value = raw.get("initial_record") or {}
	_require(isinstance(initial_record_value, Mapping), "initial_record must be an object")
	legacy_external_identity = (
		initial_family == "bagel"
		and normalize_text(initial_value.get("provenance_quality")) == "legacy_partial_model_identity"
		and not initial_model_id
		and not initial_revision
		and not initial_auxiliary_model_id
		and not initial_auxiliary_revision
		and initial_record_value.get("executor_family") == "verified_external"
		and initial_record_value.get("reproducible_generation") is False
	)
	if not exact_l3_identity:
		_require(
			config.allow_legacy_external_l3_identity and legacy_external_identity,
			"initial model identity must exactly match the locked L3 executor; only explicitly enabled, "
			"hash-frozen legacy BAGEL assets with legacy_partial_model_identity may use family-only binding",
		)
	l3_identity_binding = "exact_model_identity" if exact_l3_identity else "legacy_bagel_family_only_unverified_revision"
	if config.require_verified_initial_record:
		_require(bool(initial_record_value), "a verified initial_record is required")
		_require(initial_record_value.get("status") == "ok", "initial_record status must be ok")
		_require(initial_record_value.get("executor_family") == initial_family, "initial_record executor family mismatch")
		_require(initial_record_value.get("model_id") == initial_model_id, "initial_record model_id mismatch")
		_require(initial_record_value.get("model_revision") == initial_revision, "initial_record revision mismatch")
		_require(initial_record_value.get("seed") == initial_seed, "initial_record seed mismatch")
		_require(initial_record_value.get("output_sha256") == source["sha256"], "initial_record output hash mismatch")
		_require(bool(initial_record_value.get("job_id") and initial_record_value.get("request_hash")), "initial_record job identity missing")
	state_payload = {
		"group_id": group_id,
		"lineage_id": lineage_id,
		"target_prompt": target_prompt,
		"source_sha256": source["sha256"],
		"diagnosis": diagnosis,
	}
	state_id = normalize_text(raw.get("state_id")) or stable_id("state", state_payload)
	return {
		"schema_version": SCHEMA_VERSION,
		"state_id": state_id,
		"sample_id": normalize_text(raw.get("sample_id")) or state_id,
		"group_id": group_id,
		"provided_group_id": provided_group_id,
		"lineage_id": lineage_id,
		"dataset": normalize_text(raw.get("dataset") or "unknown"),
		"source_prompt_id": normalize_text(raw.get("source_prompt_id")),
		"prompt_family_id": prompt_family_id,
		"variant_id": normalize_text(raw.get("variant_id")),
		"error_family": normalize_text(raw.get("error_family") or raw.get("geneval_tag") or "unknown"),
		"source_round": int(raw.get("source_round", raw.get("round", 0)) or 0),
		"target_prompt": target_prompt,
		"current_image": source,
		"initial_generator": {
			"family": initial_family,
			"model_id": initial_model_id,
			"model_revision": initial_revision,
			"auxiliary_model_id": initial_auxiliary_model_id,
			"auxiliary_model_revision": initial_auxiliary_revision,
			"seed": initial_seed,
			"l3_identity_binding": l3_identity_binding,
		},
		"initial_record": dict(initial_record_value),
		"diagnosis": diagnosis,
		"diagnosis_sha256": sha256_json(diagnosis),
		"l1_action_plan": l1_action_plan,
		"l1_action_plan_sha256": sha256_json(l1_action_plan) if l1_action_plan else None,
		"provenance": dict(raw.get("provenance") or {}),
	}


def validate_action_job(job: Mapping[str, Any], config: DatasetConfig) -> None:
	"""Validate that an action job matches its state, policy, prompt, and expected repair level."""
	_require(job.get("schema_version") == SCHEMA_VERSION, "job schema_version mismatch")
	level = int(job.get("action_level", -1))
	_require(level in ACTION_LEVELS, f"invalid action level: {level}")
	_require(job.get("action_name") == ACTION_NAMES[level], f"L{level} action_name mismatch")
	_require(job.get("operation") == ACTION_OPERATIONS[level], f"L{level} operation mismatch")
	expected_executor = "flux2" if level in {1, 2} else config.l3_policy.family
	_require(job.get("executor_family") == expected_executor, f"L{level} executor family mismatch")
	_require(job.get("policy_signature") == config.action_policy_signature, "job policy_signature mismatch")
	_require(
		job.get("flux2_policy_digest") == config.policy.collector_contract_digest,
		"job Flux.2 collector-policy digest mismatch",
	)
	_require(job.get("policy") == config.policy.as_dict(), "job policy does not match locked Flux.2 policy")
	_require(job.get("l3_policy_contract") == config.l3_policy.as_dict(), "job L3 policy contract mismatch")
	_require(job.get("l3_policy_digest") == config.l3_policy.digest, "job L3 policy digest mismatch")
	initial_generator = job.get("initial_generator")
	_require(isinstance(initial_generator, Mapping), "job initial_generator is required")
	_require(
		normalize_initial_generator_family(initial_generator.get("family")) == config.l3_policy.family,
		"job initial generator differs from locked L3 executor",
	)
	_require(int(job.get("rollout_index", -1)) >= 0, "rollout_index must be non-negative")
	_require(int(job.get("seed", 0)) > 0, "job seed must be positive")
	_require(job.get("action_seed") == job.get("seed"), "action_seed must match the paired rollout seed")
	_require(int(job.get("round_index", -1)) >= 0, "round_index must be non-negative")
	_require(job.get("action_plan_schema") == ACTION_PLAN_SCHEMA, f"L{level} action plan schema mismatch")
	_require(bool(normalize_text(job.get("action_plan_sha256"))), f"L{level} action plan hash is required")
	refs = job.get("reference_roles")
	if level == 1:
		edits = job.get("edits")
		expected_refs = ["editable_full_image"] + [f"localization_crop_{index + 1:02d}" for index in range(len(edits or []))]
		_require(refs == expected_refs, "L1 reference order must be editable full image followed by ordered localization crops")
		_require(isinstance(edits, list) and 1 <= len(edits) <= config.policy.l1_max_edits, "L1 job needs 1..l1_max_edits edits")
		_require(bool(normalize_text(job.get("prompt"))), "L1 joint prompt is required")
		for edit_index, edit in enumerate(edits):
			_require(isinstance(edit, Mapping), f"L1 edit {edit_index} must be an object")
			_require(bool(normalize_text(edit.get("instruction"))), f"L1 edit {edit_index} needs instruction")
			canonical_quad(edit.get("quad_points"), f"L1 edit {edit_index}.quad_points")
	elif level == 2:
		_require(refs == ["full_image"], "L2 must use exactly one full-image reference")
		_require(bool(normalize_text(job.get("prompt"))), "L2 prompt is required")
		_require(job.get("edits") == [], "L2 must not contain local edits")
	else:
		_require(refs == [], "L3 must be prompt-only with no image reference")
		_require(bool(normalize_text(job.get("prompt"))), "L3 prompt is required")
		_require(job.get("edits") == [], "L3 must not contain local edits")
		_require(job.get("executor_family") == normalize_initial_generator_family(initial_generator.get("family")), "L3 executor must follow initial generator")


def canonical_feature_diagnosis(state: Mapping[str, Any]) -> str:
	"""Return the canonical internal diagnosis stored with a route-feature row."""

	return canonical_diagnosis_text(state["diagnosis"])


def canonical_router_diagnosis_text(value: Any, *, already_canonical: bool = False) -> str:
	"""Shared training/online text schema for the frozen-Qwen route encoder.

	Online inference should parse the reflection JSON and call this function
	before ``predict_routes`` instead of encoding the model's raw response bytes.
	"""

	diagnosis = dict(value) if already_canonical else canonicalize_diagnosis(value)
	return model_visible_diagnosis_text(diagnosis)
